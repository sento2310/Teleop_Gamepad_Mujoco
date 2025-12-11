"""
@file genericteleoperation.py
@brief Main teleoperation module for Panda and UR5 robots
@details Handles gamepad input processing, IK solving, and simulation control
         for 6DOF robotic arms using inverse kinematics.
"""
import sys
import pygame
import numpy as np
import platform
from simulation import Simulation
from generic_ik_solver import GenericVelocityIKSolver
from movement_helper import MovementHelper
from config import Configuration


class GenericTeleoperationLin:
    """
    @brief Teleoperation system for Panda and UR5 robotic arms
    @details Provides complete 6DOF control using inverse kinematics
             with smooth filtering and gripper control.
    """

    def __init__(self, robot_name='panda'):
        self.robot_name = robot_name
        self.running = False
        self.sim = None
        self.joystick = None
        self.movement = None
        self.ik_solver = None

        self.filtered_twist = np.zeros(6)
        self.gripper_state = "closed"
        self.last_a_state = False

        self.is_linux = platform.system() == "Linux"
        self.axis_center = []

        self.robot_config = Configuration.get_robot_config(robot_name)
        self.scales = Configuration.get_movement_scales(robot_name)

    def initialize_systems(self):
        print(f"Initializing teleoperation for {self.robot_name}...")

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected!")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Gamepad connected: {self.joystick.get_name()}")

        self.layout = self.detect_gamepad_layout()

        # Linux axis center calibration
        if self.is_linux:
            self.axis_center = [self.joystick.get_axis(i) for i in range(min(6, self.joystick.get_numaxes()))]
            print(f"Linux controller axis centers calibrated: {self.axis_center}")

        # Initialize simulation and movement helper
        self.sim = Simulation(robot_name=self.robot_name, show_viewer=True)
        ee_body = self.robot_config['end_effector_body']
        initial_pos, initial_quat = self.sim.get_object_state(ee_body)

        self.movement = MovementHelper(self.sim, robot_name=self.robot_name,
                                       dt=self.sim.model.opt.timestep)
        self.movement.set_initial_pose(initial_pos, initial_quat)

        self.ik_solver = GenericVelocityIKSolver(self.sim.model, self.sim.data, ee_body,
                                                 self.robot_config.get('arm_joint_count'))

        print("All systems initialized successfully")
        return True

    def detect_gamepad_layout(self):
        axis_count = self.joystick.get_numaxes()
        if axis_count >= 6:
            print("Detected XInput-style controller")
            return "xinput"
        elif axis_count == 4:
            print("Detected DirectInput-style controller")
            return "dinput"
        else:
            raise RuntimeError(f"Unknown controller axis count: {axis_count}")

    def remap_twist(self, twist):
        remap_rules = self.robot_config.get('axis_remap', {})
        if not remap_rules:
            return twist.copy()

        vx, vy, vz, roll, pitch, yaw = twist
        values = {
            'vx': vx, 'vy': vy, 'vz': vz, 'roll': roll, 'pitch': pitch, 'yaw': yaw,
            '-vx': -vx, '-vy': -vy, '-vz': -vz, '-roll': -roll, '-pitch': -pitch, '-yaw': -yaw,
            '0': 0.0
        }

        remapped_twist = np.array([
            values.get(remap_rules.get('vx', 'vx'), 0),
            values.get(remap_rules.get('vy', 'vy'), 0),
            values.get(remap_rules.get('vz', 'vz'), 0),
            values.get(remap_rules.get('roll', 'roll'), 0),
            values.get(remap_rules.get('pitch', 'pitch'), 0),
            values.get(remap_rules.get('yaw', 'yaw'), 0)
        ])
        return remapped_twist

    def get_twist_from_gamepad(self):
        """
        @brief Convert gamepad inputs to end-effector twist commands
        @details Handles both XInput and DInput layouts with Linux axis-center calibration.
                 Applies deadzone filtering and scaling.
        """
        DEADZONE_THRESHOLD = self.scales['deadzone_threshold']

        def deadzone(val):
            return 0.0 if abs(val) < DEADZONE_THRESHOLD else val

        # -----------------------------
        # Read raw axes/buttons per layout
        # -----------------------------
        if self.layout == "xinput":
            left_x_raw = self.joystick.get_axis(0)
            left_y_raw = self.joystick.get_axis(1)
            l2_raw = self.joystick.get_axis(2)
            right_x_raw = -self.joystick.get_axis(3)
            right_y_raw = self.joystick.get_axis(4)
            r2_raw = self.joystick.get_axis(5)
        else:  # dinput
            left_x_raw = self.joystick.get_axis(0)
            left_y_raw = self.joystick.get_axis(1)
            right_x_raw = -self.joystick.get_axis(2)
            right_y_raw = self.joystick.get_axis(3)
            l2_raw = float(self.joystick.get_button(6))
            r2_raw = float(self.joystick.get_button(7))

        # -----------------------------
        # Apply Linux axis-center calibration
        # -----------------------------
        if self.is_linux and self.axis_center:
            if self.layout == "xinput":
                left_x = left_x_raw - self.axis_center[0]
                left_y = left_y_raw - self.axis_center[1]
                l2 = l2_raw - self.axis_center[2]
                right_x = right_x_raw - self.axis_center[3]
                right_y = right_y_raw - self.axis_center[4]
                r2 = r2_raw - self.axis_center[5]
            else:  # dinput
                left_x = left_x_raw - self.axis_center[0]
                left_y = left_y_raw - self.axis_center[1]
                right_x = right_x_raw - self.axis_center[2]
                right_y = right_y_raw - self.axis_center[3]
                l2 = l2_raw  # buttons, no calibration
                r2 = r2_raw
        else:
            left_x, left_y, right_x, right_y = left_x_raw, left_y_raw, right_x_raw, right_y_raw
            l2 = l2_raw
            r2 = r2_raw

        # -----------------------------
        # Apply deadzone
        # -----------------------------
        left_x = -deadzone(left_x)  # invert left X for convention
        left_y = deadzone(left_y)
        right_x = deadzone(right_x)
        right_y = deadzone(right_y)
        l2 = deadzone(l2)
        r2 = deadzone(r2)

        # -----------------------------
        # Buttons
        # -----------------------------
        l1 = self.joystick.get_button(4)
        r1 = self.joystick.get_button(5)
        #start = self.joystick.get_button(7) if self.layout == "xinput" else 0
        a_button = self.joystick.get_button(0)

        # -----------------------------
        # Compute twist
        # -----------------------------
        vx = right_y * self.scales['translation']  # forward/backward
        vy = left_x * self.scales['translation']  # left/right
        vz = left_y * self.scales['translation']  # up/down
        roll = right_x * self.scales['rotation']  # roll
        pitch = (r1 - l1) * self.scales['tilt']  # tilt
        yaw = -(r2 - l2) * self.scales['rotation']  # rotation

        twist = np.array([vx, vy, vz, roll, pitch, yaw])

        # -----------------------------
        # Debug
        # -----------------------------
        print(f"[DEBUG] Raw axes: Lx={left_x_raw:.3f} Ly={left_y_raw:.3f} "
              f"Rx={right_x_raw:.3f} Ry={right_y_raw:.3f}")
        print(f"[DEBUG] Adjusted axes: Lx={left_x:.3f} Ly={left_y:.3f} "
              f"Rx={right_x:.3f} Ry={right_y:.3f}")
        print(f"[DEBUG] Triggers: L2={l2:.3f} R2={r2:.3f} | "
              f"L1={l1} R1={r1} START={start} A={a_button}")

        return self.remap_twist(twist), start, a_button

    def process_movement(self, twist_command):
        alpha = 0.3
        self.filtered_twist = (1 - alpha) * self.filtered_twist + alpha * twist_command

        if np.linalg.norm(self.filtered_twist) > 0.01:
            try:
                target_pos, target_quat = self.movement.integrate_twist(self.filtered_twist)
                target_joint_pos, success = self.ik_solver.solve(target_pos, target_quat)
                if success:
                    n_arm_joints = len(target_joint_pos)
                    actual_joints = min(n_arm_joints, len(self.sim.data.ctrl))
                    self.sim.data.ctrl[:actual_joints] = target_joint_pos[:actual_joints]
                else:
                    print("IK solution failed to converge")
            except Exception as e:
                print(f"Movement/IK error: {e}")

    def process_gripper(self, a_button):
        GRIPPER_OPEN_POS = self.scales['gripper_open_pos']
        GRIPPER_CLOSE_POS = self.scales['gripper_close_pos']
        GRIPPER_SPEED = self.scales['gripper_speed']

        if a_button and not self.last_a_state:
            if self.gripper_state == "open":
                self.movement.move_gripper(GRIPPER_CLOSE_POS, GRIPPER_SPEED)
                self.gripper_state = "closed"
                print("Gripper closing")
            else:
                self.movement.move_gripper(GRIPPER_OPEN_POS, GRIPPER_SPEED)
                self.gripper_state = "open"
                print("Gripper opening")
        self.last_a_state = a_button
        self.movement.update_gripper()

    def run(self):
        if not self.initialize_systems():
            return

        self.running = True
        if self.layout == "xinput":
            print(f"{self.robot_name.upper()} Teleoperation active. Press START to exit.")

        try:
            while self.running:
                if sys.platform != 'darwin':
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False

                twist, start, a_button = self.get_twist_from_gamepad()

                if start:
                    print("Exiting teleoperation...")
                    break

                self.process_movement(twist)
                self.process_gripper(a_button)

                self.sim.step()

        except KeyboardInterrupt:
            print("Teleoperation interrupted by user")
        except Exception as e:
            print(f"Error in teleoperation loop: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        print("Cleaning up resources...")
        if self.sim and self.sim.show_viewer:
            self.sim.viewer.close()
        pygame.quit()
        print("Teleoperation ended.")


def main():
    robot_name = 'panda'
    teleop_system = GenericTeleoperationLin(robot_name)
    teleop_system.run()


if __name__ == "__main__":
    main()
