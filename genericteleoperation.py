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
from config import Configuration  # Updated import


class GenericTeleoperation:
    """
    @brief Teleoperation system for Panda and UR5 robotic arms
    @details Provides complete 6DOF control using inverse kinematics
             with smooth filtering and gripper control.
    """

    def __init__(self, robot_name='panda'):
        """
        @brief Initialize teleoperation system for specified robot

        @param robot_name: Name of the robot ('panda' or 'ur5')

        @note Default robot is 'panda' if not specified
        """
        self.robot_name = robot_name
        self.running = False
        self.sim = None
        self.joystick = None
        self.movement = None
        self.ik_solver = None

        # Control state for smooth operation
        self.filtered_twist = np.zeros(6)
        self.gripper_state = "closed"
        self.last_a_state = False

        # OS detection
        self.is_linux = platform.system() == "Linux"

        # Center offsets for Linux DInput
        self.axis_center = []

        # Get robot-specific configuration
        self.robot_config = Configuration.get_robot_config(robot_name)
        self.scales = Configuration.get_movement_scales(robot_name)

    def initialize_systems(self):
        """
        @brief Initialize all required systems for teleoperation

        @return: True if all systems initialized successfully, False otherwise

        @throws RuntimeError: If no gamepad detected

        @note Initializes pygame, gamepad, simulation, and control systems
        """
        print(f"Initializing teleoperation for {self.robot_name}...")

        # Initialize pygame and gamepad
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected!")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Gamepad connected: {self.joystick.get_name()}")
        self.layout = self.detect_gamepad_layout()

        # If Linux + DirectInput, calibrate axis centers
        if self.is_linux and self.layout == "dinput":
            self.axis_center = [self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())]
            print(f"Linux DInput axis centers calibrated: {self.axis_center}")

        # Initialize simulation
        self.sim = Simulation(robot_name=self.robot_name, show_viewer=True)

        # Get end effector and initialize systems
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
        """
        @brief Detect gamepad type based on axis count

        @return: 'xinput' or 'dinput'
        """
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
        """
        @brief Apply robot-specific axis remapping to twist command

        @param twist: Original twist command [vx, vy, vz, roll, pitch, yaw]
        @return: Remapped twist command based on robot configuration
        """
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
        @details Supports XInput (Windows) and DirectInput (Linux) controllers
                 with Linux axis center calibration.
        @note Applies deadzone filtering and axis normalization
        """
        DEADZONE_THRESHOLD = self.scales['deadzone_threshold']

        def deadzone(value):
            return 0.0 if abs(value) < DEADZONE_THRESHOLD else value

        # -------------------------------------------------------
        #      AXIS MAPPING
        # -------------------------------------------------------
        if self.layout == "xinput":
            left_x = deadzone(-self.joystick.get_axis(0))
            left_y = deadzone(self.joystick.get_axis(1))
            right_x = deadzone(-self.joystick.get_axis(2))
            right_y = deadzone(self.joystick.get_axis(3))
            l2_raw = (self.joystick.get_axis(4) + 1) / 2
            r2_raw = (self.joystick.get_axis(5) + 1) / 2
        else:  # dinput (Linux)
            # Read raw axes
            left_x_raw = self.joystick.get_axis(0)
            left_y_raw = self.joystick.get_axis(1)
            right_x_raw = self.joystick.get_axis(2)
            right_y_raw = self.joystick.get_axis(3)

            # Apply Linux axis center calibration if available
            if getattr(self, 'is_linux', False) and getattr(self, 'axis_center', None):
                left_x = left_x_raw - self.axis_center[0]
                left_y = left_y_raw - self.axis_center[1]
                right_x = right_x_raw - self.axis_center[2]
                right_y = right_y_raw - self.axis_center[3]
            else:
                left_x = left_x_raw
                left_y = left_y_raw
                right_x = right_x_raw
                right_y = right_y_raw

            # Apply deadzone
            left_x = -deadzone(left_x)
            left_y = deadzone(left_y)
            right_x = deadzone(right_x)
            right_y = deadzone(right_y)

            # Triggers are digital buttons
            l2_raw = float(self.joystick.get_button(6))
            r2_raw = float(self.joystick.get_button(7))

        # -------------------------------------------------------
        # Deadzone for triggers
        # -------------------------------------------------------
        l2 = 0.0 if l2_raw < DEADZONE_THRESHOLD else l2_raw
        r2 = 0.0 if r2_raw < DEADZONE_THRESHOLD else r2_raw

        # -------------------------------------------------------
        # Buttons
        # -------------------------------------------------------
        l1 = self.joystick.get_button(4)
        r1 = self.joystick.get_button(5)
        start = self.joystick.get_button(7) if self.layout == "xinput" else 0
        a_button = self.joystick.get_button(0)

        # -------------------------------------------------------
        # Compute twist
        # -------------------------------------------------------
        vx = right_y * self.scales['translation']
        vy = left_x * self.scales['translation']
        vz = left_y * self.scales['translation']
        roll = right_x * self.scales['rotation']
        pitch = (r1 - l1) * self.scales['tilt']
        yaw = -(r2 - l2) * self.scales['rotation']

        twist = np.array([vx, vy, vz, roll, pitch, yaw])

        return self.remap_twist(twist), start, a_button

    def process_movement(self, twist_command):
        """
        @brief Process movement command using inverse kinematics

        @param twist_command: 6D twist command for end-effector movement
        """
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
        """
        @brief Process gripper control input with toggle behavior

        @param a_button: Current state of A button
        """
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
        """
        @brief Main teleoperation loop
        """
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
        """
        @brief Clean up resources and shutdown systems
        """
        print("Cleaning up resources...")
        if self.sim and self.sim.show_viewer:
            self.sim.viewer.close()
        pygame.quit()
        print("Teleoperation ended.")


def main():
    """
    @brief Main function for direct execution of generic teleoperation
    """
    robot_name = 'panda'
    teleop_system = GenericTeleoperation(robot_name)
    teleop_system.run()


if __name__ == "__main__":
    main()
