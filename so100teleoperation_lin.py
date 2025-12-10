"""
@file so100teleoperation_lin.py
@brief SO100-specific teleoperation supporting XInput and DirectInput (Linux)
@details Implements hybrid control using joint-space for rotations/wrist and
         IK for translations with conflict prevention. Works with XInput and DInput.
"""
import sys
import pygame
import numpy as np
from simulation import Simulation
from movement_helper import MovementHelper
from config import Configuration
from so100_ik_solver import SO100IKSolver
import mujoco


class SO100TeleoperationLin:
    """
    @brief SO100-specific teleoperation system for Linux
    @details Hybrid control: joint-space rotations/wrist + IK translations.
             Supports XInput and DirectInput gamepads.
    """

    def __init__(self):
        self.robot_name = 'so100'
        self.running = False
        self.sim = None
        self.joystick = None
        self.joint_controller = None
        self.movement = None
        self.ik_solver = None

        self.gripper_state = "closed"
        self.last_a_state = False
        self.layout = None
        self.is_linux = sys.platform.startswith('linux')
        self.axis_center = []

        self.robot_config = Configuration.get_robot_config(self.robot_name)
        self.scales = Configuration.get_movement_scales(self.robot_name)
        self.joint_multipliers = Configuration.get_joint_multipliers(self.robot_name)

    def initialize_systems(self):
        print("Initializing SO100 teleoperation (Linux)...")

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected!")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Gamepad connected: {self.joystick.get_name()}")

        # Detect controller layout
        axis_count = self.joystick.get_numaxes()
        if axis_count >= 6:
            self.layout = "xinput"
            print("Detected XInput-style controller")
        else:
            self.layout = "dinput"
            print("Detected DirectInput-style controller")

        # Linux axis center calibration for DInput
        if self.is_linux:
            num_axes = min(self.joystick.get_numaxes(), 6 if self.layout == "xinput" else 4)
            self.axis_center = [self.joystick.get_axis(i) for i in range(num_axes)]
            print(f"Linux controller axis centers calibrated: {self.axis_center}")

        # Initialize simulation
        self.sim = Simulation(robot_name=self.robot_name, show_viewer=True)

        # Initialize joint controller
        self.joint_controller = SO100JointController(self.sim)

        # Initialize IK and movement helper
        ee_body = self.robot_config['end_effector_body']
        self.ik_solver = SO100IKSolver(self.sim.model, self.sim.data, ee_body, self.joint_controller)
        initial_pos, initial_quat = self.sim.get_object_state(ee_body)
        self.movement = MovementHelper(self.sim, robot_name=self.robot_name,
                                       dt=self.sim.model.opt.timestep)
        self.movement.set_initial_pose(initial_pos, initial_quat)

        print("SO100 systems initialized successfully")
        return True

    def get_input_from_gamepad(self):
        """
        @brief Read gamepad inputs (XInput and DInput) with Linux calibration
        @return Dictionary with stick positions, buttons, triggers.
        """
        DEADZONE_THRESHOLD = self.scales['deadzone_threshold']

        def deadzone(val):
            return 0.0 if abs(val) < DEADZONE_THRESHOLD else val

        # -----------------------------
        # Read raw axes/buttons
        # -----------------------------
        if self.layout == "xinput":
            left_x_raw = self.joystick.get_axis(0)
            left_y_raw = self.joystick.get_axis(1)
            l2_raw = self.joystick.get_axis(2)
            right_x_raw = self.joystick.get_axis(3)
            right_y_raw = self.joystick.get_axis(4)
            r2_raw = self.joystick.get_axis(5)
        else:  # dinput
            left_x_raw = self.joystick.get_axis(0)
            left_y_raw = self.joystick.get_axis(1)
            right_x_raw = self.joystick.get_axis(2)
            right_y_raw = self.joystick.get_axis(3)
            l2_raw = float(self.joystick.get_button(6))
            r2_raw = float(self.joystick.get_button(7))

        # -----------------------------
        # Linux axis-center calibration
        # -----------------------------
        if self.is_linux and self.axis_center:
            if self.layout == "xinput":
                left_x  = left_x_raw - self.axis_center[0]
                left_y  = left_y_raw - self.axis_center[1]
                l2      = l2_raw - self.axis_center[2]
                right_x = right_x_raw - self.axis_center[3]
                right_y = right_y_raw - self.axis_center[4]
                r2      = r2_raw - self.axis_center[5]
            else:  # dinput
                left_x  = left_x_raw - self.axis_center[0]
                left_y  = left_y_raw - self.axis_center[1]
                right_x = right_x_raw - self.axis_center[2]
                right_y = right_y_raw - self.axis_center[3]
                l2 = l2_raw
                r2 = r2_raw
        else:
            left_x, left_y, right_x, right_y = left_x_raw, left_y_raw, right_x_raw, right_y_raw
            l2, r2 = l2_raw, r2_raw

        # -----------------------------
        # Apply deadzone
        # -----------------------------
        left_x = -deadzone(left_x)
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
        start = self.joystick.get_button(7) if self.layout == "xinput" else 0
        a_button = self.joystick.get_button(0)

        return {
            'left_x': left_x,
            'left_y': left_y,
            'right_x': right_x,
            'right_y': right_y,
            'l1': l1, 'r1': r1, 'l2': l2, 'r2': r2,
            'start': start,
            'a_button': a_button
        }

    def process_joint_control(self, input_data):
        self.joint_controller.control_rotation_joint(input_data['left_x'])
        self.joint_controller.control_wrist_roll(input_data['l2'], input_data['r2'])
        self.joint_controller.control_wrist_pitch(input_data['l1'], input_data['r1'])

    def process_translation_control(self, input_data):
        if abs(input_data['left_x']) > 0.2:
            return
        if abs(input_data['right_y']) > 0.01 or abs(input_data['left_y']) > 0.01:
            ee_body = self.robot_config['end_effector_body']
            current_pos, _ = self.sim.get_object_state(ee_body)
            target_pos = self.calculate_pure_translation(current_pos, input_data['right_y'],
                                                        input_data['left_y'], self.scales['translation'],
                                                        self.sim.model.opt.timestep)
            rotation_joint_pos = self.get_rotation_joint_position()
            position_joint_values, success = self.ik_solver.solve_position_only(
                target_pos,
                self.sim.data.ctrl[self.joint_controller.joint_map.get('rotation', 0)],
                self.sim.data.ctrl[self.joint_controller.joint_map.get('wrist_roll', 0)],
                self.sim.data.ctrl[self.joint_controller.joint_map.get('wrist_pitch', 0)]
            )
            if success:
                for joint_name, joint_value in position_joint_values.items():
                    if joint_name in self.joint_controller.joint_map:
                        self.sim.data.ctrl[self.joint_controller.joint_map[joint_name]] = joint_value

    def calculate_pure_translation(self, current_pos, right_y, left_y, translation_scale, dt):
        """
        Calculate end-effector target position based on right stick Y and left stick Y.
        Movement along right_y is relative to the robot rotation joint to maintain local forward/back.
        """
        target_pos = np.array(current_pos, dtype=np.float64)
        rotation_joint_pos = self.get_rotation_joint_position()

        # Forward/backward relative to rotation joint
        if abs(right_y) > 0.01:
            horizontal_vec = np.array([current_pos[0] - rotation_joint_pos[0],
                                       current_pos[1] - rotation_joint_pos[1]])
            horizontal_dist = np.linalg.norm(horizontal_vec)
            if horizontal_dist > 0.001:
                horizontal_dir = horizontal_vec / horizontal_dist
                movement = right_y * translation_scale * dt * 50
                target_pos[0] += horizontal_dir[0] * movement
                target_pos[1] += horizontal_dir[1] * movement
            else:
                target_pos[0] += right_y * translation_scale * dt * 50

        # Vertical movement along Z-axis
        if abs(left_y) > 0.01:
            target_pos[2] += -left_y * translation_scale * dt * 50

        return target_pos

    def get_rotation_joint_position(self):
        rotation_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "Rotation_Pitch")
        if rotation_body_id != -1:
            return self.sim.data.xpos[rotation_body_id].copy()
        return np.array([0.0, 0.0, 0.0])

    def process_gripper(self, a_button):
        GRIPPER_OPEN_POS = self.scales['gripper_open_pos']
        GRIPPER_CLOSE_POS = self.scales['gripper_close_pos']
        GRIPPER_SPEED = self.scales['gripper_speed']

        if a_button and not self.last_a_state:
            target_pos = GRIPPER_OPEN_POS if self.gripper_state == "closed" else GRIPPER_CLOSE_POS
            self.movement.move_gripper(target_pos, GRIPPER_SPEED)
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            print(f"Gripper: {self.gripper_state.upper()}")

        self.last_a_state = a_button
        self.movement.update_gripper()

    def run(self):
        if not self.initialize_systems():
            return

        self.running = True
        print("SO100 Teleoperation Linux active. Press START (XInput) to exit.")

        try:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                input_data = self.get_input_from_gamepad()
                if self.layout == "xinput" and input_data['start']:
                    print("Exiting teleoperation...")
                    break

                self.process_joint_control(input_data)
                self.process_translation_control(input_data)
                self.process_gripper(input_data['a_button'])

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
        print("SO100 teleoperation ended.")


class SO100JointController:
    def __init__(self, sim):
        self.sim = sim
        self.robot_name = 'so100'
        self.movement_scales = Configuration.get_movement_scales(self.robot_name)
        self.joint_multipliers = Configuration.get_joint_multipliers(self.robot_name)
        self.joint_map = self._build_joint_map()

    def _build_joint_map(self):
        joint_map = {}
        for i in range(self.sim.model.nu):
            name_bytes = bytearray()
            j = self.sim.model.name_actuatoradr[i]
            while j < len(self.sim.model.names) and self.sim.model.names[j] != 0:
                name_bytes.append(self.sim.model.names[j])
                j += 1
            act_name = name_bytes.decode('utf-8') if name_bytes else f"actuator_{i}"
            if 'Rotation' in act_name:
                joint_map['rotation'] = i
            elif 'Pitch' in act_name and 'Wrist' not in act_name:
                joint_map['pitch'] = i
            elif 'Elbow' in act_name:
                joint_map['elbow'] = i
            elif 'Wrist_Pitch' in act_name:
                joint_map['wrist_pitch'] = i
            elif 'Wrist_Roll' in act_name:
                joint_map['wrist_roll'] = i
            elif 'Jaw' in act_name:
                joint_map['gripper'] = i
        return joint_map

    def control_rotation_joint(self, left_stick_x):
        if 'rotation' in self.joint_map:
            scale = -self.movement_scales['rotation'] * self.joint_multipliers['rotation']
            current = self.sim.data.ctrl[self.joint_map['rotation']]
            self.sim.data.ctrl[self.joint_map['rotation']] = current - left_stick_x * scale

    def control_wrist_roll(self, l2, r2):
        if 'wrist_roll' in self.joint_map:
            scale = -self.movement_scales['rotation'] * self.joint_multipliers['wrist_roll']
            current = self.sim.data.ctrl[self.joint_map['wrist_roll']]
            delta = (l2 - r2) * scale
            self.sim.data.ctrl[self.joint_map['wrist_roll']] = current + delta

    def control_wrist_pitch(self, l1, r1):
        if 'wrist_pitch' in self.joint_map:
            scale = self.movement_scales['tilt'] * self.joint_multipliers['wrist_pitch']
            current = self.sim.data.ctrl[self.joint_map['wrist_pitch']]
            delta = (l1 - r1) * scale
            self.sim.data.ctrl[self.joint_map['wrist_pitch']] = current + delta


def main():
    teleop_system = SO100TeleoperationLin()
    teleop_system.run()


if __name__ == "__main__":
    main()

