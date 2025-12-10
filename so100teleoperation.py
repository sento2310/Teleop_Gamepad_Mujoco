"""
@file so100teleoperation.py
@brief SO100-specific teleoperation with hybrid control scheme supporting XInput and DirectInput
@details Implements hybrid control using joint-space control for rotations/wrist and
         inverse kinematics for translations with conflict prevention.
"""
import sys
import pygame
import numpy as np
from simulation import Simulation
from movement_helper import MovementHelper
from config import Configuration
from so100_ik_solver import SO100IKSolver
import mujoco


class SO100Teleoperation:
    """
    @brief SO100-specific teleoperation system with hybrid control.
    @details Combines joint control for rotations and wrist with IK-based translations.
             Supports XInput (Windows) and DirectInput (Linux) controllers.
    """

    def __init__(self):
        """
        @brief Initialize SO100 teleoperation system.
        @note Loads robot configuration, movement scales, and joint multipliers.
        """
        self.robot_name = 'so100'
        self.running = False
        self.sim = None
        self.joystick = None
        self.joint_controller = None
        self.movement = None
        self.ik_solver = None

        # Control state
        self.gripper_state = "closed"
        self.last_a_state = False
        self.layout = None
        self.is_linux = False
        self.axis_center = None

        # Robot-specific configuration
        self.robot_config = Configuration.get_robot_config(self.robot_name)
        self.scales = Configuration.get_movement_scales(self.robot_name)
        self.joint_multipliers = Configuration.get_joint_multipliers(self.robot_name)

    def initialize_systems(self):
        """
        @brief Initialize all required systems for SO100 teleoperation.
        @return True if all systems initialized successfully, False otherwise.
        @throws RuntimeError if no gamepad is detected.
        @note Sets up pygame, joystick, simulation, joint controller, IK, and movement helper.
        """
        print("Initializing SO100 teleoperation...")

        # Initialize pygame and joystick
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
            print("Detected XInput-style controller (Xbox layout)")
        else:
            self.layout = "dinput"
            print("Detected DirectInput-style controller (Logitech D-mode)")

        # Linux DInput calibration
        self.is_linux = sys.platform.startswith('linux') and self.layout == 'dinput'
        if self.is_linux:
            self.axis_center = [self.joystick.get_axis(i) for i in range(4)]
            print(f"Linux DInput axis centers calibrated: {self.axis_center}")

        # Initialize simulation
        self.sim = Simulation(robot_name=self.robot_name, show_viewer=True)

        # Initialize joint controller
        self.joint_controller = SO100JointController(self.sim)

        # Initialize IK and movement systems
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
        @brief Read and normalize input from the gamepad.
        @return Dictionary containing processed input values:
                - left_x, left_y, right_y: stick positions
                - l1, r1, l2, r2: button/trigger states
                - start, a_button: special buttons
        @note Applies deadzone filtering and Linux axis center calibration.
        """
        DEADZONE_THRESHOLD = self.scales['deadzone_threshold']

        def deadzone(value):
            """Apply deadzone filtering to analog inputs."""
            return 0.0 if abs(value) < DEADZONE_THRESHOLD else value

        # XInput controller mapping
        if self.layout == "xinput":
            left_x = deadzone(-self.joystick.get_axis(0))
            left_y = deadzone(self.joystick.get_axis(1))
            right_y = deadzone(self.joystick.get_axis(3))
            l1, r1 = self.joystick.get_button(4), self.joystick.get_button(5)
            l2_raw = (self.joystick.get_axis(4) + 1) / 2
            r2_raw = (self.joystick.get_axis(5) + 1) / 2

        # DirectInput controller mapping (Linux)
        else:
            left_x_raw = self.joystick.get_axis(0)
            left_y_raw = self.joystick.get_axis(1)
            right_y_raw = self.joystick.get_axis(3)

            # Apply axis center calibration
            if self.is_linux and self.axis_center:
                left_x = left_x_raw - self.axis_center[0]
                left_y = left_y_raw - self.axis_center[1]
                right_y = right_y_raw - self.axis_center[3]
            else:
                left_x = left_x_raw
                left_y = left_y_raw
                right_y = right_y_raw

            left_x = deadzone(left_x)
            left_y = deadzone(left_y)
            right_y = deadzone(right_y)

            l1, r1 = self.joystick.get_button(4), self.joystick.get_button(5)
            l2_raw = float(self.joystick.get_button(6))
            r2_raw = float(self.joystick.get_button(7))

        l2 = 0.0 if l2_raw < DEADZONE_THRESHOLD else l2_raw
        r2 = 0.0 if r2_raw < DEADZONE_THRESHOLD else r2_raw

        start = self.joystick.get_button(7) if self.layout == "xinput" else 0
        a_button = self.joystick.get_button(0)



        return {
            'right_y': right_y,
            'left_y': left_y,
            'left_x': left_x,
            'l1': l1,
            'r1': r1,
            'l2': l2,
            'r2': r2,
            'start': start,
            'a_button': a_button
        }

    def process_joint_control(self, input_data):
        """
        @brief Process joint-space control for manual joints.
        @param input_data Dictionary from get_input_from_gamepad()
        """
        self.joint_controller.control_rotation_joint(input_data['left_x'])
        self.joint_controller.control_wrist_roll(input_data['l2'], input_data['r2'])
        self.joint_controller.control_wrist_pitch(input_data['l1'], input_data['r1'])

    def process_translation_control(self, input_data):
        """
        @brief Process translational control using IK with conflict prevention.
        @param input_data Dictionary from get_input_from_gamepad()
        @note Skips translation when significant rotation input is detected.
        """
        if abs(input_data['left_x']) > 0.2:
            return
        if abs(input_data['right_y']) > 0.01 or abs(input_data['left_y']) > 0.01:
            ee_body = self.robot_config['end_effector_body']
            current_pos, _ = self.sim.get_object_state(ee_body)
            target_pos = self.calculate_pure_translation(
                current_pos, input_data['right_y'], input_data['left_y'],
                self.scales['translation'], self.sim.model.opt.timestep
            )

            current_rotation = self.sim.data.ctrl[self.joint_controller.joint_map['rotation']]
            current_wrist_roll = self.sim.data.ctrl[self.joint_controller.joint_map['wrist_roll']]
            current_wrist_pitch = self.sim.data.ctrl[self.joint_controller.joint_map['wrist_pitch']]

            position_joint_values, success = self.ik_solver.solve_position_only(
                target_pos, current_rotation, current_wrist_roll, current_wrist_pitch
            )

            if success:
                for joint_name, joint_value in position_joint_values.items():
                    if joint_name in self.joint_controller.joint_map:
                        self.sim.data.ctrl[self.joint_controller.joint_map[joint_name]] = joint_value

    def calculate_pure_translation(self, current_pos, right_y, left_y, translation_scale, dt):
        """
        @brief Calculate target position using rotation joint as reference.
        @param current_pos Current end-effector position.
        @param right_y Right stick vertical input (horizontal movement).
        @param left_y Left stick vertical input (vertical movement).
        @param translation_scale Movement scaling factor.
        @param dt Time step.
        @return Target position as 3D numpy array.
        """
        target_pos = np.array(current_pos, dtype=np.float64)
        rotation_joint_pos = self.get_rotation_joint_position()

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

        if abs(left_y) > 0.01:
            target_pos[2] += -left_y * translation_scale * dt * 50

        return target_pos

    def get_rotation_joint_position(self):
        """
        @brief Get world position of rotation joint for movement reference.
        @return Position of rotation joint as 3D numpy array.
        """
        rotation_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "Rotation_Pitch")
        if rotation_body_id != -1:
            return self.sim.data.xpos[rotation_body_id].copy()
        rotation_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "Rotation")
        if rotation_joint_id != -1:
            body_id = self.sim.model.jnt_bodyid[rotation_joint_id]
            return self.sim.data.xpos[body_id].copy()
        base_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "Base")
        if base_body_id != -1:
            return self.sim.data.xpos[base_body_id].copy()
        return np.array([0.0, 0.0, 0.0])

    def process_gripper(self, a_button):
        """
        @brief Toggle gripper open/close based on A button.
        @param a_button Current state of the A button.
        """
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
        """
        @brief Main teleoperation loop.
        """
        if not self.initialize_systems():
            return

        self.running = True
        if self.layout == "xinput":
            print("SO100 Teleoperation active. Press START to exit.")

        try:
            while self.running:
                if sys.platform != 'darwin':
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
        """
        @brief Clean up resources and shutdown systems.
        """
        print("Cleaning up resources...")
        if self.sim and self.sim.show_viewer:
            self.sim.viewer.close()
        pygame.quit()
        print("SO100 teleoperation ended.")


class SO100JointController:
    """
    @brief Joint-space controller for SO100-specific movements.
    """

    def __init__(self, sim):
        self.sim = sim
        self.robot_name = 'so100'
        self.movement_scales = Configuration.get_movement_scales(self.robot_name)
        self.joint_multipliers = Configuration.get_joint_multipliers(self.robot_name)
        self.joint_map = self._build_joint_map()

    def _build_joint_map(self):
        """
        @brief Build mapping of joint names to simulation actuator indices.
        @return Dictionary mapping joint names to actuator indices.
        """
        joint_map = {}
        for i in range(self.sim.model.nu):
            name_id = self.sim.model.name_actuatoradr[i]
            name_bytes = bytearray()
            j = name_id
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
        """
        @brief Control rotation joint based on left stick X input.
        @param left_stick_x Horizontal stick input.
        """
        if 'rotation' in self.joint_map:
            scale = self.movement_scales['rotation'] * self.joint_multipliers['rotation']
            current = self.sim.data.ctrl[self.joint_map['rotation']]
            self.sim.data.ctrl[self.joint_map['rotation']] = current - left_stick_x * scale

    def control_wrist_roll(self, l2, r2):
        """
        @brief Control wrist roll using triggers L2 and R2.
        @param l2 L2 trigger value.
        @param r2 R2 trigger value.
        """
        if 'wrist_roll' in self.joint_map:
            roll_scale = self.movement_scales['rotation'] * self.joint_multipliers['wrist_roll']
            current = self.sim.data.ctrl[self.joint_map['wrist_roll']]
            delta = (l2 - r2) * roll_scale
            self.sim.data.ctrl[self.joint_map['wrist_roll']] = current + delta

    def control_wrist_pitch(self, l1, r1):
        """
        @brief Control wrist pitch using buttons L1 and R1.
        @param l1 L1 button state.
        @param r1 R1 button state.
        """
        if 'wrist_pitch' in self.joint_map:
            pitch_scale = self.movement_scales['tilt'] * self.joint_multipliers['wrist_pitch']
            current = self.sim.data.ctrl[self.joint_map['wrist_pitch']]
            delta = (l1 - r1) * pitch_scale
            self.sim.data.ctrl[self.joint_map['wrist_pitch']] = current + delta


def main():
    """
    @brief Entry point for direct script execution.
    """
    teleop_system = SO100Teleoperation()
    teleop_system.run()


if __name__ == "__main__":
    main()
