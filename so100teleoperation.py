"""
@file so100teleoperation.py
@brief SO100-specific teleoperation with hybrid control scheme
"""

import pygame
import numpy as np
from simulation import Simulation
from movement_helper import MovementHelper
from config import Configuration
from so100_ik_solver import SO100IKSolver
import mujoco


class SO100Teleoperation:
    """
    @brief SO100-specific teleoperation system with hybrid control
    @details Implements hybrid control scheme using joint control for rotations
             and inverse kinematics for translations with conflict prevention.
    """

    def __init__(self):
        """
        @brief Initialize SO100 teleoperation system

        @note Loads SO100-specific configuration and control parameters
        """
        self.robot_name = 'so100'
        self.running = False
        self.sim = None
        self.joystick = None
        self.joint_controller = None
        self.movement = None
        self.ik_solver = None

        # Control state management
        self.gripper_state = "closed"
        self.last_a_state = False

        # Get robot-specific configuration
        self.robot_config = Configuration.get_robot_config(self.robot_name)
        self.scales = Configuration.get_movement_scales(self.robot_name)
        self.joint_multipliers = Configuration.get_joint_multipliers(self.robot_name)  # Fixed this line

    def initialize_systems(self):
        """
        @brief Initialize all required systems for SO100 teleoperation

        @return: True if all systems initialized successfully, False otherwise

        @throws RuntimeError: If no gamepad detected

        @note Sets up pygame, simulation, joint controller, and IK systems
        """
        print("Initializing SO100 teleoperation...")

        # Initialize pygame and gamepad
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected!")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Gamepad connected: {self.joystick.get_name()}")

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
        @brief Get modified input structure for SO100 pure translational control

        @return: Dictionary containing all gamepad input values

        @note Applies deadzone filtering and returns structured input data
        """
        DEADZONE_THRESHOLD = self.scales['deadzone_threshold']

        def deadzone(value):
            """Apply deadzone filtering to analog inputs"""
            return 0.0 if abs(value) < DEADZONE_THRESHOLD else value

        # Get raw inputs with deadzone filtering
        left_y = deadzone(self.joystick.get_axis(1))    # Left stick vertical (pure Z)
        right_y = deadzone(self.joystick.get_axis(3))   # Right stick vertical (horizontal in/out)
        left_x = deadzone(-self.joystick.get_axis(0))   # Left stick horizontal (rotation)

        # Button states for joint control
        l1, r1 = self.joystick.get_button(4), self.joystick.get_button(5)
        l2_raw, r2_raw = (self.joystick.get_axis(4) + 1) / 2, (self.joystick.get_axis(5) + 1) / 2
        l2 = 0.0 if l2_raw < DEADZONE_THRESHOLD else l2_raw
        r2 = 0.0 if r2_raw < DEADZONE_THRESHOLD else r2_raw

        return {
            'right_y': right_y,      # Horizontal in/out relative to base
            'left_y': left_y,        # Pure vertical movement
            'left_x': left_x,        # Rotation joint control
            'l1': l1, 'r1': r1,      # Wrist pitch control
            'l2': l2, 'r2': r2,      # Wrist roll control
            'start': self.joystick.get_button(7),
            'a_button': self.joystick.get_button(0)
        }

    def process_joint_control(self, input_data):
        """
        @brief Process joint-space control for manual joints

        @param input_data: Dictionary containing gamepad input values

        @note Controls rotation, wrist_roll, and wrist_pitch joints directly
        """
        self.joint_controller.control_rotation_joint(input_data['left_x'])
        self.joint_controller.control_wrist_roll(input_data['l2'], input_data['r2'])
        self.joint_controller.control_wrist_pitch(input_data['l1'], input_data['r1'])

    def process_translation_control(self, input_data):
        """
        @brief Process translational control using IK with conflict prevention

        @param input_data: Dictionary containing gamepad input values

        @note Skips translation when significant rotation input is detected
              to prevent control conflicts between manual and IK control
        """
        # Conflict prevention: skip translation during active rotation
        if abs(input_data['left_x']) > 0.2:  # Rotation threshold
            return  # Ignore translation when rotating

        # Process translation only if significant input detected
        if abs(input_data['right_y']) > 0.01 or abs(input_data['left_y']) > 0.01:
            ee_body = self.robot_config['end_effector_body']
            current_pos, current_quat = self.sim.get_object_state(ee_body)

            # Calculate target position based on pure translation
            target_pos = self.calculate_pure_translation(
                current_pos, input_data['right_y'], input_data['left_y'],
                self.scales['translation'], self.sim.model.opt.timestep
            )

            # Preserve manual joint values for IK solution
            current_rotation = self.sim.data.ctrl[self.joint_controller.joint_map['rotation']]
            current_wrist_roll = self.sim.data.ctrl[self.joint_controller.joint_map['wrist_roll']]
            current_wrist_pitch = self.sim.data.ctrl[self.joint_controller.joint_map['wrist_pitch']]

            # Solve IK for position only with fixed manual joints
            position_joint_values, success = self.ik_solver.solve_position_only(
                target_pos, current_rotation, current_wrist_roll, current_wrist_pitch
            )

            if success:
                # Apply solved joint positions to simulation
                for joint_name, joint_value in position_joint_values.items():
                    if joint_name in self.joint_controller.joint_map:
                        self.sim.data.ctrl[self.joint_controller.joint_map[joint_name]] = joint_value

    def calculate_pure_translation(self, current_pos, right_y, left_y, translation_scale, dt):
        """
        @brief Calculate target position using rotation joint as reference

        @param current_pos: Current end-effector position
        @param right_y: Right stick Y input (horizontal movement)
        @param left_y: Left stick Y input (vertical movement)
        @param translation_scale: Movement scaling factor
        @param dt: Time step for integration

        @return: Target position as 3D vector

        @note Uses rotation joint position as reference for radial movement
        """
        target_pos = np.array(current_pos, dtype=np.float64)

        # Get rotation joint position for radial movement reference
        rotation_joint_pos = self.get_rotation_joint_position()

        # Horizontal movement along radial direction from rotation joint
        if abs(right_y) > 0.01:
            horizontal_vec = np.array([current_pos[0] - rotation_joint_pos[0],
                                     current_pos[1] - rotation_joint_pos[1]])
            horizontal_dist = np.linalg.norm(horizontal_vec)

            if horizontal_dist > 0.001:
                # Move along radial direction
                horizontal_dir = horizontal_vec / horizontal_dist
                movement = right_y * translation_scale * dt * 50
                target_pos[0] += horizontal_dir[0] * movement
                target_pos[1] += horizontal_dir[1] * movement
            else:
                # Fallback for zero distance
                target_pos[0] += right_y * translation_scale * dt * 50

        # Pure vertical movement (independent of rotation)
        if abs(left_y) > 0.01:
            target_pos[2] += -left_y * translation_scale * dt * 50

        return target_pos

    def get_rotation_joint_position(self):
        """
        @brief Get world position of rotation joint for movement reference

        @return: Position of rotation joint as 3D vector

        @note Attempts multiple methods to find rotation joint position
        """
        # Try different methods to find rotation joint position
        rotation_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "Rotation_Pitch")
        if rotation_body_id != -1:
            return self.sim.data.xpos[rotation_body_id].copy()

        rotation_joint_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, "Rotation")
        if rotation_joint_id != -1:
            body_id = self.sim.model.jnt_bodyid[rotation_joint_id]
            return self.sim.data.xpos[body_id].copy()

        # Fallback to base position
        base_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "Base")
        if base_body_id != -1:
            return self.sim.data.xpos[base_body_id].copy()

        return np.array([0.0, 0.0, 0.0])  # Default origin

    def process_gripper(self, a_button):
        """
        @brief Process gripper control input with toggle behavior

        @param a_button: Current state of A button

        @note Toggles gripper state on button press with configurable positions
        """
        GRIPPER_OPEN_POS = self.scales['gripper_open_pos']
        GRIPPER_CLOSE_POS = self.scales['gripper_close_pos']
        GRIPPER_SPEED = self.scales['gripper_speed']

        # Toggle gripper on A button press (rising edge detection)
        if a_button and not self.last_a_state:
            target_pos = GRIPPER_OPEN_POS if self.gripper_state == "closed" else GRIPPER_CLOSE_POS
            self.movement.move_gripper(target_pos, GRIPPER_SPEED)
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            print(f"Gripper: {self.gripper_state.upper()}")

        self.last_a_state = a_button
        self.movement.update_gripper()

    def run(self):
        """
        @brief Main teleoperation loop for SO100

        @note Handles input processing, control execution, and simulation stepping
              with comprehensive user feedback and error handling
        """
        if not self.initialize_systems():
            return

        self.running = True
        print("SO100 Teleoperation active. Press START to exit.")
        print("Controls:")
        print("  Left Stick X: Rotation joint")
        print("  Left Stick Y: Vertical movement (up/down)")
        print("  Right Stick Y: Horizontal movement (in/out)")
        print("  R1/L1: Wrist pitch")
        print("  R2/L2: Wrist roll")
        print("  A Button: Toggle gripper")

        try:
            while self.running:
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                # Get input data
                input_data = self.get_input_from_gamepad()

                # Exit on START button
                if input_data['start']:
                    print("Exiting teleoperation...")
                    break

                # Process all control inputs
                self.process_joint_control(input_data)
                self.process_translation_control(input_data)
                self.process_gripper(input_data['a_button'])

                # Advance simulation
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

        @note Closes viewer and quits pygame gracefully
        """
        print("Cleaning up resources...")
        if self.sim and self.sim.show_viewer:
            self.sim.viewer.close()
        pygame.quit()
        print("SO100 teleoperation ended.")


class SO100JointController:
    """
    @brief Joint-space controller for SO100-specific movements
    @details Provides direct joint control for rotation, wrist_pitch, and wrist_roll
             with scaling and multiplier application from configuration.
    """

    def __init__(self, sim):
        """
        @brief Initialize SO100 joint controller

        @param sim: Simulation instance for accessing model and data

        @note Builds joint mapping and loads configuration parameters
        """
        self.sim = sim
        self.robot_name = 'so100'
        self.movement_scales = Configuration.get_movement_scales(self.robot_name)
        self.joint_multipliers = Configuration.get_joint_multipliers(self.robot_name)  # Fixed this line
        self.joint_map = self._build_joint_map()

    def _build_joint_map(self):
        """
        @brief Map joint names to their control indices

        @return: Dictionary mapping joint names to actuator indices

        @note Uses actuator naming patterns to identify SO100 joints
        """
        joint_map = {}
        for i in range(self.sim.model.nu):
            # Extract actuator name from model
            name_id = self.sim.model.name_actuatoradr[i]
            name_bytes = bytearray()
            j = name_id
            while j < len(self.sim.model.names) and self.sim.model.names[j] != 0:
                name_bytes.append(self.sim.model.names[j])
                j += 1
            act_name = name_bytes.decode('utf-8') if name_bytes else f"actuator_{i}"

            # Map to expected joints based on naming patterns
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
        @brief Control base rotation joint

        @param left_stick_x: Left stick horizontal input value

        @note Applies movement scale and joint multiplier from configuration
        """
        if 'rotation' in self.joint_map:
            scale = self.movement_scales['rotation'] * self.joint_multipliers['rotation']
            current = self.sim.data.ctrl[self.joint_map['rotation']]
            self.sim.data.ctrl[self.joint_map['rotation']] = current - left_stick_x * scale

    def control_wrist_roll(self, l2, r2):
        """
        @brief Control wrist roll using triggers

        @param l2: Left trigger value (0.0 to 1.0)
        @param r2: Right trigger value (0.0 to 1.0)

        @note Differential control: l2 decreases, r2 increases roll
        """
        if 'wrist_roll' in self.joint_map:
            roll_scale = self.movement_scales['rotation'] * self.joint_multipliers['wrist_roll']
            current = self.sim.data.ctrl[self.joint_map['wrist_roll']]
            delta = (l2 - r2) * roll_scale
            self.sim.data.ctrl[self.joint_map['wrist_roll']] = current + delta

    def control_wrist_pitch(self, l1, r1):
        """
        @brief Control wrist pitch using shoulder buttons

        @param l1: Left shoulder button state (0 or 1)
        @param r1: Right shoulder button state (0 or 1)

        @note Differential control: l1 decreases, r1 increases pitch
        """
        if 'wrist_pitch' in self.joint_map:
            pitch_scale = self.movement_scales['tilt'] * self.joint_multipliers['wrist_pitch']
            current = self.sim.data.ctrl[self.joint_map['wrist_pitch']]
            delta = (l1 - r1) * pitch_scale
            self.sim.data.ctrl[self.joint_map['wrist_pitch']] = current + delta


def main():
    """
    @brief Legacy main function for direct execution without launcher

    @note Can be used for testing SO100 teleoperation independently
    """
    teleop_system = SO100Teleoperation()
    teleop_system.run()


if __name__ == "__main__":
    main()