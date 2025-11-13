"""
@file genericteleoperation.py
@brief Main teleoperation module for Panda and UR5 robots
@details Handles gamepad input processing, IK solving, and simulation control
         for 6DOF robotic arms using inverse kinematics.
"""

import pygame
import numpy as np
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

        # Available values for remapping
        values = {
            'vx': vx, 'vy': vy, 'vz': vz, 'roll': roll, 'pitch': pitch, 'yaw': yaw,
            '-vx': -vx, '-vy': -vy, '-vz': -vz, '-roll': -roll, '-pitch': -pitch, '-yaw': -yaw,
            '0': 0.0
        }

        # Apply remapping rules from configuration
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

        @return: Tuple of (twist_command, start_button, a_button_state)
                - twist_command: 6D twist vector [vx, vy, vz, roll, pitch, yaw]
                - start_button: Boolean indicating START button state
                - a_button_state: Boolean indicating A button state

        @note Applies deadzone filtering and axis scaling from configuration
        """
        DEADZONE_THRESHOLD = self.scales['deadzone_threshold']

        def deadzone(value):
            """Apply deadzone filtering to analog inputs"""
            return 0.0 if abs(value) < DEADZONE_THRESHOLD else value

        # Joystick inputs with deadzone filtering
        left_x = deadzone(-self.joystick.get_axis(0))
        left_y = deadzone(self.joystick.get_axis(1))
        right_x = deadzone(-self.joystick.get_axis(2))
        right_y = deadzone(self.joystick.get_axis(3))

        # Button states
        l1, r1 = self.joystick.get_button(4), self.joystick.get_button(5)
        start, a_button = self.joystick.get_button(7), self.joystick.get_button(0)

        # Trigger inputs with deadzone
        l2_raw, r2_raw = (self.joystick.get_axis(4) + 1) / 2, (self.joystick.get_axis(5) + 1) / 2
        l2 = 0.0 if l2_raw < DEADZONE_THRESHOLD else l2_raw
        r2 = 0.0 if r2_raw < DEADZONE_THRESHOLD else r2_raw

        # Calculate twist components with scaling
        vx = right_y * self.scales['translation']   # Forward/backward movement
        vy = left_x * self.scales['translation']    # Left/right movement
        vz = left_y * self.scales['translation']    # Up/down movement

        # Rotation components
        roll = right_x * self.scales['rotation']    # Roll from right stick
        pitch = (r1 - l1) * self.scales['tilt']     # Pitch from shoulder buttons
        yaw = (r2 - l2) * self.scales['rotation']   # Yaw from triggers

        twist = np.array([vx, vy, vz, roll, pitch, yaw])
        return self.remap_twist(twist), start, a_button

    def process_movement(self, twist_command):
        """
        @brief Process movement command using inverse kinematics

        @param twist_command: 6D twist command for end-effector movement

        @note Applies low-pass filtering for smooth movement and handles IK convergence
        """
        # Apply low-pass filter for smooth movement
        alpha = 0.3  # Smoothing factor
        self.filtered_twist = (1 - alpha) * self.filtered_twist + alpha * twist_command

        # Only process if significant movement command
        if np.linalg.norm(self.filtered_twist) > 0.01:
            try:
                # Integrate twist to get target pose
                target_pos, target_quat = self.movement.integrate_twist(self.filtered_twist)

                # Solve inverse kinematics for arm joints
                target_joint_pos, success = self.ik_solver.solve(target_pos, target_quat)

                if success:
                    # Apply joint positions to simulation
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

        @note Toggles gripper state on button press with configurable positions
        """
        GRIPPER_OPEN_POS = self.scales['gripper_open_pos']
        GRIPPER_CLOSE_POS = self.scales['gripper_close_pos']
        GRIPPER_SPEED = self.scales['gripper_speed']

        # Toggle gripper on A button press (rising edge)
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

        @note Handles gamepad input, movement processing, and simulation stepping
              Exits gracefully on START button or KeyboardInterrupt
        """
        if not self.initialize_systems():
            return

        self.running = True
        print(f"{self.robot_name.upper()} Teleoperation active. Press START to exit.")

        try:
            while self.running:
                # Process pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                # Get gamepad input
                twist, start, a_button = self.get_twist_from_gamepad()

                # Exit on START button
                if start:
                    print("Exiting teleoperation...")
                    break

                # Process movement and gripper commands
                self.process_movement(twist)
                self.process_gripper(a_button)

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
        print("Teleoperation ended.")


def main():
    """
    @brief Main function for direct execution of generic teleoperation

    @note Can be used for testing without the launcher system
    """
    # Choose between Panda and UR5
    robot_name = 'panda'  # Options: 'panda', 'ur5'

    # Create and run teleoperation system
    teleop_system = GenericTeleoperation(robot_name)
    teleop_system.run()


if __name__ == "__main__":
    main()