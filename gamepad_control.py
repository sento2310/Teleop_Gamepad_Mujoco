"""
@file gamepad_control.py
@brief Main launcher for robotic arm teleoperation system
@details Provides unified entry point with robot selection interface
         and system initialization for all supported robots.
"""

import sys
import os

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from genericteleoperation import GenericTeleoperation
from so100teleoperation import SO100Teleoperation
from config import Configuration


class GamepadControlLauncher:
    """
    @brief Main launcher class for robotic arm teleoperation systems
    @details Handles robot selection, system initialization, and provides
             command-line and interactive selection interfaces.
    """

    def __init__(self):
        """@brief Initialize the gamepad control launcher with available robots"""
        self.available_robots = ['panda', 'ur5', 'so100']
        self.selected_robot = None
        self.teleop_system = None

    def select_robot(self, robot_name):
        """
        @brief Select and validate robot for teleoperation

        @param robot_name: Name of the robot to control
        @return: True if selection valid and robot available, False otherwise

        @note Prints error message and available options if robot not found
        """
        if robot_name not in self.available_robots:
            print(f"Error: Robot '{robot_name}' not available.")
            print(f"Available robots: {', '.join(self.available_robots)}")
            return False

        self.selected_robot = robot_name
        robot_config = Configuration.get_robot_config(robot_name)  # Updated
        print(f"Selected robot: {robot_config['name']}")
        return True

    def initialize_teleop_system(self):
        """
        @brief Initialize appropriate teleoperation system based on selected robot

        @return: True if initialization successful, False otherwise

        @throws ImportError: If required teleoperation module cannot be imported
        @throws Exception: For general initialization failures

        @note Uses SO100-specific system for SO100, generic for Panda/UR5
        """
        if not self.selected_robot:
            print("Error: No robot selected. Please call select_robot() first.")
            return False

        try:
            if self.selected_robot == 'so100':
                # Use SO100-specific teleoperation
                from so100teleoperation import SO100Teleoperation
                self.teleop_system = SO100Teleoperation()
                print("Initialized SO100 teleoperation system")
            else:
                # Use generic teleoperation for Panda/UR5
                from genericteleoperation import GenericTeleoperation
                self.teleop_system = GenericTeleoperation(robot_name=self.selected_robot)
                print(f"Initialized Generic teleoperation system for {self.selected_robot}")

            return True

        except ImportError as e:
            print(f"Error importing teleoperation module: {e}")
            return False
        except Exception as e:
            print(f"Error initializing teleoperation system: {e}")
            return False

    def run(self):
        """
        @brief Execute the selected teleoperation system

        @note Handles KeyboardInterrupt and general exceptions gracefully
        """
        if not self.teleop_system:
            print("Error: Teleoperation system not initialized.")
            return

        try:
            print(f"\nStarting teleoperation for {self.selected_robot}...")
            print("Press Ctrl+C to exit gracefully")
            print("-" * 50)

            self.teleop_system.run()

        except KeyboardInterrupt:
            print("\nTeleoperation interrupted by user")
        except Exception as e:
            print(f"Error during teleoperation: {e}")
        finally:
            print("Teleoperation session ended.")

    def get_robot_info(self, robot_name):
        """
        @brief Retrieve basic information about a specific robot

        @param robot_name: Name of the robot to query
        @return: Dictionary with robot info or None if robot not found

        @note Returns None if robot configuration cannot be retrieved
        """
        try:
            config = Configuration.get_robot_config(robot_name)  # Updated
            return {
                'name': config['name'],
                'description': f"{config['name']} with {config['arm_joint_count']} arm joints",
                'end_effector': config['end_effector_body']
            }
        except:
            return None


def main():
    """
    @brief Main entry point for gamepad control application

    @details Supports both command-line arguments and interactive selection
             Displays available robots and usage instructions
    """
    launcher = GamepadControlLauncher()

    # Display available robots
    print("=" * 60)
    print("ROBOTIC ARM TELEOPERATION SYSTEM")
    print("=" * 60)
    print("Available robots:")

    for i, robot in enumerate(launcher.available_robots, 1):
        info = launcher.get_robot_info(robot)
        if info:
            print(f"  {i}. {robot.upper()} - {info['description']}")

    print("\nUsage options:")
    print("  1. Run directly: python gamepad_control.py [robot_name]")
    print("  2. Run interactively: python gamepad_control.py")
    print("=" * 60)

    # Check for command line argument
    if len(sys.argv) > 1:
        # Use command line argument
        robot_name = sys.argv[1].lower()
    else:
        # Interactive selection
        try:
            print("\nSelect a robot:")
            for i, robot in enumerate(launcher.available_robots, 1):
                info = launcher.get_robot_info(robot)
                print(f"  {i}. {robot.upper()}")

            choice = input(f"\nEnter choice (1-{len(launcher.available_robots)}): ").strip()
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(launcher.available_robots):
                    robot_name = launcher.available_robots[index]
                else:
                    print("Invalid choice. Using default (panda).")
                    robot_name = 'panda'
            else:
                print("Invalid input. Using default (panda).")
                robot_name = 'panda'

        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return

    # Validate and run
    if launcher.select_robot(robot_name):
        if launcher.initialize_teleop_system():
            launcher.run()
        else:
            print("Failed to initialize teleoperation system.")
    else:
        print("Robot selection failed.")


if __name__ == "__main__":
    main()