"""
@file movement_helper.py
@brief Movement helper for pose integration and gripper control
@details Handles smooth movement calculations, pose integration,
         and gripper control for teleoperation systems.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from config import Configuration  # Updated import


class MovementHelper:
    """
    @brief Handles pose integration and movement calculations
    @details Provides smooth end-effector pose integration from twist commands
             and gripper control with acceleration limiting.
    """

    def __init__(self, sim, robot_name, dt=0.002, max_linear_speed=0.1, max_angular_speed=0.5):
        """
        @brief Initialize movement helper with stability settings

        @param sim: Simulation instance for accessing model and data
        @param robot_name: Name of the robot for configuration
        @param dt: Time step for integration (seconds)
        @param max_linear_speed: Maximum linear speed (m/s)
        @param max_angular_speed: Maximum angular speed (rad/s)
        """
        self.sim = sim
        self.robot_name = robot_name
        self.dt = dt
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

        # Initialize gripper control
        self.gripper_control_indices = self._find_gripper_controls()
        self.gripper_target, self.gripper_speed = 0.0, 255.0  # Start closed
        self.target_pos, self.target_quat = None, None

    def _find_gripper_controls(self):
        """
        @brief Find control indices for gripper actuators

        @return: List of actuator indices controlling the gripper

        @note Uses robot-specific naming conventions to identify gripper actuators
        """
        gripper_indices = []

        for i in range(self.sim.model.nu):
            # Extract actuator name from model
            name_id = self.sim.model.name_actuatoradr[i]
            name_bytes = bytearray()
            j = name_id
            while j < len(self.sim.model.names) and self.sim.model.names[j] != 0:
                name_bytes.append(self.sim.model.names[j])
                j += 1
            act_name = name_bytes.decode('utf-8') if name_bytes else f"actuator_{i}"

            # Robot-specific gripper detection
            if self.robot_name == 'so100':
                if 'Jaw' in act_name:
                    gripper_indices.append(i)
            else:
                # Generic gripper detection for Panda/UR5
                if any(keyword in act_name.lower() for keyword in ['finger', 'gripper', 'hand']):
                    gripper_indices.append(i)

        # Default to last actuator if none found
        if not gripper_indices:
            gripper_indices = [self.sim.model.nu - 1]

        return gripper_indices

    def set_initial_pose(self, pos, quat):
        """
        @brief Set initial end-effector pose for integration

        @param pos: Initial position as 3D vector [x, y, z]
        @param quat: Initial orientation as quaternion [w, x, y, z]
        """
        self.target_pos = np.array(pos, dtype=np.float64)
        self.target_quat = np.array(quat, dtype=np.float64)

    def integrate_twist(self, twist_command):
        """
        @brief Integrate twist command to update target pose

        @param twist_command: 6D twist [vx, vy, vz, wx, wy, wz] in end-effector frame
        @return: Tuple of (new_position, new_orientation)

        @note Applies speed limits and transforms velocities to world frame
        """
        if self.target_pos is None or self.target_quat is None:
            raise ValueError("Initial pose not set")

        # Extract and limit linear/angular velocities
        linear_vel = np.array(twist_command[:3])
        angular_vel = np.array(twist_command[3:])

        # Apply speed limits
        linear_speed = np.linalg.norm(linear_vel)
        if linear_speed > self.max_linear_speed:
            linear_vel = linear_vel * (self.max_linear_speed / linear_speed)

        angular_speed = np.linalg.norm(angular_vel)
        if angular_speed > self.max_angular_speed:
            angular_vel = angular_vel * (self.max_angular_speed / angular_speed)

        # Convert current orientation to rotation object
        current_rot = R.from_quat([self.target_quat[1], self.target_quat[2],
                                   self.target_quat[3], self.target_quat[0]])

        # Transform velocities to world frame
        linear_vel_world = current_rot.apply(linear_vel)
        new_pos = self.target_pos + linear_vel_world * self.dt

        # Update orientation using exponential map
        if np.linalg.norm(angular_vel) > 1e-6:
            angular_vel_world = current_rot.apply(angular_vel)
            delta_angle = angular_vel_world * self.dt
            delta_rotation = R.from_rotvec(delta_angle)
            new_rot = delta_rotation * current_rot
            new_quat_scipy = new_rot.as_quat()  # [x, y, z, w] format
            new_quat = np.array([new_quat_scipy[3], new_quat_scipy[0],
                                 new_quat_scipy[1], new_quat_scipy[2]])  # Convert to [w, x, y, z]
        else:
            new_quat = self.target_quat.copy()

        # Normalize quaternion and update target
        new_quat /= np.linalg.norm(new_quat)
        self.target_pos, self.target_quat = new_pos, new_quat

        return new_pos, new_quat

    def get_target_pose(self):
        """
        @brief Get current target pose

        @return: Tuple of (target_position, target_orientation)
        """
        return self.target_pos, self.target_quat

    def move_gripper(self, gripper_pos, speed):
        """
        @brief Set target gripper position and movement speed

        @param gripper_pos: Target position (0.0=closed, 1.0=open)
        @param speed: Movement speed (0-1 scale)

        @note Converts normalized position to actuator control range
        """
        max_speed = 255.0  # Maximum speed in actuator units per second
        self.gripper_target = np.clip(gripper_pos * 255, 0, 255)
        self.gripper_speed = np.clip(speed * max_speed, 0, max_speed)

    def update_gripper(self):
        """
        @brief Update gripper position towards target with speed control

        @note Moves gripper incrementally towards target position each call
        """
        if not self.gripper_control_indices:
            return

        for control_index in self.gripper_control_indices:
            if control_index >= len(self.sim.data.ctrl):
                continue

            current_pos = self.sim.data.ctrl[control_index]
            delta = self.gripper_target - current_pos

            # Skip if already at target
            if abs(delta) < 0.1:
                continue

            # Calculate step size based on speed and timestep
            step_size = self.gripper_speed * self.sim.model.opt.timestep

            # Move towards target with acceleration limiting
            if abs(delta) <= step_size:
                new_pos = self.gripper_target  # Snap to target if close
            else:
                new_pos = current_pos + np.sign(delta) * step_size

            self.sim.data.ctrl[control_index] = new_pos