"""
Movement helper for pose integration and gripper control.
Handles smooth movement calculations for teleoperation.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from config import get_robot_config  # Updated import


class MovementHelper:
    """Handles pose integration and movement calculations."""

    def __init__(self, sim, robot_name, dt=0.002, max_linear_speed=0.1, max_angular_speed=0.5):
        """
        @brief Initialize movement helper with stability settings
        @param sim: Simulation instance
        @param robot_name: Name of the robot for configuration
        @param dt: Time step for integration
        @param max_linear_speed: Maximum linear speed
        @param max_angular_speed: Maximum angular speed
        """
        self.sim = sim
        self.robot_name = robot_name  # Store robot name
        self.dt = dt
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

        self.gripper_control_indices = self._find_gripper_controls()
        self.gripper_target, self.gripper_speed = 0.0, 255.0  # Start closed
        self.target_pos, self.target_quat = None, None

    def _find_gripper_controls(self):
        """@brief Find control indices for gripper actuators"""
        gripper_indices = []

        for i in range(self.sim.model.nu):
            # Extract actuator name
            name_id = self.sim.model.name_actuatoradr[i]
            name_bytes = bytearray()
            j = name_id
            while j < len(self.sim.model.names) and self.sim.model.names[j] != 0:
                name_bytes.append(self.sim.model.names[j])
                j += 1
            act_name = name_bytes.decode('utf-8') if name_bytes else f"actuator_{i}"

            # Robot-specific gripper detection
            if self.robot_name == 'so100':  # Use stored robot name
                if 'Jaw' in act_name:
                    gripper_indices.append(i)
            else:
                if any(keyword in act_name.lower() for keyword in ['finger', 'gripper', 'hand']):
                    gripper_indices.append(i)

        # Default to last actuator if none found
        if not gripper_indices:
            gripper_indices = [self.sim.model.nu - 1]

        return gripper_indices


    def set_initial_pose(self, pos, quat):
        """
        @brief Set initial end-effector pose
        @param pos: Initial position
        @param quat: Initial orientation quaternion
        """
        self.target_pos = np.array(pos, dtype=np.float64)
        self.target_quat = np.array(quat, dtype=np.float64)

    def integrate_twist(self, twist_command):
        """
        @brief Integrate twist command to update target pose
        @param twist_command: [vx, vy, vz, wx, wy, wz] in end-effector frame
        @return: New position and orientation
        """
        if self.target_pos is None or self.target_quat is None:
            raise ValueError("Initial pose not set")

        # Apply speed limits
        linear_vel = np.array(twist_command[:3])
        angular_vel = np.array(twist_command[3:])

        linear_speed = np.linalg.norm(linear_vel)
        if linear_speed > self.max_linear_speed:
            linear_vel = linear_vel * (self.max_linear_speed / linear_speed)

        angular_speed = np.linalg.norm(angular_vel)
        if angular_speed > self.max_angular_speed:
            angular_vel = angular_vel * (self.max_angular_speed / angular_speed)

        # Convert current orientation
        current_rot = R.from_quat([self.target_quat[1], self.target_quat[2],
                                   self.target_quat[3], self.target_quat[0]])

        # Transform velocities to world frame
        linear_vel_world = current_rot.apply(linear_vel)
        new_pos = self.target_pos + linear_vel_world * self.dt

        # Update orientation
        if np.linalg.norm(angular_vel) > 1e-6:
            angular_vel_world = current_rot.apply(angular_vel)
            delta_angle = angular_vel_world * self.dt
            delta_rotation = R.from_rotvec(delta_angle)
            new_rot = delta_rotation * current_rot
            new_quat_scipy = new_rot.as_quat()  # [x, y, z, w]
            new_quat = np.array([new_quat_scipy[3], new_quat_scipy[0],
                                 new_quat_scipy[1], new_quat_scipy[2]])  # [w, x, y, z]
        else:
            new_quat = self.target_quat.copy()

        # Normalize and update
        new_quat /= np.linalg.norm(new_quat)
        self.target_pos, self.target_quat = new_pos, new_quat

        return new_pos, new_quat

    def get_target_pose(self):
        """@brief Get current target pose"""
        return self.target_pos, self.target_quat

    def move_gripper(self, gripper_pos, speed):
        """
        @brief Set target gripper position and speed
        @param gripper_pos: Target position (0=closed, 1=open)
        @param speed: Movement speed (0-1 scale)
        """
        max_speed = 255.0  # Units per second
        self.gripper_target = np.clip(gripper_pos * 255, 0, 255)
        self.gripper_speed = np.clip(speed * max_speed, 0, max_speed)

    def update_gripper(self):
        """@brief Update gripper position towards target"""
        if not self.gripper_control_indices:
            return

        for control_index in self.gripper_control_indices:
            if control_index >= len(self.sim.data.ctrl):
                continue

            current_pos = self.sim.data.ctrl[control_index]
            delta = self.gripper_target - current_pos

            if abs(delta) < 0.1:
                continue

            step_size = self.gripper_speed * self.sim.model.opt.timestep
            new_pos = (self.gripper_target if abs(delta) <= step_size
                       else current_pos + np.sign(delta) * step_size)

            self.sim.data.ctrl[control_index] = new_pos