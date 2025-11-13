"""
@file so100_ik_solver.py
@brief Specialized IK solver for SO100 5DOF arm with joint preservation
@details Preserves manually controlled joints while solving for position
         using a hybrid inverse kinematics approach.
"""
import numpy as np
import mujoco
from config import Configuration


class SO100IKSolver:
    """
    @brief 5DOF IK solver that preserves manually controlled joints
    @details Implements position-only inverse kinematics while fixing
             rotation, wrist_pitch, and wrist_roll joints to prevent
             conflicts between manual control and IK solutions.
    """

    def __init__(self, model, data, end_effector_body, joint_controller):
        """
        @brief Initialize IK solver for SO100 arm

        @param model: MuJoCo model object containing robot kinematics
        @param data: MuJoCo data object for current simulation state
        @param end_effector_body: Name of end effector body for IK targeting
        @param joint_controller: SO100Teleoperation instance for joint mapping

        @throws ValueError: If end_effector_body not found in model

        @note Builds joint mapping and defines control strategy automatically
        """
        self.model, self.data = model, data
        self.joint_controller = joint_controller

        # Get end-effector body ID
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
        if self.ee_body_id == -1:
            raise ValueError(f"End effector body '{end_effector_body}' not found")

        self._build_joint_mapping()

        # IK parameters for numerical stability
        self.damping, self.max_step_size = 0.01, 0.1


    def _build_joint_mapping(self):
        """
        @brief Build mapping between joint names and their qpos indices

        @note Uses joint naming patterns to identify SO100-specific joints
              and defines control strategy based on joint functionality
        """
        self.joint_qpos_indices = {}

        for i in range(self.model.njnt):
            # Extract joint name from model
            name_id = self.model.name_jntadr[i]
            name_bytes = bytearray()
            j = name_id
            while j < len(self.model.names) and self.model.names[j] != 0:
                name_bytes.append(self.model.names[j])
                j += 1
            joint_name = name_bytes.decode('utf-8') if name_bytes else f"joint_{i}"

            # Map identifiable joints based on naming patterns
            if 'Rotation' in joint_name:
                self.joint_qpos_indices['rotation'] = i
            elif 'Pitch' in joint_name and 'Wrist' not in joint_name:
                self.joint_qpos_indices['pitch'] = i  # Shoulder pitch
            elif 'Elbow' in joint_name:
                self.joint_qpos_indices['elbow'] = i
            elif 'Wrist_Pitch' in joint_name:
                self.joint_qpos_indices['wrist_pitch'] = i
            elif 'Wrist_Roll' in joint_name:
                self.joint_qpos_indices['wrist_roll'] = i

        # Define control strategy: position joints vs manual joints
        self.position_joints = ['pitch', 'elbow']  # Controlled by IK for translation
        self.manual_joints = ['rotation', 'wrist_pitch', 'wrist_roll']  # Manual control

    def solve_position_only(self, target_pos, fixed_rotation=None, fixed_wrist_roll=None,
                           fixed_wrist_pitch=None, max_iterations=50, tolerance=1e-3):
        """
        @brief Solve IK for position only while preserving manual joints

        @param target_pos: Target position as 3D vector [x, y, z]
        @param fixed_rotation: Fixed value for rotation joint (optional)
        @param fixed_wrist_roll: Fixed value for wrist roll joint (optional)
        @param fixed_wrist_pitch: Fixed value for wrist pitch joint (optional)
        @param max_iterations: Maximum number of iterations
        @param tolerance: Convergence tolerance for position error

        @return: Tuple of (position_joint_values, success_flag)
                - position_joint_values: Dictionary mapping joint names to values
                - success_flag: Boolean indicating convergence success

        @note Only solves for pitch and elbow joints while fixing other joints
              to maintain manual control integrity
        """
        # Store original state for restoration
        original_qpos = self.data.qpos.copy()
        success = False

        # Use current values if fixed values not provided
        if fixed_rotation is None and 'rotation' in self.joint_qpos_indices:
            fixed_rotation = self.data.qpos[self.joint_qpos_indices['rotation']]
        if fixed_wrist_roll is None and 'wrist_roll' in self.joint_qpos_indices:
            fixed_wrist_roll = self.data.qpos[self.joint_qpos_indices['wrist_roll']]
        if fixed_wrist_pitch is None and 'wrist_pitch' in self.joint_qpos_indices:
            fixed_wrist_pitch = self.data.qpos[self.joint_qpos_indices['wrist_pitch']]

        for iteration in range(max_iterations):
            # Apply fixed joint values to preserve manual control
            if fixed_rotation is not None and 'rotation' in self.joint_qpos_indices:
                self.data.qpos[self.joint_qpos_indices['rotation']] = fixed_rotation
            if fixed_wrist_roll is not None and 'wrist_roll' in self.joint_qpos_indices:
                self.data.qpos[self.joint_qpos_indices['wrist_roll']] = fixed_wrist_roll
            if fixed_wrist_pitch is not None and 'wrist_pitch' in self.joint_qpos_indices:
                self.data.qpos[self.joint_qpos_indices['wrist_pitch']] = fixed_wrist_pitch

            # Update forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Calculate position error
            current_pos = self.data.xpos[self.ee_body_id].copy()
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)

            # Check convergence
            if error_norm < tolerance:
                success = True
                break

            # Compute position Jacobian only (3xN)
            jac_pos = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac_pos, None, self.ee_body_id)

            # Extract controllable joints (position joints only)
            controllable_joints = []
            for joint_name in self.position_joints:
                if joint_name in self.joint_qpos_indices:
                    joint_id = self.joint_qpos_indices[joint_name]
                    dof_adr = self.model.jnt_dofadr[joint_id]
                    controllable_joints.append(dof_adr)

            # Exit if no controllable joints found
            if not controllable_joints:
                break

            # Create reduced Jacobian for controllable joints
            jacobian_reduced = jac_pos[:, controllable_joints]

            # Damped least squares solution for position error
            jacobian_t = jacobian_reduced.T
            jjt = jacobian_reduced @ jacobian_t
            damping_matrix = self.damping * np.eye(3)

            try:
                lambda_matrix = jjt + damping_matrix
                jacobian_pinv = jacobian_t @ np.linalg.pinv(lambda_matrix)
                delta_q = jacobian_pinv @ pos_error

                # Limit step size for stability
                step_norm = np.linalg.norm(delta_q)
                if step_norm > self.max_step_size:
                    delta_q = delta_q * (self.max_step_size / step_norm)

                # Apply updates to position joints with limits
                for i, dof_adr in enumerate(controllable_joints):
                    for joint_name, qpos_idx in self.joint_qpos_indices.items():
                        if joint_name in self.position_joints:
                            joint_dof_adr = self.model.jnt_dofadr[qpos_idx]
                            if joint_dof_adr == dof_adr:
                                self.data.qpos[qpos_idx] += delta_q[i]
                                # Enforce joint limits
                                if self.model.jnt_limited[qpos_idx]:
                                    qmin, qmax = self.model.jnt_range[qpos_idx]
                                    self.data.qpos[qpos_idx] = np.clip(self.data.qpos[qpos_idx], qmin, qmax)
                                break
            except np.linalg.LinAlgError:
                # Handle singular Jacobian by breaking iteration
                break

        # Collect final position joint values
        position_joint_values = {}
        for joint_name in self.position_joints:
            if joint_name in self.joint_qpos_indices:
                position_joint_values[joint_name] = self.data.qpos[self.joint_qpos_indices[joint_name]]

        # Restore original simulation state
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)

        return position_joint_values, success


