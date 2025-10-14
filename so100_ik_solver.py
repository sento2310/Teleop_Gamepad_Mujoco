"""
Specialized IK solver for SO100 5DOF arm with joint preservation.
Preserves manually controlled joints while solving for position.
"""

import numpy as np
import mujoco


class SO100IKSolver:
    """5DOF IK solver that preserves manually controlled joints."""

    def __init__(self, model, data, end_effector_body, joint_controller):
        """
        @brief Initialize IK solver for SO100
        @param model: MuJoCo model object
        @param data: MuJoCo data object
        @param end_effector_body: Name of end effector body
        @param joint_controller: SO100Teleoperation instance for joint mapping
        """
        self.model, self.data = model, data
        self.joint_controller = joint_controller

        # Get end-effector body ID
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
        if self.ee_body_id == -1:
            raise ValueError(f"End effector body '{end_effector_body}' not found")

        self._build_joint_mapping()

        # IK parameters
        self.damping, self.max_step_size = 0.01, 0.1

    def _build_joint_mapping(self):
        """@brief Build mapping between joint names and their qpos indices"""
        self.joint_qpos_indices = {}

        for i in range(self.model.njnt):
            # Extract joint name
            name_id = self.model.name_jntadr[i]
            name_bytes = bytearray()
            j = name_id
            while j < len(self.model.names) and self.model.names[j] != 0:
                name_bytes.append(self.model.names[j])
                j += 1
            joint_name = name_bytes.decode('utf-8') if name_bytes else f"joint_{i}"

            # Map identifiable joints
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

        # Define control strategy
        self.position_joints = ['pitch', 'elbow']  # For horizontal/vertical movement
        self.manual_joints = ['rotation', 'wrist_pitch', 'wrist_roll']  # Manual control

    def solve_position_only(self, target_pos, fixed_rotation=None, fixed_wrist_roll=None,
                           fixed_wrist_pitch=None, max_iterations=50, tolerance=1e-3):
        """
        @brief Solve IK for position only while preserving manual joints
        @param target_pos: Target position (3D)
        @param fixed_rotation: Fixed rotation joint value
        @param fixed_wrist_roll: Fixed wrist roll value
        @param fixed_wrist_pitch: Fixed wrist pitch value
        @return: Tuple of (position_joint_values, success)
        """
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
            # Apply fixed joint values
            if fixed_rotation is not None and 'rotation' in self.joint_qpos_indices:
                self.data.qpos[self.joint_qpos_indices['rotation']] = fixed_rotation
            if fixed_wrist_roll is not None and 'wrist_roll' in self.joint_qpos_indices:
                self.data.qpos[self.joint_qpos_indices['wrist_roll']] = fixed_wrist_roll
            if fixed_wrist_pitch is not None and 'wrist_pitch' in self.joint_qpos_indices:
                self.data.qpos[self.joint_qpos_indices['wrist_pitch']] = fixed_wrist_pitch

            mujoco.mj_forward(self.model, self.data)

            # Position error calculation
            current_pos = self.data.xpos[self.ee_body_id].copy()
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)

            if error_norm < tolerance:
                success = True
                break

            # Jacobian for position control
            jac_pos = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac_pos, None, self.ee_body_id)

            # Extract controllable joints
            controllable_joints = []
            for joint_name in self.position_joints:
                if joint_name in self.joint_qpos_indices:
                    joint_id = self.joint_qpos_indices[joint_name]
                    dof_adr = self.model.jnt_dofadr[joint_id]
                    controllable_joints.append(dof_adr)

            if not controllable_joints:
                break

            jacobian_reduced = jac_pos[:, controllable_joints]

            # Damped least squares solution
            jacobian_t = jacobian_reduced.T
            jjt = jacobian_reduced @ jacobian_t
            damping_matrix = self.damping * np.eye(3)

            try:
                lambda_matrix = jjt + damping_matrix
                jacobian_pinv = jacobian_t @ np.linalg.pinv(lambda_matrix)
                delta_q = jacobian_pinv @ pos_error

                # Limit step size
                step_norm = np.linalg.norm(delta_q)
                if step_norm > self.max_step_size:
                    delta_q = delta_q * (self.max_step_size / step_norm)

                # Apply to position joints with limits
                for i, dof_adr in enumerate(controllable_joints):
                    for joint_name, qpos_idx in self.joint_qpos_indices.items():
                        if joint_name in self.position_joints:
                            joint_dof_adr = self.model.jnt_dofadr[qpos_idx]
                            if joint_dof_adr == dof_adr:
                                self.data.qpos[qpos_idx] += delta_q[i]
                                if self.model.jnt_limited[qpos_idx]:
                                    qmin, qmax = self.model.jnt_range[qpos_idx]
                                    self.data.qpos[qpos_idx] = np.clip(self.data.qpos[qpos_idx], qmin, qmax)
                                break
            except np.linalg.LinAlgError:
                break

        # Get final position joint values
        position_joint_values = {}
        for joint_name in self.position_joints:
            if joint_name in self.joint_qpos_indices:
                position_joint_values[joint_name] = self.data.qpos[self.joint_qpos_indices[joint_name]]

        # Restore original state
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)

        return position_joint_values, success


