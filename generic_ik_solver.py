"""
@file generic_ik_solver.py
@brief Generic velocity-based inverse kinematics solver
@details Uses damped least squares method for stable joint-space trajectories
         with joint limit enforcement and singularity handling.
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


class GenericVelocityIKSolver:
    """
    @brief Inverse kinematics solver using Jacobian pseudo-inverse method
    @details Implements iterative damped least squares for end-effector pose control
             with numerical stability and joint limit constraints.
    """

    def __init__(self, model, data, end_effector_body, arm_joint_count=None):
        """
        @brief Initialize IK solver for specified end effector

        @param model: MuJoCo model object containing robot kinematics
        @param data: MuJoCo data object for current simulation state
        @param end_effector_body: Name of end effector body for IK targeting
        @param arm_joint_count: Number of arm joints to use for IK solution

        @throws ValueError: If end_effector_body not found in model

        @note Automatically detects revolute joints for IK control
        """
        self.model, self.data = model, data

        # Get end-effector body ID
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
        if self.ee_body_id == -1:
            raise ValueError(f"End effector body '{end_effector_body}' not found")

        # Find revolute joints for IK control
        all_joint_indices = [i for i in range(model.njnt)
                           if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]

        self.arm_joint_count = min(arm_joint_count, len(all_joint_indices)) if arm_joint_count else len(all_joint_indices)
        self.joint_indices = all_joint_indices[:self.arm_joint_count]
        self.n_joints = len(self.joint_indices)

        # IK parameters for numerical stability
        self.damping, self.max_step_size = 0.01, 0.1

    def solve(self, target_pos, target_quat, max_iterations=50, tolerance=1e-3):
        """
        @brief Solve inverse kinematics for target end-effector pose

        @param target_pos: Target position as 3D vector [x, y, z]
        @param target_quat: Target orientation as quaternion [w, x, y, z]
        @param max_iterations: Maximum number of gradient descent iterations
        @param tolerance: Convergence tolerance for pose error norm

        @return: Tuple of (joint_positions, success_flag)
                - joint_positions: Array of solved joint angles
                - success_flag: Boolean indicating convergence success

        @throws LinAlgError: If Jacobian matrix is singular during inversion

        @note Uses damped least squares: $\Delta q = J^T (JJ^T + \lambda I)^{-1} e$
              where $e$ is the 6D pose error vector and $\lambda$ is damping factor
        """
        if self.n_joints == 0:
            return np.array([]), False

        # Store original state for restoration
        original_qpos = self.data.qpos.copy()
        success = False

        for iteration in range(max_iterations):
            mujoco.mj_forward(self.model, self.data)

            # Current pose and errors
            current_pos = self.data.xpos[self.ee_body_id].copy()
            current_quat = self.data.xquat[self.ee_body_id].copy()

            # Position error
            pos_error = target_pos - current_pos

            # Orientation error as rotation vector
            current_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
            target_rot = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
            orient_error = (target_rot * current_rot.inv()).as_rotvec()

            # Combined error vector
            error = np.concatenate([pos_error, orient_error])
            error_norm = np.linalg.norm(error)

            # Check convergence
            if error_norm < tolerance:
                success = True
                break

            # Compute Jacobian matrices
            jac_pos, jac_rot = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, self.ee_body_id)
            jacobian = np.vstack([jac_pos, jac_rot])

            # Extract controllable joints subspace
            jacobian_reduced = np.zeros((6, self.n_joints))
            for i, joint_id in enumerate(self.joint_indices):
                dof_adr = self.model.jnt_dofadr[joint_id]
                jacobian_reduced[:, i] = jacobian[:, dof_adr]

            # Damped least squares solution
            jacobian_t = jacobian_reduced.T
            jjt = jacobian_reduced @ jacobian_t
            damping_matrix = self.damping * np.eye(6)

            try:
                lambda_matrix = jjt + damping_matrix
                jacobian_pinv = jacobian_t @ np.linalg.pinv(lambda_matrix)
                delta_q = jacobian_pinv @ error

                # Limit step size for stability
                step_norm = np.linalg.norm(delta_q)
                if step_norm > self.max_step_size:
                    delta_q = delta_q * (self.max_step_size / step_norm)

                # Apply joint updates with limits
                for i, joint_id in enumerate(self.joint_indices):
                    self.data.qpos[joint_id] += delta_q[i]
                    if self.model.jnt_limited[joint_id]:
                        qmin, qmax = self.model.jnt_range[joint_id]
                        self.data.qpos[joint_id] = np.clip(self.data.qpos[joint_id], qmin, qmax)

            except np.linalg.LinAlgError:
                # Fallback for singularities with increased damping
                damping_matrix = 0.1 * np.eye(6)
                lambda_matrix = jjt + damping_matrix
                jacobian_pinv = jacobian_t @ np.linalg.pinv(lambda_matrix)
                delta_q = jacobian_pinv @ error * 0.1

                for i, joint_id in enumerate(self.joint_indices):
                    self.data.qpos[joint_id] += delta_q[i]

        # Get final joint positions and restore simulation state
        final_joint_pos = self.data.qpos[self.joint_indices].copy()
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)

        return final_joint_pos, success