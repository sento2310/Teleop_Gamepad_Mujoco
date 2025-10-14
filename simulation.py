"""
MuJoCo simulation wrapper for robotic arm teleoperation.
Provides simulation environment and object manipulation utilities.
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
from config import get_robot_config


class Simulation:
    """MuJoCo simulation class for robotic arm teleoperation."""

    def __init__(self, robot_name, show_viewer=True):  # Add robot_name parameter
        """
        @brief Initialize the simulation with a model and viewer settings
        @param robot_name: Name of the robot to use ('panda', 'ur5', 'so100')
        @param show_viewer: Whether to display the visualizer
        """
        # Load robot-specific configuration
        self.config = robot_name  # Use passed robot name

        # Robot configurations with initial positions/controls
        configs = {
            "panda": {
                "world": "franka_emika_panda/demo_scene.xml",
                "qpos": [0.075, -0.798, -0.047, -2.335, -0.033, 1.529, 0.826, 0.0, 0.0, 0.39, 0.0, 0.013, 1.0, 0.0, 0.0, 0.0],
                "ctrl": [0.075, -0.88, -0.046, -2.328, -0.033, 1.530, 0.826, 0.0]
            },
            "ur5": {
                "world": "universal_robots_ur5e/scene.xml",
                "qpos": [-0.281, -1.192, 1.826, 0.976, 1.574, -0.284, 0.0, 0.39, 0.0, 0.013, 1.0, 0.0, 0.0, 0.0, 0.39, 0, 0.018, 0, 0, 0, 0],
                "ctrl": [-0.305, -1.56, 1.9, 1.23, 1.57, -0.305, 0]
            },
            "so100": {
                "world": "trs_so_arm100/scene.xml",
                "qpos": [0.0, -1.57, 1.57, 1.57, -1.57, 0.0],
                "ctrl": [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]
            }
        }

        # Validate and load configuration
        if self.config not in configs:
            raise ValueError(f"Unknown robot configuration: {self.config}")

        config = configs[self.config]
        self.model = mujoco.MjModel.from_xml_path(config["world"])
        self.data = mujoco.MjData(self.model)

        # Set initial state
        qpos, ctrl = config["qpos"], config["ctrl"]
        self.data.qpos[:min(len(qpos), self.model.nq)] = qpos[:self.model.nq]
        self.data.ctrl[:min(len(ctrl), self.model.nu)] = ctrl[:self.model.nu]

        mujoco.mj_step(self.model, self.data, nstep=100)

        # Viewer setup
        self.show_viewer = show_viewer
        self.step_count = 0

        if show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer_freq = 60
            self.viewer_interval = 1.0 / self.viewer_freq
            self.viewer_nth = int(1 / (self.model.opt.timestep * self.viewer_freq))
            self.wall_start_time = time.time()


    def step(self):
        """
        @brief Advance simulation by one step with real-time synchronization
        """
        mujoco.mj_step(self.model, self.data, nstep=1)
        self.step_count += 1

        # Real-time synchronization for viewer
        if self.show_viewer:
            t_sim, t_wall = self.data.time, time.time() - self.wall_start_time
            if t_sim - t_wall > 0:
                time.sleep(t_sim - t_wall)
            if self.step_count % self.viewer_nth == 0:
                self.viewer.sync()

    def get_object_state(self, body_name):
        """
        @brief Get world-frame position and orientation of a body
        @param body_name: Name of the body to query
        @return: Tuple of (position, quaternion) arrays
        """
        # Find body ID
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise KeyError(f"Body '{body_name}' not found")

        # Try to find free joint for direct access
        for j in range(self.model.njnt):
            if (self.model.jnt_bodyid[j] == bid and
                self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
                adr = self.model.jnt_qposadr[j]
                return (self.data.qpos[adr:adr+3].copy(),
                        self.data.qpos[adr+3:adr+7].copy())

        # Fallback to cached pose
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()


