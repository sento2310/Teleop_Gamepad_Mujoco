# Robotic Arm Teleoperation with Gamepad in MuJoCo

A professional teleoperation system for robotic arms in MuJoCo simulation with intuitive gamepad control. Supports multiple robot platforms with both inverse kinematics and hybrid control schemes.

## Features

### Multi-Robot Support
- **Franka Emika Panda** - Full 6DOF inverse kinematics control
- **UR5e with 2F-85 Gripper** - Industrial arm teleoperation  
- **SO100 Robotic Arm** - Hybrid joint+IK control (5DOF)

### Dual Control Paradigms
- **Generic IK Control** (Panda/UR5): 6DOF end-effector control using damped least squares inverse kinematics
- **Hybrid Control** (SO100): Joint-space rotation control combined with IK translational control

### Advanced Capabilities
- Real-time trajectory smoothing with low-pass filtering
- Singularity handling with adaptive damping
- Joint limit enforcement and collision prevention
- Configuration-driven robot parameter management

## Documentation

**Complete API Documentation:** [sento2310.github.io/Teleop_Gamepad_Mujoco](https://sento2310.github.io/Teleop_Gamepad_Mujoco/index.html)

The documentation provides:
- Detailed class hierarchies and method specifications
- Mathematical formulations for inverse kinematics algorithms
- Configuration parameters for all supported robots
- Control mapping explanations and usage examples

## Installation

### Prerequisites
```bash
pip install mujoco pygame numpy scipy
