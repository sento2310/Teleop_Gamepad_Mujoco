# Teleop Gamepad Mujoco
Real-time teleoperation of Panda, UR5, and SO100 robotic arms in MuJoCo using a standard gamepad.

This project provides a complete, modular teleoperation pipeline for controlling several robot arms in MuJoCo using a USB gamepad. It supports both 6-DoF IK-based control (Panda & UR5) and a hybrid joint/IK control scheme (SO100).

Full documentation is available here:  
https://sento2310.github.io/Teleop_Gamepad_Mujoco

---

## Features

### Supported Robots
- Franka Emika Panda (7-DoF)
- UR5e with Robotiq 2F-85 (6-DoF)
- SO100 Arm (5-DoF, hybrid IK/joint control)

### Control Modes

**Panda & UR5 (Generic Teleoperation)**
- Full 6-DoF end-effector control:
  - Translation (X, Y, Z)
  - Orientation (roll, pitch, yaw)
- Toggle-based gripper control
- Smooth filtered motion and twist integration

**SO100 (Hybrid Control)**
- Manual joint control:
  - Base rotation
  - Wrist pitch
  - Wrist roll
- IK-based translation using the pitch and elbow joints
- Conflict-safe blending between manual and IK movement

### Technical Highlights
- Damped-least-squares IK with joint limits
- Axis remapping and movement scaling per robot
- Twist integration with velocity and acceleration limits
- Automatic actuator and joint indexing
- Real-time MuJoCo viewer synchronization
- Modular and extendable architecture

---


## Installation

### 1. Clone the repository
```bash
git clone https://github.com/sento2310/Teleop_Gamepad_Mujoco.git
cd Teleop_Gamepad_Mujoco
```

### 2. Install dependencies

Requirements:
- Python 3.8+
- MuJoCo
- pygame
- numpy
- scipy

Install using the provided requirements file:
```bash
pip install -r requirements.txt
```

If MuJoCo is not already installed:
```bash
pip install mujoco
```



## Running the Teleoperation System

### Option 1 — Direct command
Launch the teleoperation system for a specific robot:

```bash
python gamepad_control.py panda
```

Other supported robots:

```bash
python gamepad_control.py ur5
python gamepad_control.py so100
```

### Option 2 — Interactive selection
Run without arguments to choose a robot from a menu:

```bash
python gamepad_control.py
```

A prompt will appear to select one of the available robots (Panda, UR5, or SO100).


## Controls Overview

### Panda & UR5 (Generic Teleoperation)

| Input           | Function                    |
|-----------------|------------------------------|
| Left Stick X    | Move left / right            |
| Left Stick Y    | Move up / down               |
| Right Stick Y   | Move forward / back          |
| Right Stick X   | Roll                         |
| L1 / R1         | Pitch                        |
| L2 / R2         | Yaw                          |
| A Button        | Toggle gripper               |
| START           | Exit teleoperation           |


### SO100 (Hybrid Control)

| Input           | Function                          |
|-----------------|------------------------------------|
| Left Stick X    | Base rotation (manual)             |
| Left Stick Y    | Vertical translation (IK)          |
| Right Stick Y   | Horizontal in/out (IK, radial)     |
| L1 / R1         | Wrist pitch                        |
| L2 / R2         | Wrist roll                         |
| A Button        | Toggle gripper                     |
| START           | Exit teleoperation                 |




## Project Structure

```
Teleop_Gamepad_Mujoco/
│
├── config.py                # Robot configuration manager
├── gamepad_control.py       # Main launcher interface
├── simulation.py            # MuJoCo simulation wrapper
│
├── genericteleoperation.py  # Teleoperation system for Panda/UR5
├── so100teleoperation.py    # Teleoperation system for SO100
│
├── generic_ik_solver.py     # Generic damped-least-squares IK solver
├── so100_ik_solver.py       # SO100 position-only IK solver
│
├── movement_helper.py       # Twist integration + gripper control
└── README.md
```

---

## Documentation

Full API documentation and detailed module explanations can be found here:

https://sento2310.github.io/Teleop_Gamepad_Mujoco

---

## Notes

- Requires a standard USB gamepad (Xbox, PS, or Logitech controllers work well).
- MuJoCo XML models must be accessible via the paths configured in `simulation.py`.
- Real-time viewer performance depends on your hardware; running on a machine with a dedicated GPU is recommended.
- The teleoperation system uses robot-specific configurations, scaling, and axis remapping defined in `config.py`.


