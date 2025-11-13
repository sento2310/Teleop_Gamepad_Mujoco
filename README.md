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


## Adding a New Robot

The framework is designed to be extendable, and adding support for a new robot follows a clear set of steps. This section describes how to integrate a new robot model, including configuration, file organization, and simulation setup.

### 1. Create a robot folder with XMLs and assets
Inside the main project directory (`Teleop_Gamepad_Mujoco`), create a new folder named after your robot, similar to:

```
franka_emika_panda/
universal_robots_ur5e/
trs_so_arm100/
```

For example:
```
my_custom_robot/
```

Inside this folder you should place:
- The robot’s main XML model (`robot.xml`)
- A scene XML that loads the robot (`scene.xml`)
- A folder containing required assets (textures, meshes, URDF/MJCF includes, STL files, etc.)

The structure might look like:

```
Teleop_Gamepad_Mujoco/
└── my_custom_robot/
    ├── myrobot.xml
    ├── scene.xml
    ├── assets/
    │   ├── base.stl
    │   ├── link1.stl
    │   └── textures/
    └── materials/
```

Your scene XML must reference the robot XML and any assets with relative paths.

### 2. Add a configuration entry
Edit `config.py` and add your robot definition to `ROBOT_CONFIGS`:

```python
'myrobot': {
    'name': 'My Custom Robot',
    'xml_path': 'my_custom_robot/myrobot.xml',
    'end_effector_body': 'name', # define name of robots end effector body
    'arm_joint_count': n, #define number of arm joints, so kinematic chain can be built
    'axis_remap': {
        'vx': 'vx', 'vy': 'vy', 'vz': 'vz',
        'roll': 'roll', 'pitch': 'pitch', 'yaw': 'yaw'
    },
    'movement_scales': {
        'translation': 0.2,
        'rotation': 0.3,
        'tilt': 0.3,
        'gripper_open_pos': 1.0,
        'gripper_close_pos': 0.0,
        'gripper_speed': 1.0,
        'deadzone_threshold': 0.1
    }
}
```

This tells the teleoperation system how to interpret commands for your robot.

### 3. Register your robot in the simulation
In `simulation.py`, add a new entry inside the `configs` dictionary:

```python
"myrobot": {
    "world": "my_custom_robot/scene.xml",
    "qpos": [...],   # Initial joint state (length = nq)
    "ctrl": [...],   # Initial actuator values (length = nu)
}
```

Make sure:
- The `world` path points to your newly added `scene.xml`
- `qpos` and `ctrl` arrays match your model’s joint and actuator counts

### 4. Verify the end-effector name
Ensure the value of `end_effector_body` in your config matches exactly the name in your MJCF model.

To list all bodies in the model:

```python
[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
```

### 5. Check actuator naming for gripper detection
`movement_helper.py` looks for actuator names containing common keywords:
- finger
- gripper
- hand
- Jaw (SO100-style)

If your robot uses different naming conventions, either:
- Rename actuators in your XMLs, or
- Adjust the detection logic in `_find_gripper_controls()`

### 6. Choose the teleoperation backend
Most robots will work out of the box with `GenericTeleoperation`.

If your robot has:
- fewer than 6 controllable DOF  
- extra joints  
- non-standard



## Documentation

Full API documentation and detailed module explanations can be found here:

https://sento2310.github.io/Teleop_Gamepad_Mujoco

---

## Notes

- Requires a standard USB gamepad (Xbox, PS, or Logitech controllers work well).
- MuJoCo XML models must be accessible via the paths configured in `simulation.py`.
- Real-time viewer performance depends on your hardware; running on a machine with a dedicated GPU is recommended.
- The teleoperation system uses robot-specific configurations, scaling, and axis remapping defined in `config.py`.


