"""
Centralized configuration for the teleoperation system.
Holds global settings accessible by all modules.
"""

# Robot configurations
ROBOT_CONFIGS = {
    'panda': {
        'name': 'Franka Emika Panda',
        'xml_path': 'franka_emika_panda/panda.xml',
        'end_effector_body': 'hand',
        'arm_joint_count': 7,
        'axis_remap': {
            'vx': 'vx', 'vy': 'vy', 'vz': 'vz',
            'roll': 'roll', 'pitch': 'pitch', 'yaw': 'yaw'
        },
        'movement_scales': {
            'translation': 0.3, 'rotation': 0.5, 'tilt': 0.5,
            'gripper_open_pos': 1.0, 'gripper_close_pos': 0.0,
            'gripper_speed': 1, 'deadzone_threshold': 0.1
        }
    },
    'ur5': {
        'name': 'UR5e with 2F-85 Gripper',
        'xml_path': 'universal_robots_ur5e/ur5e_with_gripper.xml',
        'end_effector_body': 'robotiq_base',
        'arm_joint_count': 6,
        'axis_remap': {
            'vx': 'vx', 'vy': 'vy', 'vz': 'vz',
            'roll': 'roll', 'pitch': 'pitch', 'yaw': 'yaw'
        },
        'movement_scales': {
            'translation': 0.2, 'rotation': 0.3, 'tilt': 0.3,
            'gripper_open_pos': 1.0, 'gripper_close_pos': 0.0,
            'gripper_speed': 1, 'deadzone_threshold': 0.1
        }
    },
    'so100': {
        'name': 'SO100 Robotic Arm',
        'xml_path': 'trs_so_arm100/so_arm100.xml',
        'end_effector_body': 'Fixed_Jaw',
        'arm_joint_count': 5,
        'axis_remap': {
            'vx': 'vy', 'vy': 'vz', 'vz': 'vx',  # Axis swapping
            'roll': 'pitch', 'pitch': 'yaw', 'yaw': 'roll'
        },
        'movement_scales': {
            'translation': 0.15, 'rotation': 0.4, 'tilt': 0.4,
            'gripper_open_pos': 0.0123, 'gripper_close_pos': 0.0,
            'gripper_speed': 1, 'deadzone_threshold': 0.1
        },
        'joint_multipliers': {
            'rotation': 0.005, 'wrist_roll': 0.01, 'wrist_pitch': 0.02,
            'pitch': 1.0, 'elbow': 1.0
        }
    }
}

def get_robot_config(robot_name):
    """@brief Get configuration for a specific robot"""
    if robot_name not in ROBOT_CONFIGS:
        available = list(ROBOT_CONFIGS.keys())
        raise ValueError(f"Robot '{robot_name}' not found. Available: {available}")
    return ROBOT_CONFIGS[robot_name].copy()

def get_movement_scales(robot_name):
    """@brief Get movement scales for specified robot"""
    config = get_robot_config(robot_name)
    return config.get('movement_scales', {
        'translation': 0.2, 'rotation': 0.3, 'tilt': 0.3,
        'gripper_open_pos': 1.0, 'gripper_close_pos': 0.0,
        'gripper_speed': 1, 'deadzone_threshold': 0.1
    })

def get_joint_multipliers(robot_name):
    """@brief Get joint control multipliers for specified robot"""
    config = get_robot_config(robot_name)
    return config.get('joint_multipliers', {
        'rotation': 1.0, 'wrist_roll': 1.0, 'wrist_pitch': 1.0,
        'pitch': 1.0, 'elbow': 1.0
    })