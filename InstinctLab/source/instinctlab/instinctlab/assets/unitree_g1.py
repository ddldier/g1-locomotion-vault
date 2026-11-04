"""
Customized Unitree G1 asset for Isaac Sim
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import G1_CFG

from instinctlab.motion_reference import NoCollisionPropertiesCfg

# 当前文件所在目录，用于拼接资源路径
__file_dir__ = os.path.dirname(os.path.realpath(__file__))

"""
joint name order:
[
    'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint',
    'waist_pitch_joint',
    'left_shoulder_roll_joint',
    'right_shoulder_roll_joint',
    'waist_roll_joint',
    'left_shoulder_yaw_joint',
    'right_shoulder_yaw_joint',
    'waist_yaw_joint',
    'left_elbow_joint',
    'right_elbow_joint',
    'left_hip_pitch_joint',
    'right_hip_pitch_joint',
    'left_wrist_roll_joint',
    'right_wrist_roll_joint',
    'left_hip_roll_joint',
    'right_hip_roll_joint',
    'left_wrist_pitch_joint',
    'right_wrist_pitch_joint',
    'left_hip_yaw_joint',
    'right_hip_yaw_joint',
    'left_wrist_yaw_joint',
    'right_wrist_yaw_joint',
    'left_knee_joint',
    'right_knee_joint',
    'left_ankle_pitch_joint',
    'right_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_ankle_roll_joint',
]
"""

# 从官方 G1 配置复制一份，作为 29 自由度 TorsoBase 配置的基础
G1_29DOF_TORSOBASE_CFG = G1_CFG.copy()
# 使用自定义 URDF 作为生成源，并设置仿真碰撞、接触传感器等属性
G1_29DOF_TORSOBASE_CFG.spawn = sim_utils.UrdfFileCfg(
    asset_path=os.path.join(__file_dir__, "resources/unitree_g1/urdf/g1_29dof_torsobase_simplified.urdf"),
    replace_cylinders_with_capsules=False,
    merge_fixed_joints=False,
    fix_base=False,
    self_collision=True,
    activate_contact_sensors=True,
)
# 关节驱动刚度使用 URDF 文件中的值，而不是在配置里覆盖
G1_29DOF_TORSOBASE_CFG.spawn.joint_drive.gains.stiffness = None  # use value from the URDF file
# 软关节位置限制系数，用于限制关节位置范围
G1_29DOF_TORSOBASE_CFG.soft_joint_pos_limit_factor = 0.95
# 定义不同身体部位的执行器配置
G1_29DOF_TORSOBASE_CFG.actuators = {
    # NOTE: checked, delayed PD actuator has same time-lag when computing torques; and no lag when
    # the num_pushes does not reach the lag time.
    # 腿部与腰部执行器：使用带延迟的 PD 执行器
    "legs": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            "waist_.*_joint",
        ],
        effort_limit={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 88.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_knee_joint": 139.0,
            "waist_roll_joint": 50.0,
            "waist_pitch_joint": 50.0,
            "waist_yaw_joint": 88.0,
        },
        velocity_limit=60.0,
        stiffness={
            ".*_hip_yaw_joint": 90.0,
            ".*_hip_roll_joint": 90.0,
            ".*_hip_pitch_joint": 90.0,
            ".*_knee_joint": 140.0,
            "waist_roll_joint": 60.0,
            "waist_pitch_joint": 60.0,
            "waist_yaw_joint": 90.0,
        },
        damping={
            ".*_hip_yaw_joint": 2.0,
            ".*_hip_roll_joint": 2.0,
            ".*_hip_pitch_joint": 2.0,
            ".*_knee_joint": 2.5,
            "waist_.*_joint": 2.5,
        },
        armature=0.03,
        min_delay=0,
        max_delay=1,
    ),
    # 脚踝执行器：控制脚踝 pitch 和 roll 关节
    "feet": DelayedPDActuatorCfg(
        effort_limit=20,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=20.0,
        damping=1.0,
        velocity_limit=60.0,
        armature=0.03,
        min_delay=0,
        max_delay=1,
    ),
    # 手臂执行器：控制肩部和肘部关节
    "arms": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
        ],
        effort_limit=25,
        velocity_limit=60.0,
        stiffness=25,
        damping={
            ".*_shoulder_.*_joint": 1.0,
            ".*_elbow_joint": 1.0,
        },
        armature=0.03,
        min_delay=0,
        max_delay=1,
    ),
    # 手腕执行器：控制手腕 roll、pitch、yaw 关节
    "wrist": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*wrist_roll_joint",
            ".*wrist_pitch_joint",
            ".*wrist_yaw_joint",
        ],
        effort_limit={
            ".*wrist_roll_joint": 25.0,
            ".*wrist_pitch_joint": 5.0,
            ".*wrist_yaw_joint": 5.0,
        },
        velocity_limit=25.0,
        stiffness={
            ".*wrist_roll_joint": 25.0,
            ".*wrist_pitch_joint": 5.0,
            ".*wrist_yaw_joint": 5.0,
        },
        damping={
            ".*wrist_roll_joint": 1.0,
            ".*wrist_pitch_joint": 0.5,
            ".*wrist_yaw_joint": 0.5,
        },
        armature=0.03,
        min_delay=0,
        max_delay=1,
    ),
}
# 从官方 G1 初始状态复制，并覆盖部分关节初始位置
G1_29DOF_TORSOBASE_CFG.init_state = G1_CFG.init_state.copy()
G1_29DOF_TORSOBASE_CFG.init_state.joint_pos = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    ".*_wrist_roll_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
}

# 复制一份带 CLOG 碰撞体配置的版本
G1_29DOF_TORSOBASE_CLOG_CFG = G1_29DOF_TORSOBASE_CFG.copy()
# 使用 clog 版本的 URDF，并设置碰撞体类型为凸分解
G1_29DOF_TORSOBASE_CLOG_CFG.spawn = sim_utils.UrdfFileCfg(
    asset_path=os.path.join(__file_dir__, "resources/unitree_g1/urdf/g1_29dof_torsobase_clog.urdf"),
    replace_cylinders_with_capsules=False,
    merge_fixed_joints=False,
    fix_base=False,
    self_collision=True,
    activate_contact_sensors=True,
    collider_type="convex_decomposition",
)
# 同样使用 URDF 中的关节驱动刚度
G1_29DOF_TORSOBASE_CLOG_CFG.spawn.joint_drive.gains.stiffness = None  # use value from the URDF file

# 左右对称增强时的关节重排映射
G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping = [
    1,
    0,
    2,  # waist pitch
    4,
    3,
    5,  # waist roll
    7,
    6,
    8,  # waist yaw
    10,
    9,
    12,
    11,
    14,
    13,
    16,
    15,
    18,
    17,
    20,
    19,
    22,
    21,
    24,
    23,
    26,
    25,
    28,
    27,
]
# 左右对称增强时对应关节的符号翻转缓冲
G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf = [
    1,
    1,
    1,  # waist pitch
    -1,
    -1,  # shoulder roll
    -1,  # waist roll
    -1,
    -1,  # shoulder yaw
    -1,  # waist yaw
    1,
    1,
    1,
    1,
    -1,
    -1,  # wrist roll
    -1,
    -1,  # hip roll
    1,
    1,  # wrist pitch
    -1,
    -1,  # hip yaw
    -1,
    -1,  # wrist yaw
    1,
    1,
    1,
    1,
    -1,
    -1,  # ankle roll
]


# Following the principles of BeyondMimic, and the kp/kd computation logic.
# NOTE: These logic are still being tested, so we put them here for substitution in users Cfg class
# 以下常量参考 BeyondMimic 的电机建模与 kp/kd 计算逻辑
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# 目标自然频率：10 Hz
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
# 阻尼比
DAMPING_RATIO = 2.0

# 根据电机转子惯量和自然频率计算刚度
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

# 根据电机转子惯量、阻尼比和自然频率计算阻尼
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# BeyondMimic 风格的 G1 29DOF 执行器配置，使用隐式执行器
beyondmimic_g1_29dof_actuators = {
    # 腿部执行器
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_knee_joint": 139.0,
        },
        velocity_limit_sim={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 32.0,
            ".*_knee_joint": 20.0,
        },
        stiffness={
            ".*_hip_pitch_joint": STIFFNESS_7520_14,
            ".*_hip_roll_joint": STIFFNESS_7520_22,
            ".*_hip_yaw_joint": STIFFNESS_7520_14,
            ".*_knee_joint": STIFFNESS_7520_22,
        },
        damping={
            ".*_hip_pitch_joint": DAMPING_7520_14,
            ".*_hip_roll_joint": DAMPING_7520_22,
            ".*_hip_yaw_joint": DAMPING_7520_14,
            ".*_knee_joint": DAMPING_7520_22,
        },
        armature={
            ".*_hip_pitch_joint": ARMATURE_7520_14,
            ".*_hip_roll_joint": ARMATURE_7520_22,
            ".*_hip_yaw_joint": ARMATURE_7520_14,
            ".*_knee_joint": ARMATURE_7520_22,
        },
    ),
    # 脚踝执行器
    "feet": ImplicitActuatorCfg(
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
    ),
    # 腰部 roll/pitch 执行器
    "waist": ImplicitActuatorCfg(
        effort_limit_sim=50,
        velocity_limit_sim=37.0,
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
    ),
    # 腰部 yaw 执行器
    "waist_yaw": ImplicitActuatorCfg(
        effort_limit_sim=88,
        velocity_limit_sim=32.0,
        joint_names_expr=["waist_yaw_joint"],
        stiffness=STIFFNESS_7520_14,
        damping=DAMPING_7520_14,
        armature=ARMATURE_7520_14,
    ),
    # 手臂执行器，包括肩、肘、腕
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        stiffness={
            ".*_shoulder_pitch_joint": STIFFNESS_5020,
            ".*_shoulder_roll_joint": STIFFNESS_5020,
            ".*_shoulder_yaw_joint": STIFFNESS_5020,
            ".*_elbow_joint": STIFFNESS_5020,
            ".*_wrist_roll_joint": STIFFNESS_5020,
            ".*_wrist_pitch_joint": STIFFNESS_4010,
            ".*_wrist_yaw_joint": STIFFNESS_4010,
        },
        damping={
            ".*_shoulder_pitch_joint": DAMPING_5020,
            ".*_shoulder_roll_joint": DAMPING_5020,
            ".*_shoulder_yaw_joint": DAMPING_5020,
            ".*_elbow_joint": DAMPING_5020,
            ".*_wrist_roll_joint": DAMPING_5020,
            ".*_wrist_pitch_joint": DAMPING_4010,
            ".*_wrist_yaw_joint": DAMPING_4010,
        },
        armature={
            ".*_shoulder_pitch_joint": ARMATURE_5020,
            ".*_shoulder_roll_joint": ARMATURE_5020,
            ".*_shoulder_yaw_joint": ARMATURE_5020,
            ".*_elbow_joint": ARMATURE_5020,
            ".*_wrist_roll_joint": ARMATURE_5020,
            ".*_wrist_pitch_joint": ARMATURE_4010,
            ".*_wrist_yaw_joint": ARMATURE_4010,
        },
    ),
}


# BeyondMimic 风格的 G1 29DOF 延迟执行器配置
beyondmimic_g1_29dof_delayed_actuators = {
    # 腿部延迟执行器
    "legs": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_knee_joint": 139.0,
        },
        velocity_limit_sim={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 32.0,
            ".*_knee_joint": 20.0,
        },
        stiffness={
            ".*_hip_pitch_joint": STIFFNESS_7520_14,
            ".*_hip_roll_joint": STIFFNESS_7520_22,
            ".*_hip_yaw_joint": STIFFNESS_7520_14,
            ".*_knee_joint": STIFFNESS_7520_22,
        },
        damping={
            ".*_hip_pitch_joint": DAMPING_7520_14,
            ".*_hip_roll_joint": DAMPING_7520_22,
            ".*_hip_yaw_joint": DAMPING_7520_14,
            ".*_knee_joint": DAMPING_7520_22,
        },
        armature={
            ".*_hip_pitch_joint": ARMATURE_7520_14,
            ".*_hip_roll_joint": ARMATURE_7520_22,
            ".*_hip_yaw_joint": ARMATURE_7520_14,
            ".*_knee_joint": ARMATURE_7520_22,
        },
        min_delay=0,
        max_delay=2,
    ),
    # 脚踝延迟执行器
    "feet": DelayedPDActuatorCfg(
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
        min_delay=0,
        max_delay=2,
    ),
    # 腰部 roll/pitch 延迟执行器
    "waist": DelayedPDActuatorCfg(
        effort_limit_sim=50,
        velocity_limit_sim=37.0,
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
        min_delay=0,
        max_delay=2,
    ),
    # 腰部 yaw 延迟执行器
    "waist_yaw": DelayedPDActuatorCfg(
        effort_limit_sim=88,
        velocity_limit_sim=32.0,
        joint_names_expr=["waist_yaw_joint"],
        stiffness=STIFFNESS_7520_14,
        damping=DAMPING_7520_14,
        armature=ARMATURE_7520_14,
        min_delay=0,
        max_delay=2,
    ),
    # 手臂延迟执行器，包括肩、肘、腕
    "arms": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        stiffness={
            ".*_shoulder_pitch_joint": STIFFNESS_5020,
            ".*_shoulder_roll_joint": STIFFNESS_5020,
            ".*_shoulder_yaw_joint": STIFFNESS_5020,
            ".*_elbow_joint": STIFFNESS_5020,
            ".*_wrist_roll_joint": STIFFNESS_5020,
            ".*_wrist_pitch_joint": STIFFNESS_4010,
            ".*_wrist_yaw_joint": STIFFNESS_4010,
        },
        damping={
            ".*_shoulder_pitch_joint": DAMPING_5020,
            ".*_shoulder_roll_joint": DAMPING_5020,
            ".*_shoulder_yaw_joint": DAMPING_5020,
            ".*_elbow_joint": DAMPING_5020,
            ".*_wrist_roll_joint": DAMPING_5020,
            ".*_wrist_pitch_joint": DAMPING_4010,
            ".*_wrist_yaw_joint": DAMPING_4010,
        },
        armature={
            ".*_shoulder_pitch_joint": ARMATURE_5020,
            ".*_shoulder_roll_joint": ARMATURE_5020,
            ".*_shoulder_yaw_joint": ARMATURE_5020,
            ".*_elbow_joint": ARMATURE_5020,
            ".*_wrist_roll_joint": ARMATURE_5020,
            ".*_wrist_pitch_joint": ARMATURE_4010,
            ".*_wrist_yaw_joint": ARMATURE_4010,
        },
        min_delay=0,
        max_delay=2,
    ),
}


# 根据执行器力矩上限和刚度，计算 BeyondMimic 风格的 action scale
beyondmimic_action_scale = {}
for a in beyondmimic_g1_29dof_actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            beyondmimic_action_scale[n] = 0.25 * e[n] / s[n]

# POPSICLE 版本的 G1 29DOF 配置，使用显式 ArticulationCfg 定义
G1_29DOF_TORSOBASE_POPSICLE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{__file_dir__}/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.82),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=beyondmimic_g1_29dof_actuators,
)

# 官方 Unitree G1 rev1.0：pelvis 为浮动根。与 torso-base 资产并存，换根任务
# 使用此配置并重新训练，旧 torso-root checkpoint 不直接复用。
G1_29DOF_REV10_PELVIS_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{__file_dir__}/resources/unitree_g1/official_rev10/g1_29dof_rev_1_0.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False,
            linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7585),
        joint_pos={
            ".*_hip_pitch_joint": -0.312, ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363, ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=beyondmimic_g1_29dof_actuators,
)

# G1 29DOF 的链接名称列表，顺序不保证
G1_29DOF_LINKS = [  # Order not guaranteed.
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "waist_yaw_link",
    "waist_roll_link",
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
