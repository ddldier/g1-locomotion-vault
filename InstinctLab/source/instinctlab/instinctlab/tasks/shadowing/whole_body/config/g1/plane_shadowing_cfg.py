# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 版权所有 (c) 2022-2024，The Isaac Lab Project Developers。
# 保留所有权利。
# SPDX 许可证标识：BSD-3-Clause

# 导入 NumPy，用于数值计算
import numpy as np
# 导入操作系统相关功能，例如路径展开
import os
# 导入 YAML 库，用于读写 YAML 配置文件
import yaml
# 导入 partial，用于创建偏函数
from functools import partial

# 导入 Isaac Lab 的环境 MDP（马尔可夫决策过程）模块，通常包含观测、奖励、终止等函数
import isaaclab.envs.mdp as mdp
# 导入 Isaac Lab 的 ViewerCfg，用于配置可视化查看器
from isaaclab.envs import ViewerCfg
# 导入课程学习相关的配置类
from isaaclab.managers import CurriculumTermCfg
# 导入奖励项配置类，并重命名为 RewTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
# 导入场景实体配置类，用于指定机器人、运动参考等实体
from isaaclab.managers import SceneEntityCfg
# 导入终止项配置类，并重命名为 DoneTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
# 导入 configclass 装饰器，用于定义配置类
from isaaclab.utils import configclass

# 导入 instinctlab 的环境 MDP 模块
import instinctlab.envs.mdp as instinct_mdp
# 导入 instinctlab 的 shadowing 任务 MDP 模块
import instinctlab.tasks.shadowing.mdp as shadowing_mdp
# 导入 instinctlab 的全身 shadowing 环境配置模块，并重命名为 shadowing_cfg
import instinctlab.tasks.shadowing.whole_body.shadowing_env_cfg as shadowing_cfg

##
# Pre-defined configs
##
# 预定义配置
##
# 从 instinctlab.assets.unitree_g1 导入 G1 机器人的相关配置
from instinctlab.assets.unitree_g1 import (
    # G1 29 自由度、带躯干基座、Popsicle 风格的机器人配置
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    # 29 自由度 G1 的对称增强关节映射
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    # 29 自由度 G1 的对称增强关节反向缓冲区
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    # BeyondMimic 动作缩放
    beyondmimic_action_scale,
    # BeyondMimic G1 29 自由度执行器配置
    beyondmimic_g1_29dof_actuators,
)
# 导入多奖励配置类
from instinctlab.managers import MultiRewardCfg
# 导入监控器相关配置
from instinctlab.monitors import (
    # 执行器监控项
    ActuatorMonitorTerm,
    # 监控项配置类
    MonitorTermCfg,
    # 奖励总和监控项
    RewardSumMonitorTerm,
    # Shadowing 基座位置监控项
    ShadowingBasePosMonitorTerm,
    # Shadowing 关节参考监控项
    ShadowingJointReferenceMonitorTerm,
)
# 导入运动参考管理器配置类
from instinctlab.motion_reference import MotionReferenceManagerCfg
# 导入 AIST++ 运动配置文件，并重命名为 AistppMotionCfgBase
from instinctlab.motion_reference.motion_files.aistpp_motion_cfg import AistppMotionCfg as AistppMotionCfgBase
# 导入 AMASS 运动配置文件，并重命名为 AmassMotionCfgBase
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
# 导入运动插值工具函数
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
# 导入人形机器人 SMPL 旋转 IK 工具
from instinctlab.utils.humanoid_ik import HumanoidSmplRotationalIK

# 定义组合方法为 "prod"（乘积），可能用于多奖励组合
combine_method = "prod"
# 将 G1 配置赋值给 G1_CFG，便于后续引用
G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

# 以下是被注释掉的不同运动名称和文件选择，用于切换不同的运动数据集
# MOTION_NAME = "AccadRun" # success
# _hacked_selected_file_ = "ACCAD/Male2Running_c3d/C5 - walk to run_retargetted.npz"

# MOTION_NAME = "AccadMartialBounce"
# _hacked_selected_file_ = "ACCAD/MartialArtsWalksTurns_c3d/E8 - bounce_retargetted.npz"
# MOTION_NAME = "KitStomp"
# _hacked_selected_file_ = "KIT/3/stomp_left03_retargetted.npz"

# MOTION_NAME = "AccadMartialSpin"
# _hacked_selected_file_ = "ACCAD/Male2MartialArtsKicks_c3d/G20_-__reverse_spin_cresent_right_retargetted.npz"
# MOTION_NAME = "KitStretch"  # requires balancing
# _hacked_selected_file_ = "KIT/3/streching_leg01_retargetted.npz"

# 当前选择的运动名称：LafanKungfu1
MOTION_NAME = "LafanKungfu1"
# 当前选择的运动文件列表
_hacked_selected_files_ = ["fightAndSports1_subject1_retargetted.npz"]
# MOTION_NAME = "LafanSprint1"
# _hacked_selected_files_ = ["sprint1_subject2_retargetted.npz"]

# MOTION_NAME = "test"
# _hacked_selected_files_ = ["CMU/90/90_26_retargetted.npz"]

# 重新定义运动名称为 LafanFight5Files
MOTION_NAME = "LafanFight5Files"
# 定义数据集路径（展开用户主目录）
_path_ = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
# 选择 5 个格斗相关的运动文件
_hacked_selected_files_ = [
    "fight1_subject2_retargetted.npz",
    "fight1_subject3_retargetted.npz",
    "fight1_subject5_retargetted.npz",
    "fightAndSports1_subject1_retargetted.npz",
    "fightAndSports1_subject4_retargetted.npz",
]

# 再次重新定义运动名称为 LafanFiltered
MOTION_NAME = "LafanFiltered"
# 数据集路径同上
_path_ = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
# 选择大量经过筛选的运动文件，注释 "O" 表示可用，"X" 表示不可用或有问题
_hacked_selected_files_ = [
    "aiming1_subject1_retargetted.npz",  # O
    "aiming1_subject4_retargetted.npz",  # O
    "aiming2_subject2_retargetted.npz",  # O
    "aiming2_subject3_retargetted.npz",  # O
    "aiming2_subject5_retargetted.npz",  # O
    "dance1_subject1_retargetted.npz",  # O
    "dance1_subject2_retargetted.npz",  # O
    "dance1_subject3_retargetted.npz",  # O
    "dance2_subject1_retargetted.npz",  # O
    "dance2_subject2_retargetted.npz",  # O
    "dance2_subject3_retargetted.npz",  # O
    "dance2_subject4_retargetted.npz",  # O
    "dance2_subject5_retargetted.npz",  # O
    "fallAndGetUp1_subject1_retargetted.npz",  # O
    "fallAndGetUp1_subject4_retargetted.npz",  # O
    "fallAndGetUp1_subject5_retargetted.npz",  # O
    "fallAndGetUp2_subject2_retargetted.npz",  # O
    "fallAndGetUp2_subject3_retargetted.npz",  # O
    "fallAndGetUp3_subject1_retargetted.npz",  # O
    "fight1_subject2_retargetted.npz",  # O
    "fight1_subject3_retargetted.npz",  # O
    "fight1_subject5_retargetted.npz",  # O
    "fightAndSports1_subject1_retargetted.npz",  # O
    "fightAndSports1_subject4_retargetted.npz",  # O
    "ground1_subject1_retargetted.npz",  # O
    "ground1_subject4_retargetted.npz",  # O
    "ground1_subject5_retargetted.npz",  # O
    "ground2_subject2_retargetted.npz",  # O
    "ground2_subject3_retargetted.npz",  # O
    "jumps1_subject1_retargetted.npz",  # O
    "jumps1_subject2_retargetted.npz",  # O
    "jumps1_subject5_retargetted.npz",  # O
    "multipleActions1_subject1_retargetted.npz",  # O
    "multipleActions1_subject2_retargetted.npz",  # O
    # "multipleActions1_subject3_retargetted.npz", # X
    # "multipleActions1_subject4_retargetted.npz", # - (some sitting pose, but seems torlerable)
    # "obstacles1_subject1_retargetted.npz", # X
    # "obstacles1_subject2_retargetted.npz", # X
    # "obstacles1_subject5_retargetted.npz", # X
    # "obstacles2_subject1_retargetted.npz", # X
    # "obstacles2_subject2_retargetted.npz", # X
    # "obstacles2_subject5_retargetted.npz", # disable all obstacles
    # "obstacles3_subject3_retargetted.npz", # disable all obstacles
    # "obstacles3_subject4_retargetted.npz", # disable all obstacles
    # "obstacles4_subject2_retargetted.npz", # disable all obstacles
    # "obstacles4_subject3_retargetted.npz", # X  # disable all obstacles
    # "obstacles4_subject4_retargetted.npz", # disable all obstacles
    # "obstacles5_subject2_retargetted.npz", # disable all obstacles
    # "obstacles5_subject3_retargetted.npz", # disable all obstacles
    # "obstacles5_subject4_retargetted.npz", # disable all obstacles
    # "obstacles6_subject1_retargetted.npz", # disable all obstacles
    # "obstacles6_subject4_retargetted.npz", # disable all obstacles
    # "obstacles6_subject5_retargetted.npz", # disable all obstacles
    "push1_subject2_retargetted.npz",  # O
    "pushAndFall1_subject1_retargetted.npz",  # O
    "pushAndFall1_subject4_retargetted.npz",  # O
    "pushAndStumble1_subject2_retargetted.npz",  # O
    "pushAndStumble1_subject3_retargetted.npz",  # O
    "pushAndStumble1_subject5_retargetted.npz",  # O
    "run1_subject2_retargetted.npz",
    "run1_subject5_retargetted.npz",
    "run2_subject1_retargetted.npz",
    "run2_subject4_retargetted.npz",
    "sprint1_subject2_retargetted.npz",
    "sprint1_subject4_retargetted.npz",
    "walk1_subject1_retargetted.npz",
    "walk1_subject2_retargetted.npz",
    "walk1_subject5_retargetted.npz",
    "walk2_subject1_retargetted.npz",
    "walk2_subject3_retargetted.npz",
    "walk2_subject4_retargetted.npz",
    "walk3_subject1_retargetted.npz",
    "walk3_subject2_retargetted.npz",
    "walk3_subject3_retargetted.npz",
    "walk3_subject4_retargetted.npz",
    "walk3_subject5_retargetted.npz",
    "walk4_subject1_retargetted.npz",  # O
]

# 以下是被注释掉的其他运动配置
# MOTION_NAME = "LafanGetup2S3"
# _path_ = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
# _hacked_selected_files_ = [
#     "fallAndGetUp2_subject3_retargetted.npz",
# ]

# 将选中的文件列表写入临时 YAML 文件 /tmp/{MOTION_NAME}.yaml
with open(f"/tmp/{MOTION_NAME}.yaml", "w") as f:
    # 使用 yaml.dump 将字典写入文件
    yaml.dump(
        {
            "selected_files": _hacked_selected_files_,
        },
        f,
    )


# 定义 AmassMotionCfg 配置类，继承自 AmassMotionCfgBase
@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    # 以下是被注释掉的路径选项，用于不同数据集
    # path = os.path.expanduser("~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702")
    # path = os.path.expanduser("~/Datasets/AMASS_SMPLX-NG_GMR_29dof_g1_torsoBase_retargetted_20250825_instinctnpz")
    # path = os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
    # path = os.path.expanduser("~/Datasets/AMASS_SMPLX-NG_GMR_29dof_g1_torsoBase_retargetted_20250901_instinctnpz")
    # path = _path_
    # 当前使用的数据集路径
    path = os.path.expanduser("~/Datasets/NoKov-Marslab-Motions-instinctnpz/20251016_diveroll4_single")
    # 重定向函数，这里设为 None
    retargetting_func = None
    # 过滤后的运动选择文件路径，这里为 None
    filtered_motion_selection_filepath = None
    # 运动从中间范围开始的概率范围 [0.0, 0.8]
    motion_start_from_middle_range = [0.0, 0.8]
    # 运动起始高度偏移
    motion_start_height_offset = 0.0
    # 是否确保链接低于零地面，这里为 False
    ensure_link_below_zero_ground = False
    # 环境起始片段采样策略：注释掉了 "concat_motion_bins"，使用 "independent"
    # env_starting_stub_sampling_strategy = "concat_motion_bins"
    env_starting_stub_sampling_strategy = "independent"
    # 缓冲区设备设置为 "output_device"
    buffer_device = "output_device"
    # 运动插值函数
    motion_interpolate_func = motion_interpolate_bilinear
    # 速度估计方法
    velocity_estimation_method = "frontbackward"
    # 运动片段长度（秒）
    motion_bin_length_s = 1.0
    # 以下是被注释掉的文件列表
    # _hacked_selected_files_ = _hacked_selected_files_


# 定义运动参考管理器配置
motion_reference_cfg = MotionReferenceManagerCfg(
    # 机器人主链接的 Prim 路径
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    # 机器人模型路径
    robot_model_path=G1_CFG.spawn.asset_path,
    # 参考机器人的 Prim 路径
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    # 感兴趣的链接列表，用于观测或奖励计算
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    # 以下是被注释掉的对称增强链接映射
    # symmetric_augmentation_link_mapping=[
    #     0,
    #     1,
    #     3,
    #     2,
    #     5,
    #     4,
    #     7,
    #     6,
    #     9,
    #     8,
    #     11,
    #     10,
    #     13,
    #     12,
    # ],
    # symmetric_augmentation_joint_mapping=G1_29Dof_torsoBase_symmetric_augmentation_joint_mapping,
    # symmetric_augmentation_joint_reverse_buf=G1_29Dof_torsoBase_symmetric_augmentation_joint_reverse_buf,
    # 对称增强相关配置设为 None
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    # 帧间隔（秒）
    frame_interval_s=0.02,
    # 更新周期（秒）
    update_period=0.02,
    # 帧数
    num_frames=10,
    # 数据起始时间：从当前时间开始
    data_start_from="current_time",
    # 可视化机器人偏移
    visualizing_robot_offset=(0.0, 1.5, 0.0),
    # 可视化机器人参考系
    visualizing_robot_from="reference_frame",
    # 运动缓冲区配置，这里使用 MOTION_NAME 作为键，值为 AmassMotionCfg 实例
    motion_buffers={
        # 以下是被注释掉的其他运动缓冲区配置
        #     "CMU_KIT": AmassMotionCfg(
        #         path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_retargetted_20250702"),  # type: ignore
        #         filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
        #             "~/Datasets/AMASS_selections/CMU_KIT_weighted_retargetted_20250702.yaml",
        #         ),
        #     ),
        # "CMU_KIT_DanceDB_BioMotionLab": AmassMotionCfg(
        #     path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_DanceDB_BioMotionLab_retargetted_20250702"),  # type: ignore
        #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
        #         "~/Datasets/AMASS_selections/CMU_KIT_DanceDB_BioMotionLab_weighted_retargetted_20250702.yaml",
        #     ),
        # ),
        # "CMU_KIT_ACCAD_DanceDB_HumanEva": AmassMotionCfg(
        #     path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702"),  # type: ignore
        #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
        #         # "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_retargetted_20250702.yaml",
        #         "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_moverange_20250724_retargetted_20250702.yaml",
        #     ),
        # ),
        # "UbisoftLAFAN1_GMR": AmassMotionCfg(
        #     path=os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz"),  # type: ignore
        #     filtered_motion_selection_filepath=None,
        # ),
        # 使用当前 MOTION_NAME 作为键，创建默认的 AmassMotionCfg
        MOTION_NAME: AmassMotionCfg(),
    },
    # 多进程分割方法为 "Even"
    mp_split_method="Even",
)


# 定义 G1 平面 Shadowing 环境配置类，继承自 shadowing_cfg.ShadowingEnvCfg
@configclass
class G1PlaneShadowingEnvCfg(shadowing_cfg.ShadowingEnvCfg):
    # 场景配置：继承 ShadowingSceneCfg，设置环境数量为 4096
    scene: shadowing_cfg.ShadowingSceneCfg = shadowing_cfg.ShadowingSceneCfg(
        num_envs=4096,
        # 机器人配置，替换 prim_path
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        # 运动参考配置
        motion_reference=motion_reference_cfg,
    )

    # 后初始化方法
    def __post_init__(self):
        # 调用父类的后初始化
        super().__post_init__()

        # 将 link_of_interests 添加到策略观测中
        if self.observations.policy.__dict__.get("link_pos", None) is not None:
            # 设置 link_pos 的 body_names 为运动参考中的感兴趣链接
            self.observations.policy.link_pos.params["asset_cfg"].body_names = (
                self.scene.motion_reference.link_of_interests
            )
        if self.observations.policy.__dict__.get("link_rot", None) is not None:
            # 设置 link_rot 的 body_names
            self.observations.policy.link_rot.params["asset_cfg"].body_names = (
                self.scene.motion_reference.link_of_interests
            )
        # 如果存在 critic 观测，也进行相同设置
        if hasattr(self.observations, "critic") and self.observations.critic is not None:
            if self.observations.critic.__dict__.get("link_pos", None) is not None:
                self.observations.critic.link_pos.params["asset_cfg"].body_names = (
                    self.scene.motion_reference.link_of_interests
                )
            if self.observations.critic.__dict__.get("link_rot", None) is not None:
                self.observations.critic.link_rot.params["asset_cfg"].body_names = (
                    self.scene.motion_reference.link_of_interests
                )

        # 设置机器人执行器为 beyondmimic_g1_29dof_actuators
        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        # 以下是被注释掉的执行器延迟设置
        # for actuator in self.scene.robot.actuators.values():
        #     actuator.min_delay = 0
        #     actuator.max_delay = 0
        # 设置动作的关节位置缩放
        self.actions.joint_pos.scale = beyondmimic_action_scale

        # 断言：目前只支持单个运动缓冲区
        assert (
            len(list(self.scene.motion_reference.motion_buffers.keys())) == 1
        ), "Only support single motion buffer for now"
        # 获取唯一的运动缓冲区
        motion_buffer = list(self.scene.motion_reference.motion_buffers.values())[0]
        # 如果运动片段长度不为 None，则根据采样策略设置课程学习项
        if motion_buffer.motion_bin_length_s is not None:
            if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins":
                # 使用 BeyondConcatMotionAdaptiveWeighting
                self.curriculum.beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
                    func=instinct_mdp.BeyondConcatMotionAdaptiveWeighting,
                )
            elif motion_buffer.env_starting_stub_sampling_strategy == "independent":
                # 使用 BeyondMimicAdaptiveWeighting
                self.curriculum.beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
                    func=instinct_mdp.BeyondMimicAdaptiveWeighting,
                )
            else:
                # 不支持其他策略，抛出异常
                raise ValueError(
                    "Unsupported env starting stub sampling method:"
                    f" {motion_buffer.env_starting_stub_sampling_strategy}"
                )

        # 设置运行名称，由多个部分组成
        self.run_name: str = "".join(
            [
                "G1Shadowing",
                f"_{MOTION_NAME}",
                (
                    "_odomObs"
                    if ("base_lin_vel" in self.observations.policy.__dict__.keys())
                    and self.commands.position_b_ref_command.anchor_frame == "robot"
                    else ""
                ),
                # 以下是被注释掉的可选名称部分
                # (
                #     "_" + "-".join(self.scene.motion_reference.motion_buffers.keys())
                #     if self.scene.motion_reference.motion_buffers
                #     else ""
                # ),
                # (
                #     f"_proprioHist{self.observations.policy.joint_pos.history_length}"
                #     if self.observations.policy.joint_pos.history_length > 0
                #     else ""
                # ),
                # (
                #     f"_futureRef{self.scene.motion_reference.num_frames}"
                #     if self.scene.motion_reference.num_frames > 1
                #     else ""
                # ),
                # f"_FrameStartFrom{self.scene.motion_reference.data_start_from}",
                # "_forLoopMotionWeights",
                # "_forLoopMotionSample",
                ("_pgTermXYalso" if not self.terminations.base_pg_too_far.params["z_only"] else ""),
                (
                    "_concatMotionBins"
                    if motion_buffer.env_starting_stub_sampling_strategy == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
                "_fixFramerate_diveroll4",
            ]
        )


# 定义用于播放/测试的环境配置类，继承自 G1PlaneShadowingEnvCfg
@configclass
class G1PlaneShadowingEnvCfg_PLAY(G1PlaneShadowingEnvCfg):
    # 场景配置：环境数量为 1，环境间距 2.5
    scene: shadowing_cfg.ShadowingSceneCfg = shadowing_cfg.ShadowingSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        # 机器人配置
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        # 参考机器人配置
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        # 运动参考配置，替换 debug_vis 为 True
        motion_reference=motion_reference_cfg.replace(
            debug_vis=True,
            # 以下是被注释掉的其他运动缓冲区配置
            # motion_buffers={
            # "amass": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702"),  # type: ignore
            #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
            #         "~/Datasets/AMASS_selections/amass_test_motion_files.yaml"
            #     ),
            # ),
            # "CMU_KIT_ACCAD_DanceDB_HumanEva": AmassMotionCfg(
            #     path=os.path.expanduser(  # type: ignore
            #         "~/Datasets/AMASS_CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702"
            #     ),
            #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
            #         # "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_moverange_top50_20250724_retargetted_20250702.yaml",
            #         "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_moverange_top30-50_20250724_retargetted_20250702.yaml",
            #         # "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_weighted_orient_height_20250725_top20_retargetted_20250702.yaml",
            #         # "~/Datasets/AMASS_selections/CMU_KIT_ACCAD_DanceDB_HumanEva_retargetted_20250702_noFastVel_filterOutGimbal_filterOutSit_20250825.yaml",
            #     ),
            # ),
            # "AMASS_GMR": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/AMASS_SMPLX-NG_GMR_29dof_g1_retargetted_instinctnpz"),  # type: ignore
            #     filtered_motion_selection_filepath=os.path.expanduser(  # type: ignore
            #         "~/Datasets/AMASS_selections/AMASS_SMPLX-NG_GMR_weighted_moverange_20250724_retargetted_instinctnpz.yaml"
            #     ),
            # ),
            # "instinct_testset": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/ProjectInstinct/tests"),  # type: ignore
            #     filtered_motion_selection_filepath=None,
            # ),
            # "UbisoftLAFAN1_GMR": AmassMotionCfg(
            #     path=os.path.expanduser("~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz_selected"),  # type: ignore
            #     filtered_motion_selection_filepath=None,
            # ),
            # },
        ),
    )
    # 查看器配置
    viewer: ViewerCfg = ViewerCfg(
        eye=[4.0, 0.75, 1.0],       # 相机位置
        lookat=[0.0, 0.75, 0.0],    # 注视点
        origin_type="asset_root",   # 原点类型
        asset_name="robot",         # 资产名称
    )

    # 后初始化方法
    def __post_init__(self):
        # 调用父类的后初始化
        super().__post_init__()

        # 随机在网格中生成机器人（而不是使用地形等级）
        self.scene.terrain.max_init_terrain_level = None
        # 减少地形数量以节省内存
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False

        # 禁用对称增强关节映射
        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        # 设置可视化标记类型为相对链接
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        # 禁用课程学习中的自适应采样
        self.curriculum.beyond_adaptive_sampling = None
        # 禁用推机器人事件
        self.events.push_robot = None
        # 禁用失败计数器平滑
        self.events.bin_fail_counter_smoothing = None

        # 在终止项中启用 print_reason 选项
        for term in self.terminations.__dict__.values():
            if term is None:
                continue
            if "print_reason" in term.params:
                term.params["print_reason"] = True
        # 以下是被注释掉的片段长度和终止项设置
        # self.episode_length_s = 10.0
        # for term_name, term in self.terminations.__dict__.items():
        #     if (not term_name == "dataset_exhausted") and (not term_name == "time_out"):
        #         self.terminations.__dict__[term_name] = None

        # 在命令中启用 debug_vis 选项
        for cmd in self.commands.__dict__.values():
            cmd.debug_vis = True

        # 以下是 PLAY 特定的监控项，被注释掉了
        # self.monitors.shoulder_actuator = MonitorTermCfg(
        #     func=ActuatorMonitorTerm,
        #     params={
        #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_shoulder_roll.*"),
        #         "torque_plot_scale": 1e-2,
        #         # "joint_vel_plot_scale": 1e-1,
        #         "joint_power_plot_scale": 1e-1,
        #     },
        # )
        # self.monitors.waist_actuator = MonitorTermCfg(
        #     func=ActuatorMonitorTerm,
        #     params={
        #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="waist_roll.*"),
        #         "torque_plot_scale": 1e-2,
        #         # "joint_vel_plot_scale": 1e-1,
        #         "joint_power_plot_scale": 1e-1,
        #     },
        # )
        # self.monitors.knee_actuator = MonitorTermCfg(
        #     func=ActuatorMonitorTerm,
        #     params={
        #         "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_knee.*"),
        #         "torque_plot_scale": 1e-2,
        #         # "joint_vel_plot_scale": 1e-1,
        #         "joint_power_plot_scale": 1e-1,
        #     },
        # )
        # self.monitors.reward_sum = MonitorTermCfg(
        #     func=RewardSumMonitorTerm,
        # )
        # self.monitors.reference_stat_case = MonitorTermCfg(
        #     func=ShadowingJointReferenceMonitorTerm,
        #     params=dict(
        #         reference_cfg=SceneEntityCfg(
        #             "motion_reference",
        #             joint_names=[
        #                 "left_hip_pitch.*",
        #             ],
        #         ),
        #     ),
        # )
        # self.monitors.shadowing_base_pos = MonitorTermCfg(
        #     func=ShadowingBasePosMonitorTerm,
        #     params=dict(
        #         robot_cfg=SceneEntityCfg("robot"),
        #         motion_reference_cfg=SceneEntityCfg("motion_reference"),
        #     ),
        # )