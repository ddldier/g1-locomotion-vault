# 导入操作系统相关功能
import os
# 从 dataclasses 导入 MISSING，用于标记缺失的配置项
from dataclasses import MISSING

# 导入 Isaac Lab 的环境 MDP 模块
import isaaclab.envs.mdp as mdp
# 导入 Isaac Lab 的仿真工具
import isaaclab.sim as sim_utils
# 导入 ArticulationCfg（关节配置）和 AssetBaseCfg（基础资产配置）
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# 导入课程学习、事件、观测、奖励、场景实体、终止等配置类
from isaaclab.managers import CurriculumTermCfg, EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import RewardTermCfg as RewTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTermCfg
# 导入交互式场景配置类
from isaaclab.scene import InteractiveSceneCfg
# 导入接触传感器配置类
from isaaclab.sensors import ContactSensorCfg
# 导入地形生成器、地形导入器配置类
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
# 导入配置类装饰器和均匀噪声配置
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

# 导入 instinctlab 的环境 MDP 模块
import instinctlab.envs.mdp as instinct_mdp
# 导入 instinctlab 的 shadowing 任务 MDP 模块
import instinctlab.tasks.shadowing.mdp as shadowing_mdp
# 导入 instinctlab 的基于管理器的 RL 环境配置基类
from instinctlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
# 导入多奖励配置类
from instinctlab.managers import MultiRewardCfg
# 导入各种监控项配置
from instinctlab.monitors import (
    BodyStatMonitorTerm,
    JointStatMonitorTerm,
    MonitorTermCfg,
    MotionReferenceMonitorTerm,
    ShadowingJointPosMonitorTerm,
    ShadowingJointVelMonitorTerm,
    ShadowingLinkPosMonitorTerm,
    ShadowingPositionMonitorTerm,
    ShadowingProgressMonitorTerm,
    ShadowingRotationMonitorTerm,
    ShadowingVelocityMonitorTerm,
    TorqueMonitorSensorCfg,
)
# 导入运动参考管理器配置类
from instinctlab.motion_reference import MotionReferenceManagerCfg
# 导入 Perlin 平面地形配置类
from instinctlab.terrains.height_field import PerlinPlaneTerrainCfg


# 定义 Shadowing 场景配置类，继承自 InteractiveSceneCfg
@configclass
class ShadowingSceneCfg(InteractiveSceneCfg):
    """Configuration for the BeyondMimic scene with necessary scene entities as motion reference."""

    # 环境间距为 4.0
    env_spacing = 4.0

    # robots
    # 机器人关节配置，必须提供（MISSING）
    robot: ArticulationCfg = MISSING

    # robot reference articulation
    # 参考机器人关节配置，默认为 None
    robot_reference: ArticulationCfg = None

    # motion reference
    # 运动参考管理器配置，必须提供（MISSING）
    motion_reference: MotionReferenceManagerCfg = MISSING  # type: ignore

    # terrain
    # 地形配置：使用平面地形，无地形生成器
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        # 物理材质：摩擦组合模式为 multiply，静摩擦和动摩擦均为 1.0
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # 视觉材质：使用 MDL 文件
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    # lights
    # 方向光：颜色为浅灰，强度 3000
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # 穹顶光：颜色为深灰，强度 1000
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    # 接触力传感器：监测机器人所有刚体，历史长度 3，追踪空中时间，力阈值 10.0
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )

    # 后初始化方法
    def __post_init__(self):
        # 如果运动参考配置缺失或未开启 debug_vis，则删除 robot_reference 属性
        if type(self.motion_reference) is type(MISSING) or not self.motion_reference.debug_vis:
            delattr(self, "robot_reference")


# 定义命令配置类
@configclass
class CommandCfg:
    """BeyondMimic command configuration following their approach."""

    # 位置参考命令：实时模式，非当前状态命令，锚定在机器人坐标系
    position_ref_command = instinct_mdp.PositionRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        anchor_frame="robot",
    )
    # 位置参考命令（参考坐标系）：实时模式，非当前状态命令，锚定在参考坐标系
    position_b_ref_command = instinct_mdp.PositionRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        anchor_frame="reference",
    )
    # 旋转参考命令：实时模式，非当前状态命令，基座坐标系，旋转模式为 tannorm
    rotation_ref_command = instinct_mdp.RotationRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        in_base_frame=True,
        rotation_mode="tannorm",
    )
    # 关节位置参考命令：非当前状态命令
    joint_pos_ref_command = instinct_mdp.JointPosRefCommandCfg(current_state_command=False)
    # 关节速度参考命令：非当前状态命令
    joint_vel_ref_command = instinct_mdp.JointVelRefCommandCfg(current_state_command=False)


# 定义动作配置类
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # 关节位置动作：作用于名为 "robot" 的资产，所有关节，缩放 0.5
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
    )


# 定义观测配置类
@configclass
class ObservationsCfg:
    """BeyondMimic observation configuration following their approach."""

    # 策略观测组配置
    @configclass
    class PolicyObsCfg(ObsGroupCfg):
        """Policy observations for BeyondMimic."""

        # BeyondMimic 特定的参考观测
        # 关节位置参考：来自命令 joint_pos_ref_command
        joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
        # 关节速度参考：来自命令 joint_vel_ref_command
        joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
        # 位置参考：来自命令 position_b_ref_command，添加均匀噪声 [-0.25, 0.25]
        position_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_b_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        )
        # 旋转参考：来自命令 rotation_ref_command，添加均匀噪声 [-0.05, 0.05]
        rotation_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )

        # proprioception
        # 本体感觉
        # 基座线速度（被注释掉）
        # base_lin_vel = ObsTermCfg(
        #     func=mdp.base_lin_vel,
        #     noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        # )
        # 投影重力：添加均匀噪声 [-0.05, 0.05]
        projected_gravity = ObsTermCfg(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        # 基座角速度：添加均匀噪声 [-0.2, 0.2]
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        )
        # 关节位置（相对）：添加均匀噪声 [-0.01, 0.01]
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        # 关节速度（相对）：添加均匀噪声 [-0.5, 0.5]
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        )
        # 上一步动作
        last_action = ObsTermCfg(func=mdp.last_action)

        # 后初始化：启用损坏，不拼接各项
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # 评论家观测组配置
    @configclass
    class CriticObsCfg(ObsGroupCfg):
        """Critic observations for BeyondMimic."""

        # BeyondMimic 特定的参考观测
        # 关节位置参考
        joint_pos_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
        )
        # 关节速度参考
        joint_vel_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
        )
        # 位置参考
        position_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "position_ref_command"},
        )
        # 旋转参考
        rotation_ref = ObsTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
        )

        # proprioception
        # 本体感觉
        # 链接位置（基座坐标系）：需要提供 body_names，保持顺序
        link_pos = ObsTermCfg(
            func=instinct_mdp.link_pos_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=MISSING,
                    preserve_order=True,
                ),
            },
        )
        # 链接旋转（基座坐标系，tannorm）：需要提供 body_names，保持顺序
        link_rot = ObsTermCfg(
            func=instinct_mdp.link_tannorm_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    body_names=MISSING,
                    preserve_order=True,
                ),
            },
        )
        # 基座线速度
        base_lin_vel = ObsTermCfg(
            func=mdp.base_lin_vel,
        )
        # 基座角速度
        base_ang_vel = ObsTermCfg(
            func=mdp.base_ang_vel,
        )
        # 关节位置（相对）
        joint_pos = ObsTermCfg(
            func=mdp.joint_pos_rel,
        )
        # 关节速度（相对）
        joint_vel = ObsTermCfg(
            func=mdp.joint_vel_rel,
        )
        # 上一步动作
        last_action = ObsTermCfg(
            func=mdp.last_action,
        )

        # 后初始化：不启用损坏，不拼接各项
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # 策略观测和评论家观测实例
    policy = PolicyObsCfg()
    critic = CriticObsCfg()


# 定义奖励组配置类
@configclass
class RewardGroupCfg:
    """BeyondMimic reward terms following their approach."""

    # 基座位置模仿（高斯）：权重 0.5，标准差 0.3
    base_position_imitation_gauss = RewTermCfg(
        func=instinct_mdp.base_position_imitation_gauss,
        weight=0.5,
        params={
            "std": 0.3,
        },
    )
    # 基座旋转模仿（高斯）：权重 0.5，标准差 0.4，差异类型轴角
    base_rot_imitation_gauss = RewTermCfg(
        func=instinct_mdp.base_rot_imitation_gauss,
        weight=0.5,
        params={
            "std": 0.4,
            "difference_type": "axis_angle",
        },
    )
    # 链接位置模仿（高斯）：权重 1.0，组合方法 mean_prod，不在基座坐标系，在相对世界坐标系，标准差 0.3
    link_pos_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_pos_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": 0.3,
        },
    )
    # 链接旋转模仿（高斯）：权重 1.0，组合方法 mean_prod，不在基座坐标系，在相对世界坐标系，标准差 0.4
    link_rot_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_rot_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": 0.4,
        },
    )
    # 链接线速度模仿（高斯）：权重 1.0，组合方法 mean_prod，标准差 1.0
    link_lin_vel_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_lin_vel_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "std": 1.0,
        },
    )
    # 链接角速度模仿（高斯）：权重 1.0，组合方法 mean_prod，标准差 3.14
    link_ang_vel_imitation_gauss = RewTermCfg(
        func=instinct_mdp.link_ang_vel_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "std": 3.14,
        },
    )
    # 动作变化率 L2 惩罚：权重 -0.1
    action_rate_l2 = RewTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    # 关节限位惩罚：权重 -10.0，作用于机器人所有关节
    joint_limit = RewTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # 非期望接触惩罚：权重 -0.1，使用接触力传感器，排除脚踝和手腕，阈值 1.0
    undesired_contacts = RewTermCfg(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


# 单奖励配置类，继承自 MultiRewardCfg
@configclass
class SingleRewardsCfg(MultiRewardCfg):
    """Single reward configuration for BeyondMimic."""

    # 奖励组配置
    rewards: RewardGroupCfg = RewardGroupCfg()


# 事件配置类
@configclass
class EventCfg:
    """BeyondMimic events config such as termination conditions."""

    # startup
    # 物理材质随机化：启动时，作用于机器人所有刚体，静摩擦 (0.3, 1.6)，动摩擦 (0.3, 1.2)，弹性 (0.0, 0.5)，64 个桶
    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    # 添加关节默认位置随机化：启动时，作用于机器人所有关节，偏移分布 (-0.01, 0.01)，操作 add，均匀分布
    add_joint_default_pos = EventTermCfg(
        func=instinct_mdp.randomize_default_joint_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "offset_distribution_params": (-0.01, 0.01),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # 基座质心随机化：启动时，作用于 torso_link，范围 x(-0.025, 0.025), y(-0.05, 0.05), z(-0.05, 0.05)
    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "com_range": {
                "x": (-0.025, 0.025),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    # interval
    # 推动机器人：间隔事件，间隔 (1.0, 3.0) 秒，设置速度范围
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )

    # for motion initialization and reset
    # 将运动参考与场景匹配：启动时
    match_motion_ref_with_scene = EventTermCfg(
        func=instinct_mdp.match_motion_ref_with_scene,
        mode="startup",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
        },
    )
    # 根据参考重置机器人状态：重置时
    reset_robot = EventTermCfg(
        func=instinct_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("robot"),
            "position_offset": [0.0, 0.0, 0.0],
            "dof_vel_ratio": 1.0,
            "base_lin_vel_ratio": 1.0,
            "base_ang_vel_ratio": 1.0,
            # Pose randomization (+-5cm position, +-6degrees rotation)
            # 位姿随机化（位置 ±5cm，旋转 ±6 度）
            "randomize_pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            # Velocity randomization (+-0.1 m/s linear, +-0.1 rad/s angular)
            # 速度随机化（线速度 ±0.1 m/s，角速度 ±0.1 rad/s）
            "randomize_velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
            # Joint position randomization (+-0.1 rad)
            # 关节位置随机化（±0.1 rad）
            "randomize_joint_pos_range": (-0.1, 0.1),
        },
    )
    # 失败计数器平滑：间隔事件，每 0.02 秒（每个环境步），课程名称 beyond_adaptive_sampling
    bin_fail_counter_smoothing = EventTermCfg(
        func=instinct_mdp.beyondmimic_bin_fail_counter_smoothing,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # every environment step
        params={
            "curriculum_name": "beyond_adaptive_sampling",
        },
    )


# 课程学习配置类
@configclass
class CurriculumCfg:
    """BeyondMimic curriculum terms for the MDP."""

    # 自适应采样课程：使用 BeyondMimicAdaptiveWeighting
    beyond_adaptive_sampling = CurriculumTermCfg(  # type: ignore
        func=instinct_mdp.BeyondMimicAdaptiveWeighting,
    )


# 终止条件配置类
@configclass
class TerminationCfg:
    """BeyondMimic termination terms for the MDP."""

    # 超时终止
    time_out = DoneTermCfg(func=mdp.time_out, time_out=True)
    # 基座位置距离参考太远终止：距离阈值 0.25，仅检查高度
    base_pos_too_far = DoneTermCfg(
        func=instinct_mdp.pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "distance_threshold": 0.25,
            "check_at_keyframe_threshold": -1,
            "print_reason": False,
            "height_only": True,
        },
    )
    # 基座投影重力距离参考太远终止：投影重力阈值 0.8，不限制 z 轴
    base_pg_too_far = DoneTermCfg(
        func=instinct_mdp.projected_gravity_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "projected_gravity_threshold": 0.8,  # distance on z-axis of projected gravity
            "check_at_keyframe_threshold": -1,
            "z_only": False,  # find out useful if not z_only but beyondmimic default is z_only
            "print_reason": False,
        },
    )
    # 链接位置距离参考太远终止：检查脚踝和手腕，距离阈值 0.25，仅检查高度
    link_pos_too_far = DoneTermCfg(
        func=instinct_mdp.link_pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg(
                "motion_reference",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                preserve_order=True,
            ),
            "distance_threshold": 0.25,
            "in_base_frame": False,
            "check_at_keyframe_threshold": -1,
            "height_only": True,
            "print_reason": False,
        },
    )

    # 数据集耗尽终止
    dataset_exhausted = DoneTermCfg(
        func=instinct_mdp.dataset_exhausted,
        time_out=True,
        params={
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "print_reason": False,
        },
    )
    # 超出边界终止
    out_of_border = DoneTermCfg(
        func=instinct_mdp.terrain_out_of_bounds,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
    )


# 监控配置类
@configclass
class MonitorCfg:

    # joint_torque = SceneEntityCfg("monitor_joint_torque") # NOTE: hurt the performance, so not used.
    # 关节力矩监控（注释掉，因为影响性能）
    # upper_joint_stat = MonitorTermCfg(
    #     func=JointStatMonitorTerm,
    #     params=dict(
    #         asset_cfg=SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_.*",
    #                 ".*_elbow_.*",
    #                 ".*_wrist_.*",
    #             ],
    #         ),
    #     ),
    # )
    # lower_joint_stat = MonitorTermCfg(
    #     func=JointStatMonitorTerm,
    #     params=dict(
    #         asset_cfg=SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "waist_.*",
    #                 ".*_ankle_.*",
    #                 ".*_hip_.*",
    #             ],
    #         ),
    #     ),
    # )
    # body_stat = MonitorTermCfg(
    #     func=BodyStatMonitorTerm,
    #     params=dict(
    #         asset_cfg=SceneEntityCfg(
    #             "robot",
    #             body_names=MISSING,
    #         ),
    #     ),
    # )

    # 数据集监控：采样统计间隔 500，前 5 个样本
    dataset = MonitorTermCfg(
        func=MotionReferenceMonitorTerm,
        params=dict(
            asset_cfg=SceneEntityCfg("motion_reference"),
            sample_stat_interval=500,
            top_n_samples=5,
        ),
    )
    # Shadowing 位置监控：在基座坐标系，关键帧阈值 0.03
    shadowing_position = MonitorTermCfg(
        func=ShadowingPositionMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            check_at_keyframe_threshold=0.03,
        ),
    )
    # Shadowing 旋转监控：启用掩码
    shadowing_rotation = MonitorTermCfg(
        func=ShadowingRotationMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    # Shadowing 关节位置监控：启用掩码
    shadowing_joint_pos = MonitorTermCfg(
        func=ShadowingJointPosMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    # Shadowing 关节速度监控：启用掩码
    shadowing_joint_vel = MonitorTermCfg(
        func=ShadowingJointVelMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    # Shadowing 链接位置监控：在基座坐标系，启用掩码
    shadowing_link_pos = MonitorTermCfg(
        func=ShadowingLinkPosMonitorTerm,
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            masking=True,
        ),
    )


# Shadowing 环境配置类，继承自 InstinctLabRLEnvCfg
@configclass
class ShadowingEnvCfg(InstinctLabRLEnvCfg):
    """Configuration for the shadowing environment."""

    # 场景配置：4096 个环境
    scene: ShadowingSceneCfg = ShadowingSceneCfg(num_envs=4096)
    # 动作配置
    actions: ActionsCfg = ActionsCfg()
    # 观测配置
    observations: ObservationsCfg = ObservationsCfg()
    # 命令配置
    commands: CommandCfg = CommandCfg()
    # 奖励配置
    rewards: SingleRewardsCfg = SingleRewardsCfg()
    # 事件配置
    events: EventCfg = EventCfg()
    # 课程配置
    curriculum: CurriculumCfg = CurriculumCfg()
    # 终止配置
    terminations: TerminationCfg = TerminationCfg()
    # 监控配置
    monitors: MonitorCfg = MonitorCfg()

    # 后初始化方法
    def __post_init__(self):
        # general settings
        # 通用设置
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        # 仿真设置
        self.sim.dt = 1.0 / 50.0 / self.decimation
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # 运行名称初始化为空字符串
        self.run_name = "".join([])