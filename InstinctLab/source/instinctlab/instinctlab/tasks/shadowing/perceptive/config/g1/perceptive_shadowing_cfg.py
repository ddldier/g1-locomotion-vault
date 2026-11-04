"""G1 感知式 shadowing 任务的具体配置。

这个文件只负责把通用的感知式 shadowing 环境配置成 G1 版本：

* 指定 G1 机器人和执行器；
* 指定动作参考数据（motion）以及与动作匹配的地形；
* 指定训练环境和单环境可视化（PLAY）环境的差异。

奖励、观测项、相机噪声和终止条件等通用定义位于
``perceptive/perceptive_env_cfg.py``。因此，阅读本文件时应把它看成
“机器人/数据集适配层”，而不是完整的 MDP 实现。
"""

# 当前文件中 numpy 暂未参与配置计算，但保留它是为了兼容历史实验代码。
# 如果后续确认没有其他分支使用它，可以删除该 import。
import numpy as np
import os

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# instinct_mdp 提供深度图等 InstinctLab 扩展的 MDP 函数；shadowing_mdp 是
# shadowing 任务专用函数。它们中的部分函数由父类配置间接使用，因此不要仅凭
# 本文件中是否直接出现名称来判断 import 是否可以删除。
import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.mdp as shadowing_mdp
import instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
    beyondmimic_g1_29dof_delayed_actuators,
)
from instinctlab.monitors import ActuatorMonitorTerm, MonitorTermCfg, ShadowingBasePosMonitorTerm
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.motion_files.terrain_motion_cfg import TerrainMotionCfg as TerrainMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets

# G1_CFG 是本文件中所有 robot/references 的统一机器人模型配置。
# “TORSOBASE”表示基座选择 torso_link；动作空间是 G1 的 29 个自由度。
G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

# TODO: 将这里替换为包含动作文件和 metadata.yaml 的真实目录。
# 对于 motion-matched terrain，目录中通常还需要与动作对应的地形网格文件。
MOTION_FOLDER = "{AbsolutePathOfYourDataDirectory}"


@configclass
class TerrainMotionCfg(TerrainMotionCfgBase):
    """带有配套地形的动作数据配置。

    TerrainMotionCfg 不仅读取机器人运动，还会读取 metadata.yaml 中的
    motion-terrain 对应关系，使每条翻越/攀爬动作在匹配的地形上播放。
    这是感知式 parkour/shadowing 任务的默认数据类型。
    """

    # 动作和 terrain 文件所在的根目录。
    path = os.path.expanduser(MOTION_FOLDER)
    # metadata.yaml 描述动作文件、地形文件和 terrain id 的对应关系。
    metadata_yaml = os.path.expanduser(f"{MOTION_FOLDER}/metadata.yaml")
    # 每条动作最多生成的 terrain origin 数量，影响环境并行初始化数量。
    max_origins_per_motion = 49

    # 不强制把所有链接压到地面以下；翻越动作可能暂时出现离地或特殊姿态。
    ensure_link_below_zero_ground = False
    # 训练时从动作中间开始的比例范围；[0, 0] 表示总是从动作开头开始。
    motion_start_from_middle_range = [0.0, 0.0]
    # 读取参考动作时对整体高度施加的偏移。
    motion_start_height_offset = 0.0
    # 把动作切成 1 秒一个的 bin，用于失败统计和自适应采样。
    motion_bin_length_s = 1.0
    # 将 motion buffer 放到环境的输出设备（通常是 GPU）上，减少拷贝。
    buffer_device = "output_device"
    # 在离散动作帧之间进行双线性插值，得到连续时间的参考姿态。
    motion_interpolate_func = motion_interpolate_bilinear
    # 用前向/后向差分估计参考线速度和角速度。
    velocity_estimation_method = "frontbackward"
    # 多环境训练时，把动作 bin 拼接后作为起始采样池。
    env_starting_stub_sampling_strategy = "concat_motion_bins"


@configclass
class AMASSMotionCfg(AmassMotionCfgBase):
    """只包含 AMASS/GMR 动作的配置，作为平地模式的备用方案。

    当前默认路径走 TerrainMotionCfg；当 ``PLANE_TERRAIN=True`` 时，代码会
    删除带地形的 motion buffer 并切换到这里。它只提供动作，不提供与动作
    匹配的 terrain mesh。
    """

    path = os.path.expanduser(MOTION_FOLDER)
    # None 表示不额外筛选动作文件，读取目录中可用的动作。
    filtered_motion_selection_filepath = None
    # 以下参数与 TerrainMotionCfg 保持一致，保证切换数据类型时参考动作的
    # 插值、速度估计和起始片段采样方式一致。
    ensure_link_below_zero_ground = False
    motion_start_from_middle_range = [0.0, 0.0]
    motion_start_height_offset = 0.0
    motion_bin_length_s = 1.0
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    env_starting_stub_sampling_strategy = "concat_motion_bins"


# MotionReferenceManager 是 shadowing 的“参考轨迹播放器”。它在每个仿真步
# 根据当前时间戳读取参考帧，并把参考位置、旋转、关节角和速度暴露给命令、奖励
# 以及 critic。
motion_reference_cfg = MotionReferenceManagerCfg(
    # 真实机器人在场景中的参考 prim。{ENV_REGEX_NS} 会被替换成每个环境的命名空间。
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    # 用哪个机器人 USD/资产模型来解释动作数据中的 link 和 joint。
    robot_model_path=G1_CFG.spawn.asset_path,
    # debug_vis=True 时，参考机器人通常放在 RobotReference prim 下显示。
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    # 只跟踪这些关键 link。它们同时影响 link-level observation、reward 和 monitor。
    link_of_interests=[
        # 躯干/骨盆：约束全身的主干运动。
        "pelvis",
        "torso_link",
        # 上肢：肩、肘和腕，用于翻箱子时的手臂姿态和接触动作。
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        # 下肢：髋、膝和踝，用于支撑、跨越和落地动作。
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    # None 表示不做左右对称动作增强；如使用镜像动作数据，需要提供对应映射。
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    # 相邻参考帧间隔 0.1 秒；仿真中的参考管理器每 0.02 秒更新一次。
    frame_interval_s=0.1,
    update_period=0.02,
    # 每次给策略提供当前帧之后的 10 帧参考信息。
    num_frames=10,
    data_start_from="current_time",
    # 以下偏移和可视化只用于 PLAY：把参考机器人放到真实机器人旁边，便于比较。
    # 训练时不需要显示参考机器人；如果改变场景布局，应同步检查这个偏移。
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["relative_links", "links"],
    # 当前只有一个 motion buffer，名字会在 __post_init__ 中动态读取。
    motion_buffers={
        "TerrainMotion": TerrainMotionCfg(),
    },
    # 暂不对多个 motion buffer 做 multiprocessing 切分。
    mp_split_method="None",
)


@configclass
class G1PerceptiveShadowingEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    """训练用的 G1 感知式 shadowing 环境。

    父类 ``PerceptiveShadowingEnvCfg`` 提供通用的相机、深度图、奖励、事件、
    终止条件和 terrain generator；这里仅替换成 G1 机器人及其 motion 数据。
    """

    # 4096 个并行环境用于 PPO 采样；显存不足时应通过 task 参数降低它。
    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )

    def __post_init__(self):
        # 先执行父类初始化，建立 camera、terrain、observations、rewards 等通用配置。
        super().__post_init__()

        # 相机默认只看地面；把 G1 的 link prim 加入可见网格列表后，深度图中才会
        # 正确出现机器人自身，训练出的视觉策略才能学习到身体与障碍物的关系。
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS))

        # 使用 BeyondMimic 风格的 G1 执行器和动作缩放。
        # policy 输出的是归一化动作，scale 决定它转换成关节位置目标时的幅度。
        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        # 该选项曾用于限制刚体穿透时的最大修正速度；只有遇到严重穿模时才考虑打开。
        # self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 0.3
        self.actions.joint_pos.scale = beyondmimic_action_scale

        # 当前 motion_reference 只有一个 buffer，因此取出它的名字即可完成后续绑定。
        MOTION_NAME = list(self.scene.motion_reference.motion_buffers.keys())[0]
        # 统一从 motion 文件目录读取 metadata，避免配置中的占位路径失效。
        self.scene.motion_reference.motion_buffers[MOTION_NAME].metadata_yaml = os.path.join(
            self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
        )
        # False：使用与动作匹配的 terrain mesh（默认，适合翻箱子/翻越动作）。
        # True：改为平地 + AMASS motion，适合先验证动作跟踪，不需要 terrain 文件。
        PLANE_TERRAIN = False
        if PLANE_TERRAIN:
            # 移除 TerrainMotion，换成只包含运动数据的 AMASSMotion。
            self.scene.motion_reference.motion_buffers.pop(MOTION_NAME)
            self.scene.motion_reference.motion_buffers["AMASSMotion"] = AMASSMotionCfg()
            # 平地模式不创建 terrain generator。
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
        else:
            # 默认模式：让 terrain generator 使用同一 motion 目录中的匹配地形，
            # 这样每个参考动作会在对应的箱子/障碍物布局上播放。
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].path = (
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path
            )
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].metadata_yaml = os.path.join(
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
            )

        # critic 的 link observation 必须和 motion_reference 的 link_of_interests
        # 保持完全相同、顺序也一致，否则 link 误差的维度或对应关系会错位。
        self.observations.critic.link_pos.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests
        self.observations.critic.link_rot.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests

        # 将关键实验设置写入 run_name，便于从日志目录判断使用了哪种起始采样策略。
        self.run_name = "g1Perceptive" + "".join(
            [
                (
                    "_concatMotionBins"
                    if self.scene.motion_reference.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy
                    == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
            ]
        )


@configclass
class G1PerceptiveShadowingEnvCfg_PLAY(G1PerceptiveShadowingEnvCfg):
    """单环境可视化/评估配置。

    PLAY 不用于大规模训练，而是把环境数降到 1，关闭会妨碍观察的随机性，
    从动作开头播放整段轨迹，并打开相机和参考机器人的可视化。
    """

    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(debug_vis=True),
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=[1.5, 0.0, 1.5],
        lookat=[0.0, 0.0, 0.0],
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        # 先继承训练配置中的 G1、motion 和 terrain 绑定，再覆盖 PLAY 专用设置。
        super().__post_init__()

        # 评估时关闭自适应采样和失败计数更新，保证每次都从相同的动作起点播放，
        # 这样不同 checkpoint 的视频和误差统计才具有可比性。
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        MOTION_NAME = list(self.scene.motion_reference.motion_buffers.keys())[0]
        # 从动作第 0 秒开始，不在训练时随机选动作中间片段。
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        # None 表示播放整条动作，而不是限制在一个 1 秒 motion bin 中。
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_bin_length_s = None
        # 单环境评估使用独立起点；训练时的 concat 策略不适合严格复现整条轨迹。
        self.scene.motion_reference.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy = "independent"
        # 可替换为另一套动作目录，用于快速对比不同数据集。启用前要确保该目录有
        # 对应的 metadata.yaml；如果 terrain 不是匹配地形，还要同步修改 terrain 配置。
        # self.scene.motion_reference.motion_buffers[MOTION_NAME].path = (
        #     "/localhdd/Datasets/NoKov-Marslab-Motions-instinctnpz/20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall"
        # )
        self.scene.motion_reference.motion_buffers[MOTION_NAME].metadata_yaml = os.path.join(
            self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
        )
        # 评估时再次同步 terrain 的动作目录，防止上面的 path 被替换后仍读取旧地形。
        if self.scene.terrain.terrain_type == "hacked_generator":
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].path = (
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path
            )
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].metadata_yaml = os.path.join(
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
            )

        # 减少可视化时的地形规模，降低启动时间和显存占用。
        # 这里仍保留 terrain generator，因为默认 PLAY 仍然使用 motion-matched terrain。
        self.scene.terrain.terrain_generator.num_rows = 6
        self.scene.terrain.terrain_generator.num_cols = 6
        # 下面几行是“改成平地播放”的备用方案：先移除带 terrain 的 motion buffer，
        # 再创建 AMASSMotion，并把 terrain generator 关闭。它们当前被注释是因为默认
        # 需要观察动作和箱子/障碍物地形的对应关系。
        # 注意：若打开这些行，TerrainMotion 相关的 metadata 和 terrain generator 也应同步处理。
        # self.scene.motion_reference.motion_buffers.pop(MOTION_NAME)
        # self.scene.motion_reference.motion_buffers["AMASSMotion"] = AMASSMotionCfg()
        # self.scene.motion_reference.motion_buffers["AMASSMotion"].motion_start_from_middle_range = [0.0, 0.0]
        # self.scene.motion_reference.motion_buffers["AMASSMotion"].motion_bin_length_s = None
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        # 显示相机射线/深度观测，并在 policy observation 上打开调试可视化。
        # 这会增加渲染开销，只建议在 PLAY 时打开。
        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True

        # 如果参考动作的机器人朝向 -y，且初始姿态需要向下俯身，可以增大 roll 范围。
        # 当前保持注释，避免评估时改变动作初始状态。
        # self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.6)

        # 评估时暂时关闭这三个“偏离参考就终止”的条件，便于完整观察动作和定位问题。
        # 训练时不建议照搬，否则机器人偏离后可能长时间继续运行并产生无效数据。
        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None

        # 参考机器人默认由 motion_reference.debug_vis 自动显示。下面的备用设置可把
        # 参考机器人单独移开，或者把 viewer 的观察对象切换成 RobotReference。
        # 适合检查“参考动作本身是否正确”，而不是检查真实机器人跟踪效果。
        # self.events.reset_robot.params["position_offset"] = [0.0, 1.0, 2.0]
        # self.scene.motion_reference.visualizing_robot_offset = (0.0, 0.0, 0.0)
        # self.viewer.asset_name = "robot_reference"

        # 关闭 PLAY 中的摩擦、质心、推力和初始状态随机化，让每次回放尽量确定。
        # 这些 term 设为 None 后，父类中的事件管理器不会执行对应事件。
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None
        self.events.reset_robot.params["randomize_pose_range"]["x"] = [0.0] * 2  # x 方向不随机
        self.events.reset_robot.params["randomize_pose_range"]["y"] = [0.0] * 2  # y 方向不随机
        self.events.reset_robot.params["randomize_pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["x"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["y"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["z"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["roll"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)

        # 增加额外监控项：记录基座 shadowing 误差，以及容易在翻越/翻箱动作中
        # 过载的脚踝和膝关节执行器状态。
        self.monitors.shadowing_position_stats = MonitorTermCfg(
            func=ShadowingBasePosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
            ),
        )
        self.monitors.right_ankle_pitch_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_ankle_pitch.*"),
            ),
        )
        self.monitors.left_ankle_pitch_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_ankle_pitch.*"),
            ),
        )
        self.monitors.right_knee_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_knee.*"),
            ),
        )
        self.monitors.left_knee_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_knee.*"),
            ),
        )

        # 下面是一组未启用的“额外障碍物/箱子”示例，用于测试视觉泛化。
        # 它们当前全部注释，因此场景中不会生成这些物体。
        # 若要启用，需要额外导入 RigidObjectCfg 和 sim_utils（当前文件没有导入），
        # 并将新物体的 prim 路径加入 camera.mesh_prim_paths，否则深度相机看不到它。
        # 同时还要检查物体是否有碰撞体、是否会挡住参考动作，以及 reset 时的摆放位置。
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(4.8, 0.6, 0.5),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.35, 0.25)),
        # )
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/cube/.*")
        # 第二个箱子示例：更宽的横向障碍物。size 的单位是米，pos 是世界坐标初始位置。
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(4.8, 2.6, 0.5),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.35, 0.25)),
        # )
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/cube/.*")
        # 第三个箱子示例：窄而高的墙体，用于测试绕行或翻越行为。
        # self.scene.distractor = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cube",
        #     spawn=sim_utils.MeshCuboidCfg(
        #         size=(0.1, 4.6, 1.8),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.8, -1.35, 0.25)),
        # )
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/cube/.*")

        # 圆锥障碍物示例。这里名称是 distractor1，避免覆盖上面的 cube。
        # self.scene.distractor1 = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/cone1",
        #     spawn=sim_utils.MeshConeCfg(
        #         radius=0.22,
        #         height=0.55,
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             kinematic_enabled=True,
        #             disable_gravity=False,
        #             max_depenetration_velocity=1.0,
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.3)),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.0, 0.3)),
        # )
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/cone1/.*")

        # 将参考机器人加入相机可见网格列表，便于检查相机是否能看到 reference。
        # 通常训练时不应加入它，否则视觉策略可能把参考机器人当成障碍物或捷径。
        # self.scene.camera.mesh_prim_paths.append("/World/envs/env_.*/RobotReference/.*")
