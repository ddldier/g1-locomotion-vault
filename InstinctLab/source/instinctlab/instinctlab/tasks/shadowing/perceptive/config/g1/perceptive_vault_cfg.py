# =============================================================================
# G1 感知式翻越（Perceptive Vault）任务的环境配置
# -----------------------------------------------------------------------------
# 本文件在 InstinctLab 的 "perceptive"（带深度相机感知）shadowing 框架上，
# 组合出一个具体的任务：Unitree G1（29 DoF）助跑并翻越一个箱子（VaultBox）。
#
# 主要包含四类配置，每一类都有 "Teacher" 和 "Student" 两个版本：
#   1. PhpTeacher*        —— 教师（带特权观测：高度图 + 全局状态，无深度图）
#   2. VaultStudent*      —— 学生（只有参考动作 + 深度图 + 本体感知）
#   3. G1PerceptiveVault* —— torso 为根 / pelvis 为根 的教师环境
#   4. *EnvCfg_PLAY       —— 单环境回放配置（关闭随机化、固定起点）
#
# 设计思路（与 agent.yaml 的双教师 DAgger 对应）：
#   - 教师用 privileged（特权）信息训练出高质量翻越策略；
#   - 学生只用深度图 + 本体感知，通过 DAgger 向教师蒸馏；
#   - "pelvis 版本" 把动作参考根从 torso 换成 pelvis，保证与教师 checkpoint
#     (expected_teacher_root=pelvis) 以及 critic 观测维度一致。
# =============================================================================

import math
import os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.noise import UniformNoiseCfg

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg as perceptual_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_REV10_PELVIS_CFG,
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.tasks.shadowing.beyondmimic.config.g1 import beyondmimic_vault_cfg as vault_cfg
from instinctlab.sensors import get_link_prim_targets
from instinctlab.sensors import NoisyGroupedRayCasterCameraCfg


# =============================================================================
# 一、全局常量 / 超参数
# =============================================================================

# 教师的 proprio（本体感知）观测不做历史堆叠，与 BeyondMimic 保持一致。
# 注意：这里为 0 意味着 obs 里每一项都是单帧。
TEACHER_PROPRIO_HISTORY_LENGTH = 0

# 机器人资产配置：默认使用以 torso 为根（torso-base）的 G1 29DoF 配置。
# （后面 G1PerceptiveVaultPelvisEnvCfg 会替换成 pelvis 根版本）
G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

# 深度相机参数：
#   - 俯仰 34°，垂直视场角 70°
#   - 深度范围 0~3 m
#   - 分辨率 48x32（降低分辨率以节省显存，同时保持约 1.5 的宽高比）
#   - 更新周期 0.02 s，与策略步频 50 Hz 对齐（sim.dt=0.005, decimation=4）
VAULT_CAMERA_PITCH_DEG = 34
VAULT_CAMERA_FOVY_DEG = 70
VAULT_CAMERA_DEPTH_RANGE = (0.0, 3.0)
VAULT_CAMERA_WIDTH = 48
VAULT_CAMERA_HEIGHT = 32
VAULT_CAMERA_UPDATE_PERIOD = 0.02

# 箱子（VaultBox）用于深度图射线检测的 prim 路径表达式
VAULT_BOX_RAYCAST_PATH = "/World/envs/env_.*/VaultBox/.*"
# 箱子根 prim 表达式（用于高度图射线检测目标）
VAULT_BOX_PRIM_EXPR = "/World/envs/env_.*/VaultBox"

# 高度扫描参数：3.0m x 0.8m，分辨率 0.1m -> 31 x 9 = 279 个点
# 这与教师 checkpoint 训练时的 height_scan 维度 (279) 必须一致
VAULT_HEIGHT_SCAN_SIZE = (3.0, 0.8)
VAULT_HEIGHT_SCAN_RESOLUTION = 0.1
VAULT_HEIGHT_SCAN_DIM = 31 * 9

# 环境间距 20m：为了容纳 10m 宽的箱子（y∈[-4.5,5.5]）和行走范围
VAULT_ENV_SPACING = 20.0

# 教师跟踪终止阈值：机器人骨盆/关键 link 偏离参考超过 0.5m 就终止
# 这个阈值也会被 DAgger 蒸馏时用于 mask（只保留有效样本）
TEACHER_TRACKING_DISTANCE_THRESHOLD = 0.5

# 箱子位姿随机化开关：环境变量 PERCEPTIVE_VAULT_BOX_RANDOMIZATION=0 可关闭
VAULT_BOX_RANDOMIZATION_ENABLED = os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZATION", "1") != "0"

# 箱子位姿随机化范围：
#   - x/y 各 ±0.05m（可通过环境变量覆盖）
#   - z/roll/pitch 固定为 0
#   - yaw 随机 ±45°（与 PHP 论文设置一致）
VAULT_BOX_RANDOMIZE_POSE_RANGE = {
    "x": (
        -float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_X", "0.05")),
        float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_X", "0.05")),
    ),
    "y": (
        -float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_Y", "0.05")),
        float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_Y", "0.05")),
    ),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (
        -float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_YAW", str(math.radians(45)))),
        float(os.environ.get("PERCEPTIVE_VAULT_BOX_RANDOMIZE_YAW", str(math.radians(45)))),
    ),
}

# 动作参考配置：基于 BeyondMimic 的 vault 配置，做以下定制：
#   - update_period=0.02（与策略步频一致）
#   - 保留单帧参考（num_frames=1, frame_interval_s=0.0）
#   - 可视化偏移、标记类型等
#   - mp_split_method="None"（不做多进程切分）
# 注意：不继承 shadowing 的 10 帧未来参考栈或缩短的 link 列表。
VAULT_MOTION_REFERENCE_CFG = vault_cfg.motion_reference_cfg.replace(
    update_period=0.02,
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["relative_links", "links"],
    mp_split_method="None",
)
# 强制指定过滤后的动作选择文件（只保留 vault 相关动作）
VAULT_MOTION_REFERENCE_CFG.motion_buffers[
    vault_cfg.MOTION_NAME
].filtered_motion_selection_filepath = vault_cfg.MOTION_SELECTION_FILE

# Pelvis 根版本的动作参考配置：
#   - prim_path 挂到 Robot/pelvis（而不是 torso）
#   - FK 显式绑定到 vault 使用的 URDF
#   - 参考帧 prim 路径为 RobotReference/pelvis
#   - 动作选择文件换成 pelvis 重定向版本（原始 NPZ 的 base 在 torso_link，
#     pelvis 版本是显式转换后的文件）
#   - motion_start_from_middle_range 固定为 [0.0, 0.0]（不从中段开始）
VAULT_PELVIS_MOTION_REFERENCE_CFG = VAULT_MOTION_REFERENCE_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot/pelvis",
    robot_model_path=vault_cfg.VAULT_FK_URDF,
    reference_prim_path="/World/envs/env_.*/RobotReference/pelvis",
)
VAULT_PELVIS_MOTION_REFERENCE_CFG.motion_buffers[
    vault_cfg.MOTION_NAME
].filtered_motion_selection_filepath = vault_cfg.PELVIS_MOTION_SELECTION_FILE
VAULT_PELVIS_MOTION_REFERENCE_CFG.motion_buffers[
    vault_cfg.MOTION_NAME
].motion_start_from_middle_range = [0.0, 0.0]

# 关注 link 数量（18 个），用于 critic / 奖励。必须与 link_of_interests 长度一致。
VAULT_NUM_LINK_OF_INTERESTS = len(VAULT_MOTION_REFERENCE_CFG.link_of_interests)


# =============================================================================
# 二、辅助函数
# =============================================================================

def _append_unique(items: list[str], item: str):
    """向列表中追加元素（如果不存在），用于去重。"""
    if item not in items:
        items.append(item)


def global_pelvis_position(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """返回每个环境世界坐标系下的骨盆（根）位置。

    注意：减去了 env_origins，得到的是相对环境原点的位置，
    避免不同环境之间的偏移导致数值过大。
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def global_pelvis_velocity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """返回世界坐标系下骨盆（根）的线速度。"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def _apply_vault_camera_settings(camera_cfg):
    """对相机配置应用 vault 任务专用的参数（内参/外参/分辨率/噪声）。"""
    pitch_rad = math.radians(VAULT_CAMERA_PITCH_DEG)
    fovy_rad = math.radians(VAULT_CAMERA_FOVY_DEG)

    # 相机外参（相对 torso_link 的偏移）：位置微调 + 俯仰旋转
    camera_cfg.offset.pos = (
        0.04764571478 + 0.0039635 - 0.0042 * math.cos(pitch_rad),
        0.015,
        0.46268178553 - 0.044 + 0.0042 * math.sin(pitch_rad) + 0.016,
    )
    camera_cfg.offset.rot = (
        math.cos(pitch_rad / 2),
        0.0,
        math.sin(pitch_rad / 2),
        0.0,
    )

    # 相机内参：垂直视场角 -> 垂直孔径
    camera_cfg.pattern_cfg.vertical_aperture = 2 * math.tan(fovy_rad / 2)
    # 方形像素：水平孔径 = 垂直孔径 * (宽/高)，这样 fovx ≈ 92.8°
    # （如果沿用父类 87° 会导致像素非方形）
    camera_cfg.pattern_cfg.horizontal_aperture = camera_cfg.pattern_cfg.vertical_aperture * (
        VAULT_CAMERA_WIDTH / VAULT_CAMERA_HEIGHT
    )
    camera_cfg.pattern_cfg.width = VAULT_CAMERA_WIDTH
    camera_cfg.pattern_cfg.height = VAULT_CAMERA_HEIGHT

    # 更新周期与策略步频对齐
    camera_cfg.update_period = VAULT_CAMERA_UPDATE_PERIOD

    # 噪声管线：深度归一化范围
    camera_cfg.noise_pipeline["normalize"].depth_range = VAULT_CAMERA_DEPTH_RANGE
    # 移除父类的 crop/resize 到 18x32 的步骤（我们直接用 48x32 原生分辨率）
    camera_cfg.noise_pipeline.pop("crop_and_resize", None)

    # 把机器人所有 link 的 prim 路径加入相机的 mesh_prim_paths（用于遮挡/自碰撞检测）
    for mesh_prim_path in get_link_prim_targets(G1_29DOF_LINKS):
        _append_unique(camera_cfg.mesh_prim_paths, mesh_prim_path)
    # 加入箱子，这样深度图里能看到箱子
    _append_unique(camera_cfg.mesh_prim_paths, VAULT_BOX_RAYCAST_PATH)


def enable_vault_camera(env_cfg, debug_vis: bool = False):
    """为环境恢复/启用 vault 深度相机（学生/蒸馏环境需要）。"""
    # 从父类 perceptive 配置取出相机，应用 vault 参数
    camera_cfg = perceptual_cfg.PerceptiveShadowingSceneCfg().camera
    _apply_vault_camera_settings(camera_cfg)
    camera_cfg.debug_vis = debug_vis
    env_cfg.scene.camera = camera_cfg

    # 如果父类已有相机外参随机化事件，这里确保它存在（训练时用于 sim2real）
    if getattr(env_cfg.events, "randomize_ray_offsets", None) is None:
        env_cfg.events.randomize_ray_offsets = perceptual_cfg.EventsCfg().randomize_ray_offsets


def _vault_box_cfg(debug_vis: bool = False) -> RigidObjectCfg:
    """构造箱子（VaultBox）刚体配置。

    - 尺寸来自 vault_cfg.VAULT_BOX_SIZE（0.4 x 10 x 0.6）
    - kinematic_enabled=True：箱子不动（固定障碍物）
    - 初始位置来自 vault_cfg.VAULT_BOX_POS
    - debug_vis 时颜色变蓝，便于可视化
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/VaultBox",
        spawn=sim_utils.MeshCuboidCfg(
            size=vault_cfg.VAULT_BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3) if not debug_vis else (0.0, 0.6, 0.9)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=vault_cfg.VAULT_BOX_POS),
    )


def vault_box_cfg(debug_vis: bool = False) -> RigidObjectCfg:
    """对外暴露的箱子配置构造函数。"""
    return _vault_box_cfg(debug_vis=debug_vis)


def reset_vault_box_pose_by_motion_randomized(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("vault_box"),
    active_pos: tuple[float, float, float] = vault_cfg.VAULT_BOX_POS,
    pose_range: dict[str, tuple[float, float]] | None = None,
    # 以下参数为兼容旧配置保留，实际未使用
    motion_ref_cfg: SceneEntityCfg | None = None,
    motion_buffer_name: str | None = None,
    vault_motion_file: str | None = None,
    hidden_pos: tuple[float, float, float] | None = None,
):
    """在每个回合重置时，把箱子放置到带随机化的位姿。

    动作缓冲区只包含 vault 动作（包括速度变体），所以这里不需要根据
    动作帧去查表，直接用固定位置 + 随机扰动即可。
    """
    _ = (motion_ref_cfg, motion_buffer_name, vault_motion_file, hidden_pos)
    if asset_cfg.name not in env.scene.rigid_objects:
        return

    vault_box = env.scene[asset_cfg.name]
    root_state = vault_box.data.default_root_state[env_ids].clone()

    # 1) 基准位置：active_pos + 环境原点偏移
    root_state[:, :3] = torch.tensor(active_pos, dtype=root_state.dtype, device=env.device)
    root_state[:, :3] += env.scene.env_origins[env_ids]

    # 2) 在基准位置上叠加随机偏移（x/y/z）和随机旋转（roll/pitch/yaw）
    pose_range = pose_range or {}
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, dtype=root_state.dtype, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    root_state[:, :3] += rand_samples[:, :3]
    quat_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    root_state[:, 3:7] = math_utils.quat_mul(quat_delta, root_state[:, 3:7])

    # 3) 速度清零（箱子是 kinematic，但保持数据一致）
    root_state[:, 7:] = 0.0

    # 4) 写回仿真
    vault_box.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    vault_box.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


# =============================================================================
# 三、教师观测配置
# =============================================================================

@configclass
class PhpTeacherPolicyObsCfg(ObsGroupCfg):
    """PHP 教师的特权 policy 观测（高度图 + 全局状态，无深度图）。

    注意：这里的 policy 观测是 privileged 的，因为教师训练时可以直接读取
    高度扫描和全局骨盆状态。学生版本 (VaultStudentPolicyObsCfg) 会去掉这些。
    """

    # 参考动作（来自 command 管理器，实时从 motion_reference 读取）
    joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
    joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
    # 参考位置/朝向，带均匀噪声（模拟传感器/参考误差）
    position_ref = ObsTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "position_ref_command"},
        noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
    )
    rotation_ref = ObsTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "rotation_ref_command"},
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    )
    # 高度扫描（279 维，来自 height_scanner）
    height_scan = ObsTermCfg(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=[-1.0, 1.0],
    )
    # 全局骨盆位置/速度（特权信息）
    global_pelvis_pos = ObsTermCfg(func=global_pelvis_position)
    global_pelvis_vel = ObsTermCfg(func=global_pelvis_velocity)
    # 本体感知（带噪声，模拟真实传感器）
    base_lin_vel = ObsTermCfg(
        func=mdp.base_lin_vel,
        noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    projected_gravity = ObsTermCfg(
        func=mdp.projected_gravity,
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    base_ang_vel = ObsTermCfg(
        func=mdp.base_ang_vel,
        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    joint_pos = ObsTermCfg(
        func=mdp.joint_pos_rel,
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    joint_vel = ObsTermCfg(
        func=mdp.joint_vel_rel,
        noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        history_length=TEACHER_PROPRIO_HISTORY_LENGTH,
    )
    last_action = ObsTermCfg(func=mdp.last_action, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)

    def __post_init__(self):
        self.enable_corruption = True   # 训练时启用噪声
        self.concatenate_terms = False  # 不拼接成一个向量，保留 dict 结构


@configclass
class PhpTeacherCriticObsCfg(ObsGroupCfg):
    """教师的 critic 观测，顺序显式排列，必须与蒸馏教师的 obs_format 一致。

    agent.yaml 中 teacher_policy.obs_format.critic 的顺序是：
      joint_pos_ref, joint_vel_ref, position_ref, rotation_ref, height_scan,
      global_pelvis_pos, global_pelvis_vel, base_lin_vel, projected_gravity,
      base_ang_vel, joint_pos, joint_vel, last_action, link_pos, link_rot
    这里逐项对应，确保 DAgger 查教师时维度对齐。
    """

    joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
    joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
    position_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "position_ref_command"})
    rotation_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "rotation_ref_command"})
    height_scan = ObsTermCfg(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=[-1.0, 1.0],
    )
    global_pelvis_pos = ObsTermCfg(func=global_pelvis_position)
    global_pelvis_vel = ObsTermCfg(func=global_pelvis_velocity)
    base_lin_vel = ObsTermCfg(func=mdp.base_lin_vel, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    projected_gravity = ObsTermCfg(func=mdp.projected_gravity, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    base_ang_vel = ObsTermCfg(func=mdp.base_ang_vel, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    joint_pos = ObsTermCfg(func=mdp.joint_pos_rel, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    joint_vel = ObsTermCfg(func=mdp.joint_vel_rel, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    last_action = ObsTermCfg(func=mdp.last_action, history_length=TEACHER_PROPRIO_HISTORY_LENGTH)
    # 18 个关注 link 的位姿（位置 + tannorm 旋转），保持 link_of_interests 顺序
    link_pos = ObsTermCfg(
        func=instinct_mdp.link_pos_b,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=VAULT_MOTION_REFERENCE_CFG.link_of_interests,
                preserve_order=True,
            ),
        },
    )
    link_rot = ObsTermCfg(
        func=instinct_mdp.link_tannorm_b,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=VAULT_MOTION_REFERENCE_CFG.link_of_interests,
                preserve_order=True,
            ),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False  # critic 用干净的 privileged 观测
        self.concatenate_terms = False


@configclass
class PhpTeacherObservationsCfg(perceptual_cfg.ObservationsCfg):
    """教师环境的观测组合：privileged policy + 标准 critic。"""

    policy: PhpTeacherPolicyObsCfg = PhpTeacherPolicyObsCfg()
    critic: PhpTeacherCriticObsCfg = PhpTeacherCriticObsCfg()


# =============================================================================
# 四、学生观测配置
# =============================================================================

@configclass
class VaultStudentPolicyObsCfg(ObsGroupCfg):
    """学生 policy 观测：参考动作 + 深度图 + 本体感知。

    与教师的关键区别：
      - 保留 joint_pos_ref / joint_vel_ref / position_ref / rotation_ref
        （这些是"告诉学生目标是什么"的参考，不是特权信息）
      - 用 depth_image 替代 height_scan（学生没有特权高度扫描）
      - 去掉 global_pelvis_pos / global_pelvis_vel（学生不知道全局位置）
    注意：critic 仍然沿用教师的 privileged 版本，方便第一阶段先用 PPO 训练，
    后续再接入教师查询和 DAgger。
    """

    # 参考动作（与学生共享，不算特权）
    joint_pos_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_pos_ref_command"})
    joint_vel_ref = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "joint_vel_ref_command"})
    position_ref = ObsTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "position_ref_command"},
        noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
    )
    rotation_ref = ObsTermCfg(
        func=mdp.generated_commands,
        params={"command_name": "rotation_ref_command"},
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    )
    # 深度图（48x32，来自 camera，取 distance_to_image_plane_noised）
    depth_image = ObsTermCfg(
        func=instinct_mdp.visualizable_image,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane_noised"},
    )
    # 本体感知（带噪声）
    projected_gravity = ObsTermCfg(
        func=mdp.projected_gravity,
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    )
    base_ang_vel = ObsTermCfg(
        func=mdp.base_ang_vel,
        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    )
    joint_pos = ObsTermCfg(func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTermCfg(func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5))
    last_action = ObsTermCfg(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class VaultStudentObservationsCfg(PhpTeacherObservationsCfg):
    """学生环境的观测组合：学生 policy + 教师的 privileged critic。"""

    policy: VaultStudentPolicyObsCfg = VaultStudentPolicyObsCfg()


# =============================================================================
# 五、Vault 任务通用设置
# =============================================================================

def _apply_perceptive_vault_settings(env_cfg, debug_vis: bool = False):
    """把 vault 任务的所有通用设置应用到 env_cfg 上。

    这个函数会被各个 EnvCfg 的 __post_init__ 调用，负责：
      - 场景元素（箱子、地形、执行器、相机）
      - 高度扫描（MultiMesh 版本，能扫到箱子）
      - 摩擦/扰动等域随机化
      - 动作参考的采样策略
      - 终止/奖励权重
    """
    # --- 场景元素 ---
    # 箱子：尺寸/位置来自 vault_cfg
    env_cfg.scene.vault_box = _vault_box_cfg(debug_vis=debug_vis)
    # 地形：平面（不用地形生成器）
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    # 执行器：使用 BeyondMimic 的 G1 29DoF 执行器配置
    env_cfg.scene.robot.actuators = beyondmimic_g1_29dof_actuators
    # 动作缩放：PHP 所有专家都用 scale=1（不用 BeyondMimic 的力矩/刚度启发式）
    env_cfg.actions.joint_pos.scale = 1.0

    # 特权教师不用深度相机，移除以节省计算；同时移除相机外参随机化事件
    env_cfg.scene.camera = None
    env_cfg.events.randomize_ray_offsets = None

    # 标准 RayCaster 只支持单 mesh，改用 MultiMesh 以便扫描到箱子
    # 注意：高度扫描挂在 torso_link（vault 原始版本），pelvis 版本会改为 pelvis
    env_cfg.scene.height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=VAULT_HEIGHT_SCAN_RESOLUTION,
            size=list(VAULT_HEIGHT_SCAN_SIZE),
        ),
        debug_vis=debug_vis,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr=VAULT_BOX_PRIM_EXPR,
                track_mesh_transforms=True,
            ),
        ],
    )

    # 如果 critic 里有 height_scan，设置 clip 范围 [-1, 1]
    if getattr(env_cfg.observations.critic, "height_scan", None) is not None:
        env_cfg.observations.critic.height_scan.clip = [-1.0, 1.0]

    # 与 BeyondMimic 一致：critic 的本体感知也是单帧（不做历史堆叠）
    for term_name in ("base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "last_action"):
        term = getattr(env_cfg.observations.critic, term_name, None)
        if term is not None:
            term.history_length = TEACHER_PROPRIO_HISTORY_LENGTH

    # --- 域随机化 ---
    # PHP Table V：机器人摩擦随机化（覆盖父类默认值）
    if getattr(env_cfg.events, "physics_material", None) is not None:
        env_cfg.events.physics_material.params["static_friction_range"] = (0.4, 1.3)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.1)

    # PHP Table V：每 1~3 秒对根速度加扰动（只在训练时启用）
    if debug_vis:
        # 回放时关闭自适应采样、失败计数、push_robot，确保从固定起点回放整段动作
        env_cfg.events.push_robot = None
        env_cfg.events.bin_fail_counter_smoothing = None
        env_cfg.curriculum.beyond_adaptive_sampling = None
    else:
        # 训练时保留父类的 BeyondMimic 自适应采样：
        # 失败的 motion bin 会获得更高的后续采样概率，帮助策略反复练习难点
        env_cfg.events.push_robot = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1),
                    "z": (-0.05, 0.05),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.1, 0.1),
                },
            },
        )

    # --- 动作参考采样策略 ---
    # 固定从头开始（不从中段开始）
    env_cfg.scene.motion_reference.motion_buffers[
        vault_cfg.MOTION_NAME
    ].motion_start_from_middle_range = [0.0, 0.0]
    # motion bin 长度：训练时 1.0s（分 bin 采样），回放时不分 bin
    env_cfg.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_bin_length_s = (
        None if debug_vis else 1.0
    )
    # 起始采样策略：训练时 concat_motion_bins（按 bin 拼接采样），回放时 independent
    env_cfg.scene.motion_reference.motion_buffers[
        vault_cfg.MOTION_NAME
    ].env_starting_stub_sampling_strategy = "independent" if debug_vis else "concat_motion_bins"

    # 箱子重置事件：根据随机化开关决定是否加扰动
    env_cfg.events.reset_vault_box = EventTermCfg(
        func=reset_vault_box_pose_by_motion_randomized,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("vault_box"),
            "active_pos": vault_cfg.VAULT_BOX_POS,
            "pose_range": VAULT_BOX_RANDOMIZE_POSE_RANGE
            if (VAULT_BOX_RANDOMIZATION_ENABLED and not debug_vis)
            else {},
        },
    )

    # --- 终止条件 ---
    # 骨盆/关键 link 偏离参考 > 0.5m 终止
    env_cfg.terminations.base_pos_too_far.params["distance_threshold"] = TEACHER_TRACKING_DISTANCE_THRESHOLD
    env_cfg.terminations.base_pos_too_far.params["height_only"] = False
    env_cfg.terminations.link_pos_too_far.params["distance_threshold"] = TEACHER_TRACKING_DISTANCE_THRESHOLD
    env_cfg.terminations.link_pos_too_far.params["height_only"] = False
    # 只跟踪脚踝和手腕（关键接触点）
    env_cfg.terminations.link_pos_too_far.params["reference_cfg"].body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]

    # --- 奖励权重（PHP Table IV）---
    env_cfg.rewards.rewards.base_position_imitation_gauss.weight = 1.0
    env_cfg.rewards.rewards.base_rot_imitation_gauss.weight = 1.0
    env_cfg.rewards.rewards.undesired_contacts.weight = -0.5
    env_cfg.rewards.rewards.applied_torque_limits_by_ratio = None  # 不使用力矩比奖励
    # 非期望接触的 body 正则：排除脚踝 roll 和手腕各 link（这些部位允许接触）
    env_cfg.rewards.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        (
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
            r"(?!left_wrist_roll_link$)(?!left_wrist_pitch_link$)(?!left_wrist_yaw_link$)"
            r"(?!right_wrist_roll_link$)(?!right_wrist_pitch_link$)(?!right_wrist_yaw_link$).+$"
        )
    ]
    # 非法接触终止使用同样的 body 正则
    env_cfg.terminations.illegal_reset_contact.params["sensor_cfg"].body_names = (
        env_cfg.rewards.rewards.undesired_contacts.params["sensor_cfg"].body_names
    )

    # critic 的 link_pos/link_rot 使用动作参考的 link_of_interests（18 个 link）
    env_cfg.observations.critic.link_pos.params["asset_cfg"].body_names = (
        env_cfg.scene.motion_reference.link_of_interests
    )
    env_cfg.observations.critic.link_rot.params["asset_cfg"].body_names = (
        env_cfg.scene.motion_reference.link_of_interests
    )


# =============================================================================
# 六、教师环境配置（torso 根）
# =============================================================================

@configclass
class G1PerceptiveVaultEnvCfg(perceptual_cfg.PerceptiveShadowingEnvCfg):
    """G1 感知式翻越教师环境（torso 根版本）。

    - 4096 个并行环境，间距 20m
    - 机器人用 torso 为根的 G1_CFG
    - 观测用 PhpTeacherObservationsCfg（特权 policy + privileged critic）
    """

    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=4096,
        env_spacing=VAULT_ENV_SPACING,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=VAULT_MOTION_REFERENCE_CFG,
    )
    observations: PhpTeacherObservationsCfg = PhpTeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_perceptive_vault_settings(self, debug_vis=False)
        self.run_name = "g1PerceptiveVaultPhpTeacher_concatMotionBins"


@configclass
class G1PerceptiveVaultEnvCfg_PLAY(G1PerceptiveVaultEnvCfg):
    """教师回放配置（torso 根版本）：单环境、固定起点、关闭随机化。"""

    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=VAULT_ENV_SPACING,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=VAULT_MOTION_REFERENCE_CFG.replace(debug_vis=True),
    )
    viewer: ViewerCfg = ViewerCfg(
        # 世界坐标系视角，用户可以自由旋转/平移/缩放，不会被 viewer 强制拉回机器人
        eye=[7.0, -9.0, 4.5],
        lookat=[0.0, 0.5, 0.8],
        origin_type="world",
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_perceptive_vault_settings(self, debug_vis=True)

        # 回放用很长的 episode（6000s），避免中途终止
        self.episode_length_s = 6000.0
        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_bin_length_s = None
        self.scene.motion_reference.motion_buffers[
            vault_cfg.MOTION_NAME
        ].env_starting_stub_sampling_strategy = "independent"

        # 关闭跟踪终止条件（回放时不想被终止）
        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None

        # 关闭所有域随机化
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None  # type: ignore
        # 重置时的位姿/速度/关节随机化范围全部置零
        self.events.reset_robot.params["randomize_pose_range"]["x"] = [0.0] * 2
        self.events.reset_robot.params["randomize_pose_range"]["y"] = [0.0] * 2
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


# =============================================================================
# 七、教师环境配置（pelvis 根）—— 与 agent.yaml 的 teacher_checkpoint 对应
# =============================================================================

@configclass
class G1PerceptiveVaultPelvisEnvCfg(G1PerceptiveVaultEnvCfg):
    """使用官方 pelvis 根 rev1.0 机器人资产的 Vault 教师环境。

    与 torso 根版本的区别：
      - 机器人资产换成 G1_29DOF_REV10_PELVIS_CFG
      - 动作参考换成 VAULT_PELVIS_MOTION_REFERENCE_CFG（base 在 pelvis）
      - height_scanner 挂到 pelvis
      - run_name = g1PerceptiveVaultPelvisPhpTeacher_concatMotionBins
        （与 agent.yaml 中 teacher_checkpoint 路径一致）
    """

    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=4096,
        env_spacing=VAULT_ENV_SPACING,
        robot=G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=VAULT_PELVIS_MOTION_REFERENCE_CFG,
    )

    def __post_init__(self):
        super().__post_init__()
        # 父类会重新写入旧 scene 配置，所以最后再固定一次 pelvis-root 资产和 FK 根
        self.scene.robot = G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.motion_reference = VAULT_PELVIS_MOTION_REFERENCE_CFG
        self.scene.motion_reference.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        self.scene.motion_reference.reference_prim_path = "/World/envs/env_.*/RobotReference/pelvis"
        # 高度扫描挂到 pelvis（与动作参考根一致）
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        self.run_name = "g1PerceptiveVaultPelvisPhpTeacher_concatMotionBins"


@configclass
class G1PerceptiveVaultPelvisEnvCfg_PLAY(G1PerceptiveVaultPelvisEnvCfg):
    """pelvis 根教师回放配置：单环境、固定起点、关闭随机化。"""

    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=VAULT_ENV_SPACING,
        robot=G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=VAULT_PELVIS_MOTION_REFERENCE_CFG.replace(debug_vis=True),
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot_reference = G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference")
        self.scene.motion_reference = VAULT_PELVIS_MOTION_REFERENCE_CFG.replace(debug_vis=True)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        self.episode_length_s = 6000.0
        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_bin_length_s = None
        self.scene.motion_reference.motion_buffers[
            vault_cfg.MOTION_NAME
        ].env_starting_stub_sampling_strategy = "independent"
        # 回放使用确定性顺序播放。关闭自适应 bin 采样和 motion bins；
        # 否则 curriculum 构造函数会访问 _motion_bin_weights（在 bin_length_s=None 时不会创建）
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None
        # 重置随机化全部置零
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.events.reset_robot.params["randomize_pose_range"][key] = (0.0, 0.0)
            self.events.reset_robot.params["randomize_velocity_range"][key] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)


# =============================================================================
# 八、学生环境配置（pelvis 根）—— 与 agent.yaml 的主训练对应
# =============================================================================

@configclass
class G1PerceptiveVaultPelvisStudentEnvCfg(G1PerceptiveVaultPelvisEnvCfg):
    """pelvis 根的视觉学生环境；在教师训练完成后使用。

    - policy 观测换成 VaultStudentPolicyObsCfg（参考 + 深度图 + 本体感知）
    - critic 仍用教师的 privileged 观测
    - 启用 vault 深度相机
    - run_name = g1PerceptiveVaultPelvisStudent_depth32x48
    """

    observations: VaultStudentObservationsCfg = VaultStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 教师设置里移除了相机，这里重新启用
        enable_vault_camera(self, debug_vis=False)
        self.run_name = "g1PerceptiveVaultPelvisStudent_depth32x48"


@configclass
class G1PerceptiveVaultPelvisStudentEnvCfg_PLAY(G1PerceptiveVaultPelvisStudentEnvCfg):
    """pelvis 根学生回放配置：单环境、深度图可视化、固定起点。"""

    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=VAULT_ENV_SPACING,
        robot=G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=VAULT_PELVIS_MOTION_REFERENCE_CFG.replace(debug_vis=True),
    )

    def __post_init__(self):
        super().__post_init__()
        # 训练父类会重新设置 motion_reference；播放需要恢复独立的 pelvis 引用配置
        self.scene.robot_reference = G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference")
        self.scene.motion_reference = VAULT_PELVIS_MOTION_REFERENCE_CFG.replace(debug_vis=True)
        enable_vault_camera(self, debug_vis=True)
        self.episode_length_s = 6000.0
        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_bin_length_s = None
        # 无 motion bin 时不能继续使用 bin 采样、课程或失败计数事件
        self.scene.motion_reference.motion_buffers[
            vault_cfg.MOTION_NAME
        ].env_starting_stub_sampling_strategy = "independent"
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        # 与 pelvis Teacher PLAY 对齐，固定起点并关闭训练扰动
        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.events.reset_robot.params["randomize_pose_range"][key] = (0.0, 0.0)
            self.events.reset_robot.params["randomize_velocity_range"][key] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)


# =============================================================================
# 九、学生环境配置（torso 根）—— 旧版本，保留兼容
# =============================================================================

@configclass
class G1PerceptiveVaultStudentEnvCfg(G1PerceptiveVaultEnvCfg):
    """视觉学生环境（torso 根）：Actor 只接收参考动作、深度图和本体状态。

    Critic 继续沿用 PhpTeacherCriticObsCfg 的 privileged 输入，便于第一阶段
    先用 PPO 跑通视觉策略，后续再接入 Teacher 查询和 DAgger 数据聚合。
    """

    observations: VaultStudentObservationsCfg = VaultStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Teacher 的设置会主动移除 camera；Student 在最后恢复视觉传感器
        enable_vault_camera(self, debug_vis=False)
        self.run_name = "g1PerceptiveVaultStudent_depth32x48"


@configclass
class G1PerceptiveVaultStudentEnvCfg_PLAY(G1PerceptiveVaultStudentEnvCfg):
    """torso 根学生播放配置：固定单环境并打开深度图可视化。"""

    scene: perceptual_cfg.PerceptiveShadowingSceneCfg = perceptual_cfg.PerceptiveShadowingSceneCfg(
        num_envs=1,
        env_spacing=VAULT_ENV_SPACING,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=VAULT_MOTION_REFERENCE_CFG.replace(debug_vis=True),
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=[7.0, -9.0, 4.5],
        lookat=[0.0, 0.5, 0.8],
        origin_type="world",
    )

    def __post_init__(self):
        super().__post_init__()
        # Student 父类先按训练配置初始化；Play 必须切回固定动作段，避免
        # concat_motion_bins 在单环境初始化时访问尚未建立的 bin 权重
        _apply_perceptive_vault_settings(self, debug_vis=True)
        enable_vault_camera(self, debug_vis=True)
        self.episode_length_s = 6000.0
        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[vault_cfg.MOTION_NAME].motion_bin_length_s = None
        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            self.events.reset_robot.params["randomize_pose_range"][key] = (0.0, 0.0)
            self.events.reset_robot.params["randomize_velocity_range"][key] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)
