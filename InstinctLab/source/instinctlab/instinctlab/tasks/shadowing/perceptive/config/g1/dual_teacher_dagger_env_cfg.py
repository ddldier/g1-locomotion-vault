"""Single-scene task mixture for locomotion + pelvis-root Vault DAgger.

Teacher routing uses a privileged task identifier. Autonomous Student variants
can omit it and Vault reference commands from their policy observations.
"""

# =============================================================================
# 文件说明：
# 该文件定义了一个“单场景混合任务”环境配置，用于在一个 Isaac Lab 场景中
# 同时训练两类任务：
#   1. locomotion（平地行走任务）；
#   2. Vault（以 pelvis 为 root 的翻越障碍任务）。
#
# 混合策略：
#   - 偶数编号环境（env_id % 2 == 0）→ locomotion 样本；
#   - 奇数编号环境（env_id % 2 == 1）→ Vault 样本。
#
# 这样做的目的：
#   - 让 Student（学生策略）同时学习两个任务；
#   - 同时保证 rollout 中两类任务样本数量均衡；
#   - 两个 Teacher 专家网络仍然接收各自原始的观测格式，互不干扰；
#   - task_id 仅用于 critic / Teacher 标签路由；可见 Student 配置会移除它。
#
# 关键实现方式：
#   - 所有 Vault 专用的终止条件/重置逻辑都通过 _task_mask 过滤，
#     只对 Vault 环境生效，避免影响 locomotion 样本；
#   - locomotion 专用的终止/接触条件也通过 task mask 过滤；
#   - 重置机器人时，两种任务使用各自 Teacher 训练时约定的重置方式；
#   - Vault 环境保留障碍物（vault_box），locomotion 环境把障碍物隐藏。
#   - Student 在 Vault 样本中接收固定的 1 m/s 前进速度命令，locomotion 命令不变。
# =============================================================================

# 从标准库复制模块中导入 deepcopy。
# 该函数用于深拷贝配置对象，避免多个环境实例共享同一份可变配置。
from copy import deepcopy

# 导入 PyTorch，用于张量生成、条件选择等。
import torch

# 导入 Isaac Lab 的 MDP 常用函数集合，例如命令生成、重置、高度扫描等。
import isaaclab.envs.mdp as mdp
# 从 Isaac Lab 管理器模块中导入事件、观测组、观测项、场景实体和终止项配置类。
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg as ObsGroupCfg,
    ObservationTermCfg as ObsTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
# 导入射线扫描相关配置：RayCasterCfg 和网格扫描模式。
from isaaclab.sensors import RayCasterCfg, patterns
# 导入多网格射线扫描配置，用于自定义 height scanner。
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
# 导入 configclass 装饰器，用于声明 Isaac Lab 配置类。
from isaaclab.utils import configclass
# 导入均匀噪声配置，用于给观测添加噪声。
from isaaclab.utils.noise import UniformNoiseCfg

# 导入 instinctlab 的 MDP 扩展函数，例如参考轨迹相关重置与终止条件。
import instinctlab.envs.mdp as instinct_mdp
# 导入 G1 OmniLoco 的命令配置基类，混合命令配置会继承它。
from instinctlab.tasks.locomotion.config.g1.omni_loco_env_cfg import G1OmniLocoCommandsCfg
# 导入 Vault 配置模块，并以 vault 作为别名使用其中的配置和函数。
from . import perceptive_vault_cfg as vault
# 从 Vault 配置中导入若干具体类、函数和常量。
from .perceptive_vault_cfg import (
    # G1 感知 Vault、以 pelvis 为 root 的 Student 环境配置基类。
    G1PerceptiveVaultPelvisStudentEnvCfg,
    # Vault Student 策略观测配置类。
    VaultStudentPolicyObsCfg,
    # PHP Teacher critic 观测配置类。
    PhpTeacherCriticObsCfg,
    # 根据参考运动随机重置 Vault 障碍物位姿的函数。
    reset_vault_box_pose_by_motion_randomized,
    # Vault 运动参考配置常量。
    VAULT_MOTION_REFERENCE_CFG,
    # Vault 环境间距常量。
    VAULT_ENV_SPACING,
)


# 任务标识常量：
# LOCO_MASK_VALUE = 0 表示 locomotion 任务；
# VAULT_MASK_VALUE = 1 表示 Vault 任务。
LOCO_MASK_VALUE = 0
VAULT_MASK_VALUE = 1
# Vault Student 的目标前进速度（m/s）；横向和偏航命令固定为 0。
VAULT_STUDENT_FORWARD_SPEED = 1.0

# 从 Vault 配置中复制一份命令配置对象，供混合命令配置类使用。
# 注意：这里只是先实例化 Vault 的 CommandCfg，后续在 DualCommandsCfg 中会逐项 deepcopy。
_VAULT_COMMANDS = vault.perceptual_cfg.CommandCfg()


# 根据环境编号固定分配任务 ID。
# 偶数 env_id → 0（locomotion）；
# 奇数 env_id → 1（Vault）。
# 返回形状为 (num_envs, 1) 的 float32 张量。
def dual_task_id(env) -> torch.Tensor:
    """Fixed task assignment: 0 locomotion, 1 Vault, balanced per rollout."""
    # 生成 [0, num_envs-1] 的浮点环境 ID。
    # 使用 env.device 保证张量位于与环境相同的设备上。
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.float32)

    # 对 2 取余：偶数得 0，奇数得 1。
    # unsqueeze(-1) 把形状变为 (num_envs, 1)，方便作为观测项使用。
    return (ids.remainder(2.0)).unsqueeze(-1)


# 根据给定 task 值返回布尔 mask。
# 例如 value=VAULT_MASK_VALUE 时，返回哪些环境属于 Vault 任务。
def _task_mask(env, value: int) -> torch.Tensor:
    # dual_task_id(env)[:, 0] 取到每个环境的 task id；
    # eq(value) 得到与 value 相等的布尔 mask。
    # 返回形状为 (num_envs,) 的布尔张量。
    return dual_task_id(env)[:, 0].eq(value)


# 双任务 Student 速度命令函数。
# locomotion 样本继续使用原本采样的 base_velocity 命令；
# Vault 样本则覆盖为固定前进速度命令 (vx, vy, yaw) = (1, 0, 0)。
def dual_student_velocity_command(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Keep locomotion commands sampled, but give Vault Students a fixed 1 m/s forward command."""
    # The shared base_velocity generator remains unchanged for locomotion.  Only the
    # Student-facing observation is overridden for Vault samples: (vx, vy, yaw) = (1, 0, 0).
    #
    # 中文说明：
    # 1. 先读取环境中共享的 base_velocity 命令；
    # 2. 构造一个与 command 形状相同的全零张量；
    # 3. 将第 0 维（前进速度 vx）设为固定值 1.0；
    # 4. 使用 torch.where 按 task mask 选择：
    #    - Vault 环境使用固定命令；
    #    - locomotion 环境保留原始随机命令。
    command = mdp.generated_commands(env, command_name)
    # 创建与原始命令同形状、同设备、同数据类型的全零命令张量。
    vault_command = torch.zeros_like(command)
    # 将 Vault 固定命令的前进速度设为 VAULT_STUDENT_FORWARD_SPEED。
    vault_command[:, 0] = VAULT_STUDENT_FORWARD_SPEED
    # _task_mask(...).unsqueeze(-1) 将 (num_envs,) 扩展为 (num_envs, 1)，
    # 以便与 command 的最后一维广播匹配。
    return torch.where(_task_mask(env, VAULT_MASK_VALUE).unsqueeze(-1), vault_command, command)


# locomotion 专用非法接触终止条件：
# 只有 locomotion 环境（task_id==0）才检查非法接触，
# 避免 Vault 环境的接触被 locomotion 终止条件误触发。
def dual_loco_contact(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # mdp.illegal_contact 检查接触力是否超阈值；
    # 再与 locomotion task mask 相与，只保留 locomotion 环境的结果。
    #
    # 返回值形状为 (num_envs,)，其中只有 locomotion 环境可能为 True。
    return mdp.illegal_contact(env, threshold=threshold, sensor_cfg=sensor_cfg) & _task_mask(env, LOCO_MASK_VALUE)


# Vault 启动接触检测的包装类。
# 用于防止 Vault 启动阶段的接触保护逻辑错误地终止 locomotion 样本。
class DualIllegalResetContact(instinct_mdp.illegal_reset_contact):
    """Keep the Vault startup-contact guard from resetting locomotion samples."""

    # 重写 __call__，在父类逻辑结果上再与 Vault task mask 相与。
    def __call__(self, env, threshold, sensor_cfg, print_reason=False, episode_length_threshold=1):
        # 调用父类的 __call__ 得到原始的 Vault 启动接触检测结果，
        # 再与 Vault task mask 相与，只保留 Vault 环境的结果。
        #
        # 这样即使 locomotion 环境在启动阶段发生接触，也不会被该 Vault 专用保护逻辑重置。
        return super().__call__(env, threshold, sensor_cfg, print_reason, episode_length_threshold) & _task_mask(
            env, VAULT_MASK_VALUE
        )


# Vault 专用终止条件：base 位置偏离参考轨迹过远。
# 只在 Vault 环境上生效，避免 locomotion 环境因没有参考轨迹而被误终止。
def dual_vault_base_pos_too_far(
    env, asset_cfg: SceneEntityCfg, reference_cfg: SceneEntityCfg, distance_threshold: float,
    check_at_keyframe_threshold: float, print_reason: bool, height_only: bool,
) -> torch.Tensor:
    # 调用 instinct_mdp.pos_far_from_ref 检测位置偏离，
    # 再与 Vault task mask 相与，只保留 Vault 环境的结果。
    #
    # 参数含义与原 Vault 终止条件保持一致，本包装不改变检测逻辑，只增加任务过滤。
    return instinct_mdp.pos_far_from_ref(
        env, asset_cfg, reference_cfg, distance_threshold, check_at_keyframe_threshold, print_reason, height_only
    ) & _task_mask(env, VAULT_MASK_VALUE)


# Vault 专用终止条件：base 投影重力偏离参考过远。
def dual_vault_base_pg_too_far(
    env, asset_cfg: SceneEntityCfg, reference_cfg: SceneEntityCfg, projected_gravity_threshold: float,
    check_at_keyframe_threshold: float, z_only: bool, print_reason: bool,
) -> torch.Tensor:
    # 调用 projected_gravity_far_from_ref 检测投影重力偏离，
    # 再与 Vault task mask 相与，只保留 Vault 环境。
    #
    # 该终止条件用于判断机器人 base 姿态是否严重偏离参考运动。
    return instinct_mdp.projected_gravity_far_from_ref(
        env, asset_cfg, reference_cfg, projected_gravity_threshold, check_at_keyframe_threshold, z_only, print_reason
    ) & _task_mask(env, VAULT_MASK_VALUE)


# Vault 专用终止条件：link 位置偏离参考过远。
def dual_vault_link_pos_too_far(
    env, asset_cfg: SceneEntityCfg, reference_cfg: SceneEntityCfg, distance_threshold: float,
    in_base_frame: bool, check_at_keyframe_threshold: float, height_only: bool, print_reason: bool,
) -> torch.Tensor:
    # 调用 link_pos_far_from_ref 检测 link 位置偏离，
    # 再与 Vault task mask 相与，只保留 Vault 环境。
    #
    # 该终止条件用于判断某个机器人 link 是否偏离参考运动中的对应位置。
    return instinct_mdp.link_pos_far_from_ref(
        env, asset_cfg, reference_cfg, distance_threshold, in_base_frame,
        check_at_keyframe_threshold, height_only, print_reason,
    ) & _task_mask(env, VAULT_MASK_VALUE)


# Vault 专用终止条件：数据集中的参考轨迹已耗尽。
def dual_vault_dataset_exhausted(
    env, reference_cfg: SceneEntityCfg, print_reason: bool,
) -> torch.Tensor:
    # 调用 dataset_exhausted 检测数据集耗尽，
    # 再与 Vault task mask 相与，只保留 Vault 环境。
    #
    # locomotion 环境没有 Vault 参考轨迹，因此必须过滤掉，避免误终止。
    return instinct_mdp.dataset_exhausted(env, reference_cfg, print_reason=print_reason) & _task_mask(env, VAULT_MASK_VALUE)


# 双任务机器人重置函数。
# 根据任务的两种约定分别重置：
#   - locomotion 环境：随机初始位置/速度 + 按比例缩放关节；
#   - Vault 环境：按参考运动重置 robot state。
def reset_dual_robot(env, env_ids: torch.Tensor, motion_ref_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """Reset each half with the reset convention used by its Teacher."""
    # 从 env_ids 中筛出 locomotion 环境 ID。
    # 这里先用 _task_mask 得到全局 mask，再用 [env_ids] 取出本次需要重置的环境对应的子集。
    loco_ids = env_ids[_task_mask(env, LOCO_MASK_VALUE)[env_ids]]

    # 从 env_ids 中筛出 Vault 环境 ID。
    vault_ids = env_ids[_task_mask(env, VAULT_MASK_VALUE)[env_ids]]

    # 如果有 locomotion 环境需要重置：
    if loco_ids.numel():
        # 以均匀随机的方式重置根状态：位置 x/y/偏航 + 速度 x/y/z/roll/pitch/yaw。
        #
        # 这是 locomotion Teacher 训练时使用的重置约定。
        mdp.reset_root_state_uniform(
            env, loco_ids,
            pose_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            velocity_range={
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.1, 0.1),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            }, asset_cfg=asset_cfg,
        )

        # 按比例随机缩放关节位置和速度。
        #
        # position_range=(0.8, 1.2) 表示关节位置围绕默认值按比例缩放；
        # velocity_range=(-1.0, 1.0) 表示关节速度随机化范围。
        mdp.reset_joints_by_scale(
            env, loco_ids, position_range=(0.8, 1.2), velocity_range=(-1.0, 1.0), asset_cfg=asset_cfg
        )

    # 如果有 Vault 环境需要重置：
    if vault_ids.numel():
        # 按参考运动重置 robot state：
        # - position_offset: 位置偏移；
        # - dof_vel_ratio / base_lin_vel_ratio / base_ang_vel_ratio: 速度缩放比例；
        # - randomize_pose_range: 位置随机化范围；
        # - randomize_velocity_range: 速度随机化范围（此处为空）；
        # - randomize_joint_pos_range: 关节位置随机化范围。
        #
        # 这是 Vault Teacher 训练时使用的重置约定，保证 Student 看到的初始状态分布一致。
        instinct_mdp.reset_robot_state_by_reference(
            env, vault_ids, motion_ref_cfg=motion_ref_cfg, asset_cfg=asset_cfg,
            position_offset=[0.0, 0.0, 0.0], dof_vel_ratio=1.0,
            base_lin_vel_ratio=1.0, base_ang_vel_ratio=1.0,
            randomize_pose_range={"x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            randomize_velocity_range={}, randomize_joint_pos_range=(-0.1, 0.1),
        )


# 双任务障碍物重置函数。
# Vault 环境根据参考运动随机重置障碍物位置；
# locomotion 环境把障碍物移到一个很远的隐藏位置，相当于“隐藏”障碍物。
def reset_dual_vault_box(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("vault_box")):
    """Keep the obstacle only for Vault environments; hide it for locomotion."""
    # 从 env_ids 中筛出 task_id > 0.5 的 Vault 环境。
    # 因为 task_id 是 float32，所以这里用 > 0.5 判断是否为 1。
    vault_ids = env_ids[dual_task_id(env)[env_ids, 0] > 0.5]

    # 从 env_ids 中筛出 task_id < 0.5 的 locomotion 环境。
    # 因为 task_id 是 float32，所以这里用 < 0.5 判断是否为 0。
    loco_ids = env_ids[dual_task_id(env)[env_ids, 0] < 0.5]

    # Vault 环境：按参考运动随机化障碍物位置。
    if vault_ids.numel():
        reset_vault_box_pose_by_motion_randomized(env, vault_ids, asset_cfg=asset_cfg)

    # locomotion 环境：把障碍物移到 (100, 100, -10)，并清空随机化范围，相当于隐藏。
    #
    # active_pos 指定障碍物的固定位置；
    # pose_range={} 表示不再进行随机化。
    if loco_ids.numel():
        reset_vault_box_pose_by_motion_randomized(
            env, loco_ids, asset_cfg=asset_cfg, active_pos=(100.0, 100.0, -10.0), pose_range={}
        )


# 混合任务命令配置。
# 继承自 locomotion 的命令配置，并额外加入 Vault 的参考命令项：
# - position_ref_command；
# - position_b_ref_command；
# - rotation_ref_command；
# - joint_pos_ref_command；
# - joint_vel_ref_command。
@configclass
class DualCommandsCfg(G1OmniLocoCommandsCfg):
    """Locomotion command plus the Vault reference commands."""

    # 以下命令项都从 Vault 配置中深拷贝，避免与原配置共享引用。
    # 这些参考命令主要供 Vault Teacher / critic 使用。
    position_ref_command = deepcopy(_VAULT_COMMANDS.position_ref_command)
    position_b_ref_command = deepcopy(_VAULT_COMMANDS.position_b_ref_command)
    rotation_ref_command = deepcopy(_VAULT_COMMANDS.rotation_ref_command)
    joint_pos_ref_command = deepcopy(_VAULT_COMMANDS.joint_pos_ref_command)
    joint_vel_ref_command = deepcopy(_VAULT_COMMANDS.joint_vel_ref_command)


# 混合任务 Student policy 观测配置。
# 继承自 Vault 的 Student policy 观测配置，并加入：
# - velocity_command：locomotion 速度命令；
# - task_id：任务标识（0 表示 locomotion，1 表示 Vault）。
@configclass
class DualStudentPolicyObsCfg(VaultStudentPolicyObsCfg):
    """The visual Student input with locomotion command and task mode."""

    # Vault 样本固定接收 1 m/s 前进命令；locomotion 样本仍接收随机 base_velocity 命令。
    #
    # 注意：虽然这里观测项名字叫 velocity_command，但实际数据由 dual_student_velocity_command 生成。
    velocity_command = ObsTermCfg(func=dual_student_velocity_command, params={"command_name": "base_velocity"})

    # 任务标识观测项：使用 dual_task_id 生成每个环境的 task_id。
    # 该观测项用于 Teacher 路由或 critic 输入；可见 Student 配置会移除它。
    task_id = ObsTermCfg(func=dual_task_id)


# 混合任务 critic 观测配置。
# 继承自 Vault 的 teacher critic 观测配置，并在其后追加 locomotion 的 623 维输入。
# 这样 critic 观测 = Vault critic 前缀 + locomotion 输入 + task_id。
@configclass
class DualCriticObsCfg(PhpTeacherCriticObsCfg):
    """Vault critic prefix followed by the exact 623-D locomotion input."""

    # locomotion 专用高度扫描观测项。
    # 使用名为 loco_height_scanner 的传感器，输出裁剪到 [-1.0, 1.0]。
    loco_height_scan = ObsTermCfg(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("loco_height_scanner")},
        clip=[-1.0, 1.0],
    )

    # locomotion 速度命令观测项。
    # 直接读取 base_velocity 命令生成器的输出。
    loco_velocity_command = ObsTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    # locomotion 投影重力观测项。
    # 使用 instinct_mdp.projected_gravity 获取 base 投影重力。
    loco_projected_gravity = ObsTermCfg(func=instinct_mdp.projected_gravity)

    # locomotion base 角速度观测项。
    loco_base_ang_vel = ObsTermCfg(func=instinct_mdp.base_ang_vel)

    # locomotion 关节相对位置观测项。
    loco_joint_pos_rel = ObsTermCfg(func=mdp.joint_pos_rel)

    # locomotion 关节速度观测项。
    loco_joint_vel = ObsTermCfg(func=mdp.joint_vel)

    # locomotion 上一步动作观测项。
    loco_last_action = ObsTermCfg(func=instinct_mdp.last_action)

    # 任务标识观测项。
    # 用于让 critic / 双 Teacher DAgger 知道当前样本属于哪个任务。
    task_id = ObsTermCfg(func=dual_task_id)


# 混合任务的整体观测配置。
# policy 使用 DualStudentPolicyObsCfg；
# critic 使用 DualCriticObsCfg。
@configclass
class DualObservationsCfg:
    # policy 组：Student 策略可见观测。
    policy: DualStudentPolicyObsCfg = DualStudentPolicyObsCfg()
    # critic 组：critic / Teacher 标签路由使用的观测。
    critic: DualCriticObsCfg = DualCriticObsCfg()


# 自动翻越 Student 的 Actor 观测：不包含任务标识和 Vault 参考动作，
# 仅依赖前视深度、速度命令与机器人本体状态决定动作。
@configclass
class AutonomousStudentPolicyObsCfg(ObsGroupCfg):
    # Vault 样本固定接收 1 m/s 前进命令；locomotion 样本仍接收随机 base_velocity 命令。
    velocity_command = ObsTermCfg(func=dual_student_velocity_command, params={"command_name": "base_velocity"})
    # 深度图像观测项。
    # 使用 camera 传感器的 distance_to_image_plane_noised 数据；
    # visualizable_image 用于可视化或图像型观测。
    depth_image = ObsTermCfg(
        func=instinct_mdp.visualizable_image,
        params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "distance_to_image_plane_noised"},
    )
    # 投影重力观测项，并添加均匀噪声 [-0.05, 0.05]。
    projected_gravity = ObsTermCfg(
        func=mdp.projected_gravity,
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    )
    # base 角速度观测项，并添加均匀噪声 [-0.2, 0.2]。
    base_ang_vel = ObsTermCfg(
        func=mdp.base_ang_vel,
        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    )
    # 关节相对位置观测项，并添加均匀噪声 [-0.01, 0.01]。
    joint_pos = ObsTermCfg(
        func=mdp.joint_pos_rel,
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    # 关节相对速度观测项，并添加均匀噪声 [-0.5, 0.5]。
    joint_vel = ObsTermCfg(
        func=mdp.joint_vel_rel,
        noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
    )
    # 上一步动作观测项。
    last_action = ObsTermCfg(func=mdp.last_action)

    def __post_init__(self):
        # 开启观测噪声/ corruption。
        self.enable_corruption = True
        # 不将各观测项拼接成单一向量，保留分组/分项结构。
        self.concatenate_terms = False


@configclass
class AutonomousStudentObservationsCfg:
    # policy 组：仅视觉与本体状态的自主 Student 观测。
    policy: AutonomousStudentPolicyObsCfg = AutonomousStudentPolicyObsCfg()
    # Teacher 参考量和 task_id 保留在 critic，供双教师 DAgger 生成标签。
    critic: DualCriticObsCfg = DualCriticObsCfg()


# 双 Teacher DAgger 主环境配置。
# 继承自 G1PerceptiveVaultPelvisStudentEnvCfg，在 Vault 场景基础上混合 locomotion 任务。
@configclass
class G1DualTeacherDaggerEnvCfg(G1PerceptiveVaultPelvisStudentEnvCfg):
    """Balanced mixed-task scene used by ``DualTeacherDagger``."""

    # 使用混合命令配置。
    commands: DualCommandsCfg = DualCommandsCfg()

    # 使用混合观测配置。
    observations: DualObservationsCfg = DualObservationsCfg()

    def __post_init__(self):
        # 先调用父类 __post_init__，完成 Vault 场景基础初始化。
        super().__post_init__()

        # The pelvis parent assigns a module-level reference config. Own a
        # copy before train/play changes so Student settings cannot leak into
        # the frozen Teacher configuration or another environment instance.
        #
        # 中文说明：
        # 父类可能将 motion_reference 指向模块级共享配置。
        # 这里深拷贝一份，使当前环境实例拥有独立配置，
        # 防止 Student 训练/播放时的修改污染 Teacher 配置或其他环境实例。
        self.scene.motion_reference = deepcopy(self.scene.motion_reference)

        # 用混合命令配置覆盖父类配置。
        self.commands = DualCommandsCfg()

        # 用混合观测配置覆盖父类配置。
        self.observations = DualObservationsCfg()

        # The Vault scan is 31x9=279 points.  Locomotion's frozen actor uses
        # an independent pelvis scan of 31x17=527 points.
        #
        # 中文说明：
        # Vault 的高度扫描是 31x9=279 个点；
        # locomotion 冻结的 actor 使用独立的 pelvis 高度扫描，为 31x17=527 个点。
        # 因此这里为 locomotion 单独增加一个高度扫描传感器 loco_height_scanner。
        #
        # 为 locomotion 单独增加一个高度扫描传感器 loco_height_scanner。
        # 该扫描挂在 pelvis 上，使用 3.0 x 1.6 的网格，分辨率 0.1，即 31 x 17 = 527 个点。
        self.scene.loco_height_scanner = MultiMeshRayCasterCfg(
            # 传感器挂载在机器人 pelvis 上。
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            # 更新周期为 0.02 秒。
            update_period=0.02,
            # 射线起点相对 pelvis 向上偏移 20 米，然后向下扫描。
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            # 射线对齐方式为 yaw，即只随偏航角旋转。
            ray_alignment="yaw",
            # 网格模式：分辨率 0.1 米，尺寸 3.0 x 1.6 米。
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(3.0, 1.6)),
            # 扫描目标为地面。
            mesh_prim_paths=["/World/ground"],
            # 关闭调试可视化。
            debug_vis=False,
        )

        # 覆盖 reset_robot 事件：使用双任务重置函数。
        # 重置模式为 "reset"，参数包括运动参考配置和机器人实体配置。
        self.events.reset_robot = EventTermCfg(
            func=reset_dual_robot,
            mode="reset",
            params={"motion_ref_cfg": SceneEntityCfg("motion_reference"), "asset_cfg": SceneEntityCfg("robot")},
        )

        # 覆盖 reset_vault_box 事件：使用双任务障碍物重置函数。
        # Vault 环境保留障碍物，locomotion 环境隐藏障碍物。
        self.events.reset_vault_box = EventTermCfg(
            func=reset_dual_vault_box,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("vault_box")},
        )

        # Motion-reference termination is meaningful only for Vault samples;
        # locomotion samples use the flat-task fall/contact termination.
        #
        # 中文说明：
        # 基于运动参考的终止条件只对 Vault 样本有意义；
        # locomotion 样本使用平地任务的摔倒/接触终止条件。
        #
        # 将 Vault 专用的终止条件替换为双任务包装版本，只对 Vault 环境生效。
        self.terminations.base_pos_too_far.func = dual_vault_base_pos_too_far
        self.terminations.base_pg_too_far.func = dual_vault_base_pg_too_far
        self.terminations.link_pos_too_far.func = dual_vault_link_pos_too_far
        self.terminations.dataset_exhausted.func = dual_vault_dataset_exhausted
        self.terminations.illegal_reset_contact.func = DualIllegalResetContact

        # 新增 locomotion 专用的 base 接触终止条件：
        # 只有 locomotion 环境在 torso/shoulder/elbow/wrist/hip/knee 等部位发生非法接触时才终止。
        #
        # sensor_cfg 指定接触力传感器 contact_forces，并限定需要检查的 body_names；
        # threshold=1.0 表示接触力阈值。
        self.terminations.base_contact = TerminationTermCfg(
            func=dual_loco_contact,
            time_out=False,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["torso_link", ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*", ".*_hip_.*", ".*_knee_.*"],
                ),
                "threshold": 1.0,
            },
        )

        # Adaptive bins are useful for PPO Teacher training, but they make a
        # mixed DAgger rollout depend on Vault-only failure bookkeeping.
        #
        # 中文说明：
        # 自适应 bin 对 PPO Teacher 训练有用，但会让混合 DAgger rollout 依赖仅 Vault 的失败统计。
        # 因此这里关闭自适应采样和失败计数平滑：混合 DAgger 训练中不需要依赖 Vault 专用的失败统计。
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None

        # 关闭 push_robot 事件：混合 DAgger 训练中不需要外力扰动。
        self.events.push_robot = None

        # Keep the box and motion reference available for Vault envs.  The
        # locomotion half hides the box and simply routes labels to its expert.
        #
        # 中文说明：
        # Vault 环境保留障碍物和运动参考；
        # locomotion 那一半会隐藏障碍物，并直接把标签路由到其对应专家。
        #
        # 设置运行名称，便于日志区分。
        self.run_name = "g1_loco_vault_dual_dagger"


@configclass
class G1AutonomousVaultDaggerEnvCfg(G1DualTeacherDaggerEnvCfg):
    """Mixed locomotion/Vault DAgger with vision-only autonomous Student routing."""

    # 使用自主 Student 观测配置：不包含 task_id 和 Vault 参考动作，仅依赖视觉和本体状态。
    observations: AutonomousStudentObservationsCfg = AutonomousStudentObservationsCfg()

    def __post_init__(self):
        # 先执行父类初始化。
        super().__post_init__()
        # 再次显式设置观测配置，确保使用 AutonomousStudentObservationsCfg。
        self.observations = AutonomousStudentObservationsCfg()
        # 设置运行名称。
        self.run_name = "g1_loco_vault_autonomous_dagger"


# 双 Teacher DAgger 的可视化/播放模式配置。
# 在基础配置上做少量修改，便于人工观察策略行为。
@configclass
class G1DualTeacherDaggerEnvCfg_PLAY(G1DualTeacherDaggerEnvCfg):
    def __post_init__(self):
        # 先调用父类 __post_init__，完成基础配置。
        super().__post_init__()

        # 播放模式只使用 2 个环境（一 locomotion 一 Vault）。
        # 因为任务按 env_id 奇偶分配，所以 2 个环境正好各一个。
        self.scene.num_envs = 2

        # 拉长 episode 长度，便于持续观察。
        self.episode_length_s = 6000.0

        # 打开 motion_reference 的可视化。
        self.scene.motion_reference.debug_vis = True

        # 关闭 motion bin 长度限制，播放模式下不按 bin 截断轨迹。
        self.scene.motion_reference.motion_buffers[vault.vault_cfg.MOTION_NAME].motion_bin_length_s = None

        # 设置起始帧采样策略为 "independent"，每个环境独立采样起始帧。
        self.scene.motion_reference.motion_buffers[
            vault.vault_cfg.MOTION_NAME
        ].env_starting_stub_sampling_strategy = "independent"

        # 播放模式同样关闭自适应采样和失败计数平滑。
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None


@configclass
class G1AutonomousVaultDaggerEnvCfg_PLAY(G1DualTeacherDaggerEnvCfg_PLAY):
    """Two-scene visual check: one clear path and one obstacle path."""

    # 使用自主 Student 观测配置。
    observations: AutonomousStudentObservationsCfg = AutonomousStudentObservationsCfg()

    def __post_init__(self):
        # 先执行父类播放模式初始化。
        super().__post_init__()
        # 显式设置自主 Student 观测配置。
        self.observations = AutonomousStudentObservationsCfg()
        # 播放模式下关闭运动参考可视化，便于观察仅视觉 Student 的行为。
        self.scene.motion_reference.debug_vis = False


# Current Student: unified height scan; depth classes above are kept for later distillation.
#
# 中文说明：
# 当前 Student 使用统一高度扫描；
# 上面的深度图观测类保留，供后续蒸馏使用。
@configclass
class UnifiedHeightPolicyObsCfg(ObsGroupCfg):
    """Fixed order: height, command, gravity, angular velocity, joints, action."""

    # 高度扫描观测项。
    # 使用 student_height_scanner 传感器，并将输出裁剪到 [-1.0, 1.0]。
    height_scan = ObsTermCfg(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("student_height_scanner")},
        clip=[-1.0, 1.0],
    )
    # 速度命令观测项。
    # Vault 样本固定 1 m/s 前进，locomotion 样本保留随机命令。
    velocity_command = ObsTermCfg(func=dual_student_velocity_command)
    # 投影重力观测项，并添加均匀噪声 [-0.05, 0.05]。
    projected_gravity = ObsTermCfg(
        func=mdp.projected_gravity,
        noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    )
    # base 角速度观测项，并添加均匀噪声 [-0.2, 0.2]。
    base_ang_vel = ObsTermCfg(
        func=mdp.base_ang_vel,
        noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    )
    # 关节相对位置观测项，并添加均匀噪声 [-0.01, 0.01]。
    joint_pos = ObsTermCfg(
        func=mdp.joint_pos_rel,
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    # 关节速度观测项，并添加均匀噪声 [-0.5, 0.5]。
    joint_vel = ObsTermCfg(
        func=mdp.joint_vel,
        noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
    )
    # 上一步动作观测项。
    last_action = ObsTermCfg(func=mdp.last_action)

    def __post_init__(self):
        # 开启观测噪声/ corruption。
        self.enable_corruption = True
        # 不将各观测项拼接成单一向量，保留分项结构。
        self.concatenate_terms = False
        # 不使用历史长度，即当前时刻观测。

        # Preserve temporal context for the new history-frame Student run.
        # The observation manager accepts None at group level when each term
        # declares its own history length.
        self.history_length = None
        self.height_scan.history_length = 3
        self.height_scan.flatten_history_dim = True
        self.velocity_command.history_length = 0
        for name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action"):
            term = getattr(self, name)
            term.history_length = 5
            term.flatten_history_dim = True


@configclass
class UnifiedHeightObservationsCfg:
    # policy 组：统一高度扫描 Student 观测。
    policy: UnifiedHeightPolicyObsCfg = UnifiedHeightPolicyObsCfg()
    # critic 组：与双 Teacher DAgger 相同的 critic 观测。
    critic: DualCriticObsCfg = DualCriticObsCfg()


# 配置统一高度扫描 Student 的辅助函数。
# random_start 控制是否从参考运动的中间随机位置开始。
def _configure_height_student(env_cfg, random_start: bool):
    # 设置观测配置为统一高度扫描观测。
    env_cfg.observations = UnifiedHeightObservationsCfg()
    # Copy the Vault sensor to retain its moving-box target. Widen only the
    # Student grid; both frozen experts retain their original observation data.
    #
    # 中文说明：
    # 复制 Vault 的高度扫描传感器，以保留其移动障碍物目标；
    # 只加宽 Student 的扫描网格；两个冻结专家仍保留各自原始观测数据。
    scanner = deepcopy(env_cfg.scene.height_scanner)
    # Student 高度扫描挂载到机器人 pelvis 上。
    scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
    # Student 扫描网格：分辨率 0.1 米，尺寸 3.0 x 1.6 米。
    scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.1, size=(3.0, 1.6))
    # 更新周期设置为仿真步长乘以 decimation，即每个控制步更新一次。
    scanner.update_period = env_cfg.sim.dt * env_cfg.decimation
    # 关闭调试可视化。
    scanner.debug_vis = False
    # 将配置好的扫描传感器注册为 student_height_scanner。
    env_cfg.scene.student_height_scanner = scanner
    # 该 Student 场景不需要相机，置空。
    env_cfg.scene.camera = None
    # 关闭射线偏移随机化事件。
    env_cfg.events.randomize_ray_offsets = None
    # This Student scene has no RobotReference articulation. Motion reference
    # data/FK remain enabled; only the optional reference-robot display is off.
    #
    # 中文说明：
    # 该 Student 场景没有 RobotReference 关节体；
    # 运动参考数据/FK 仍然启用，只关闭可选的参考机器人显示。
    reference = env_cfg.scene.motion_reference
    # 关闭运动参考调试可视化。
    reference.debug_vis = False
    # 不设置参考机器人 prim 路径。
    reference.reference_prim_path = None
    # Select the independent sampler explicitly: a configured motion bin would
    # otherwise override motion_start_from_middle_range in AmassMotion.
    #
    # 中文说明：
    # 显式选择独立采样器；
    # 否则已配置的 motion bin 会覆盖 AmassMotion 中的 motion_start_from_middle_range。
    motion = reference.motion_buffers[vault.vault_cfg.MOTION_NAME]
    # 关闭 motion bin 长度限制。
    motion.motion_bin_length_s = None
    # 设置起始采样策略为 independent。
    motion.env_starting_stub_sampling_strategy = "independent"
    # The supplied NPZ marks frames [102, 665) as (vx, vy, wz)=(1, 0, 0).
    # Only the Student uses this active window. Teachers keep their full clips.
    #
    # 中文说明：
    # 提供的 NPZ 将帧区间 [102, 665) 标记为 (vx, vy, wz)=(1, 0, 0)。
    # 只有 Student 使用这个有效窗口；Teacher 保留完整片段。
    motion.retargeted_frame_range = (102, 665)
    # Reset robot pose AND velocities from a random reference state, not just
    # the reference clock. Keep enough remaining motion to learn a rollout.
    #
    # 中文说明：
    # 从随机参考状态重置机器人位姿和速度，而不仅仅是参考时钟；
    # 保留足够的剩余运动用于学习 rollout。
    #
    # random_start=True 时，在 [0.0, 0.9] 范围内选择起始位置；
    # random_start=False 时，从 0.0 开始。
    motion.motion_start_from_middle_range = [0.0, 0.9] if random_start else [0.0, 0.0]
    # 设置运行名称。
    env_cfg.run_name = "g1_loco_vault_unified_height_dagger"


@configclass
class G1UnifiedHeightDaggerEnvCfg(G1DualTeacherDaggerEnvCfg):
    # 使用统一高度扫描观测配置。
    observations: UnifiedHeightObservationsCfg = UnifiedHeightObservationsCfg()

    def __post_init__(self):
        # 先执行父类初始化。
        super().__post_init__()
        # 配置统一高度扫描 Student，并允许从参考运动中间随机开始。
        _configure_height_student(self, random_start=True)


@configclass
class G1UnifiedHeightDaggerEnvCfg_PLAY(G1DualTeacherDaggerEnvCfg_PLAY):
    # 播放模式使用统一高度扫描观测配置。
    observations: UnifiedHeightObservationsCfg = UnifiedHeightObservationsCfg()

    def __post_init__(self):
        # 先执行父类播放模式初始化。
        super().__post_init__()
        # 配置统一高度扫描 Student；播放模式不随机选择起始帧，从 0.0 开始。
        _configure_height_student(self, random_start=False)
