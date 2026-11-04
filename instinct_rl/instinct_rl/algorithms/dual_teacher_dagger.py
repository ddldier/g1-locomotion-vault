"""Task-routed, frozen locomotion + Vault teachers for pure online DAgger."""

# =============================================================================
# 文件说明：
# 该文件定义了一个双教师（Dual Teacher）的纯在线 DAgger 适配器类 DualTeacherDagger。
#
# 核心思想：
#   - 在一个混合任务环境中，同时存在 locomotion 任务和 Vault 任务；
#   - locomotion 环境（task_id=0）由 locomotion 教师网络给出动作标签；
#   - Vault 环境（task_id=1）由 Vault 教师网络给出动作标签；
#   - 两个教师网络都被冻结（requires_grad=False），只作为监督标签来源；
#   - Student 学生策略通过这些标签进行纯 DAgger 蒸馏（不使用 PPO loss）。
#
# 继承关系：
#   DualTeacherDagger -> VaultDaggerTPPO -> TPPO -> PPO
#
# 关键设计：
#   1. 必须显式传入 locomotion_teacher_checkpoint，并校验其与 Vault 教师的
#      动作空间、仿真步长、机器人配置是否一致，防止不兼容的教师被混用；
#   2. 只支持 using_ppo=False（纯在线 DAgger），因为两个教师的标签就是纯监督信号；
#   3. get_teacher_actions 根据 task_id 路由到对应的教师网络；
#   4. compute_losses 在基础蒸馏损失之外，额外统计两个任务的 MSE 以进行监控。
# 本文件仅添加中文注释，不修改原有逻辑。
# =============================================================================

from pathlib import Path

import torch
import yaml

import instinct_rl.modules as modules
from .tppo import TPPO
from .vault_dagger_tppo import VaultDaggerTPPO


# locomotion 教师策略的 policy 观测项定义。
# 每项为 (name, shape)，其中 shape 为该观测项的维度元组。
# 总维度 = 527 + 3 + 3 + 3 + 29 + 29 + 29 = 623。
LOCO_POLICY_TERMS = {
    "height_scan": (527,),       # 高度扫描：31 x 17 网格
    "velocity_command": (3,),    # 速度命令：x/y/yaw
    "projected_gravity": (3,),   # 投影重力：3 维
    "base_ang_vel": (3,),        # base 角速度：3 维
    "joint_pos_rel": (29,),      # 关节相对位置：29 个关节
    "joint_vel": (29,),          # 关节速度：29 个关节
    "last_action": (29,),        # 上一步动作：29 个关节
}

# locomotion 教师策略的 critic 观测项定义（此处主要供构建网络结构时使用）。
# 总维度 = 3 + 3 + 3 + 3 + 29 + 29 + 29 = 99。
LOCO_CRITIC_TERMS = {
    "base_lin_vel": (3,),          # base 线速度：3 维
    "base_ang_vel": (3,),          # base 角速度：3 维
    "projected_gravity": (3,),     # 投影重力：3 维
    "velocity_commands": (3,),     # 速度命令：3 维
    "joint_pos": (29,),            # 关节位置：29 个关节
    "joint_vel": (29,),            # 关节速度：29 个关节
    "actions": (29,),              # 动作：29 个关节
}


# 双教师 DAgger 类。
# 在 VaultDaggerTPPO 的基础上，额外加载一个 locomotion 教师，
# 并根据观测中的 task_id 路由到对应的教师网络。
# 使用约定：
#   task_id = 0 → locomotion 教师；
#   task_id = 1 → Vault 教师。
class DualTeacherDagger(VaultDaggerTPPO):
    """Use task_id=0 for locomotion and task_id=1 for Vault labels."""

    def __init__(self, *args, locomotion_teacher_checkpoint, **kwargs):
        # 该实现只支持纯 DAgger（using_ppo=False），
        # 若传入 using_ppo=True 则直接报错。
        if kwargs.get("using_ppo", True):
            raise ValueError("DualTeacherDagger currently supports using_ppo=False only")

        # 调用父类 VaultDaggerTPPO 的初始化，完成 Vault 教师加载及通用配置。
        super().__init__(*args, **kwargs)

        # 将 locomotion 教师 checkpoint 路径转为 Path 对象。
        checkpoint = Path(locomotion_teacher_checkpoint)

        # 读取 locomotion 教师的 env.yaml，用于与 Vault 教师进行兼容性校验。
        with (checkpoint.parent / "params/env.yaml").open() as stream:
            loco_env = yaml.load(stream, Loader=yaml.BaseLoader)

        # 读取 Vault 教师的 env.yaml（self.teacher_checkpoint 已在父类设置）。
        with (self.teacher_checkpoint.parent / "params/env.yaml").open() as stream:
            vault_env = yaml.load(stream, Loader=yaml.BaseLoader)

        # 校验两个教师的关键仿真配置是否一致：
        # - actions：动作空间配置；
        # - decimation：控制频率（每多少仿真步执行一次动作）。
        for key in ("actions", "decimation"):
            if loco_env[key] != vault_env[key]:
                raise ValueError(f"Teacher {key} mismatch")

        # 校验两个教师使用的机器人配置是否一致：
        # - spawn：初始生成配置；
        # - init_state：初始状态配置；
        # - actuators：执行器配置。
        for key in ("spawn", "init_state", "actuators"):
            if loco_env["scene"]["robot"][key] != vault_env["scene"]["robot"][key]:
                raise ValueError(f"Teacher robot {key} mismatch")

        # 校验仿真步长 dt 是否一致，否则两个教师观测/动作的时间尺度不一致。
        if loco_env["sim"]["dt"] != vault_env["sim"]["dt"]:
            raise ValueError("Teacher simulation dt mismatch")

        # 读取 locomotion 教师的 agent.yaml，用于构建教师网络。
        with (checkpoint.parent / "params/agent.yaml").open() as stream:
            cfg = yaml.full_load(stream)

        # 复制 policy 配置并弹出 class_name，以便分开传入构造函数。
        policy_cfg = cfg["policy"].copy()

        # 构建 locomotion 教师 actor-critic 网络：
        # - 类名由配置中的 class_name 指定；
        # - obs_format 使用本文件顶部定义的 LOCO_POLICY_TERMS / LOCO_CRITIC_TERMS；
        # - num_actions=29，num_rewards=1。
        self.locomotion_teacher = modules.build_actor_critic(
            policy_cfg.pop("class_name"), policy_cfg,
            {"policy": LOCO_POLICY_TERMS, "critic": LOCO_CRITIC_TERMS}, 29, 1,
        ).to(self.device)

        # 加载 locomotion 教师的模型权重，strict=True 要求结构完全匹配。
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.locomotion_teacher.load_state_dict(state["model_state_dict"], strict=True)

        # 设置 eval 模式并冻结参数（不参与训练）。
        self.locomotion_teacher.eval().requires_grad_(False)

        # 复制 normalizer 配置并弹出 class_name。
        norm_cfg = cfg["normalizers"]["policy"].copy()

        # 构建 locomotion 教师的观测 normalizer：
        # - 输入维度固定为 623（对应 LOCO_POLICY_TERMS 的总维度）；
        # - 类名与参数来自配置。
        self.locomotion_normalizer = modules.build_normalizer(623, norm_cfg.pop("class_name"), norm_cfg).to(self.device)

        # 加载 normalizer 权重，strict=True；并设置 eval 模式、冻结参数。
        self.locomotion_normalizer.load_state_dict(state["policy_normalizer_state_dict"], strict=True)
        self.locomotion_normalizer.eval().requires_grad_(False)

        # 打印加载信息，便于确认使用了正确的 checkpoint。
        print(f"[Dual DAgger] Frozen locomotion teacher: {checkpoint}")

    # 覆盖父类 init_storage，校验 critic 观测的项、顺序与形状是否符合约定。
    # 约定顺序：
    #   1. Vault 教师的 critic 观测项；
    #   2. locomotion 的 7 个观测项（loco_ 前缀）；
    #   3. task_id。
    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        # 取出 Vault 教师的 critic 观测项列表。
        vault_terms = list(self.teacher_policy_cfg_dict["obs_format"]["critic"].items())

        # 构造期望的 critic 观测项完整列表：
        # Vault 项 + locomotion 项 + task_id。
        expected = vault_terms + [
            ("loco_height_scan", (527,)),
            ("loco_velocity_command", (3,)),
            ("loco_projected_gravity", (3,)),
            ("loco_base_ang_vel", (3,)),
            ("loco_joint_pos_rel", (29,)),
            ("loco_joint_vel", (29,)),
            ("loco_last_action", (29,)),
            ("task_id", (1,)),
        ]

        # 若实际 critic 观测项与期望不一致（名称、顺序或形状不同），则报错。
        if list(obs_format["critic"].items()) != expected:
            raise ValueError("Dual DAgger critic order must be Vault, locomotion, task_id")

        # 调用父类 init_storage 完成 storage 构建。
        return super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)

    # 覆盖父类 get_teacher_actions，根据 task_id 路由到对应教师网络。
    # 观测约定：
    #   - critic_obs 最后一维为 1234 维原始观测（Vault critic 610 维 + loco 623 维 + task_id 1 维）；
    #   - task_id < 0.5 → locomotion；task_id > 0.5 → Vault。
    @torch.no_grad()
    def get_teacher_actions(self, obs, critic_obs=None):
        # 校验输入维度。
        if critic_obs is None or critic_obs.shape[-1] != 1234:
            raise ValueError("Dual DAgger teacher input must have 1234 raw values")

        # 提取 task_id（critic_obs 的最后一维）。
        task = critic_obs[:, -1]
        if not torch.all((task == 0) | (task == 1)):
            raise ValueError("Dual DAgger task_id must contain only 0 or 1")

        # 预分配输出动作标签张量，形状 (batch, 29)。
        labels = torch.empty((len(task), 29), device=self.device, dtype=obs.dtype)

        # 构造 locomotion 与 Vault 的布尔 mask。
        loco, vault = task == 0, task == 1

        # 对 locomotion 环境：
        # - 从 critic_obs 中切出 locomotion 的观测片段 [610:1233]（623 维，不含 task_id）；
        # - 先通过 locomotion normalizer 归一化；
        # - 再调用 locomotion 教师的 act_inference 得到确定性动作。
        if loco.any():
            labels[loco] = self.locomotion_teacher.act_inference(
                self.locomotion_normalizer(critic_obs[loco, 610:1233])
            )

        # 对 Vault 环境：
        # - 只取 critic_obs 的前 448 维作为 Vault 教师输入；
        # - 调用父类 TPPO.get_teacher_actions（注意：这里显式调用 TPPO 版本，
        #   而不是 VaultDaggerTPPO，因为 input 已经截断为 448 维）。
        if vault.any():
            labels[vault] = TPPO.get_teacher_actions(self, obs[vault], critic_obs[vault, :448])

        # 校验教师输出均为有限值，避免 NaN/Inf 污染蒸馏标签。
        if not torch.isfinite(labels).all():
            raise ValueError("Non-finite dual-teacher labels")

        return labels

    # 覆盖父类 compute_losses：
    # 在基础蒸馏损失之外，额外统计两个任务各自的 MSE，便于监控训练。
    def compute_losses(self, minibatch):
        # 先调用父类计算损失和统计信息。
        losses, variables, stats = super().compute_losses(minibatch)

        # 计算每个样本的 MSE 平方和：
        # error = sum((action_mean - action_labels)^2)。
        error = (self.actor_critic.action_mean - minibatch.action_labels).square().sum(-1)

        # 分别统计 locomotion（task_id=0）和 vault（task_id=1）的 MSE 均值。
        for task, name in ((0, "locomotion"), (1, "vault")):
            # 根据 minibatch.critic_obs 的最后一维（task_id）构造 mask。
            mask = minibatch.critic_obs[:, -1] == task

            # 若 mask 中存在至少一个样本，则计算均值；否则写入 0。
            stats[f"{name}_mse_sum"] = error[mask].mean().detach() if mask.any() else error.new_zeros(())

        return losses, variables, stats