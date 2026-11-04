# =============================================================================
# TPPO 适配器：Vault Teacher -> Visual Student 的接口层
# -----------------------------------------------------------------------------
# 这个文件在通用 TPPO（Two-stage Policy Policy Optimization / 教师-学生蒸馏
# 框架）之上，做了一层 vault 任务专用的适配，负责：
#
#   1. 固定使用指定的教师 checkpoint（而不是让 TPPO 自动扫目录找最新一代）；
#   2. 校验教师 checkpoint 的根坐标系（torso vs pelvis）是否与任务匹配；
#   3. 实现"纯 DAgger 模式"下的动作执行逻辑：教师接管的环境执行教师动作，
#      学生接管的环境执行 student 的 action_mean（不采样，因为纯 MSE 蒸馏
#      不训练 std，采样噪声会破坏标签质量）；
#   4. 冻结并加载教师策略 + 教师侧的观测 normalizer；
#   5. 校验 critic 观测的前缀与教师 policy 观测的名称/顺序/形状严格一致，
#      避免"总维度对了但切错观测"的静默 bug；
#   6. 在统计信息里额外记录当前 iteration 的 teacher_act_prob。
#
# 注意：教师用的是 privileged critic 观测的前 448 维（= teacher_actor_obs_dim），
# 因为 agent.yaml 里 teacher_policy.obs_format.critic 的前若干项与这里
# PhpTeacherCriticObsCfg 的顺序是一一对应的。
# =============================================================================

"""TPPO adapter for the Vault Teacher -> visual Student interface."""

import math
from pathlib import Path

import torch
import yaml
import instinct_rl.modules as modules
from .tppo import TPPO


class VaultDaggerTPPO(TPPO):
    """带显式 "privileged-critic -> Teacher" 投影的 TPPO。

    继承通用 TPPO，额外做的事情：
      - 构造函数强制要求 teacher_checkpoint，并做根坐标系校验；
      - 覆写 act() 实现纯 DAgger 模式下的动作执行；
      - 覆写 load_teacher_policy() 同时加载教师网络和教师观测 normalizer；
      - 覆写 init_storage() 做观测名称/顺序/形状的严格校验；
      - 覆写 get_teacher_actions() 从 critic_obs 中切片出教师需要的 448 维；
      - 覆写 compute_losses() 记录 teacher_probability 统计。
    """

    # 教师 actor 的观测维度（448）。等于教师 policy 观测各项 shape 乘积之和：
    #   joint_pos_ref(1*29) + joint_vel_ref(1*29) + position_ref(1*3)
    # + rotation_ref(1*6) + height_scan(279) + global_pelvis_pos(3)
    # + global_pelvis_vel(3) + base_lin_vel(3) + projected_gravity(3)
    # + base_ang_vel(3) + joint_pos(29) + joint_vel(29) + last_action(29)
    # = 29+29+3+6+279+3+3+3+3+3+29+29+29 = 448
    # （对应 agent.yaml 中 teacher_policy.obs_format.policy 的维度）
    teacher_actor_obs_dim = 448

    def __init__(self, *args, teacher_checkpoint, expected_teacher_root=None, **kwargs):
        # ---------------------------------------------------------------------
        # 步骤 1：显式指定教师 checkpoint（禁止 TPPO 自动选最新）
        # ---------------------------------------------------------------------
        # 必须明确选择检查点，不能让 TPPO 自动选择目录中最新的一代。
        self.teacher_checkpoint = Path(teacher_checkpoint)
        if not self.teacher_checkpoint.is_file():
            raise FileNotFoundError(self.teacher_checkpoint)

        # ---------------------------------------------------------------------
        # 步骤 2：根坐标系校验（torso vs pelvis）
        # ---------------------------------------------------------------------
        # 教师的动作参考根必须与当前任务一致，否则 obs 维度/语义会错位。
        # 从 checkpoint 旁边的 params/env.yaml 读取 motion_reference.prim_path，
        # 取最后一段作为实际根（例如 ".../Robot/pelvis" -> "pelvis"）。
        if expected_teacher_root is not None:
            # BaseLoader 只读取数据，避免 env.yaml 中的 Python slice 标签触发构造错误。
            # （env.yaml 里有 !!python/object/apply:builtins.slice 这类标签，
            # 用 full_load 会尝试构造对象，BaseLoader 则全部当字符串读）
            with (self.teacher_checkpoint.parent / 'params/env.yaml').open() as stream:
                teacher_env = yaml.load(stream, Loader=yaml.BaseLoader)
            root_path = teacher_env['scene']['motion_reference']['prim_path']
            actual_root = root_path.rstrip('/').rsplit('/', 1)[-1]
            if actual_root != expected_teacher_root:
                raise ValueError(
                    f'Vault Teacher root mismatch: expected {expected_teacher_root}, '
                    f'got {actual_root} in {self.teacher_checkpoint}. '
                    'Use the matching torso/pelvis task and checkpoint.'
                )

        # ---------------------------------------------------------------------
        # 步骤 3：交给父类 TPPO 完成通用初始化
        # ---------------------------------------------------------------------
        # teacher_logdir 固定为 checkpoint 所在目录，父类会从中找 normalizer 等。
        super().__init__(*args, teacher_logdir=str(self.teacher_checkpoint.parent), **kwargs)
        # 教师网络参数全部冻结，不参与梯度更新（只作为标签来源）
        self.teacher_actor_critic.requires_grad_(False)

    def act(self, obs, critic_obs):
        """执行一步动作。

        关键点：在纯 DAgger 模式（using_ppo=False）下，学生不采样动作，
        而是直接执行 actor 输出的均值 action_mean。原因：
          - 纯 MSE 蒸馏只训练 actor 的均值，init_noise_std=1.0 从未被训练；
          - 如果让学生采样，动作会带 ~1 rad 的随机噪声，破坏 rollout 质量。

        教师接管的环境仍执行教师动作（由父类 act() 决定），学生接管的环境
        用 action_mean 覆盖掉采样结果。
        """
        # 纯 MSE 蒸馏不训练 std，因此不能执行带初始 1 rad 噪声的 Student 采样。
        # 教师选择仍按 episode 保留；Student 环境执行均值。PPO 路径另行处理。

        # 每个 episode 开始时（第一次调用）一次性采样"哪些环境由教师接管"的 mask，
        # 并在整个 episode 内保持不变，避免同一环境在教师/学生之间来回跳变。
        if not self.using_ppo and not hasattr(self, 'use_teacher_act_mask'):
            self.use_teacher_act_mask = (
                torch.rand(obs.shape[0], device=self.device)
                < self.teacher_act_prob(self.current_learning_iteration)
            )

        # 父类 act() 会：教师接管环境 -> 查教师动作；其余环境 -> student 采样动作
        actions = super().act(obs, critic_obs)

        if not self.using_ppo:
            # 学生接管的环境：用 actor 输出的均值替换掉采样动作
            student_mask = ~self.use_teacher_act_mask
            actions[student_mask] = self.actor_critic.action_mean.detach()[student_mask]
            # 保持 transition 与实际执行动作一致；纯 DAgger 不使用 PPO 比率。
            # （PPO 会用 actions_log_prob 计算 ratio，DAgger 不需要，但为了
            #   transition 数据自洽仍然重新计算一下）
            self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(actions).detach()
        return actions

    def load_teacher_policy(self):
        """加载教师网络 + 教师侧的观测 normalizer（全部冻结）。"""
        # ---------------------------------------------------------------------
        # 1) 加载教师网络权重
        # ---------------------------------------------------------------------
        state = torch.load(self.teacher_checkpoint, map_location=self.device, weights_only=False)
        self.teacher_actor_critic.load_state_dict(state['model_state_dict'], strict=True)
        self.teacher_actor_critic.to(self.device).eval()

        # ---------------------------------------------------------------------
        # 2) 读取教师训练时的 agent.yaml，构建同款 normalizer
        # ---------------------------------------------------------------------
        # 注意：normalizer 必须与教师训练时完全一致，否则教师观测分布会错位。
        with (self.teacher_checkpoint.parent / 'params/agent.yaml').open() as stream:
            config = yaml.full_load(stream)
        normalizer_cfg = config['normalizers']['policy'].copy()
        self.teacher_policy_normalizer = modules.build_normalizer(
            input_shape=self.teacher_actor_obs_dim,
            normalizer_class_name=normalizer_cfg.pop('class_name'),
            normalizer_kwargs=normalizer_cfg,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # 3) 加载 normalizer 的统计量并冻结
        # ---------------------------------------------------------------------
        self.teacher_policy_normalizer.load_state_dict(state['policy_normalizer_state_dict'], strict=True)
        self.teacher_policy_normalizer.eval()
        self.teacher_policy_normalizer.requires_grad_(False)
        print(f'[Vault DAgger] Frozen teacher: {self.teacher_checkpoint}')

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        """初始化 rollout storage 前，严格校验教师观测与 critic 观测的对应关系。"""
        # ---------------------------------------------------------------------
        # 1) 名称 + 顺序校验：critic 的前缀必须与 teacher policy 项完全一致
        # ---------------------------------------------------------------------
        # 校验名称、顺序和形状，而不只是总维度，防止静默切错观测。
        # 例如 critic 里如果 joint_pos_ref 排到 joint_vel_ref 后面，即使总维度
        # 仍然是 448，切出来的教师输入也是错的。
        teacher_terms = list(self.teacher_policy_cfg_dict['obs_format']['policy'].items())
        critic_terms = list(obs_format['critic'].items())
        if critic_terms[:len(teacher_terms)] != teacher_terms:
            raise ValueError('Vault critic prefix does not match Teacher observation terms')

        # ---------------------------------------------------------------------
        # 2) 形状校验：teacher policy 各项 shape 乘积之和必须 == 448
        # ---------------------------------------------------------------------
        if sum(math.prod(shape) for _, shape in teacher_terms) != self.teacher_actor_obs_dim:
            raise ValueError('Teacher observation dimension must be 448')

        # ---------------------------------------------------------------------
        # 3) 动作维度校验：vault 教师必须是 29 DoF
        # ---------------------------------------------------------------------
        if num_actions != 29:
            raise ValueError('Vault Teacher requires 29 actions')

        return super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)

    def get_teacher_actions(self, obs, critic_obs=None):
        """查询教师动作标签（DAgger 的核心：用 critic 特权观测问教师）。"""
        # 教师必须看特权 critic 观测，否则无法给出准确标签
        if critic_obs is None:
            raise ValueError("VaultDaggerTPPO requires the privileged critic observation")

        # critic 观测总维度应 >= 448（实际是 610，因为后面还有 link_pos/link_rot
        # 以及 walk teacher 的 loco_* 项）。这里只要求 >= 448 即可。
        if critic_obs.shape[-1] != 610:
            raise ValueError(
                f"Vault critic observation has {critic_obs.shape[-1]} values; expected at least "
                f"{self.teacher_actor_obs_dim}"
            )

        # 关键：教师只需要 critic 观测的前 448 维（即 teacher policy 观测对应的部分）
        actions = super().get_teacher_actions(obs, critic_obs[..., : self.teacher_actor_obs_dim])

        # 防止教师输出 NaN/Inf 污染蒸馏标签
        if not torch.isfinite(actions).all():
            raise ValueError('Teacher produced non-finite action labels')
        return actions

    def compute_losses(self, minibatch):
        """在父类损失计算基础上，额外记录当前教师接管概率。"""
        losses, inter_vars, stats = super().compute_losses(minibatch)
        # 把 teacher_act_prob 作为标量 tensor 写进 stats，方便日志跟踪 DAgger 进度
        stats['teacher_probability'] = torch.as_tensor(
            self.teacher_act_prob(self.current_learning_iteration), device=self.device
        )
        return losses, inter_vars, stats