"""Actor-only transfer and explicit critic warmup for the uploaded Instinct-RL."""
from copy import deepcopy
import math
import torch
from instinct_rl.algorithms.ppo import PPO


def freeze_policy_normalizer(runner):
    """
    冻结 policy 使用的观测归一化器。

    背景：
    - Runner 在训练模式切换时可能会调用 normalizer.train()；
    - 这里通过把 EmpiricalNormalization 的 until 设为 0，使其统计量不再更新；
    - 同时调用 eval()，确保归一化器处于推理/冻结状态。
    """
    # 取出 policy 对应的归一化器。
    norm = runner.normalizers["policy"]

    # 该辅助函数依赖 EmpiricalNormalization 的 until 字段；
    # 如果归一化器实现不同，则直接报错，避免静默产生错误行为。
    if not hasattr(norm, "until"):
        raise TypeError("This helper requires EmpiricalNormalization with 'until'.")

    # until=0 表示不再进行统计更新。
    # 即使后续 Runner.train_mode() 调用 norm.train()，归一化器仍保持冻结。
    norm.until = 0  # Still frozen after Runner.train_mode() calls norm.train().

    # 切换到 eval 模式，进一步确保归一化器处于冻结/推理状态。
    norm.eval()


def load_actor_only(runner, checkpoint_path, fixed_std=0.2):
    """
    仅从检查点加载 actor，并保留当前 runner 的 critic、优化器和迭代计数。

    使用约束：
    - 必须在“全新的 runner”上调用；
    - 不允许已有学习迭代或优化器状态；
    - 会校验 actor 参数键、形状、数值有限性；
    - 会校验 policy normalizer 状态；
    - 加载后冻结 policy normalizer，并把动作标准差固定为 fixed_std。
    """
    # 仅允许在全新 runner 上做 actor-only transfer。
    # 如果已经有学习迭代，或优化器已有状态，说明不是全新 runner。
    if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
        raise RuntimeError("Actor-only transfer requires a fresh runner.")

    # fixed_std 必须是有限正数，否则固定标准差没有意义。
    if not math.isfinite(fixed_std) or fixed_std <= 0:
        raise ValueError("fixed_std must be finite and positive.")

    # 如果算法要求的最小 std 已经大于 fixed_std，则固定 std 会与约束冲突。
    if torch.as_tensor(runner.alg.clip_min_std).max().item() > fixed_std:
        raise ValueError("clip_min_std exceeds fixed_std.")

    # 当前 runner 中的 actor-critic 模型。
    model = runner.alg.actor_critic

    # 从 CPU 加载检查点，weights_only=True 表示只加载张量/安全权重。
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # 检查点中的 actor 参数键通常带有 "actor." 前缀；
    # 这里去掉前 6 个字符，得到与 model.actor.state_dict() 对齐的键。
    actor = {k[6:]: v for k, v in source["model_state_dict"].items()
             if k.startswith("actor.")}

    # 当前模型 actor 的目标状态字典。
    target = model.actor.state_dict()

    # 参数键集合必须完全一致，否则说明网络结构不匹配。
    if set(actor) != set(target):
        raise ValueError("Actor keys differ; check network architecture.")

    # 逐个校验 actor 张量：
    # - 形状必须与目标一致；
    # - 所有数值必须有限，不能有 NaN/Inf。
    for key, value in actor.items():
        if value.shape != target[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Incompatible actor tensor: {key}")

    # 读取 policy normalizer 的状态字典。
    # 注意：先在副本上校验，确认无误后再修改真实 runner。
    norm_state = source["policy_normalizer_state_dict"]
    norm = runner.normalizers["policy"]

    # 当前 policy normalizer 也必须支持 until 字段。
    if not hasattr(norm, "until"):
        raise TypeError("Unsupported policy normalizer.")

    # 使用 deepcopy 创建一个探针副本，在副本上尝试加载并校验状态，
    # 避免校验失败时污染真实 normalizer。
    probe = deepcopy(norm)
    probe.load_state_dict(norm_state, strict=True)

    # 校验 normalizer 状态中的所有张量均为有限值。
    if any(not torch.isfinite(v).all() for v in norm_state.values()):
        raise ValueError("Non-finite policy normalizer.")

    # 校验 normalizer 的 std/var 非负。
    if (probe._std < 0).any() or (probe._var < 0).any():
        raise ValueError("Negative policy variance/std.")

    # 所有校验通过后，正式加载 actor 参数。
    model.actor.load_state_dict(actor, strict=True)

    # 正式加载 policy normalizer 状态。
    norm.load_state_dict(norm_state, strict=True)

    # 冻结 policy normalizer，防止后续训练更新其统计量。
    freeze_policy_normalizer(runner)

    # 固定动作标准差，不继承 DAgger 阶段可能存在的探索噪声。
    with torch.no_grad():
        model.std.fill_(fixed_std)  # Do not inherit DAgger exploration noise.

    # 固定 std 后不再需要梯度。
    model.std.requires_grad_(False)


class WarmStartPPO(PPO):
    """
    支持 warm start 的 PPO：
    - 单个 MLP actor/critic；
    - 固定动作 std；
    - actor 与 critic 使用不同的学习率；
    - 支持 critic-only warmup 阶段。
    """
    def __init__(self, actor_critic, critic_only=True,
                 actor_learning_rate=3e-5, critic_learning_rate=3e-4, **kwargs):
        # 该实现要求：
        # - 非 recurrent 网络；
        # - 存在独立 critic 属性；
        # 否则直接报错。
        if actor_critic.is_recurrent or not hasattr(actor_critic, "critic"):
            raise TypeError("Requires the single-critic, non-recurrent ActorCritic.")

        # 为了保留 actor/critic 分开的学习率，必须使用 fixed schedule；
        # 如果外部传入其他 schedule，则拒绝。
        if kwargs.get("schedule", "fixed") != "fixed":
            raise ValueError("Use schedule='fixed' to preserve separate LRs.")

        # 校验两个学习率均为有限正数。
        for rate in (actor_learning_rate, critic_learning_rate):
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Learning rates must be finite and positive.")

        # 强制使用 fixed schedule。
        kwargs["schedule"] = "fixed"

        # 固定模式下不启用 desired_kl 的 KL 早停逻辑。
        kwargs["desired_kl"] = None  # Base PPO has no KL early-stop in fixed mode.

        # 调用父类 PPO 初始化。
        super().__init__(actor_critic, **kwargs)

        # 保存 actor/critic 的独立学习率。
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        # 兼容 Runner 的日志字段 learning_rate。
        self.learning_rate = critic_learning_rate  # Legacy Runner log field.

        # 重新构建优化器，将 actor 和 critic 参数分组，分别使用不同学习率。
        self.optimizer = torch.optim.Adam([
            {"params": actor_critic.actor.parameters(), "lr": actor_learning_rate},
            {"params": actor_critic.critic.parameters(), "lr": critic_learning_rate},
        ])

        # 根据 critic_only 参数设置初始冻结状态。
        self._set_critic_only(critic_only)

    def _set_critic_only(self, enabled):
        """
        设置是否处于 critic-only 阶段。

        enabled=True:
        - actor 不参与梯度更新；
        - 只有 critic 参与训练；
        - actor 参数组学习率设为 0。

        enabled=False:
        - actor 和 critic 都参与训练；
        - actor 参数组恢复为 actor_learning_rate。
        """
        # 保存当前模式标记。
        self.critic_only = bool(enabled)

        # 先冻结整个 actor-critic 的所有参数。
        self.actor_critic.requires_grad_(False)

        # critic 始终需要梯度。
        self.actor_critic.critic.requires_grad_(True)

        # critic-only 模式下 actor 不需要梯度；
        # 非 critic-only 模式下 actor 需要梯度。
        self.actor_critic.actor.requires_grad_(not self.critic_only)

        # 清空所有参数的已有梯度，避免模式切换时残留梯度。
        for parameter in self.actor_critic.parameters():
            parameter.grad = None

        # 第 0 个参数组是 actor：
        # critic-only 时学习率为 0，否则恢复 actor_learning_rate。
        self.optimizer.param_groups[0]["lr"] = (
            0.0 if self.critic_only else self.actor_learning_rate)

        # 第 1 个参数组是 critic，始终使用 critic_learning_rate。
        self.optimizer.param_groups[1]["lr"] = self.critic_learning_rate

    def finish_critic_warmup(self):
        """
        结束 critic warmup，解冻 actor。

        调用时机：
        - 必须在两次 rollout 之间调用；
        - 调用前应确认任务覆盖情况和 value 指标；
        - 如果已收集 rollout 但尚未 finish/update，则不允许切换。
        """
        # 如果 storage 已存在且有已收集的 step，说明 rollout 尚未处理完；
        # 此时切换模式可能导致更新逻辑不一致。
        if self.storage is not None and self.storage.step != 0:
            raise RuntimeError("Finish/update the collected rollout before unfreezing.")

        # 解冻 actor，进入 actor+critic 联合训练。
        self._set_critic_only(False)

    def compute_losses(self, minibatch):
        """
        计算一个 minibatch 的损失。

        critic-only 阶段：
        - 只对 critic 做 MSE 回归；
        - 返回占位的 surrogate_loss，保证 Runner.log 接口兼容；
        - 记录 actor_lr=0、critic_lr 等统计量。

        非 critic-only 阶段：
        - 调用父类 PPO 的损失计算；
        - 额外计算新旧策略之间的 KL，用于监控。
        """
        # critic-only 分支：只训练 critic。
        if self.critic_only:
            # 使用 critic_obs 让 critic 预测 value。
            prediction = self.actor_critic.evaluate(minibatch.critic_obs)

            # critic 的损失为预测值与 return 的均方误差；
            # returns detach，避免梯度回传到 return 计算图。
            loss = (prediction - minibatch.returns.detach()).square().mean()

            # Runner.log 即使在还没有 episode 完成时也期望存在这些键；
            # 因此这里返回 value_loss 和零 surrogate_loss。
            return {"value_loss": loss, "surrogate_loss": loss.new_zeros(())}, {}, {
                "critic_only": loss.new_tensor(1.),
                "actor_lr": loss.new_tensor(0.),
                "critic_lr": loss.new_tensor(self.critic_learning_rate),
            }

        # 非 critic-only 分支：调用父类 PPO 的完整损失计算。
        losses, variables, stats = super().compute_losses(minibatch)

        # 额外计算新旧策略之间的对角高斯 KL，仅用于监控。
        with torch.no_grad():
            # 当前策略的动作均值与标准差。
            mu, sigma = self.actor_critic.action_mean, self.actor_critic.action_std

            # 对角高斯分布 KL 公式：
            # KL(old || new) = log(sigma_new / sigma_old)
            #   + (sigma_old^2 + (mu_old - mu_new)^2) / (2 * sigma_new^2) - 0.5
            # 对动作维度求和后，再对 batch 取平均。
            kl = (torch.log(sigma / minibatch.old_sigma)
                  + (minibatch.old_sigma.square()
                     + (minibatch.old_mu - mu).square()) / (2 * sigma.square())
                  - 0.5).sum(-1).mean()

        # 把额外统计量写入 stats：
        # - critic_only=0 表示当前不是 critic-only；
        # - policy_kl 为监控 KL；
        # - actor_lr/critic_lr 为当前学习率。
        # 使用 kl.new_tensor 保持与 kl 相同的 dtype/device。
        stats.update(critic_only=kl.new_tensor(0.), policy_kl=kl,
                     actor_lr=kl.new_tensor(self.actor_learning_rate),
                     critic_lr=kl.new_tensor(self.critic_learning_rate))

        # 返回与父类相同结构的 losses、variables 和扩展后的 stats。
        return losses, variables, stats

    def state_dict(self):
        """
        保存状态字典时，额外保存当前是否处于 critic-only 阶段。
        """
        # 先取得父类 PPO 的状态字典。
        state = super().state_dict()

        # 增加 warm start 模式标记，便于恢复时还原冻结状态。
        state["warm_start_critic_only"] = self.critic_only

        # 返回扩展后的状态字典。
        return state

    def load_state_dict(self, state_dict):
        """
        恢复 WarmStartPPO 状态。

        注意：
        - 这是恢复一个新的 PPO 训练运行；
        - 不是 DAgger 的 actor-only transfer；
        - 因此需要恢复并还原 critic_only 模式。
        """
        # 从状态字典中读取保存时的 critic_only 标记。
        # This is resume of a NEW PPO run, not DAgger transfer.
        enabled = state_dict["warm_start_critic_only"]

        # 先恢复父类 PPO 的状态。
        super().load_state_dict(state_dict)

        # 再根据保存的标记恢复 critic-only / 联合训练模式。
        self._set_critic_only(enabled)