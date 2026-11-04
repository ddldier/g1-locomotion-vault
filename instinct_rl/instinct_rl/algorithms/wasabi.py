# 导入 importlib，用于动态导入模块（支持通过字符串指定类路径）。
import importlib
# 导入 Literal 类型，用于限定字符串枚举值。
from typing import Literal

# 导入 PyTorch 主库。
import torch
# 导入 PyTorch 分布式训练模块。
import torch.distributed as dist
# 导入 PyTorch 神经网络模块。
import torch.nn as nn
# 导入 PyTorch 优化器模块。
import torch.optim as optim

# 导入 instinct_rl.modules 并重命名为 instinct_modules，用于动态获取判别器类。
import instinct_rl.modules as instinct_modules
# 从 instinct_rl.algorithms.ppo 导入 PPO 算法基类。
from instinct_rl.algorithms.ppo import PPO
# 从 instinct_rl.storage.amp_storage 导入 AmpStorage，用于存储 AMP 相关的转移数据。
from instinct_rl.storage.amp_storage import AmpStorage
# 从 instinct_rl.utils.buffer 导入 buffer_func，用于对缓冲区进行操作。
from instinct_rl.utils.buffer import buffer_func
# 从 instinct_rl.utils.utils 导入 get_subobs_size，用于计算观测子段的总维度。
from instinct_rl.utils.utils import get_subobs_size


# WasabiAlgoMixin：WASABI 算法的插件混入类，同时也包含传统 AMP 算法。
# 通过设置 discriminator_loss_func 为 "BCEWithLogitsLoss" 可退化为 AMP。
class WasabiAlgoMixin:
    """A plugin algorithm for [WASABI](https://github.com/martius-lab/wasabi), which also includes traditional AMP algorithm
    (Set `discriminator_loss_func` to "BCEWithLogitsLoss" for AMP).
    """

    # 初始化混入类，接受大量与判别器相关的配置参数。
    def __init__(
        self,
        *args,
        actor_state_key="amp_policy",  # 从观测字典中获取策略状态序列的键。
        reference_state_key="amp_reference",  # 从观测字典中获取专家参考序列的键。
        discriminator_class_name="Discriminator",  # 判别器类名，默认 "Discriminator"。
        discriminator_kwargs={},  # 传递给判别器构造函数的额外关键字参数。
        discriminator_optimizer_class_name="AdamW",  # 判别器优化器类名，默认 "AdamW"。
        discriminator_optimizer_kwargs={},  # 传递给优化器的额外关键字参数。
        discriminator_reward_coef=1.0,  # 判别器奖励的系数。
        discriminator_reward_type: Literal[
            "log", "quad", "wasserstein"
        ] = "log",  # 判别器奖励类型，详见 compute_discriminator_reward。
        discriminator_loss_func: Literal[
            "WassersteinLoss", "BCEWithLogitsLoss", "MSELoss"
        ] = "BCEWithLogitsLoss",  # 判别器损失函数类型，默认 BCEWithLogitsLoss（即 AMP）。
        discriminator_loss_coef=1.0,  # 判别器损失的系数。
        discriminator_gradient_penalty_coef=10.0,  # 梯度惩罚系数。
        discriminator_weight_decay_coef=0.0,  # 权重衰减系数。
        discriminator_logit_weight_decay_coef=0.0,  # 最后一层权重的衰减系数。
        discriminator_gradient_torlerance=0.0,  # 梯度容忍度：若梯度小于此值则不惩罚（注意拼写 torlerance 为原代码保留）。
        discriminator_backbone_gradient_only=False,  # 若为 True，则只对判别器骨干网络计算梯度惩罚（要求判别器支持 encoders 和 backbone_run）。
        **kwargs,  # 其他传递给父类的关键字参数。
    ):
        # 调用父类（通常是 PPO）的初始化方法。
        super().__init__(*args, **kwargs)
        # 保存所有与判别器相关的配置到实例属性。
        self.actor_state_key = actor_state_key
        self.reference_state_key = reference_state_key
        self.discriminator_class_name = discriminator_class_name
        self.discriminator_kwargs = discriminator_kwargs
        self.discriminator_optimizer_class_name = discriminator_optimizer_class_name
        self.discriminator_optimizer_kwargs = discriminator_optimizer_kwargs
        self.discriminator_reward_coef = discriminator_reward_coef
        self.discriminator_reward_type = discriminator_reward_type
        self.discriminator_loss_func = discriminator_loss_func
        self.discriminator_loss_coef = discriminator_loss_coef
        self.discriminator_gradient_penalty_coef = discriminator_gradient_penalty_coef
        self.discriminator_weight_decay_coef = discriminator_weight_decay_coef
        self.discriminator_logit_weight_decay_coef = discriminator_logit_weight_decay_coef
        self.discriminator_gradient_torlerance = discriminator_gradient_torlerance
        self.discriminator_backbone_gradient_only = discriminator_backbone_gradient_only

    # 初始化存储：先调用父类初始化，然后构建判别器、优化器和 AMP 存储。
    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        # 调用父类的 init_storage。
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        # 构建判别器：根据类名动态导入。
        if ":" in self.discriminator_class_name:
            # 如果类名包含冒号，格式为 "模块:类名"，动态导入模块并获取类。
            DiscriminatorClass = getattr(
                importlib.import_module(self.discriminator_class_name.split(":")[0]),
                self.discriminator_class_name.split(":")[1],
            )
        else:
            # 否则从 instinct_modules 中按名称获取类。
            DiscriminatorClass = getattr(instinct_modules, self.discriminator_class_name)
        # 实例化判别器，传入 actor_state_key 对应的观测分段作为输入。
        self.discriminator = DiscriminatorClass(
            input_segment=obs_format[self.actor_state_key],
            **self.discriminator_kwargs,
        ).to(self.device)
        # 如果优化器参数中没有学习率，则使用父类的 learning_rate。
        if not "lr" in self.discriminator_optimizer_kwargs:
            self.discriminator_optimizer_kwargs = dict(lr=self.learning_rate, **self.discriminator_optimizer_kwargs)
        # 创建判别器优化器。
        self.discriminator_optimizer = getattr(optim, self.discriminator_optimizer_class_name)(
            self.discriminator.parameters(),
            **self.discriminator_optimizer_kwargs,
        )

        # 创建一个 AMP 转移对象（用于逐步填充数据）。
        self.amp_transition = AmpStorage.Transition()
        # 计算参考状态和 actor 状态的维度。
        reference_state_size = get_subobs_size(obs_format[self.actor_state_key])
        actor_state_size = get_subobs_size(obs_format[self.reference_state_key])
        # 断言两者维度相同。
        assert actor_state_size == reference_state_size, "The shape of robot state and reference state must be the same"
        # 创建 AMP 存储，用于存储 actor 状态和参考状态序列。
        self.amp_storage = AmpStorage(
            num_envs, num_transitions_per_env, [actor_state_size], [reference_state_size], device=self.device
        )

    # 处理环境步：从 infos 中提取 actor 状态和参考状态，存入 AMP 存储，然后调用父类处理。
    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        # 检查 infos["observations"] 中是否包含所需的键。
        if not (self.actor_state_key in infos["observations"] and self.reference_state_key in infos["observations"]):
            raise ValueError(
                "The key of trajectory observations ({}) or reference observations ({}) is not found in the observation"
                " dictionary".format(self.actor_state_key, self.reference_state_key)
            )

        # 提取 actor 状态和参考状态。
        actor_state = infos["observations"][self.actor_state_key]
        reference_state = infos["observations"][self.reference_state_key]
        # 填充 AMP 转移对象。
        self.amp_transition.actor_states = actor_state
        self.amp_transition.reference_states = reference_state
        # 如果判别器是循环网络，则保存隐藏状态。
        if self.discriminator.is_recurrent:
            self.amp_transition.hidden_states = self.discriminator.get_hidden_states()
        self.amp_transition.dones = dones
        # 将转移添加到 AMP 存储。
        self.amp_storage.add_transitions(self.amp_transition)
        # 清空转移对象，准备下一次使用。
        self.amp_transition.clear()

        # 注意：这里不调用 compute_auxiliary_reward，因为它会在父类函数中被调用。
        super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

    # 计算辅助奖励：在父类辅助奖励基础上增加判别器奖励。
    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 先调用父类的辅助奖励计算。
        super_auxilary_reward = super().compute_auxiliary_reward(obs_pack)
        # 获取 actor 状态。
        actor_state = obs_pack[self.actor_state_key]
        # 通过判别器计算判别值，并断开梯度。
        disc = self.discriminator(actor_state).detach()
        # 根据配置的奖励类型计算奖励。
        if self.discriminator_reward_type == "log":
            # 默认 AMP 实现：奖励 = -log(1 - sigmoid(disc))，并做数值裁剪。
            reward = -torch.log(1 - torch.clamp(torch.sigmoid(disc), 1e-6, 1 - 1e-6))
        elif self.discriminator_reward_type == "quad":
            # 来自 WASABI 的二次型奖励（假设 disc 是线性层输出）。
            reward = torch.clamp(1 - (1 / 4) * torch.square(disc - 1), min=0)
        elif self.discriminator_reward_type == "wasserstein":
            # 来自 WASABI 的 Wasserstein 奖励（假设 disc 是线性层输出）。
            reward = disc
        # 将判别器奖励添加到辅助奖励字典。
        super_auxilary_reward["discriminator_reward"] = reward
        return super_auxilary_reward

    # 更新：先调用父类 update 更新策略，然后训练判别器。
    def update(self, *args, **kwargs):
        # 调用父类 PPO 的 update，得到平均损失和统计信息。
        mean_losses, average_stats = super().update(*args, **kwargs)

        # 遍历判别器的优化步骤。
        if self.discriminator.is_recurrent:
            # 循环判别器使用循环小批量生成器（设计尚不完善）。
            amp_generator = self.amp_storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            # 非循环判别器使用普通小批量生成器。
            amp_generator = self.amp_storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # 总更新次数。
        num_updates = self.num_learning_epochs * self.num_mini_batches
        # 遍历每个 AMP 小批量。
        for amp_minibatch in amp_generator:
            # 计算 AMP 损失。
            losses, _, stats = self.compute_amp_losses(amp_minibatch)

            # 组合损失：各项损失乘以对应系数。
            loss = 0.0
            for k, v in losses.items():
                loss += getattr(self, k + "_coef", 1.0) * v
                # 累加平均损失（用于日志）。
                mean_losses[k] = mean_losses[k] + v.detach() / num_updates
            # 累加总 AMP 损失。
            mean_losses["amp_total_loss"] = mean_losses["amp_total_loss"] + loss.detach() / num_updates
            # 累加统计量。
            for k, v in stats.items():
                average_stats[k] = average_stats[k] + v.detach() / num_updates

            # 执行梯度更新。
            self.wasabi_gradient_step(loss, average_stats)

        # 清空 AMP 存储。
        self.amp_storage.clear()

        return mean_losses, average_stats

    # 计算 AMP 损失：包括判别器损失、梯度惩罚、权重衰减等。
    def compute_amp_losses(self, amp_minibatch: AmpStorage.MiniBatch):
        # 初始化损失、统计和中间变量字典。
        losses, stats, inter_vars = dict(), dict(), dict()

        # 判别器必须在前向 act() 中计算，这里再次计算。
        # TODO: 运行循环判别器。
        actor_d = self.discriminator(
            amp_minibatch.actor_states,
            masks=amp_minibatch.masks,
        )
        reference_d = self.discriminator(
            amp_minibatch.reference_states,
            masks=amp_minibatch.masks,
        )

        # 根据配置的损失函数计算判别器损失。
        if self.discriminator_loss_func == "WassersteinLoss":
            actor_d_loss = actor_d.mean()
            reference_d_loss = -reference_d.mean()
        elif self.discriminator_loss_func == "BCEWithLogitsLoss":
            # 默认 AMP：actor 状态标签为 0，参考状态标签为 1。
            actor_d_loss = torch.nn.BCEWithLogitsLoss()(actor_d, torch.zeros_like(actor_d))
            reference_d_loss = torch.nn.BCEWithLogitsLoss()(reference_d, torch.ones_like(reference_d))
        elif self.discriminator_loss_func == "MSELoss":
            # 均方误差：actor 目标为 -1，参考目标为 1。
            actor_d_loss = torch.nn.MSELoss()(actor_d, -torch.ones_like(actor_d))
            reference_d_loss = torch.nn.MSELoss()(reference_d, torch.ones_like(reference_d))
        else:
            raise NotImplementedError(f"discriminator loss function {self.discriminator_loss_func} not implemented")
        # 总判别器损失为两者平均。
        discriminator_loss = (actor_d_loss + reference_d_loss) * 0.5
        # 计算梯度惩罚（如果系数大于 0）。
        if self.discriminator_gradient_penalty_coef > 0:
            if self.discriminator_backbone_gradient_only:
                # 只对骨干网络计算梯度惩罚。
                discriminator_gradient_penalty = self.compute_discriminator_backbone_gradient(amp_minibatch)
            else:
                # 对判别器输入计算梯度惩罚。
                discriminator_gradient_penalty = self.compute_discriminator_gradient(amp_minibatch)
        else:
            discriminator_gradient_penalty = torch.zeros(1, device=self.device)[0]

        # 计算权重衰减损失（所有参数的平方和）。
        weight_decay_loss = torch.zeros(1, device=self.device)[0]
        for param in self.discriminator.parameters():
            weight_decay_loss += torch.sum(param**2)

        # 参考 ProtoMotions，计算判别器最后一层权重的衰减损失。
        logit_weight_decay_loss = torch.zeros(1, device=self.device)[0]
        if self.discriminator_logit_weight_decay_coef > 0:
            logit_weight = self.discriminator.logit_layer_weights()
            logit_weight_decay_loss = torch.sum(logit_weight**2)

        # 将各项损失和统计量填入字典。
        losses["discriminator_loss"] = discriminator_loss
        losses["discriminator_gradient_penalty"] = discriminator_gradient_penalty
        losses["discriminator_weight_decay"] = weight_decay_loss
        losses["logit_weight_decay"] = logit_weight_decay_loss
        stats["discriminator_actor"] = actor_d.mean()
        stats["discriminator_reference"] = reference_d.mean()

        return losses, inter_vars, stats

    # 计算判别器骨干网络的梯度惩罚（WASABI 中用于 Lipschitz 连续性）。
    def compute_discriminator_backbone_gradient(self, amp_minibatch: AmpStorage.MiniBatch):
        """Compute the gradient w.r.t discriminator input.
        In WASABI algorithm, it is used as a penalty to satisfy the condition of Lipschitz continuity
        """
        # 克隆参考状态和 actor 状态。
        reference_states = amp_minibatch.reference_states
        reference_states = buffer_func(reference_states, torch.clone)

        actor_states = amp_minibatch.actor_states
        actor_states = buffer_func(actor_states, torch.clone)

        # 拼接状态。
        combined_states = torch.cat([actor_states, reference_states], dim=0)

        # 注意：假设判别器有 encoders 和 backbone_run 函数。
        latent = self.discriminator.encoders(combined_states).detach()
        latent.requires_grad = True

        # 通过骨干网络得到判别输出。
        disc = self.discriminator.backbone_run(latent)

        # 计算梯度。
        ones = torch.ones_like(disc)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=latent,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 返回梯度范数减去容忍度后的平方均值（裁剪到非负）。
        return torch.clamp(grad.norm(2, dim=1) - self.discriminator_gradient_torlerance, 0).pow(2).mean()

    # 计算判别器输入的梯度惩罚。
    def compute_discriminator_gradient(self, amp_minibatch: AmpStorage.MiniBatch):
        """Compute the gradient w.r.t expert input.
        In WASABI algorithm, it is used as a penalty to satisfy the condition of Lipschitz continuity
        """
        # 克隆参考状态并设置 requires_grad。
        reference_states = amp_minibatch.reference_states
        reference_states = buffer_func(reference_states, torch.clone)
        buffer_func(reference_states, setattr, "requires_grad", True)

        # 克隆 actor 状态并设置 requires_grad。
        actor_states = amp_minibatch.actor_states
        actor_states = buffer_func(actor_states, torch.clone)
        buffer_func(actor_states, setattr, "requires_grad", True)

        # 拼接状态。
        combined_states = torch.cat([actor_states, reference_states], dim=0)

        # 通过判别器得到输出。
        disc = self.discriminator(
            combined_states,
            masks=amp_minibatch.masks,  # 掩码设计为判别器可能是循环的，但通常不是。
        )

        # 计算梯度。
        ones = torch.ones_like(disc)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=combined_states,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 返回梯度范数减去容忍度后的平方均值（裁剪到非负）。
        return torch.clamp(grad.norm(2, dim=1) - self.discriminator_gradient_torlerance, 0).pow(2).mean()

    # 返回状态字典，包含父类状态、判别器状态和优化器状态。
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["discriminator"] = self.discriminator.state_dict()
        state_dict["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
        return state_dict

    # 加载状态字典，恢复父类、判别器和优化器的状态。
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "discriminator" in state_dict:
            self.discriminator.load_state_dict(state_dict["discriminator"])
        else:
            print("[Warning] The discriminator state_dict is not found in the checkpoint")
        if "discriminator_optimizer" in state_dict:
            self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])
        else:
            print("[Warning] The discriminator_optimizer state_dict is not found in the checkpoint")

    # 执行判别器的梯度更新步骤（包括分布式梯度平均）。
    def wasabi_gradient_step(self, loss: torch.Tensor, average_stats: dict):
        # 清零梯度。
        self.discriminator_optimizer.zero_grad()
        # 反向传播。
        loss.backward()
        # 如果分布式已初始化，则对所有参数的梯度进行 all_reduce 并取平均。
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for param in self.discriminator.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
        # TODO: 添加梯度裁剪。
        # 更新参数。
        self.discriminator_optimizer.step()


# WasabiPPO：组合 WasabiAlgoMixin 和 PPO 算法。
class WasabiPPO(WasabiAlgoMixin, PPO):
    pass