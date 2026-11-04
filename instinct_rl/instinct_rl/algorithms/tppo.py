# =============================================================================
# 文件说明：
# 该文件实现了一个带教师网络（Teacher Network）的 PPO 算法，类名为 TPPO。
# 核心思想：
# 1. 学生策略（当前 PPO 的 actor_critic）正常与环境交互；
# 2. 同时加载一个预训练好的教师策略（teacher_actor_critic）；
# 3. 在交互过程中，用教师策略生成动作标签（action_labels）；
# 4. 根据一定概率决定当前 step 是使用学生动作还是教师动作；
# 5. 训练时除了 PPO loss 外，还加入蒸馏损失（distillation loss），
#    让学生策略模仿教师策略的输出。
# 本文件仅添加详细中文注释，不修改原有代码逻辑。
# =============================================================================

""" PPO with teacher network """

import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

# instinct_rl 是项目内部强化学习库：
# modules 用于构建 actor-critic、normalizer 等模块；
# PPO 是父类算法；
# ActionLabelRollout 是带动作标签的 rollout 存储；
# unpad_trajectories 用于处理变长轨迹；
# get_subobs_size 用于计算观测子项的维度大小。
import instinct_rl.modules as modules
from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage.rollout_storage import ActionLabelRollout
from instinct_rl.utils import unpad_trajectories
from instinct_rl.utils.utils import get_subobs_size


# 根据学习迭代次数生成一个概率函数。
# 该函数通常用于控制“使用教师动作的概率”或“蒸馏损失系数”随训练逐渐衰减。
# option 可选："linear"、"exp"、"tanh"；
# iteration_scale 表示大概经过多少次迭代后概率降到 0。
# assuming learning iteration is at an assumable iteration scale
def GET_PROB_FUNC(option, iteration_scale):
    # PROB_options 字典中保存三种概率衰减函数。
    # x 表示当前学习迭代次数，返回值范围为 [0, 1]。
    PROB_options = {
        # 线性衰减：x=0 时为 1，x=iteration_scale 时为 0。
        "linear": lambda x: max(0, 1 - 1 / iteration_scale * x),
        # 指数衰减：x=0 时为 1，之后按比例指数下降。
        "exp": lambda x: max(0, (1 - 1 / iteration_scale) ** x),
        # tanh 衰减：使用 tanh 平滑地从 1 衰减到 0。
        "tanh": lambda x: max(0, 0.5 * (1 - np.tanh(1 / iteration_scale * (x - iteration_scale)))),
    }
    # 根据 option 返回对应的概率函数。
    return PROB_options[option]


# TPPO 继承自 PPO，在标准 PPO 基础上加入教师策略蒸馏。
class TPPO(PPO):
    def __init__(
        self,
        *args,
        teacher_logdir=None,  # 教师策略的日志目录，里面应包含 model_*.pt 和 params/agent.yaml
        teacher_policy_class_name="ActorCritic",  # 教师策略的类名，用于 build_actor_critic
        teacher_policy=dict(),  # 教师策略的配置字典，包含 obs_format、num_actions、num_rewards 等
        label_action_with_critic_obs=True,  # 是否使用 critic obs 作为教师策略输入；否则使用 actor obs
        teacher_act_prob="exp",  # 使用教师动作的概率，可以是数字、字符串或 callable；字符串会转为概率衰减函数
        update_times_scale=100,  # 粗略估计 update 被调用的总次数，用于概率衰减函数
        using_ppo=True,  # 是否计算标准 PPO loss；False 时跳过 PPO loss，接近 DAGGER 行为
        distillation_loss_coef=1.0,  # 蒸馏损失系数；也可以是字符串，表示使用概率衰减函数
        distill_target="real",  # 蒸馏目标/损失类型，如 real、mse_sum、l1、tanh、scaled_tanh、max_log_prob、kl
        buffer_dilation_ratio=1.0,  # rollout buffer 膨胀比例，用于 ActionLabelRollout 存储
        lr_scheduler_class_name=None,  # 学习率调度器类名，如 "StepLR"；None 表示不使用
        lr_scheduler=dict(),  # 学习率调度器参数
        hidden_state_resample_prob=0.0,  # 若大于 0，则在 minibatch 中以该概率重采样 hidden state
        action_labels_from_sample=False,  # 为 True 时教师标签来自 policy.act（采样动作），否则来自 act_inference（确定性动作）
        **kwargs,
    ):
        """
        Args:
        - teacher_logdir: the log directory of the teacher policy. Must contain a model_...pt file and the
            params/agent.yaml file.
        """
        # 调用父类 PPO 的初始化，完成学生策略、优化器、storage 等基础初始化。
        super().__init__(*args, **kwargs)

        # 保存教师策略日志目录。
        self.teacher_logdir = teacher_logdir

        # 保存教师策略配置字典，后续用于构建教师网络。
        self.teacher_policy_cfg_dict = teacher_policy

        # 标记教师策略输入使用 critic obs 还是 actor obs。
        self.label_action_with_critic_obs = label_action_with_critic_obs

        # 保存教师动作使用概率的原始配置。
        self.teacher_act_prob = teacher_act_prob

        # 保存 update 次数尺度，用于概率衰减函数。
        self.update_times_scale = update_times_scale

        # 如果 teacher_act_prob 是字符串，则转换为对应的概率衰减函数。
        if isinstance(self.teacher_act_prob, str):
            self.teacher_act_prob = GET_PROB_FUNC(self.teacher_act_prob, update_times_scale)
        else:
            # 如果是固定数字或 callable，则保存到私有变量。
            # 注意：这里使用双下划线前缀，会触发 Python 名称改写。
            self.__teacher_act_prob = self.teacher_act_prob

            # 包装成只接收迭代次数 x 的 lambda，忽略 x，返回固定概率。
            self.teacher_act_prob = lambda x: self.__teacher_act_prob

        # 是否使用标准 PPO 损失。
        self.using_ppo = using_ppo

        # 保存蒸馏损失系数的原始配置。
        self.__distillation_loss_coef = distillation_loss_coef

        # 如果蒸馏损失系数是字符串，则构建对应的概率衰减函数。
        if isinstance(self.__distillation_loss_coef, str):
            self.distillation_loss_coef_func = GET_PROB_FUNC(self.__distillation_loss_coef, update_times_scale)

        # 保存蒸馏损失目标类型。
        self.distill_target = distill_target

        # 保存 buffer 膨胀比例。
        self.buffer_dilation_ratio = buffer_dilation_ratio

        # 保存学习率调度器类名和参数。
        self.lr_scheduler_class_name = lr_scheduler_class_name
        self.lr_scheduler_kwargs = lr_scheduler

        # 保存 hidden state 重采样概率。
        self.hidden_state_resample_prob = hidden_state_resample_prob

        # 保存教师动作标签是否来自采样动作。
        self.action_labels_from_sample = action_labels_from_sample

        # self.transition = ActionLabelRollout.Transition() # suppose to be built at init_storage()

        # 构建教师 actor-critic 网络。
        # 使用 teacher_policy_class_name 指定网络类，teacher_policy_cfg_dict 指定配置。
        # build_actor_critic 需要 obs_format、num_actions、num_rewards。
        self.teacher_actor_critic = modules.build_actor_critic(
            teacher_policy_class_name,
            self.teacher_policy_cfg_dict,
            self.teacher_policy_cfg_dict["obs_format"],
            self.teacher_policy_cfg_dict["num_actions"],
            self.teacher_policy_cfg_dict["num_rewards"],
        )

        # 如果提供了教师日志目录，则加载教师模型权重。
        if not self.teacher_logdir is None:
            self.load_teacher_policy()
        else:
            # 没有提供教师日志目录时给出警告：需要确保教师网络已经预训练。
            print(
                "TPPO Warning: No snapshot loaded for teacher policy. Make sure you have a pretrained teacher network"
            )

        # 如果需要学习率调度器，则根据类名动态创建。
        # initialize lr scheduler if needed
        if not self.lr_scheduler_class_name is None:
            self.lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_class_name)(
                self.optimizer,
                **self.lr_scheduler_kwargs,
            )

    # 加载教师策略模型权重，以及可选的 normalizer。
    def load_teacher_policy(self):
        # 获取教师日志目录下所有以 model_ 开头、以 .pt 结尾的模型文件。
        # acquire the latest model file in the teacher logdir
        model_files = [
            file for file in os.listdir(self.teacher_logdir) if file.endswith(".pt") and file.startswith("model_")
        ]

        # 按文件名中的数字排序，找到最新的模型文件。
        model_files.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
        model_file = model_files[-1]

        # 加载模型 state_dict 到 CPU，避免直接占用 GPU。
        state_dict = torch.load(osp.join(self.teacher_logdir, model_file), map_location="cpu")

        # 将模型权重加载到教师 actor-critic 中。
        self.teacher_actor_critic.load_state_dict(state_dict["model_state_dict"])

        # 将教师网络移动到当前设备，并设置为 eval 模式。
        self.teacher_actor_critic.to(self.device)
        self.teacher_actor_critic.eval()

        # 如果 state_dict 中包含 policy normalizer 的状态，则构建并加载 normalizer。
        # if normalizer is provided, build the normalizer for current teacher policy
        if "policy_normalizer_state_dict" in state_dict:
            # 读取教师日志目录中的 agent.yaml，获取 normalizer 配置。
            with open(osp.join(self.teacher_logdir, "params", "agent.yaml")) as f:
                env_cfg_dict = yaml.full_load(f)

            # 根据教师策略的 policy obs 维度构建 normalizer。
            self.teacher_policy_normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(self.teacher_policy_cfg_dict["obs_format"]["policy"]),
                normalizer_class_name=env_cfg_dict["normalizers"]["policy"].pop("class_name"),
                normalizer_kwargs=env_cfg_dict["normalizers"]["policy"],
            )

            # 将 normalizer 移到当前设备，加载状态并设置为 eval 模式。
            self.teacher_policy_normalizer.to(self.device)
            self.teacher_policy_normalizer.load_state_dict(state_dict["policy_normalizer_state_dict"])
            self.teacher_policy_normalizer.eval()
        else:
            # 没有 normalizer 时置为 None。
            self.teacher_policy_normalizer = None

    # 使用教师策略生成动作标签。
    # 根据 label_action_with_critic_obs 和 action_labels_from_sample 有四种组合：
    # 1. 使用 critic_obs + 采样动作 act；
    # 2. 使用 critic_obs + 确定性动作 act_inference；
    # 3. 使用 actor obs + 采样动作 act；
    # 4. 使用 actor obs + 确定性动作 act_inference。
    def get_teacher_actions(self, obs, critic_obs=None):
        # 分支 1：有 critic_obs，使用 critic_obs，且标签来自采样动作。
        if critic_obs is not None and self.label_action_with_critic_obs and self.action_labels_from_sample:
            # 如果教师有 normalizer，则先对 critic_obs 做归一化。
            if self.teacher_policy_normalizer is not None:
                critic_obs = self.teacher_policy_normalizer(critic_obs)

            # 调用教师策略的 act 采样动作，并 detach 防止梯度回传到教师网络。
            return self.teacher_actor_critic.act(critic_obs).detach()

        # 分支 2：有 critic_obs，使用 critic_obs，且标签来自确定性推理动作。
        elif critic_obs is not None and self.label_action_with_critic_obs:
            # 如果教师有 normalizer，则先对 critic_obs 做归一化。
            if self.teacher_policy_normalizer is not None:
                critic_obs = self.teacher_policy_normalizer(critic_obs)

            # 调用教师策略的 act_inference 得到确定性动作，并 detach。
            return self.teacher_actor_critic.act_inference(critic_obs).detach()

        # 分支 3：不使用 critic_obs，使用 actor obs，且标签来自采样动作。
        elif self.action_labels_from_sample:
            # 如果教师有 normalizer，则先对 obs 做归一化。
            if self.teacher_policy_normalizer is not None:
                obs = self.teacher_policy_normalizer(obs)

            # 调用教师策略的 act 采样动作，并 detach。
            return self.teacher_actor_critic.act(obs).detach()

        # 分支 4：不使用 critic_obs，使用 actor obs，且标签来自确定性推理动作。
        else:
            # 如果教师有 normalizer，则先对 obs 做归一化。
            if self.teacher_policy_normalizer is not None:
                obs = self.teacher_policy_normalizer(obs)

            # 调用教师策略的 act_inference 得到确定性动作，并 detach。
            return self.teacher_actor_critic.act_inference(obs).detach()

    # 初始化 storage。
    # 与标准 PPO 不同，这里使用 ActionLabelRollout，因为需要存储教师动作标签。
    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        # 创建 ActionLabelRollout 的 Transition 对象。
        self.transition = ActionLabelRollout.Transition()

        # 计算 policy obs 的维度。
        obs_size = get_subobs_size(obs_format["policy"])

        # 如果 obs_format 中有 critic，则计算 critic obs 的维度，否则为 None。
        critic_obs_size = get_subobs_size(obs_format.get("critic")) if "critic" in obs_format else None

        # 构建 ActionLabelRollout 存储。
        # 传入观测维度、critic 观测维度、动作维度、奖励维度、buffer 膨胀比例和设备。
        self.storage = ActionLabelRollout(
            num_envs,
            num_transitions_per_env,
            [obs_size],
            [critic_obs_size],
            [num_actions],
            num_rewards=num_rewards,
            buffer_dilation_ratio=self.buffer_dilation_ratio,
            device=self.device,
        )

    # 执行一步动作。
    # 先调用父类 PPO.act 得到学生动作，再计算教师动作标签，
    # 最后根据 use_teacher_act_mask 决定哪些环境使用教师动作。
    def act(self, obs, critic_obs):
        # 获取学生策略动作。
        # get actions
        return_ = super().act(obs, critic_obs)

        # 使用教师策略生成动作标签，并保存到 transition 中。
        self.transition.action_labels = self.get_teacher_actions(obs, critic_obs)

        # 如果还没有初始化 use_teacher_act_mask，则初始化为全 True。
        # 即默认所有环境都使用教师动作，后续在 process_env_step 中按概率重采样。
        # decide whose action to use
        if not hasattr(self, "use_teacher_act_mask"):
            self.use_teacher_act_mask = torch.ones(obs.shape[0], device=self.device, dtype=torch.bool)

        # 对于 mask 为 True 的环境，用教师动作覆盖学生动作。
        return_[self.use_teacher_act_mask] = self.transition.action_labels[self.use_teacher_act_mask]

        # 返回最终执行的动作。
        return return_

    # 处理环境 step 的返回结果。
    # 在父类逻辑之外，还需要重置教师网络的 hidden state，
    # 并为 done 的环境重新采样“是否使用教师动作”的 mask。
    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        # 调用父类 PPO 的 process_env_step。
        return_ = super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

        # 对 done 的环境重置教师 actor-critic 的 hidden state。
        self.teacher_actor_critic.reset(dones)

        # 对 done 的环境重新采样 use_teacher_act_mask。
        # 使用当前学习迭代次数对应的教师动作概率 teacher_act_prob。
        # resample teacher action mask for those dones env
        self.use_teacher_act_mask[dones.to(bool)] = torch.rand(dones.sum(), device=self.device) < self.teacher_act_prob(
            self.current_learning_iteration
        )
        return return_

    # 从数据集中收集 transition，而不是从环境交互中收集。
    # 该接口通常用于离线数据或 DAGGER 式数据收集。
    def collect_transition_from_dataset(self, transition, infos):
        """The interface to collect transition from dataset rather than env"""
        # 使用数据集中的 observation 和 privileged_observation 调用父类 act。
        super().act(transition.observation, transition.privileged_observation)

        # 直接使用数据集中的 action 作为动作标签。
        self.transition.action_labels = transition.action

        # 调用父类 process_env_step 处理 reward、done、next_obs 等。
        super().process_env_step(
            transition.reward,
            transition.done,
            infos,
            transition.next_observation,
            transition.next_privileged_observation,
        )

    # 计算 return。
    # 如果不使用 PPO loss（using_ppo=False），则跳过 return 计算。
    def compute_returns(self, last_critic_obs):
        if not self.using_ppo:
            return
        return super().compute_returns(last_critic_obs)

    # 执行一次更新。
    # 在父类 update 之后，如果存在学习率调度器，则 step 一次并更新 learning_rate 记录。
    def update(self, *args, **kwargs):
        return_ = super().update(*args, **kwargs)
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return return_

    # 计算蒸馏损失：让学生策略输出 action_mean 尽量接近教师动作标签 action_labels。
    # 支持多种蒸馏目标：
    # - real: L2 范数；
    # - mse_sum: 逐元素 MSE 后求和；
    # - l1: L1 范数；
    # - tanh: 将动作映射到 [0,1] 后做二元交叉熵；
    # - scaled_tanh: 在 tanh 交叉熵基础上乘以 L1 距离缩放；
    # - max_log_prob: 最大化教师动作在当前策略下的对数概率；
    # - kl: 暂未实现。
    def compute_distill_loss(self, action_mean, action_labels):
        """Compute distillation loss between predicted actions and teacher action labels.

        Args:
            action_mean: Predicted actions from the current policy
            action_labels: Target actions from the teacher policy

        Returns:
            dist_loss: Computed distillation loss
        """
        # 目标类型 real：使用 L2 范数（欧氏距离）。
        if self.distill_target == "real":
            dist_loss = torch.norm(action_mean - action_labels, dim=-1)

        # 目标类型 mse_sum：逐元素 MSE，然后在动作维度上求和。
        elif self.distill_target == "mse_sum":
            dist_loss = F.mse_loss(
                action_mean,
                action_labels,
                reduction="none",
            ).sum(-1)

        # 目标类型 l1：使用 L1 范数。
        elif self.distill_target == "l1":
            dist_loss = torch.norm(
                action_mean - action_labels,
                dim=-1,
                p=1,
            )

        # 目标类型 tanh：将动作从 [-1,1] 映射到 [0,1]，做二元交叉熵，再乘 2。
        elif self.distill_target == "tanh":
            # for tanh, similar to loss function for sigmoid, refer to https://stats.stackexchange.com/questions/12754/matching-loss-function-for-tanh-units-in-a-neural-net
            dist_loss = (
                F.binary_cross_entropy(
                    (action_mean + 1) * 0.5,
                    (action_labels + 1) * 0.5,
                )
                * 2
            )

        # 目标类型 scaled_tanh：tanh 交叉熵逐元素结果取均值，再乘以 L1 距离和维度缩放。
        elif self.distill_target == "scaled_tanh":
            l1 = torch.norm(
                action_mean - action_labels,
                dim=-1,
                p=1,
            )
            dist_loss = (
                F.binary_cross_entropy(
                    (action_mean + 1) * 0.5,
                    (action_labels + 1) * 0.5,  # (n, t, d)
                    reduction="none",
                ).mean(-1)
                * 2
                * l1
                / action_mean.shape[-1]
            )  # (n, t)

        # 目标类型 max_log_prob：最大化教师动作在当前策略下的对数概率，因此损失取负。
        elif self.distill_target == "max_log_prob":
            action_labels_log_prob = self.actor_critic.get_actions_log_prob(action_labels)
            dist_loss = -action_labels_log_prob

        # 目标类型 kl：暂未实现。
        elif self.distill_target == "kl":
            raise NotImplementedError()

        # 其他未知类型报错。
        else:
            raise ValueError(f"Unknown distill_target: {self.distill_target}")

        return dist_loss

    # 计算损失。
    # 流程：
    # 1. 可选：对 minibatch 中的 hidden state 进行随机重采样；
    # 2. 如果 using_ppo=True，则调用父类 PPO 计算标准 PPO loss；
    #    否则只做一次 actor 前向，得到 action_mean；
    # 3. 计算蒸馏损失；
    # 4. 如果使用 tanh 类蒸馏目标，记录额外统计量；
    # 5. 更新蒸馏损失系数；
    # 6. 将蒸馏损失加入 losses。
    def compute_losses(self, minibatch):
        # 如果 hidden_state_resample_prob > 0，则对 minibatch 的 hidden state 进行随机重采样。
        if self.hidden_state_resample_prob > 0.0:
            # 假设 hidden states 来自 LSTM 或 GRU，值范围在 -1 到 1 之间。
            # 取出一个 hidden state 示例，用于确定 batch 维度。
            # assuming the hidden states are from LSTM or GRU, which are always betwein -1 and 1
            hidden_state_example = (
                minibatch.hidden_states[0][0]
                if isinstance(minibatch.hidden_states[0], tuple)
                else minibatch.hidden_states[0]
            )

            # 对每个环境/batch 元素采样一个 mask，决定是否重采样其 hidden state。
            resample_mask = (
                torch.rand(hidden_state_example.shape[1], device=self.device) < self.hidden_state_resample_prob
            )

            # 如果 hidden state 是 tuple，说明可能是 LSTM（包含 h 和 c）。
            # 对每个 hidden state 从 -1 到 1 均匀重采样。
            # for each hidden state, resample from -1 to 1
            if isinstance(minibatch.hidden_states[0], tuple):
                # for LSTM not tested
                # iterate through actor and critic hidden state
                minibatch = minibatch._replace(
                    hidden_states=tuple(
                        tuple(
                            torch.where(
                                resample_mask.unsqueeze(-1).unsqueeze(-1),
                                torch.rand_like(minibatch.hidden_states[i][j], device=self.device) * 2 - 1,
                                minibatch.hidden_states[i][j],
                            )
                            for j in range(len(minibatch.hidden_states[i]))
                        )
                        for i in range(len(minibatch.hidden_states))
                    )
                )
            else:
                # 对于 GRU 等非 tuple hidden state，同样进行重采样。
                # for GRU
                # iterate through actor and critic hidden state
                minibatch = minibatch._replace(
                    hidden_states=tuple(
                        torch.where(
                            resample_mask.unsqueeze(-1),
                            torch.rand_like(minibatch.hidden_states[i], device=self.device) * 2 - 1,
                            minibatch.hidden_states[i],
                        )
                        for i in range(len(minibatch.hidden_states))
                    )
                )

        # 如果使用 PPO，则调用父类 compute_losses 计算标准 PPO 损失和统计量。
        if self.using_ppo:
            losses, inter_vars, stats = super().compute_losses(minibatch)
        else:
            # 不使用 PPO 时，只初始化空字典，并手动执行一次 actor 前向。
            losses = dict()
            inter_vars = dict()
            stats = dict()

            # 调用 actor_critic.act 进行前向，得到 action_mean 等内部变量。
            self.actor_critic.act(
                minibatch.obs,
                masks=minibatch.masks,
                hidden_states=getattr(minibatch.hidden_states, "actor", None),
            )

        # 计算蒸馏损失：当前策略动作均值与教师动作标签之间的损失。
        # distillation loss (with teacher actor)
        dist_loss = self.compute_distill_loss(self.actor_critic.action_mean, minibatch.action_labels)

        # 如果蒸馏目标包含 "tanh"，额外记录 L1 距离统计量。
        if "tanh" in self.distill_target:
            # 记录 action_mean 与 action_labels 之间的 L1 距离均值。
            stats["l1distance"] = (
                torch.norm(
                    self.actor_critic.action_mean - minibatch.action_labels,
                    dim=-1,
                    p=1,
                )
                .mean()
                .detach()
            )

            # 记录 tanh 变换后的 L1 距离均值。
            stats["l1_before_tanh"] = (
                torch.norm(torch.tan(self.actor_critic.action_mean) - torch.tan(minibatch.action_labels), dim=-1, p=1)
                .mean()
                .detach()
            )

        # 如果蒸馏损失系数是动态函数，则根据当前学习迭代次数更新系数；
        # 否则使用固定的私有系数。
        # update distillation loss coef if applicable
        self.distillation_loss_coef = (
            self.distillation_loss_coef_func(self.current_learning_iteration)
            if hasattr(self, "distillation_loss_coef_func")
            else self.__distillation_loss_coef
        )

        # 将蒸馏损失取均值后加入总损失字典。
        losses["distillation_loss"] = dist_loss.mean()

        # 返回损失、中间变量和统计信息。
        return losses, inter_vars, stats