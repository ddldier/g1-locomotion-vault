# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# 从 collections 导入 namedtuple，用于定义轻量级的数据结构 MiniBatch。
from collections import namedtuple

# 导入 PyTorch。
import torch

# 从 instinct_rl.utils 导入 split_and_pad_trajectories，用于处理 RNN 轨迹的拆分与填充。
from instinct_rl.utils import split_and_pad_trajectories
# 从 instinct_rl.utils.buffer 导入若干缓冲区操作工具：
# - buffer_expand：扩展缓冲区；
# - buffer_from_example：根据样例创建缓冲区；
# - buffer_method：对缓冲区调用指定方法；
# - buffer_swap：交换缓冲区内容。
from instinct_rl.utils.buffer import (
    buffer_expand,
    buffer_from_example,
    buffer_method,
    buffer_swap,
)
# 从 instinct_rl.utils.collections 导入 is_namedarraytuple，用于判断是否为命名数组元组。
from instinct_rl.utils.collections import is_namedarraytuple


# RolloutStorage 类：用于存储 on-policy 算法（如 PPO）在 rollout 阶段收集的转移数据。
class RolloutStorage:
    # Transition 内部类：表示单步转移数据，包含观测、动作、奖励、值等字段。
    class Transition:
        # 初始化所有字段为 None。
        def __init__(self):
            self.observations = None          # 策略观测
            self.critic_observations = None   # 评论家观测（特权观测）
            self.actions = None               # 动作
            self.rewards = None               # 奖励
            self.dones = None                 # 终止标志
            self.values = None                # 状态值
            self.actions_log_prob = None      # 动作对数概率
            self.action_mean = None           # 动作分布均值
            self.action_sigma = None          # 动作分布标准差
            self.hidden_states = None         # RNN 隐藏状态

        # 清空转移数据：重新调用 __init__ 将所有字段置为 None。
        def clear(self):
            self.__init__()

    # MiniBatch 命名元组：定义从存储中采样出的小批量数据的字段。
    MiniBatch = namedtuple(
        "MiniBatch",
        [
            "obs",                     # 策略观测
            "critic_obs",              # 评论家观测
            "actions",                 # 动作
            "values",                  # 值
            "advantages",              # 优势
            "returns",                 # 回报
            "old_actions_log_prob",    # 旧动作对数概率
            "old_mu",                  # 旧动作均值
            "old_sigma",               # 旧动作标准差
            "hidden_states",           # 隐藏状态
            "masks",                   # 掩码（用于 RNN）
        ],
    )

    # 初始化 RolloutStorage。
    def __init__(
        self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape, num_rewards=1, device="cpu"
    ):
        # 保存设备。
        self.device = device

        # 保存观测形状、评论家观测形状、动作形状。
        self.obs_shape = obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape
        # 奖励项数量。
        self.num_rewards = num_rewards

        # Core
        # 核心缓冲区：观测，形状 [T, N, *obs_shape]。
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        # 如果评论家观测形状的第一个维度不为 None，则创建评论家观测缓冲区。
        if critic_obs_shape[0] is not None:
            self.critic_observations = torch.zeros(
                num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device
            )
        else:
            self.critic_observations = None
        # 奖励缓冲区，形状 [T, N, num_rewards]。
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)
        # 动作缓冲区，形状 [T, N, *actions_shape]。
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # 终止标志缓冲区，形状 [T, N, 1]，使用 byte 类型。
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        # 动作对数概率缓冲区，形状 [T, N, 1]。
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # 状态值缓冲区，形状 [T, N, num_rewards]。
        self.values = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)
        # 回报缓冲区，形状 [T, N, num_rewards]。
        self.returns = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)
        # 优势缓冲区，形状 [T, N, num_rewards]。
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, num_rewards, device=self.device)
        # 动作分布均值缓冲区，形状 [T, N, *actions_shape]。
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        # 动作分布标准差缓冲区，形状 [T, N, *actions_shape]。
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # 每个环境的转移步数。
        self.num_transitions_per_env = num_transitions_per_env
        # 环境数量。
        self.num_envs = num_envs

        # rnn
        # 保存的隐藏状态，初始为 None。
        self.saved_hidden_states = None

        # 当前写入位置（步索引）。
        self.step = 0

    # 向存储中添加一条转移。
    def add_transitions(self, transition: Transition):
        # 如果当前步索引超出容量，则抛出异常。
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        # 将转移中的各字段复制到对应缓冲区当前步的位置。
        self.observations[self.step].copy_(transition.observations)
        if self.critic_observations is not None:
            self.critic_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        # 奖励展平为 [N, num_rewards]。
        self.rewards[self.step].copy_(transition.rewards.view(-1, self.num_rewards))
        # 终止标志展平为 [N, 1]。
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        # 动作对数概率展平为 [N, 1]。
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        # 保存隐藏状态（如果有）。
        self._save_hidden_states(transition.hidden_states)
        # 步索引加一。
        self.step += 1

    # 保存隐藏状态到缓冲区。
    def _save_hidden_states(self, hidden_states):
        """Assuming hidden_states is a torch tensor or a namedarraytuple of torch tensor"""
        # 如果隐藏状态为 None，直接返回。
        if hidden_states is None:
            return
        # 如果是命名数组元组，尝试获取前导维度。
        if is_namedarraytuple(hidden_states):
            try:
                leading_dims = hidden_states.get_leading_dims()
            except AttributeError as e:
                # 如果异常信息中包含 "None"，则直接返回。
                if "None" in str(e):
                    return

        # initialize if needed
        # 如果尚未初始化保存隐藏状态的缓冲区，则根据样例创建。
        if self.saved_hidden_states is None:
            self.saved_hidden_states = buffer_from_example(hidden_states, self.observations.shape[0])
        # copy the states
        # 将当前隐藏状态复制到缓冲区的当前步。
        self.saved_hidden_states[self.step] = hidden_states

    # 清空存储：将步索引重置为 0。
    def clear(self):
        self.step = 0

    # 计算回报和优势（GAE）。
    def compute_returns(self, last_values, gamma, lam):
        # 初始化优势为 0。
        advantage = 0
        # 从最后一步向前反向遍历。
        for step in reversed(range(self.num_transitions_per_env)):
            # 如果是最后一步，下一个值使用传入的 last_values；否则使用下一步的值。
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 下一个状态是否不是终止状态（1 表示非终止，0 表示终止）。
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD 误差 delta。
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # 优势递推。
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # 回报 = 优势 + 值。
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        # 计算优势 = 回报 - 值。
        self.advantages = self.returns - self.values
        # 对优势进行归一化。
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # 获取统计信息：平均轨迹长度和平均奖励。
    def get_statistics(self):
        # 获取终止标志。
        done = self.dones
        # 将最后一步的终止标志设为 1，以确保能计算最后一段轨迹。
        done[-1] = 1
        # 展平终止标志。
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        # 找到所有 done 的位置，并在开头添加 -1 作为起始。
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        # 轨迹长度 = 相邻 done 索引之差。
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        # 返回平均轨迹长度和平均奖励。
        return trajectory_lengths.float().mean(), self.rewards.mean()

    # 生成小批量数据（非 RNN 情况）。
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # 总批量大小 = 环境数 * 每环境步数。
        batch_size = self.num_envs * self.num_transitions_per_env
        # 每个小批量的样本数。
        mini_batch_size = batch_size // num_mini_batches
        # 随机排列索引。
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        # 时间索引 = 索引整除环境数。
        T_indices = (indices // self.num_envs).to(torch.long)
        # 环境索引 = 索引对模型环境数取模。
        B_indices = (indices % self.num_envs).to(torch.long)

        # 遍历每个 epoch。
        for epoch in range(num_epochs):
            # 遍历每个小批量。
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                T_idx = T_indices[start:end]
                B_idx = B_indices[start:end]

                # 生成并返回一个小批量。
                yield self.get_minibatch_from_selection(T_idx, B_idx)

    # for RNNs only
    # 生成小批量数据（RNN 情况）。
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # 将观测按轨迹拆分并填充。
        self._padded_obs_trajectories, self._trajectory_masks = split_and_pad_trajectories(
            self.observations, self.dones
        )
        # 如果有评论家观测，同样处理。
        if self.critic_observations is not None:
            self._padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.critic_observations, self.dones)

        # 每个小批量的环境数。
        mini_batch_size = self.num_envs // num_mini_batches
        # 遍历每个 epoch。
        for ep in range(num_epochs):
            first_traj = 0
            # 遍历每个小批量。
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                # 获取 done 标志并去掉最后一维。
                dones = self.dones.squeeze(-1)
                # 创建 last_was_done 掩码，表示每个时间步的前一步是否 done。
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                # 计算当前小批量中包含的轨迹数量。
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                # 生成并返回一个小批量。
                yield self.get_minibatch_from_selection(
                    slice(None),
                    slice(start, stop),
                    padded_B_slice=slice(first_traj, last_traj),
                    prev_done_mask=last_was_done,
                )

                first_traj = last_traj

    # 根据选择的索引/切片提取小批量数据。
    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        """Extract minibatch based on selected indices/slice.
        An independent method allows override by subclasses.
        Args:
            - padded_B_slice: For recurrent trajectories, the observations are already expanded and padded with zeros.
            - prev_done_mask: For recurrent trajectories,
        Outputs:
            - MiniBatch:
                only batch dimension if not padded_B_slice (non-recurrent case)
                with time, batch dimension if padded_B_slice (recurrent case)
        """
        # 如果是非 RNN 情况（padded_B_slice 为 None）。
        if padded_B_slice is None:
            # 直接按时间索引和环境索引选择观测。
            obs_batch = self.observations[T_select, B_select]
            # 评论家观测：如果没有单独的评论家观测，则使用策略观测。
            critic_obs_batch = (
                obs_batch if self.critic_observations is None else self.critic_observations[T_select, B_select]
            )
            hid_batch = None
            obs_mask_batch = None
        else:
            # RNN 情况：从填充后的轨迹中选择。
            obs_batch = self._padded_obs_trajectories[T_select, padded_B_slice]
            critic_obs_batch = (
                obs_batch
                if self.critic_observations is None
                else self._padded_critic_obs_trajectories[T_select, padded_B_slice]
            )

            # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
            # then take only time steps after dones (flattens num envs and time dimensions),
            # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
            # 将 prev_done_mask 从 (T, B) 转置为 (B, T)。
            prev_done_mask = prev_done_mask.permute(1, 0)  # (T, B) -> (B, T)
            # 对隐藏状态进行重排和选择，以匹配轨迹批次。
            hid_batch = buffer_method(
                buffer_method(
                    buffer_method(self.saved_hidden_states, "permute", 2, 0, 1, 3)[prev_done_mask][padded_B_slice],
                    "transpose",
                    1,
                    0,
                ),
                "contiguous",
            )
            # 获取观测掩码。
            obs_mask_batch = self._trajectory_masks[T_select, padded_B_slice]

        # 选择动作、值、回报、旧对数概率、优势、旧均值、旧标准差。
        action_batch = self.actions[T_select, B_select]
        target_value_batch = self.values[T_select, B_select]
        return_batch = self.returns[T_select, B_select]
        old_action_log_prob_batch = self.actions_log_prob[T_select, B_select]
        advantage_batch = self.advantages[T_select, B_select]
        old_mu_batch = self.mu[T_select, B_select]
        old_sigma_batch = self.sigma[T_select, B_select]

        # 返回 MiniBatch 命名元组。
        return RolloutStorage.MiniBatch(
            obs_batch,
            critic_obs_batch,
            action_batch,
            target_value_batch,
            advantage_batch,
            return_batch,
            old_action_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_batch,
            obs_mask_batch,
        )


# QueueRolloutStorage 类：继承 RolloutStorage，允许缓冲区比 rollout 长度更大，实现类似队列的循环存储。
class QueueRolloutStorage(RolloutStorage):
    # 初始化。
    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        *args,
        buffer_dilation_ratio=1.0,
        **kwargs,
    ):
        """This rollout storage allows the buffer to be larger than the rollout length.
        NOTE: num_transitions_per_env is no longer a constant representing the buffer temporal length.

        Args:
            size_dilation_ratio: float, for the size of buffer bigger than num_transitions_per_env
        """
        # 每次 rollout 的时间步数。
        self.num_timesteps_each_rollout = num_transitions_per_env
        # 缓冲区膨胀比例。
        self.buffer_dilation_ratio = buffer_dilation_ratio
        # 缓冲区是否已满。
        self.buffer_full = False
        # 调用父类初始化。
        super().__init__(
            num_envs,
            num_transitions_per_env,
            *args,
            **kwargs,
        )

    # 扩展一次缓冲区大小。
    @torch.no_grad()
    def expand_buffer_once(self):
        """Expand the buffer size in this way so that the mini_batch_generator will not output
        the buffer where no data has been stored
        """
        # 计算扩展大小：膨胀比例 * 每次 rollout 步数 - 当前步数，再与每次 rollout 步数取小。
        expand_size = int(self.buffer_dilation_ratio * self.num_timesteps_each_rollout - self.num_transitions_per_env)
        expand_size = min(expand_size, self.num_timesteps_each_rollout)
        # 更新总时间步数。
        self.num_transitions_per_env += expand_size

        # expand the buffer by concatenating
        # Core
        # 扩展观测缓冲区。
        self.observations = torch.cat(
            [
                self.observations,
                torch.zeros(expand_size, self.num_envs, *self.obs_shape, device=self.device),
            ],
            dim=0,
        ).contiguous()
        # 扩展评论家观测缓冲区（如果有）。
        if self.critic_obs_shape[0] is not None:
            self.critic_observations = torch.cat(
                [
                    self.critic_observations,
                    torch.zeros(expand_size, self.num_envs, *self.critic_obs_shape, device=self.device),
                ],
                dim=0,
            ).contiguous()
        # 扩展奖励缓冲区。
        self.rewards = torch.cat(
            [
                self.rewards,
                torch.zeros(expand_size, self.num_envs, 1, device=self.device),
            ],
            dim=0,
        ).contiguous()
        # 扩展动作缓冲区。
        self.actions = torch.cat(
            [
                self.actions,
                torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
            ],
            dim=0,
        ).contiguous()
        # 扩展 done 缓冲区。
        self.dones = torch.cat(
            [
                self.dones,
                torch.zeros(expand_size, self.num_envs, 1, device=self.device).byte(),
            ],
            dim=0,
        ).contiguous()

        # For PPO
        # 扩展动作对数概率缓冲区。
        self.actions_log_prob = torch.cat(
            [
                self.actions_log_prob,
                torch.zeros(expand_size, self.num_envs, 1, device=self.device),
            ],
            dim=0,
        ).contiguous()
        # 扩展值缓冲区。
        self.values = torch.cat(
            [
                self.values,
                torch.zeros(expand_size, self.num_envs, 1, device=self.device),
            ],
            dim=0,
        ).contiguous()
        # 扩展回报缓冲区。
        self.returns = torch.cat(
            [
                self.returns,
                torch.zeros(expand_size, self.num_envs, 1, device=self.device),
            ],
            dim=0,
        ).contiguous()
        # 扩展优势缓冲区。
        self.advantages = torch.cat(
            [
                self.advantages,
                torch.zeros(expand_size, self.num_envs, 1, device=self.device),
            ],
            dim=0,
        ).contiguous()
        # 扩展动作均值缓冲区。
        self.mu = torch.cat(
            [
                self.mu,
                torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
            ],
            dim=0,
        ).contiguous()
        # 扩展动作标准差缓冲区。
        self.sigma = torch.cat(
            [
                self.sigma,
                torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
            ],
            dim=0,
        ).contiguous()

        # For hidden_states
        # 扩展隐藏状态缓冲区（如果有）。
        if not self.saved_hidden_states is None:
            self.saved_hidden_states = buffer_expand(
                self.saved_hidden_states,
                expand_size,
                dim=0,
                contiguous=True,
            )

        # 返回扩展大小。
        return expand_size

    # 添加转移，并处理循环写入。
    def add_transitions(self, transition: RolloutStorage.Transition):
        # 调用父类方法添加转移。
        return_ = super().add_transitions(transition)
        # 如果步索引达到总时间步数。
        if self.step >= self.num_transitions_per_env:
            # 检查缓冲区是否已满。
            self.buffer_full = self.num_transitions_per_env >= int(
                self.buffer_dilation_ratio * self.num_timesteps_each_rollout
            )
            # 如果已满，则循环回到开头。
            if self.buffer_full:
                self.step = self.step % self.num_transitions_per_env
        return return_

    # 清空时不一定重置步索引，而是检查是否需要扩展缓冲区。
    def clear(self):
        """Not return the self.step to 0 but check whether it needs to expand the buffer."""
        # 如果步索引达到总时间步数且缓冲区未满，则扩展一次。
        if self.step >= self.num_transitions_per_env and not self.buffer_full:
            _ = self.expand_buffer_once()  # Then self.num_transitions_per_env is updated
            print("QueueRolloutStorage: rollout storage expanded.")

    # 从游标位置交换缓冲区，使数据顺序连续。
    @torch.no_grad()
    def swap_from_cursor(self, buffer):
        """This returns a new buffer (not necessarily new memory)"""
        # 如果游标在开头或结尾，直接返回原缓冲区。
        if self.step == buffer.shape[0] or self.step == 0:
            return buffer
        # 否则将游标之后的部分拼接到前面，形成连续数据。
        return (
            torch.cat(
                [
                    buffer[self.step :],
                    buffer[: self.step],
                ],
                dim=0,
            )
            .detach()
            .contiguous()
        )

    # 解开缓冲区循环，使数据按时间顺序排列。
    def untie_buffer_loop(self):
        # 对所有缓冲区调用 swap_from_cursor。
        self.observations = self.swap_from_cursor(self.observations)
        if self.critic_observations is not None:
            self.critic_observations = self.swap_from_cursor(self.critic_observations)
        self.actions = self.swap_from_cursor(self.actions)
        self.rewards = self.swap_from_cursor(self.rewards)
        self.dones = self.swap_from_cursor(self.dones)
        self.values = self.swap_from_cursor(self.values)
        self.actions_log_prob = self.swap_from_cursor(self.actions_log_prob)
        self.mu = self.swap_from_cursor(self.mu)
        self.sigma = self.swap_from_cursor(self.sigma)
        # 隐藏状态使用 buffer_swap。
        if not self.saved_hidden_states is None:
            with torch.no_grad():
                self.saved_hidden_states = buffer_swap(self.saved_hidden_states, self.step, contiguous=True)
        # 重置步索引为 0。
        self.step = 0

    # 重写 RNN 小批量生成器，在采样前重新对齐缓冲区。
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Re-align all the buffer to make the transitions continuous before the sampling.
        [5,6,7,8,9,0,1,2,3,4] -> [0,1,2,3,4,5,6,7,8,9] where 9 is where the latest transition stored.
        """
        # 如果膨胀比例大于 1 且缓冲区已满，则解开循环。
        if self.buffer_dilation_ratio > 1.0 and self.buffer_full:
            self.untie_buffer_loop()
        # 调用父类方法。
        return super().recurrent_mini_batch_generator(num_mini_batches, num_epochs)


# ActionLabelRollout 类：继承 QueueRolloutStorage，额外存储动作标签。
class ActionLabelRollout(QueueRolloutStorage):
    # 扩展 Transition，增加 action_labels 字段。
    class Transition(QueueRolloutStorage.Transition):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.action_labels = None

    # 扩展 MiniBatch，增加 action_labels 字段。
    MiniBatch = namedtuple(
        "MiniBatch",
        [
            *RolloutStorage.MiniBatch._fields,
            "action_labels",
        ],
    )

    # 初始化。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 创建与动作同形状的动作标签缓冲区。
        self.action_labels = torch.zeros_like(self.actions)

    # 扩展缓冲区时同时扩展动作标签。
    def expand_buffer_once(self):
        expand_size = super().expand_buffer_once()
        self.action_labels = torch.cat(
            [
                self.action_labels,
                torch.zeros(expand_size, self.num_envs, *self.actions_shape, device=self.device),
            ],
            dim=0,
        ).contiguous()
        return expand_size

    # 添加转移时保存动作标签。
    def add_transitions(self, transition: Transition):
        self.action_labels[self.step] = transition.action_labels
        return super().add_transitions(transition)

    # 解开循环时同时处理动作标签。
    def untie_buffer_loop(self):
        self.action_labels = self.swap_from_cursor(self.action_labels)
        return super().untie_buffer_loop()

    # 获取小批量时包含动作标签。
    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        minibatch = super().get_minibatch_from_selection(T_select, B_select, padded_B_slice, prev_done_mask)
        action_label_batch = self.action_labels[T_select, B_select]

        return ActionLabelRollout.MiniBatch(*minibatch, action_label_batch)


# SarsaRolloutStorage 类：继承 RolloutStorage，用于 SARSA 等需要下一状态观测的算法。
class SarsaRolloutStorage(RolloutStorage):
    """The rollout storage for SARSA algorithm and those who need the next state"""

    # 扩展 Transition，增加下一观测和下一评论家观测。
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.next_observations = None
            self.next_critic_observations = None

    # 扩展 MiniBatch，增加 next_obs 和 next_critic_obs。
    MiniBatch = namedtuple(
        "MiniBatch",
        [
            *RolloutStorage.MiniBatch._fields,
            "next_obs",
            "next_critic_obs",
        ],
    )

    # 初始化。
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use a longer buffer to store the next observation and perform the safety checking
        # 创建一个更长的观测缓冲区，用于同时存储当前观测和下一观测。
        self.all_observations = torch.cat(
            [
                self.observations,
                torch.zeros_like(self.observations[:1]),
            ]
        ).contiguous()
        # 当前观测是 all_observations 的前 T 步。
        self.observations = self.all_observations[:-1]
        # 下一观测是 all_observations 的后 T 步。
        self.next_observations = self.all_observations[1:]
        # 如果有评论家观测，同样处理。
        if not self.critic_observations is None:
            self.all_critic_observations = torch.cat(
                [
                    self.critic_observations,
                    torch.zeros_like(self.critic_observations[:1]),
                ]
            ).contiguous()
            self.critic_observations = self.all_critic_observations[:-1]
            self.next_critic_observations = self.all_critic_observations[1:]

    # 添加转移。
    def add_transitions(self, transition: Transition):
        # 如果为 True，则进行断言检查（这里为 False，跳过）。
        if False:
            # For the running efficiency, will not check obs[step] == next_obs[step-1]
            assert (transition.observations == self.next_observations[self.step - 1]).all(), (
                "It is the user's responsibility to make sure that that the next_obs[step-1] == obs[step] (error in"
                " observation) "
            )
            assert (transition.critic_observations == self.next_critic_observations[self.step - 1]).all(), (
                "It is the user's responsibility to make sure that that the next_obs[step-1] == obs[step] (error in"
                " privileged observation) "
            )
        # 如果是最后一步，则复制下一观测。
        if self.step == (self.num_transitions_per_env - 1):
            # For the running efficiency, only copy next_observation at the end of the rollout.
            # Because next_obs and obs shares the same memory.
            # Also assuming each rollout (by the runner) fills the rollout storage.
            self.next_observations[self.step].copy_(transition.next_observations)
            self.next_critic_observations[self.step].copy_(transition.next_critic_observations)
        # 调用父类方法。
        return super().add_transitions(transition)

    # 重写 RNN 小批量生成器，处理下一观测。
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # 对下一观测按轨迹拆分并填充。
        self._padded_next_obs_trajectories, _ = split_and_pad_trajectories(
            self.next_observations,
            self.dones,
        )
        # 对下一评论家观测同样处理。
        if not self.critic_observations is None:
            self._padded_next_critic_obs_trajectories, _ = split_and_pad_trajectories(
                self.next_critic_observations,
                self.dones,
            )
        # 调用父类方法。
        return super().recurrent_mini_batch_generator(num_mini_batches, num_epochs)

    # 获取小批量时包含下一观测。
    def get_minibatch_from_selection(self, T_slice, B_slice, padded_B_slice=None, prev_done_mask=None):
        minibatch = super().get_minibatch_from_selection(T_slice, B_slice, padded_B_slice, prev_done_mask)

        # 根据是否为 RNN 情况选择下一观测。
        if padded_B_slice is None:
            next_obs_batch = self.next_observations[T_slice, B_slice]
            next_critic_obs_batch = (
                next_obs_batch if self.critic_observations is None else self.next_critic_observations[T_slice, B_slice]
            )
        else:
            next_obs_batch = self._padded_next_obs_trajectories[T_slice, B_slice]
            next_critic_obs_batch = (
                next_obs_batch
                if self.critic_observations is None
                else self._padded_next_critic_obs_trajectories[T_slice, B_slice]
            )

        # 返回扩展后的 MiniBatch。
        return SarsaRolloutStorage.MiniBatch(
            *minibatch,
            next_obs_batch,
            next_critic_obs_batch,
        )