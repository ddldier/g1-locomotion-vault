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

# ---------------------------------------------------------------------------
# 本文件实现 PPO 的 Actor-Critic 网络。
# 结构上与 legged_gym / rsl_rl 的 ActorCritic 同源（ETH Zurich, Nikita Rudin），
# 但经过 Instinct 的改造，主要差异：
#   1. 观测按"命名段（obs_segments）"切分，policy 与 critic 可以有不同的观测格式；
#   2. 支持 num_rewards > 1 时构建多个独立 critic（多奖励头）；
#   3. 支持 actor 最后一层接 mu_activation（例如 tanh / sigmoid 把输出压到有界区间）；
#   4. std（动作噪声）是每个动作维度的独立可学习参数，而不是全局标量。
# ---------------------------------------------------------------------------

import os                        # 用于拼接 ONNX 导出路径
from typing import Dict          # 仅用于类型注解

import numpy as np               # 本文件未直接使用，保留与上游一致（可能子类/工具函数用到）
import torch
import torch.nn as nn
from torch.distributions import Normal              # 高斯策略分布
from torch.nn.modules import rnn                     # 本文件未使用，保留兼容性

from instinct_rl.utils.utils import get_subobs_size  # 统计"命名观测段"总维度


# ---------------------------------------------------------------------------
# 1) ActorCritic 网络主体
# 说明：
#   - 不是 RNN（is_recurrent = False），是纯 MLP 策略。
#   - 同时包含 actor（策略均值）与 critic（状态价值），以及可学习的动作噪声 std。
#   - 训练时用 act() 采样动作；推理时用 act_inference() 取均值（确定性策略）。
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    is_recurrent = False      # 标记：本网络不是循环网络，Runner 据此不做 hidden state 管理

    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],   # 观测格式：{"policy": {name: shape}, "critic": {name: shape}}
        num_actions,                                # 动作维度（如 12 个关节）
        actor_hidden_dims=[256, 256, 256],          # actor MLP 各隐藏层宽度
        critic_hidden_dims=[256, 256, 256],         # critic MLP 各隐藏层宽度
        activation="elu",                           # 隐藏层激活函数名（见下方 get_activation）
        init_noise_std=1.0,                         # 动作噪声 std 的初始值（每个动作维度一份）
        num_rewards=1,                              # 奖励头数量；>1 时构建多个独立 critic
        mu_activation=None,  # If set, the last layer will be added with a special activation layer.
        **kwargs,                                   # 吸收多余参数，避免上游配置不一致时报错
    ):
        # 1.1) 忽略未知参数（只打印一次提示，不抛异常，保持向后兼容）
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()   # 初始化 nn.Module

        # 1.2) 保存观测格式，并拆出 policy / critic 两套命名段
        #      obs_format 形如 {"policy": {"base_lin_vel": (3,), "joint_pos": (12,), ...},
        #                       "critic": {...}}
        #      若没有 "critic" 键，则 critic 复用 policy 的观测段。
        # obs_segmnets is a ordered dict that contains the observation components (string) and its shape.
        # We use this to slice the observations into the correct components.
        self.__obs_format = obs_format
        self.__obs_segments = obs_format["policy"]
        self.__critic_obs_segments = obs_format.get("critic", obs_format["policy"])

        # 1.3) 保存激活函数相关配置（实际 nn.Module 实例在构建网络时才创建）
        self.activation = activation
        self.mu_activation = mu_activation
        # 计算 actor / critic 的输入维度 = 对应命名段所有 shape 的元素数之和
        self.mlp_input_dim_a = get_subobs_size(self.__obs_segments)
        self.actor_hidden_dims = actor_hidden_dims
        self.mlp_input_dim_c = get_subobs_size(self.__critic_obs_segments)
        self.critic_hidden_dims = critic_hidden_dims

        # 1.4) 构建策略网络（输出维度 = num_actions，即均值 μ）
        # Policy
        self.actor = self._build_actor(num_actions)

        # 1.5) 构建价值网络（输出 1 维 value）
        #      - num_rewards == 1：只有一个 self.critic
        #      - num_rewards > 1 ：self.critics 是 ModuleList，每个奖励一个独立 critic
        #        （避免多奖励共享参数导致梯度冲突）
        # Value function
        critic = self._build_critic(1)
        if num_rewards > 1:
            self.critics = nn.ModuleList([critic] + [self._build_critic(1) for _ in range(num_rewards - 1)])
        else:
            self.critic = critic

        # 1.6) 打印网络结构，便于训练日志里确认维度是否正确
        print(f"Actor MLP: {self.actor}")
        if num_rewards > 1:
            print(f"Multiple Critics MLP: {len(self.critics)} in total.")
        else:
            print(f"Critic MLP: {self.critic}")

        # 1.7) 动作噪声 std：每个动作维度一个可学习标量
        #      nn.Parameter 使它能被优化器更新（PPO 中通常也会被"裁剪"以限制探索幅度）
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        # 1.8) 当前分布对象（每次 update_distribution 时覆盖）
        self.distribution = None
        # disable args validation for speedup
        #      关闭 Normal 的参数校验以加速（PPO 高频调用，校验开销可观）
        Normal.set_default_validate_args = False

        # 1.9) 上游注释：经验上初始化 memory 权重反而降低性能，故保留注释但默认不执行
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    # -----------------------------------------------------------------------
    # 2) 静态工具：按 scales 用正交初始化给 Sequential 里每个 Linear 赋权
    # 当前未被调用（上游注释 "not used at the moment"），保留以兼容子类/实验。
    # -----------------------------------------------------------------------
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    # -----------------------------------------------------------------------
    # 3) 构建 actor MLP
    # 结构：Linear(input, h0) -> act -> Linear(h0, h1) -> act -> ... -> Linear(h_last, num_actions)
    # 可选：最后一层 Linear 之后再接 mu_activation（如 tanh），用于把均值限制在有界范围。
    # -----------------------------------------------------------------------
    def _build_actor(self, num_actions):
        activation = get_activation(self.activation)   # 隐藏层激活（注意：每次构建都新建实例）
        actor_layers = []
        actor_layers.append(nn.Linear(self.mlp_input_dim_a, self.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(self.actor_hidden_dims)):
            if l == len(self.actor_hidden_dims) - 1:
                # 3.1) 最后一层：映射到动作维度，不再接隐藏层激活
                actor_layers.append(nn.Linear(self.actor_hidden_dims[l], num_actions))
                # 3.2) 若配置了 mu_activation，则对输出均值再做一次激活（如 tanh 压到 [-1,1]）
                if self.mu_activation:
                    actor_layers.append(get_activation(self.mu_activation))
            else:
                # 3.3) 中间层：宽 -> 宽，接激活
                actor_layers.append(nn.Linear(self.actor_hidden_dims[l], self.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        return nn.Sequential(*actor_layers)

    # -----------------------------------------------------------------------
    # 4) 构建 critic MLP（默认输出 1 维 value）
    # 结构与 actor 类似，但没有 mu_activation。
    # num_values 预留给未来"单 critic 多值头"的扩展，当前调用处都传 1。
    # -----------------------------------------------------------------------
    def _build_critic(self, num_values=1):
        """Build the critic network, enabling multi-value prediction. Predict 1 value by default."""
        activation = get_activation(self.activation)
        critic_layers = []
        critic_layers.append(nn.Linear(self.mlp_input_dim_c, self.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(self.critic_hidden_dims)):
            if l == len(self.critic_hidden_dims) - 1:
                # 4.1) 最后一层输出 num_values 维（默认 1 维 state value）
                critic_layers.append(nn.Linear(self.critic_hidden_dims[l], num_values))
            else:
                # 4.2) 中间层
                critic_layers.append(nn.Linear(self.critic_hidden_dims[l], self.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        return nn.Sequential(*critic_layers)

    # -----------------------------------------------------------------------
    # 5) 生命周期钩子
    # 非循环网络不需要在 episode 结束时清 hidden state，故 reset 为空实现。
    # forward 未实现：Runner 不会直接调用 model(x)，而是显式调用 act / evaluate。
    # -----------------------------------------------------------------------
    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # 6) 分布相关属性：从 self.distribution 里惰性取值
    # -----------------------------------------------------------------------
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        # 6.1) 各动作维度熵求和，得到每个样本的标量熵（用于 PPO 熵正则）
        return self.distribution.entropy().sum(dim=-1)

    # -----------------------------------------------------------------------
    # 7) 根据观测构建高斯策略分布：N(actor(obs), std)
    # 注意 std 的写法：mean * 0.0 + self.std
    #   - mean * 0.0 得到与 mean 同形状的全零张量（不改变数值）
    #   - 再加上 self.std（形状 (num_actions,)），利用广播得到 (N, num_actions)
    # 这样写而不是 self.std.expand_as(mean)，是为了避免后续 in-place 操作报错。
    # -----------------------------------------------------------------------
    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    # -----------------------------------------------------------------------
    # 8) 训练时采样动作（随机策略）
    # -----------------------------------------------------------------------
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    # -----------------------------------------------------------------------
    # 9) 计算给定动作在"当前分布"下的 log 概率（PPO ratio 需要）
    # 对各动作维度求和 -> 每个样本一个标量 log_prob。
    # -----------------------------------------------------------------------
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # -----------------------------------------------------------------------
    # 10) 推理时取确定性动作（均值），不采样、不加噪声
    # -----------------------------------------------------------------------
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    # -----------------------------------------------------------------------
    # 11) 计算状态价值
    # 三种情况：
    #   a) 多 critic 且观测按 critic 分段传入 list：每个 critic 吃自己的观测，输出拼接；
    #   b) 多 critic 但只给一份观测：所有 critic 吃同一份观测，输出拼接；
    #   c) 单 critic：直接调用 self.critic。
    # 返回值形状：(N, num_rewards)。
    # -----------------------------------------------------------------------
    def evaluate(self, critic_observations, **kwargs):
        if hasattr(self, "critics") and isinstance(critic_observations, list):
            # 11.a) 多 critic + 分段观测
            value = torch.cat(
                [critic(critic_obs) for critic, critic_obs in zip(self.critics, critic_observations)],
                dim=-1,
            )
        elif hasattr(self, "critics"):
            # 11.b) 多 critic + 共享观测
            value = torch.cat(
                [critic(critic_observations) for critic in self.critics],
                dim=-1,
            )
        else:
            # 11.c) 单 critic
            value = self.critic(critic_observations)
        return value

    # -----------------------------------------------------------------------
    # 12) 原地裁剪动作噪声 std（训练中限制探索幅度）
    # 用 copy_ 而非重新赋值，保证 nn.Parameter 的身份不变（优化器引用不失效）。
    # @torch.no_grad()：不参与梯度。
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def clip_std(self, min=None, max=None):
        self.std.copy_(self.std.clip(min=min, max=max))

    # -----------------------------------------------------------------------
    # 13) 只读属性：暴露观测段定义（用于 Runner / 包装器切分观测）
    # NOTE: make sure to copy these methods in subclasses if the subclass has its own obs_segments
    # -----------------------------------------------------------------------
    @property
    def obs_segments(self):
        """NOTE: make sure to copy these methods in subclasses if the subclass has its own obs_segments"""
        return self.__obs_segments

    @property
    def critic_obs_segments(self):
        return self.__critic_obs_segments

    # -----------------------------------------------------------------------
    # 14) 导出 actor 为 ONNX（部署用：C++/推理引擎可加载）
    # 说明：
    #   - self.eval() 切到推理模式（影响 Dropout / BatchNorm 等，本网络没有，但语义正确）；
    #   - 输入 observations 必须是 batchsize=1 的 batch-wise 张量；
    #   - 只导出 actor，不导出 critic 和 std（部署只需要确定性策略）。
    # -----------------------------------------------------------------------
    def export_as_onnx(self, observations, filedir):
        """Export the model as an ONNX file. Input should be batch-wise observations with batchsize 1."""
        self.eval()
        with torch.no_grad():
            exported_program = torch.onnx.export(
                self.actor,                                    # 只导出 actor
                observations,                                  # 示例输入（决定输入形状）
                os.path.join(filedir, "actor.onnx"),           # 输出路径
                input_names=["input"],
                output_names=["output"],
                opset_version=12,                              # ONNX opset 版本
            )
            print(f"Exported ActorCritic model to {os.path.join(filedir, 'actor.onnx')}")


# ---------------------------------------------------------------------------
# 15) 激活函数名 -> nn.Module 实例的映射
# 注意：
#   - "crelu" 实际返回 nn.ReLU()（上游笔误/遗留，并非真正的 CReLU），这里保持原样不改；
#   - 未知名字会打印 "invalid activation function!" 并返回 None，
#     若误用会导致 Sequential 里出现 None，构建/前向时报错；
#   - 每次调用都新建实例：nn.ELU 等无参数，重复实例化没有副作用。
# ---------------------------------------------------------------------------
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None