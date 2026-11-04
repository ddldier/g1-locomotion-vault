# 从 typing 导入 Literal，用于类型注解（此处未直接使用，保留原导入）。
from typing import Literal

# 导入 PyTorch 主库。
import torch
# 导入 PyTorch 神经网络模块。
import torch.nn as nn
# 导入 PyTorch 函数式接口，用于 softmax 等操作。
import torch.nn.functional as F

# 从当前包的 actor_critic 模块导入 get_activation 函数，用于根据名称获取激活函数。
from .actor_critic import get_activation  # Assuming get_activation is needed


# MoeLayer 类：实现混合专家（Mixture of Experts, MoE）层。
# 该层包含一个门控网络（gate）和多个专家网络（experts）。
# 门控网络根据输入为每个专家分配权重，最终输出是所有专家输出的加权和。
class MoeLayer(nn.Module):
    # 初始化 MoE 层。
    # 参数：
    #   input_dim: 输入特征维度。
    #   num_experts: 专家数量。
    #   output_dim: 每个专家的输出维度，若为 None 则专家只输出隐藏层特征。
    #   activation: 激活函数名称，默认为 "elu"。
    #   expert_hidden_dims: 每个专家网络的隐藏层维度列表。
    #   gate_hidden_dims: 门控网络的隐藏层维度列表。
    def __init__(
        self,
        input_dim,
        num_experts,
        output_dim=None,
        activation="elu",
        expert_hidden_dims=[],
        gate_hidden_dims=[],
    ):
        # 调用父类 nn.Module 的初始化。
        super().__init__()
        # 根据激活函数名称获取激活函数实例。
        self.act_fn = get_activation(activation)
        # 构建门控网络。
        self.gate = self._build_gate(input_dim, num_experts, gate_hidden_dims)
        # 构建多个专家网络，使用 nn.ModuleList 以便正确注册参数。
        self.experts = nn.ModuleList(
            [self._build_expert(input_dim, output_dim, expert_hidden_dims) for _ in range(num_experts)]
        )

    # 构建门控网络。
    # 门控网络是一个多层感知机，输出维度为 num_experts。
    def _build_gate(self, input_dim, num_experts, hidden_dims):
        # 用于存放网络层的列表。
        layers = []
        # 当前层输入维度初始为 input_dim。
        curr_dim = input_dim
        # 遍历每个隐藏层维度。
        for h in hidden_dims:
            # 添加线性层。
            layers.append(nn.Linear(curr_dim, h))
            # 添加激活函数。
            layers.append(self.act_fn)
            # 更新当前维度。
            curr_dim = h
        # 最后添加输出层，输出维度为专家数量 num_experts。
        layers.append(nn.Linear(curr_dim, num_experts))
        # 使用 nn.Sequential 组合所有层并返回。
        return nn.Sequential(*layers)

    # 构建单个专家网络。
    # 每个专家是一个多层感知机。
    def _build_expert(self, input_dim, output_dim, hidden_dims):
        # 用于存放网络层的列表。
        layers = []
        # 当前层输入维度初始为 input_dim。
        curr_dim = input_dim
        # 遍历每个隐藏层维度。
        for h in hidden_dims:
            # 添加线性层。
            layers.append(nn.Linear(curr_dim, h))
            # 添加激活函数。
            layers.append(self.act_fn)
            # 更新当前维度。
            curr_dim = h
        # 如果指定了 output_dim，则添加输出层。
        if output_dim is not None:
            # 添加线性层，输出维度为 output_dim。
            # 注意：最后一层不加激活函数。
            layers.append(nn.Linear(curr_dim, output_dim))  # no activation for the last layer
        # 使用 nn.Sequential 组合所有层并返回。
        return nn.Sequential(*layers)

    # 前向传播。
    def forward(self, x):
        # 计算门控分数：通过门控网络得到每个专家的 logits，再沿最后一维做 softmax。
        # 形状：[batch, num_experts]
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [batch, num_experts] # gate the expert outputs
        # 计算每个专家的输出：对每个专家调用其前向传播。
        expert_outputs = [expert(x) for expert in self.experts]
        # 将所有专家输出堆叠成一个张量。
        # 形状：[batch, num_experts, output_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_dim]
        # 使用爱因斯坦求和将门控分数与专家输出加权混合。
        # 公式：output[b, o] = sum_e gate_scores[b, e] * expert_outputs[b, e, o]
        # 结果形状：[batch, output_dim]
        output = torch.einsum("be,beo->bo", gate_scores, expert_outputs)  # mix the expert outputs
        # 返回最终输出。
        return output