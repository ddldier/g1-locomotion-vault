# 导入 PyTorch 主库。
import torch
# 导入 PyTorch 神经网络模块，用于构建网络层。
import torch.nn as nn
# 导入 PyTorch 函数式接口，此处虽未直接使用，但保留原导入。
import torch.nn.functional as F

# 从当前包的 actor_critic 模块导入 ActorCritic 基类和 get_activation 激活函数获取工具。
from .actor_critic import ActorCritic, get_activation
# 从当前包的 moe 模块导入 MoeLayer（混合专家层）。
from .moe import MoeLayer


# MoEActorCritic 类：继承自 ActorCritic，使用混合专家（MoE）层来构建 actor 和 critic 网络。
# 与普通 ActorCritic 的区别在于，其 actor 和 critic 不再使用简单的 MLP，而是使用 MoeLayer。
class MoEActorCritic(ActorCritic):
    # 初始化方法。
    def __init__(
        self,
        obs_format,                              # 观测格式字典，描述各观测组的名称与维度。
        num_actions,                             # 动作维度。
        actor_hidden_dims=[256, 256, 256],       # actor 专家网络的隐藏层维度列表，默认三层 256。
        critic_hidden_dims=[256, 256, 256],      # critic 专家网络的隐藏层维度列表，默认三层 256。
        activation="elu",                        # 激活函数名称，默认为 "elu"。
        init_noise_std=1.0,                      # 动作分布初始噪声标准差。
        num_rewards=1,                           # 奖励项数量。
        mu_activation=None,                      # 动作均值输出后的激活函数，默认为 None。
        num_moe_experts=8,                       # 混合专家层中专家的数量，默认为 8。
        moe_gate_hidden_dims=[],                 # 门控网络的隐藏层维度列表，默认为空（即线性门控）。
        **kwargs,                                # 其他传递给父类的关键字参数。
    ):
        # 保存专家数量，供后续构建 MoE 层使用。
        self.num_moe_experts = num_moe_experts
        # 保存门控网络隐藏层维度，供后续构建 MoE 层使用。
        self.moe_gate_hidden_dims = moe_gate_hidden_dims
        # 调用父类 ActorCritic 的初始化方法，传入必要的参数。
        # 注意：父类初始化会调用 _build_actor 和 _build_critic 方法，而这两个方法在子类中被重写。
        super().__init__(
            obs_format,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            num_rewards,
            mu_activation,
            **kwargs,
        )

    # 重写 _build_actor 方法：使用 MoeLayer 构建 actor 网络。
    def _build_actor(self, num_actions):
        # 创建 MoeLayer 实例：
        # - 输入维度为 self.mlp_input_dim_a（由父类根据观测格式计算得到）。
        # - 专家数量为 self.num_moe_experts。
        # - 输出维度为 num_actions。
        # - 激活函数为 self.activation。
        # - 每个专家的隐藏层维度为 self.actor_hidden_dims。
        # - 门控网络的隐藏层维度为 self.moe_gate_hidden_dims。
        moe = MoeLayer(
            self.mlp_input_dim_a,
            self.num_moe_experts,
            output_dim=num_actions,
            activation=self.activation,
            expert_hidden_dims=self.actor_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )
        # 如果指定了 mu_activation（动作均值输出后的激活函数），
        # 则将 MoE 层与激活函数串联成 Sequential 返回。
        if self.mu_activation:
            return nn.Sequential(moe, get_activation(self.mu_activation))
        # 否则直接返回 MoE 层。
        return moe

    # 重写 _build_critic 方法：使用 MoeLayer 构建 critic 网络。
    def _build_critic(self, num_values=1):
        # 创建并返回 MoeLayer 实例：
        # - 输入维度为 self.mlp_input_dim_c（由父类根据 critic 观测格式计算得到）。
        # - 专家数量为 self.num_moe_experts。
        # - 输出维度为 num_values（通常为奖励项数量）。
        # - 激活函数为 self.activation。
        # - 每个专家的隐藏层维度为 self.critic_hidden_dims。
        # - 门控网络的隐藏层维度为 self.moe_gate_hidden_dims。
        return MoeLayer(
            self.mlp_input_dim_c,
            self.num_moe_experts,
            output_dim=num_values,
            activation=self.activation,
            expert_hidden_dims=self.critic_hidden_dims,
            gate_hidden_dims=self.moe_gate_hidden_dims,
        )