# ============================================================
# 文件：from isaaclab.utils import configclass.txt
# 说明：G1 跑酷任务的强化学习训练配置，包括策略网络、
#       AMP 算法以及 On-Policy Runner 的设置。
# 以下仅添加注释，不修改任何原有代码。
# ============================================================

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


# 深度图像卷积编码器配置，继承自 InstinctRlConv2dHeadCfg
@configclass
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size = 128              # 编码器输出特征维度
    channels = [4]                 # 卷积通道数（可能对应输入深度图的堆叠帧数）
    kernel_sizes = [3]             # 卷积核大小
    strides = [1]                  # 卷积步长
    hidden_sizes = [256, 256]      # 全连接隐藏层维度
    paddings = [1]                 # 卷积填充
    nonlinearity = "ReLU"          # 激活函数
    use_maxpool = True             # 是否使用最大池化
    component_names = [            # 该编码器处理的观测组件名称
        "depth_image",
    ]


# 编码器配置集合，可包含多个编码器（此处仅深度编码器）
@configclass
class EncoderConfigs:
    depth_encoder = DepthEncoderConv2dCfg()


# 混合专家（MoE）策略配置，继承自 InstinctRlEncoderMoEActorCriticCfg
@configclass
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std = 1.0                     # 动作噪声初始标准差
    num_moe_experts = 4                      # MoE 专家数量
    actor_hidden_dims = [256, 128, 64]       # Actor 网络隐藏层维度
    critic_hidden_dims = [256, 128, 64]      # Critic 网络隐藏层维度
    activation = "elu"                       # 激活函数
    encoder_configs = EncoderConfigs()       # Actor 使用的编码器配置
    critic_encoder_configs = EncoderConfigs()# Critic 使用的编码器配置


# AMP（对抗运动先验）算法配置，继承自 InstinctRlPpoAlgorithmCfg
@configclass
class AmpAlgoCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "WasabiPPO"                 # 使用的 PPO 变体类名
    discriminator_kwargs = {                 # 判别器网络参数
        "hidden_sizes": [1024, 512],
        "nonlinearity": "ReLU",
    }

    discriminator_reward_coef = 0.25         # 判别器奖励系数
    discriminator_reward_type = "quad"       # 判别器奖励类型（二次型）
    discriminator_loss_func = "MSELoss"      # 判别器损失函数
    discriminator_gradient_penalty_coef = 5.0# 梯度惩罚系数
    discriminator_optimizer_class_name = "AdamW"  # 判别器优化器
    discriminator_weight_decay_coef = 3e-4   # 权重衰减系数
    discriminator_logit_weight_decay_coef = 0.04  # logit 权重衰减系数
    discriminator_optimizer_kwargs = {       # 判别器优化器参数
        "lr": 1.0e-4,
        "betas": [0.9, 0.999],
    }
    value_loss_coef = 1.0                    # 价值损失系数
    use_clipped_value_loss = True            # 是否使用裁剪后的价值损失
    clip_param = 0.2                         # PPO 裁剪参数
    entropy_coef = 0.006                     # 熵正则化系数
    num_learning_epochs = 5                  # 每次迭代的学习轮数
    num_mini_batches = 4                     # 小批量数量
    learning_rate = 1.0e-3                   # 学习率
    schedule = "adaptive"                    # 学习率调度方式
    gamma = 0.99                             # 折扣因子
    lam = 0.95                               # GAE 的 lambda
    desired_kl = 0.01                        # 目标 KL 散度
    max_grad_norm = 1.0                      # 梯度裁剪最大范数


# G1 跑酷任务的 On-Policy Runner 配置，继承自 InstinctRlOnPolicyRunnerCfg
@configclass
class G1ParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24                   # 每个环境每次迭代采集的步数
    max_iterations = 30000                   # 最大训练迭代次数
    save_interval = 1000                     # 每 1000 次训练迭代保存一次模型
    experiment_name = "g1_parkour"           # 实验名称
    resume = False                           # 是否从检查点恢复
    load_run = ""                            # 要加载的运行目录
    empirical_normalization = False          # 是否使用经验归一化
    policy = MoEPolicyCfg()                  # 策略网络配置
    algorithm = AmpAlgoCfg()                 # 算法配置
