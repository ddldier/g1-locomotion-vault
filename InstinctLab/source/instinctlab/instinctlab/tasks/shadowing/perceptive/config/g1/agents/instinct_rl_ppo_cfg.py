# 导入操作系统接口，用于读取环境变量、展开路径等操作
import os

# 从 IsaacLab 导入 configclass 装饰器，用于声明配置类
from isaaclab.utils import configclass

# 从外部感知观测模块导入可视化图像函数；当前文件未直接使用，通常用于注册或扩展观测项
from instinctlab.envs.mdp.observations.exteroception import visualizable_image
# 从 InstinctRL 包装器中导入强化学习相关配置基类
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,          # Actor-Critic 配置基类；当前文件未直接使用
    InstinctRlConv2dHeadCfg,           # 二维卷积头配置基类，用于图像编码器
    InstinctRlEncoderActorCriticCfg,   # 带编码器的 Actor-Critic 配置基类
    InstinctRlNormalizerCfg,           # 观测归一化器配置基类
    InstinctRlOnPolicyRunnerCfg,       # On-Policy 运行器配置基类
    InstinctRlPpoAlgorithmCfg,         # PPO 算法配置基类
)


# 定义卷积头编码器配置容器，内部包含深度图像编码器配置
@configclass
class Conv2dHeadEncoderCfg:
    # 深度图像编码器配置，继承 InstinctRlConv2dHeadCfg
    @configclass
    class DepthImageEncoderCfg(InstinctRlConv2dHeadCfg):
        # 卷积层输出通道数：两层卷积，每层输出 32 通道
        channels = [32, 32]
        # 每层卷积核大小均为 3x3
        kernel_sizes = [3, 3]
        # 每层卷积步长均为 1
        strides = [1, 1]
        # 每层卷积填充均为 1，用于保持空间尺寸
        paddings = [1, 1]
        # 卷积之后的全连接隐藏层维度为 32
        hidden_sizes = [
            32,
        ]
        # 非线性激活函数使用 ReLU
        nonlinearity = "ReLU"
        # 不使用最大池化
        use_maxpool = False
        # 编码器最终输出特征维度为 32
        output_size = 32
        # 该编码器处理的输入组件名为 depth_image
        component_names = ["depth_image"]
        # 从输入观测中取出该组件后，不再将其传给后续网络，避免重复使用原始输入
        takeout_input_components = True

    # 实例化一个深度图像编码器配置，供外部引用
    depth_image = DepthImageEncoderCfg()


# 定义 Actor-Critic 策略配置，使用编码器处理观测
@configclass
class PolicyCfg(InstinctRlEncoderActorCriticCfg):
    # 初始化时策略噪声标准差
    init_noise_std = 1.0
    # Actor 网络隐藏层维度
    actor_hidden_dims = [512, 256, 128]
    # Critic 网络隐藏层维度
    critic_hidden_dims = [512, 256, 128]
    # 激活函数使用 ELU
    activation = "elu"

    # Actor 使用的编码器配置，使用卷积头编码器处理深度图像
    encoder_configs = Conv2dHeadEncoderCfg()
    # Critic 不使用单独的编码器配置
    critic_encoder_configs = None  # No encoder for critic


# 定义 PPO 算法配置，继承 InstinctRlPpoAlgorithmCfg
@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    # 算法类名为 PPO
    class_name = "PPO"
    # 价值损失系数
    value_loss_coef = 1.0
    # 是否使用裁剪后的价值损失
    use_clipped_value_loss = True
    # PPO 裁剪参数
    clip_param = 0.2
    # 熵正则系数
    entropy_coef = 0.005
    # 每次学习迭代的 epoch 数
    num_learning_epochs = 5
    # 每个 epoch 划分的 mini-batch 数
    num_mini_batches = 4
    # 学习率
    learning_rate = 1e-3
    # 学习率调度方式为自适应
    schedule = "adaptive"
    # 折扣因子
    gamma = 0.99
    # GAE lambda
    lam = 0.95
    # 目标 KL 散度
    desired_kl = 0.01
    # 梯度裁剪最大范数
    max_grad_norm = 1.0


# 定义归一化器配置
@configclass
class NormalizersCfg:
    # 策略观测使用的归一化器
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    # 评论家观测使用的归一化器
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


# 定义 G1 感知影子 PPO 运行器配置，继承 InstinctRlOnPolicyRunnerCfg
@configclass
class G1PerceptiveShadowingPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    # 策略配置使用 PolicyCfg
    policy: PolicyCfg = PolicyCfg()
    # 算法配置使用 AlgorithmCfg
    algorithm: AlgorithmCfg = AlgorithmCfg()
    # 归一化器配置使用 NormalizersCfg
    normalizers: NormalizersCfg = NormalizersCfg()

    # 每个环境每次采集的步数
    num_steps_per_env = 24
    # 最大训练迭代次数
    max_iterations = 50000
    # 保存间隔（迭代次数）
    save_interval = 1000
    # 日志记录间隔（迭代次数）
    log_interval = 10
    # 实验名称
    experiment_name = "g1_perceptive_shadowing"

    # 要加载的运行目录；None 表示不加载已有运行，从头开始训练
    load_run = None

    # 初始化后处理函数
    def __post_init__(self):
        # 调用父类的初始化后处理
        super().__post_init__()
        # 如果 load_run 不为 None，则 resume 为 True，表示续训
        self.resume = self.load_run is not None
        # 根据 CUDA_VISIBLE_DEVICES 环境变量生成运行名称后缀
        self.run_name = "".join(
            [
                f"_GPU{os.environ.get('CUDA_VISIBLE_DEVICES')}" if "CUDA_VISIBLE_DEVICES" in os.environ else "",
            ]
        )