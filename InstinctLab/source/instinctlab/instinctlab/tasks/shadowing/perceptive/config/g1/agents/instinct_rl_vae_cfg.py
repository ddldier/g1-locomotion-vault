# 导入操作系统接口，用于读取环境变量、展开用户目录等
import os

# 从 IsaacLab 导入 configclass 装饰器，用于把类声明为可配置的 dataclass 风格配置
from isaaclab.utils import configclass

# 导入外部感知观测中的可视化图像函数；当前文件未直接使用，可能是为注册或后续扩展保留
from instinctlab.envs.mdp.observations.exteroception import visualizable_image
# 从 InstinctRL 包装器中导入各种强化学习配置基类
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg,
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderActorCriticCfg,
    InstinctRlEncoderVaeActorCriticCfg,
    InstinctRlNormalizerCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


# 定义卷积头编码器配置容器，内部包含深度图像编码器配置
@configclass
class Conv2dHeadEncoderCfg:
    # 深度图像编码器配置，继承 InstinctRlConv2dHeadCfg
    @configclass
    class DepthImageEncoderCfg(InstinctRlConv2dHeadCfg):
        # 卷积通道数：第一层卷积输出 32 通道，第二层卷积输出 32 通道
        channels = [32, 32]
        # 每层卷积核大小均为 3x3
        kernel_sizes = [3, 3]
        # 每层卷积步长均为 1
        strides = [1, 1]
        # 每层卷积填充均为 1，用于保持空间尺寸不变
        paddings = [1, 1]
        # 卷积之后全连接隐藏层维度为 32
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
        # 从输入观测中取出这些组件后，不再把它们传给后续网络，避免重复使用原始输入
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

    # Actor 使用的编码器配置，使用上面定义的卷积头编码器处理深度图像
    encoder_configs = Conv2dHeadEncoderCfg()
    # Critic 不使用单独的编码器配置
    critic_encoder_configs = None


# 定义 VAE 策略配置，继承 InstinctRlEncoderVaeActorCriticCfg
@configclass
class VaePolicyCfg(InstinctRlEncoderVaeActorCriticCfg):
    # 编码器配置，使用卷积头编码器处理深度图像
    encoder_configs = Conv2dHeadEncoderCfg()

    # VAE 编码器网络参数：隐藏层维度和激活函数
    vae_encoder_kwargs = {
        "hidden_sizes": [256, 128, 64],
        "nonlinearity": "ELU",
    }
    # VAE 解码器网络参数：隐藏层维度和激活函数
    vae_decoder_kwargs = {
        "hidden_sizes": [512, 256, 128],
        "nonlinearity": "ELU",
    }
    # VAE 潜变量维度为 16
    vae_latent_size = 16
    # VAE 输入使用的子观测组件列表
    vae_input_subobs_components = [
        "parallel_latent_0_depth_image",  # based on the encoder_configs in Conv2dHeadEncoderCfg
        # "projected_gravity",
        # "base_ang_vel",
        # "joint_pos",
        # "joint_vel",
        # "last_action",
    ]
    # VAE 辅助子观测组件列表，通常用于辅助损失或重构目标
    vae_aux_subobs_components = [
        # "parallel_latent_0_depth_image",
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "last_action",
    ]


# 定义算法配置，继承 InstinctRlPpoAlgorithmCfg
@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    # 算法类名为 VaeDistill，表示使用 VAE 蒸馏方法
    class_name = "VaeDistill"
    # KL 损失函数类型为 kl_divergence
    kl_loss_func = "kl_divergence"
    # KL 损失系数为 1.0
    kl_loss_coef = 1.0
    # 不使用标准 PPO 更新
    using_ppo = False
    # 每次学习迭代的 epoch 数
    num_learning_epochs = 5
    # 每个 epoch 划分的 mini-batch 数
    num_mini_batches = 4
    # 学习率
    learning_rate = 1e-3
    # PPO parameters should not affect anything.
    # 调度方式为自适应
    schedule = "adaptive"
    # 折扣因子
    gamma = 0.99
    # GAE lambda
    lam = 0.95
    # 目标 KL 散度
    desired_kl = 0.01
    # 梯度裁剪最大范数
    max_grad_norm = 1.0

    # 教师动作概率，可能用于蒸馏时混合教师动作
    teacher_act_prob = 0.2
    # update_times_scale = 20 * int(1e3)

    # 教师策略类名，从 InstinctRlEncoderActorCriticCfg 实例中获取
    teacher_policy_class_name = InstinctRlEncoderActorCriticCfg().class_name
    # 教师策略配置字典，包含网络结构、编码器、观测格式、动作/奖励维度等
    teacher_policy: dict = {
        # 教师策略初始化噪声标准差
        "init_noise_std": 1.0,
        # 教师 Actor 隐藏层维度
        "actor_hidden_dims": [512, 256, 128],
        # 教师 Critic 隐藏层维度
        "critic_hidden_dims": [512, 256, 128],
        # 教师激活函数
        "activation": "elu",
        # 教师编码器配置：深度图像使用 Conv2dHeadModel
        "encoder_configs": {
            "depth_image": {
                # 编码器类名
                "class_name": "Conv2dHeadModel",
                # 处理的输入组件
                "component_names": ["depth_image"],
                # 输出维度
                "output_size": 32,
                # 取出输入组件后不再传递
                "takeout_input_components": True,
                # 卷积通道
                "channels": [32, 32],
                # 卷积核大小
                "kernel_sizes": [3, 3],
                # 卷积步长
                "strides": [1, 1],
                # 全连接隐藏层
                "hidden_sizes": [32],
                # 卷积填充
                "paddings": [1, 1],
                # 非线性激活
                "nonlinearity": "ReLU",
                # 不使用最大池化
                "use_maxpool": False,
            }
        },
        # 教师 Critic 不使用单独编码器
        "critic_encoder_configs": None,
        # 教师策略的观测格式，分为 policy 和 critic 两部分
        "obs_format": {
            # 策略观测格式：组件名 -> 形状/维度
            "policy": {
                "joint_pos_ref": (10, 29),
                "joint_vel_ref": (10, 29),
                "position_ref": (10, 3),
                "rotation_ref": (10, 6),
                "depth_image": (1, 18, 32),
                "projected_gravity": (24,),
                "base_ang_vel": (24,),
                "joint_pos": (232,),
                "joint_vel": (232,),
                "last_action": (232,),
            },
            # 评论家观测格式：包含更多特权信息，如 link_pos、link_rot、height_scan、base_lin_vel 等
            "critic": {
                "joint_pos_ref": (10, 29),
                "joint_vel_ref": (10, 29),
                "position_ref": (10, 3),
                "link_pos": (14, 3),
                "link_rot": (14, 6),
                "height_scan": (187,),
                "base_lin_vel": (24,),
                "base_ang_vel": (24,),
                "joint_pos": (232,),
                "joint_vel": (232,),
                "last_action": (232,),
            },
        },
        # 动作维度为 29
        "num_actions": 29,
        # 奖励维度为 1
        "num_rewards": 1,
    }
    # 教师策略日志目录，使用 os.path.expanduser 展开 ~
    teacher_logdir = os.path.expanduser(
        "~/Data/instinctlab_logs/instinct_rl/g1_perceptive_shadowing/20260111_103654_g1Perceptive_4MotionsKneelClimbStep1_concatMotionBins__GPU0_from20260108_032900"
    )


# 定义归一化器配置
@configclass
class NormalizersCfg:
    # 策略观测使用默认的 InstinctRlNormalizerCfg
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    # critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    # NOTE: No critic normalizer, must be loaded from the teacher policy.
    # 说明：没有 critic 归一化器，必须从教师策略加载


# 定义 G1 感知 VAE PPO 运行器配置，继承 InstinctRlOnPolicyRunnerCfg
@configclass
class G1PerceptiveVaePPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    # 策略配置使用 VaePolicyCfg
    policy: VaePolicyCfg = VaePolicyCfg()
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
    experiment_name = "g1_perceptive_vae"

    # 要加载的运行目录，None 表示不加载已有运行
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