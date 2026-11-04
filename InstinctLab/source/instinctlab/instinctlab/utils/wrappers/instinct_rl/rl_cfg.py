# 导入 MISSING，用于标记必须由用户提供的配置项
from dataclasses import MISSING
# 导入类型提示：Literal 用于限定字面量，Sequence 用于序列类型
from typing import Literal, Sequence

# 导入配置类装饰器
from isaaclab.utils import configclass

# 导入并行块配置基类（此处未直接使用，可能用于其他模块）
from instinctlab.utils.wrappers.instinct_rl.module_cfg import InstinctRlParallelBlockCfg


# 配置类：PPO actor-critic 网络基础配置
@configclass
class InstinctRlActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    # 策略类名，默认使用 ActorCritic
    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    # 策略初始噪声标准差，必须指定
    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    # Actor 网络隐藏层维度，必须指定
    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    # Critic 网络隐藏层维度，必须指定
    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    # 激活函数，必须指定
    activation: str = MISSING
    """The activation function for the actor and critic networks."""


# 配置类：带循环层的 actor-critic 网络配置，继承自基础配置
@configclass
class InstinctRlActorCriticRecurrentCfg(InstinctRlActorCriticCfg):
    # 类名改为 ActorCriticRecurrent
    class_name: str = "ActorCriticRecurrent"

    # RNN 类型，默认 GRU
    rnn_type: str = "gru"
    """The type of RNN to use. Default is GRU."""

    # RNN 隐藏层大小
    rnn_hidden_size: int = 256
    """The hidden size of the RNN."""

    # RNN 层数
    rnn_num_layers: int = 1
    """The number of layers in the RNN."""

    # 是否使用多个 RNN critic 对应多个奖励
    multireward_multirnn: bool = False
    """Whether to use multiple RNN critics for multiple rewards."""


# 混入类：编码器配置
@configclass
class EncoderCfgMixin:
    # 编码器配置，必须指定
    encoder_configs: object = MISSING
    """ A configclass containing InstinctRlEncoderCfg.
    """

    # Critic 编码器配置，可以是共享、None 或具体配置
    critic_encoder_configs: object | Literal["shared"] | None = None
    """ A configclass containing InstinctRlEncoderCfg for building the critic encoders.
    """


# 混入类：状态估计器配置
@configclass
class EstimatorCfgMixin:
    """Mixin for state estimator."""

    # 用于估计器的观测组件列表，必须指定
    estimator_obs_components: list[str] = MISSING
    """The components of the observation used for the estimator."""

    # 作为估计器目标的观测组件列表，必须指定
    estimator_target_components: list[str] = MISSING
    """The components of the observation used as the target for the estimator."""

    # 估计器配置，必须指定
    estimator_configs: object = MISSING
    """A configclass containing GalaxeaRlEncoderCfg for building the estimator."""

    # 替换状态的概率
    replace_state_prob: float = 1.0


# 配置类：带估计器的 actor-critic 网络
@configclass
class EstimatorActorCriticCfg(
    EstimatorCfgMixin,
    InstinctRlActorCriticCfg,
):
    """Configuration for the estimator actor-critic networks."""

    # 类名设为 EstimatorActorCritic
    class_name = "EstimatorActorCritic"


# 配置类：带估计器的循环 actor-critic 网络
@configclass
class EstimatorActorCriticRecurrentCfg(
    EstimatorCfgMixin,
    InstinctRlActorCriticRecurrentCfg,
):
    """Configuration for the estimator actor-critic-recurrent networks."""

    # 类名设为 EstimatorActorCriticRecurrent
    class_name = "EstimatorActorCriticRecurrent"


# 配置类：混合专家（MoE）actor-critic 网络
@configclass
class InstinctRlMoEActorCriticCfg(InstinctRlActorCriticCfg):
    # 类名设为 MoEActorCritic
    class_name: str = "MoEActorCritic"

    # MoE 专家数量，默认 8
    num_moe_experts: int = 8
    """The number of Mixture of Experts (MoE) experts."""

    # MoE 门控网络隐藏层维度，默认为空
    moe_gate_hidden_dims: list[int] = []
    """The hidden dimensions of the MoE gate network."""


# 配置类：VAE actor-critic 网络
@configclass
class InstinctRlVaeActorCriticCfg(InstinctRlActorCriticCfg):
    # 类名设为 VaeActor
    class_name: str = "VaeActor"

    # VAE 编码器参数，必须指定
    vae_encoder_kwargs: dict = MISSING
    """ A dict building the MLP-based VAE encoder."""

    # VAE 解码器参数，必须指定
    vae_decoder_kwargs: dict = MISSING
    """ A dict building the MLP-based VAE decoder."""

    # VAE 潜在空间大小，默认 16
    vae_latent_size: int = 16
    """ The latent size of the VAE."""

    # Critic 隐藏层维度（VAE 蒸馏通常不用）
    critic_hidden_dims: list[int] = [512, 256, 128]
    """The hidden dimensions of the critic network (typically not used for VAE distillation)."""

    # 初始噪声标准差（VAE 蒸馏通常不用）
    init_noise_std: float = 1e-4
    """The initial noise standard deviation for the critic network (typically not used for VAE distillation)."""

    # 激活函数（VAE 蒸馏通常不用）
    activation: str = "elu"
    """The activation function for the critic network (typically not used for VAE distillation)."""


# 配置类：带编码器的 actor-critic 网络
@configclass
class InstinctRlEncoderActorCriticCfg(
    EncoderCfgMixin,
    InstinctRlActorCriticCfg,
):
    """Configuration for the encoder actor-critic networks."""

    # 类名设为 EncoderActorCritic
    class_name = "EncoderActorCritic"


# 配置类：带编码器的循环 actor-critic 网络
@configclass
class InstinctRlEncoderActorCriticRecurrentCfg(
    EncoderCfgMixin,
    InstinctRlActorCriticRecurrentCfg,
):
    """Configuration for the encoder actor-critic-recurrent networks."""

    # 类名设为 EncoderActorCriticRecurrent
    class_name = "EncoderActorCriticRecurrent"


# 配置类：带编码器的 MoE actor-critic 网络
@configclass
class InstinctRlEncoderMoEActorCriticCfg(
    EncoderCfgMixin,
    InstinctRlMoEActorCriticCfg,
):
    """Configuration for the encoder actor-critic networks."""

    # 类名设为 EncoderMoEActorCritic
    class_name = "EncoderMoEActorCritic"


# 配置类：带编码器的 VAE actor-critic 网络
@configclass
class InstinctRlEncoderVaeActorCriticCfg(
    EncoderCfgMixin,
    InstinctRlVaeActorCriticCfg,
):
    """Configuration for the encoder actor networks."""

    # 类名设为 EncoderVaeActorCritic
    class_name = "EncoderVaeActorCritic"


# 配置类：PPO 算法配置
@configclass
class InstinctRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    # 算法类名，默认 PPO
    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    # 价值损失系数，必须指定
    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    # 是否使用裁剪后的价值损失，必须指定
    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    # 策略裁剪参数，必须指定
    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    # 熵损失系数，必须指定
    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    # 每次更新的学习轮数，必须指定
    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    # 每次更新的小批量数量，必须指定
    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    # 学习率，必须指定
    learning_rate: float = MISSING
    """The learning rate for the policy."""

    # 优化器类名，默认 AdamW
    optimizer_class_name: str = "AdamW"
    """The optimizer class name. Default is AdamW."""

    # 学习率调度方式，必须指定
    schedule: str = MISSING
    """The learning rate schedule."""

    # 折扣因子，必须指定
    gamma: float = MISSING
    """The discount factor."""

    # GAE 的 lambda 参数，必须指定
    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    # 多奖励时优势混合权重，默认 1.0
    advantage_mixing_weights: float | Sequence[float] = 1.0
    """The weights for the mixing advantages and compute surrogate loss when multiple rewards are returned."""

    # 目标 KL 散度，必须指定
    desired_kl: float = MISSING
    """The desired KL divergence."""

    # 梯度裁剪最大范数，必须指定
    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    # 计算分布时策略的最小标准差，默认 1e-12 防止数值不稳定
    clip_min_std: float = 1e-12
    """The minimum standard deviation for the policy when computing distribution.
    Default: 1e-12 to prevent numerical instability.
    """


# 配置类：经验归一化器
@configclass
class InstinctRlNormalizerCfg:
    # 类名设为 EmpiricalNormalization
    class_name: str = "EmpiricalNormalization"


# 配置类：On-Policy 算法的 Runner 配置
@configclass
class InstinctRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    # 实验随机种子，默认 42
    seed: int = 42
    """The seed for the experiment. Default is 42."""

    # RL 代理使用的设备，默认 cuda:0
    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    # 每次更新每个环境采集的步数，必须指定
    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    # 最大迭代次数，必须指定
    max_iterations: int = MISSING
    """The maximum number of iterations."""

    # 检查点操纵器，加载时调用，默认为 None
    ckpt_manipulator: str | None = None
    """A string calling the checkpoint manipulator when loading. Typically, user has to implement their own
    checkpoint manipulator to load the model weights if the loaded model is different from the current training
    model. Default is None for no manipulation.
    """

    # 策略配置，必须指定
    policy: InstinctRlActorCriticCfg = MISSING
    """The policy configuration."""

    # 算法配置，必须指定
    algorithm: InstinctRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    # 各观测组的归一化配置，空字典表示不运行归一化
    normalizers = dict()
    """The configs for each observation group when they are still flattened tensors
    Empty dict for no normalizer running in the RL runner.
    """

    ##
    # 检查点参数
    ##

    # 保存间隔（迭代次数），必须指定
    save_interval: int = MISSING
    """The number of iterations between saves."""

    # 日志记录间隔（迭代次数），默认 1
    log_interval: int = 1
    """The number of iterations between logs."""

    # 实验名称，必须指定
    experiment_name: str = MISSING
    """The experiment name."""

    # 运行名称，默认为空字符串
    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # 日志参数
    ##

    # logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    # """The logger to use. Default is tensorboard."""

    # neptune_project: str = "instinctlab"
    # """The neptune project name. Default is "instinctlab"."""

    # wandb_project: str = "instinctlab"
    # """The wandb project name. Default is "instinctlab"."""

    ##
    # 加载参数
    ##

    # 是否恢复训练，默认 False
    resume: bool = False
    """Whether to resume. Default is False."""

    # 要加载的运行目录，默认为 ".*"（所有）
    load_run: str | None = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    # 要加载的检查点文件，默认为 "model_.*.pt"
    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """