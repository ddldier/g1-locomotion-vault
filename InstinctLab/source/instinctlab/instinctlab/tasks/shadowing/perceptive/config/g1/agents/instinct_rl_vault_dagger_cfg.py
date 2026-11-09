"""Vault 在线蒸馏入口；独立于原来的 Student PPO 配置。"""

# 标准库：用于读取环境变量，从而允许通过外部环境覆盖 Teacher checkpoint 路径。
import os

# IsaacLab 的配置类装饰器：用于把普通类声明为可组合的配置对象。
from isaaclab.utils import configclass

# InstinctRL 的 normalizer 配置：用于 policy / critic 观测归一化。
from instinctlab.utils.wrappers.instinct_rl import InstinctRlNormalizerCfg

# 复用 Student PPO 的算法基础配置和 Runner 配置。
# 当前文件只针对 Vault 在线蒸馏做增量覆盖，不直接修改原 Student PPO 配置。
from .instinct_rl_vault_student_cfg import AlgorithmCfg, G1PerceptiveVaultStudentPPORunnerCfg


# Teacher policy 的观测项及其形状定义。
# 该字典用于构造 Teacher 的 obs_format，确保 Student/蒸馏侧按 Teacher 所需观测组织输入。
TEACHER_POLICY_TERMS = {
    # 参考关节位置 / 速度：1 个 batch 维度 + 29 个关节维度。
    'joint_pos_ref': (1, 29), 'joint_vel_ref': (1, 29),
    # 参考位置 / 旋转：位置 3 维，旋转使用 6 维表示。
    'position_ref': (1, 3), 'rotation_ref': (1, 6),
    # 高度扫描观测：279 维。
    'height_scan': (279,),
    # 全局骨盆位置 / 速度：各 3 维。
    'global_pelvis_pos': (3,), 'global_pelvis_vel': (3,),
    # 机器人本体感知观测：线速度、投影重力、角速度各 3 维。
    'base_lin_vel': (3,), 'projected_gravity': (3,), 'base_ang_vel': (3,),
    # 关节位置 / 速度 / 上一动作：各 29 维，对应 29 个关节。
    'joint_pos': (29,), 'joint_vel': (29,), 'last_action': (29,),
}

# Pin the two pelvis-root experts selected for distillation.  The current
# VaultDaggerTPPO implementation consumes only the Vault checkpoint; the
# locomotion checkpoint is reserved for the upcoming multi-teacher router.
#
# 中文说明：
# 这里固定两个以 pelvis 为根节点的专家 checkpoint，用于后续蒸馏。
# 当前 VaultDaggerTPPO 实际只消费 Vault Teacher checkpoint；
# locomotion Teacher checkpoint 预留给后续 multi-teacher router 使用。
# 通过环境变量可覆盖默认路径，便于不同实验机器或不同 checkpoint 复用代码。
LOCOMOTION_TEACHER_CHECKPOINT = os.environ.get(
    'LOCOMOTION_TEACHER_CHECKPOINT',
    # 默认 locomotion Teacher checkpoint 路径。
    '/root/gpufree-data/InstinctLab/logs/instinct_rl/g1_unified_omni_loco_pelvis_rev10/'
    '20261021_101015_G1UnifiedOmniLocoPelvisRev10_obs623_scale1_stand02_from24000_from20261020_115051/'
    'model_35000.pt',
)

# Vault Teacher checkpoint 路径。
# 当前蒸馏入口主要加载该 checkpoint；同样支持通过 VAULT_TEACHER_CHECKPOINT 环境变量覆盖。
VAULT_TEACHER_CHECKPOINT = os.environ.get(
    'VAULT_TEACHER_CHECKPOINT',
    '/root/gpufree-data/InstinctLab/logs/instinct_rl/g1_perceptive_vault_pelvis_corrected/'
    '20261019_192913_g1PerceptiveVaultPelvisPhpTeacher_concatMotionBins/model_15001.pt',
)


@configclass
class VaultDaggerAlgorithmCfg(AlgorithmCfg):
    """Vault DAgger / 在线蒸馏算法配置。

    该配置继承基础 AlgorithmCfg，只覆盖与 Teacher 标注、蒸馏损失、
    Teacher 执行概率和 Teacher policy 结构相关的字段。
    """

    # 指定使用 VaultDaggerTPPO 算法实现。
    class_name = 'VaultDaggerTPPO'

    # 第一阶段只拟合 Teacher 动作，不训练 PPO value/surrogate loss。
    # 因此关闭 PPO 训练路径，仅走蒸馏损失。
    using_ppo = False

    # 学习率调度方式固定，不随训练过程变化。
    schedule = 'fixed'

    # 蒸馏阶段学习率。
    learning_rate = 3e-4

    # 蒸馏目标使用 MSE 的 sum 形式。
    distill_target = 'mse_sum'

    # 蒸馏损失系数为 1.0，即直接使用蒸馏损失作为主要优化目标。
    distillation_loss_coef = 1.0

    # 动作标签不从 sample 中直接读取，而是通过 critic obs 由 Teacher 标注。
    action_labels_from_sample = False
    label_action_with_critic_obs = True

    # 按 episode 选择执行 Teacher 或 Student。
    # teacher_act_prob='linear' 表示 Teacher 执行概率线性衰减；
    # update_times_scale=10000 表示经过约 10000 次更新后概率降到 0。
    teacher_act_prob = 'linear'
    update_times_scale = 10000

    # 保留最近 4 个 rollout 的标注；buffer_dilation_ratio=2.0 表示有限队列聚合，
    # 不是无限离线数据集，避免标注缓存无限增长。
    buffer_dilation_ratio = 2.0

    # 当前蒸馏使用的 Vault Teacher checkpoint。
    teacher_checkpoint = VAULT_TEACHER_CHECKPOINT

    # 与 pelvis Student 入口绑定；加载 Teacher 时检查 Teacher 保存的 FK 根节点。
    expected_teacher_root = 'pelvis'

    # Teacher policy 类名，与 checkpoint 中保存的网络结构对应。
    teacher_policy_class_name = 'ActorCritic'

    # Teacher policy 网络与观测格式配置。
    teacher_policy = {
        # Actor / Critic 初始化噪声标准差。
        'init_noise_std': 1.0,
        # Actor 隐藏层维度。
        'actor_hidden_dims': [512, 256, 128],
        # Critic 隐藏层维度。
        'critic_hidden_dims': [512, 256, 128],
        # 激活函数使用 ELU。
        'activation': 'elu',
        # 动作维度：29 个关节。
        'num_actions': 29,
        # reward 输出维度。
        'num_rewards': 1,
        # 观测格式：
        # policy 使用 TEACHER_POLICY_TERMS；
        # critic 在 policy 观测基础上增加 link_pos 和 link_rot。
        'obs_format': {
            'policy': TEACHER_POLICY_TERMS,
            'critic': {**TEACHER_POLICY_TERMS, 'link_pos': (18, 3), 'link_rot': (18, 6)},
        },
    }


@configclass
class DaggerNormalizersCfg:
    """DAgger / 蒸馏 Runner 使用的 normalizer 配置。"""

    # policy 使用默认的 InstinctRlNormalizerCfg 进行归一化。
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()

    # critic 必须保持原始值：Teacher 内部使用自己的 checkpoint normalizer。
    # 因此这里不对 critic 额外配置 normalizer，避免对 critic obs 做二次归一化。


@configclass
class G1PerceptiveVaultDaggerRunnerCfg(G1PerceptiveVaultStudentPPORunnerCfg):
    """G1 感知 Vault 在线蒸馏 Runner 配置。

    继承 Student PPO Runner 配置，仅替换 algorithm、normalizers 和实验名等
    蒸馏相关字段，其余 Runner 环境配置沿用父类。
    """

    # 使用 Vault DAgger / 蒸馏算法配置覆盖父类 algorithm。
    algorithm: VaultDaggerAlgorithmCfg = VaultDaggerAlgorithmCfg()

    # 使用 DAgger 专用 normalizer 配置覆盖父类 normalizers。
    normalizers: DaggerNormalizersCfg = DaggerNormalizersCfg()

    # 实验名称，用于日志目录和实验区分。
    experiment_name = 'g1_perceptive_vault_dagger'

    # 最大训练迭代次数；12001 表示包含 0 到 12000 共 12001 次迭代。
    max_iterations = 12001


@configclass
class LegacyVaultDaggerAlgorithmCfg(VaultDaggerAlgorithmCfg):
    """旧版 torso 入口的 Vault DAgger 算法配置。

    旧 torso 入口不能复用 pelvis 的 VAULT_TEACHER_CHECKPOINT，
    因此这里单独覆盖 teacher_checkpoint 和 expected_teacher_root。
    """

    # 旧 torso 入口使用独立环境变量 LEGACY_VAULT_TEACHER_CHECKPOINT。
    # 注意：不能继承 pelvis 的 VAULT_TEACHER_CHECKPOINT。
    teacher_checkpoint = os.environ.get(
        'LEGACY_VAULT_TEACHER_CHECKPOINT',
        # 旧 torso Teacher checkpoint 默认路径。
        '/root/gpufree-data/IsaacLab/logs/instinct_rl/g1_perceptive_vault_shadowing/'
        '20261010_222813_g1PerceptiveVaultPhpTeacher_concatMotionBins/model_10001.pt',
    )

    # 旧 torso 入口对应的 Teacher FK 根节点为 torso_link。
    expected_teacher_root = 'torso_link'


@configclass
class G1PerceptiveVaultLegacyDaggerRunnerCfg(G1PerceptiveVaultDaggerRunnerCfg):
    """旧版 torso 入口的 Vault DAgger Runner 配置。

    继承通用 Vault Dagger Runner 配置，仅将 algorithm 替换为 Legacy 版本，
    从而使用旧 torso Teacher checkpoint 与 torso_link 根节点校验。
    """

    # 使用 LegacyVaultDaggerAlgorithmCfg 覆盖父类 algorithm。
    algorithm: LegacyVaultDaggerAlgorithmCfg = LegacyVaultDaggerAlgorithmCfg()