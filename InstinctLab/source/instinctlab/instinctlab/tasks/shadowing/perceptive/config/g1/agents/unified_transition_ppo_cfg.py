"""Ordinary PPO warm-start configuration; no teachers, AMP, or imitation loss."""
from isaaclab.utils import configclass
from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlActorCriticCfg, InstinctRlNormalizerCfg, InstinctRlOnPolicyRunnerCfg, InstinctRlPpoAlgorithmCfg,
)


@configclass
class PolicyCfg(InstinctRlActorCriticCfg):
    class_name = "ActorCritic"
    actor_hidden_dims = [512,256,128]
    critic_hidden_dims = [512,256,128]
    activation = "elu"
    init_noise_std = 0.2


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "WarmStartPPO"
    critic_only: bool = True
    actor_learning_rate: float = 3e-5
    critic_learning_rate: float = 3e-4
    value_loss_coef = 1.
    use_clipped_value_loss = True
    clip_param = 0.1
    entropy_coef = 0.
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 3e-4
    optimizer_class_name = "Adam"
    schedule = "fixed"
    gamma = 0.995
    lam = 0.95
    desired_kl = None
    max_grad_norm = 1.


@configclass
class NormalizersCfg:
    policy: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()
    critic: InstinctRlNormalizerCfg = InstinctRlNormalizerCfg()


@configclass
class G1UnifiedTransitionPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    policy: PolicyCfg = PolicyCfg()
    algorithm: AlgorithmCfg = AlgorithmCfg()
    normalizers: NormalizersCfg = NormalizersCfg()
    num_steps_per_env = 24
    max_iterations = 50
    save_interval = 10
    log_interval = 1
    experiment_name = "g1_unified_transition_ppo"
    run_name = "critic_warmup"
    resume = False
    load_run = None
    ckpt_manipulator = None
