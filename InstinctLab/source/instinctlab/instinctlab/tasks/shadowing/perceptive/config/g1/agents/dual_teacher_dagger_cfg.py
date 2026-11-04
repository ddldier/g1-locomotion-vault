"""Runner configuration for pure dual-teacher DAgger."""

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import InstinctRlActorCriticCfg

from .instinct_rl_vault_dagger_cfg import (
    DaggerNormalizersCfg,
    G1PerceptiveVaultDaggerRunnerCfg,
    LOCOMOTION_TEACHER_CHECKPOINT,
    VaultDaggerAlgorithmCfg,
)


@configclass
class DualTeacherAlgorithmCfg(VaultDaggerAlgorithmCfg):
    class_name = "DualTeacherDagger"
    locomotion_teacher_checkpoint = LOCOMOTION_TEACHER_CHECKPOINT


@configclass
class G1DualTeacherDaggerRunnerCfg(G1PerceptiveVaultDaggerRunnerCfg):
    algorithm: DualTeacherAlgorithmCfg = DualTeacherAlgorithmCfg()
    experiment_name = "g1_loco_vault_dual_dagger"
    load_run = None


@configclass
class G1AutonomousVaultDaggerRunnerCfg(G1DualTeacherDaggerRunnerCfg):
    """Fresh DAgger run for the vision-conditioned autonomous Student."""

    experiment_name = "g1_loco_vault_autonomous_dagger"
    load_run = None


@configclass
class UnifiedHeightPolicyCfg(InstinctRlActorCriticCfg):
    class_name = "ActorCritic"
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


@configclass
class G1UnifiedHeightDaggerRunnerCfg(G1DualTeacherDaggerRunnerCfg):
    policy: UnifiedHeightPolicyCfg = UnifiedHeightPolicyCfg()
    experiment_name = "g1_loco_vault_unified_height_dagger"
    load_run = None
    resume = False
