"""Isolated PPO+walking-anchor experiment. Original task configuration stays intact."""
from isaaclab.utils import configclass
from .unified_transition_ppo_cfg import AlgorithmCfg, G1UnifiedTransitionPPORunnerCfg


@configclass
class WalkAnchorAlgorithmCfg(AlgorithmCfg):
    class_name = "WalkAnchorPPO"
    walk_anchor_loss_coef: float = 0.1


@configclass
class G1WalkAnchorPPORunnerCfg(G1UnifiedTransitionPPORunnerCfg):
    algorithm: WalkAnchorAlgorithmCfg = WalkAnchorAlgorithmCfg()
    experiment_name = "g1_unified_transition_walk_anchor_ppo"
    run_name = "walk_anchor_critic_warmup_v1"
    save_interval = 100
