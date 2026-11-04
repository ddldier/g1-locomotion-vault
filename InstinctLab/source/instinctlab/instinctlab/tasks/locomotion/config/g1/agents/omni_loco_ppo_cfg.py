"""From-scratch PPO configuration for the unified G1 locomotion teacher."""

from isaaclab.utils import configclass

from .instinct_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1OmniLocoPPORunnerCfg(G1FlatPPORunnerCfg):
    max_iterations = 5000
    save_interval = 1000
    experiment_name = "g1_unified_omni_loco"
    load_run = None

    def __post_init__(self):
        super().__post_init__()
        self.resume = False
        self.load_run = None
        self.run_name = "from_scratch_obs623_scale1"
