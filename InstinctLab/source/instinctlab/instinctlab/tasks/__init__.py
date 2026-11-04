"""The current G1 pipeline: two Teachers and one height-scan Student."""

import gymnasium as gym

_LOCO = "instinctlab.tasks.locomotion.config.g1"
_VAULT = "instinctlab.tasks.shadowing.perceptive.config.g1"

# Only register tasks used by the current project. Keep configuration imports
# lazy so listing tasks does not recursively register unrelated experiments.
_TASKS = (
    (
        "Instinct-Unified-OmniLoco-Pelvis-G1-v0",
        "Instinct-Unified-OmniLoco-Pelvis-G1-Play-v0",
        f"{_LOCO}.omni_loco_env_cfg",
        "G1OmniLocoPelvisEnvCfg", "G1OmniLocoPelvisEnvCfg_PLAY",
        f"{_LOCO}.agents.omni_loco_ppo_cfg:G1OmniLocoPPORunnerCfg",
        f"{_LOCO}.omni_loco_env_cfg:G1OmniLocoPelvisTeacherEnv",
    ),
    (
        "Instinct-Perceptive-Vault-Pelvis-G1-v0",
        "Instinct-Perceptive-Vault-Pelvis-G1-Play-v0",
        f"{_VAULT}.perceptive_vault_cfg",
        "G1PerceptiveVaultPelvisEnvCfg", "G1PerceptiveVaultPelvisEnvCfg_PLAY",
        f"{_VAULT}.agents.instinct_rl_vault_cfg:G1PerceptiveVaultPPORunnerCfg",
        "instinctlab.envs:InstinctRlEnv",
    ),
    (
        "Instinct-Unified-Height-Dagger-G1-v0",
        "Instinct-Unified-Height-Play-G1-v0",
        f"{_VAULT}.dual_teacher_dagger_env_cfg",
        "G1UnifiedHeightDaggerEnvCfg", "G1UnifiedHeightDaggerEnvCfg_PLAY",
        f"{_VAULT}.agents.dual_teacher_dagger_cfg:G1UnifiedHeightDaggerRunnerCfg",
        "instinctlab.envs:InstinctRlEnv",
    ),
)

for _train_id, _play_id, _module, _train_cfg, _play_cfg, _runner, _entry in _TASKS:
    for _task_id, _config in ((_train_id, _train_cfg), (_play_id, _play_cfg)):
        gym.register(
            id=_task_id,
            entry_point=_entry,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{_module}:{_config}",
                "instinct_rl_cfg_entry_point": _runner,
            },
        )

# BEGIN unified_transition_ppo
_TRANSITION_MODULE = "instinctlab.tasks.shadowing.perceptive.config.g1.unified_transition_ppo_env_cfg"
_TRANSITION_AGENT = "instinctlab.tasks.shadowing.perceptive.config.g1.agents.unified_transition_ppo_cfg:G1UnifiedTransitionPPORunnerCfg"
for _transition_id, _transition_cfg in (
    ("Instinct-Unified-Transition-PPO-G1-v0", "G1UnifiedTransitionEnvCfg"),
    ("Instinct-Unified-Transition-PPO-Play-G1-v0", "G1UnifiedTransitionEnvCfg_PLAY"),
):
    gym.register(
        id=_transition_id,
        entry_point=f"{_TRANSITION_MODULE}:G1UnifiedTransitionEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point":f"{_TRANSITION_MODULE}:{_transition_cfg}",
                "instinct_rl_cfg_entry_point":_TRANSITION_AGENT},
    )
# END unified_transition_ppo

# BEGIN walk_anchor_ppo_v1
_ANCHOR_AGENT = "instinctlab.tasks.shadowing.perceptive.config.g1.agents.walk_anchor_ppo_cfg:G1WalkAnchorPPORunnerCfg"
for _anchor_id, _anchor_env_cfg in (
    ("Instinct-Unified-Transition-WalkAnchor-PPO-G1-v0", "G1UnifiedTransitionEnvCfg"),
    ("Instinct-Unified-Transition-WalkAnchor-PPO-Play-G1-v0", "G1UnifiedTransitionEnvCfg_PLAY"),
):
    gym.register(
        id=_anchor_id,
        entry_point=f"{_TRANSITION_MODULE}:G1UnifiedTransitionEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{_TRANSITION_MODULE}:{_anchor_env_cfg}",
                "instinct_rl_cfg_entry_point": _ANCHOR_AGENT},
    )
# END walk_anchor_ppo_v1
