# G1 locomotion and vault research

This repository packages a modified InstinctLab environment and the matching
`instinct_rl` algorithm code used to train a Unitree G1 pelvis-root locomotion
teacher, a vault teacher, a dual-teacher DAgger student, and a subsequent PPO
transition policy. It is a research snapshot, not an official Project Instinct
release.

## Layout

| Path | Contents |
| --- | --- |
| `InstinctLab/` | G1 assets, motion data, environments, training/play scripts, project handoff report, and selected checkpoints |
| `instinct_rl/` | Modified on-policy runner, DAgger, WarmStartPPO, and WalkAnchorPPO algorithms |
| `InstinctLab/model/` | Selected teachers, DAgger student, PPO checkpoint, and per-run YAML snapshots |

The original training logs and TensorBoard event files are not included. The
`model/` directory contains the checkpoints needed to inspect the main policy
lineage. The original InstinctLab attributes store the four checkpoints and
one PNG asset in Git LFS; install Git LFS before cloning to fetch their data.

## Requirements and installation

- Clone the combined repository:

```bash
git lfs install
git clone git@github.com:ddldier/g1-locomotion-vault.git
cd g1-locomotion-vault
```

- Linux with an NVIDIA GPU and an Isaac Sim/Isaac Lab installation compatible
  with this snapshot. The work was developed with Isaac Sim 5.1, Python 3.11,
  and a local Isaac Lab checkout near the upstream `f73c331738` revision.
- Activate the Python environment in which Isaac Lab and Isaac Sim work.
- Install both projects in editable mode from this repository's root:

```bash
python -m pip install -e ./instinct_rl
python -m pip install -e ./InstinctLab/source/instinctlab
cd InstinctLab
```

Isaac Lab/Isaac Sim are external prerequisites and are not vendored here. If
the upstream API has changed, use a compatible Isaac Lab checkout before
attempting to run the environments.

## Policy lineage

1. `model/teachers/locomotion_pelvis_35000/model_35000.pt`: velocity-controlled
   pelvis-root locomotion teacher.
2. `model/teachers/vault_pelvis_15001/model_15001.pt`: vault motion tracking
   teacher trained against the pelvis-root retargeted NPZ.
3. `model/student/autonomous_dagger_vault1ms_14000/model_14000.pt`: dual-teacher
   DAgger student. This is a separate earlier student experiment.
4. `model/ppo/transition_1502/model_1502.pt`: current transition PPO experiment.
   It uses a height-scan history and a 29-joint action, not the earlier
   depth-camera student observation contract.

The DAgger teacher defaults resolve to the two files under `model/teachers/`.
Override them with `LOCOMOTION_TEACHER_CHECKPOINT` and
`VAULT_TEACHER_CHECKPOINT` when using other weights. The legacy torso teacher
entry still requires `LEGACY_VAULT_TEACHER_CHECKPOINT` if used.

## Play the transition policy

From `InstinctLab/`, run:

```bash
python scripts/instinct_rl/play.py \
  --task Instinct-Unified-Transition-PPO-Play-G1-v0 \
  --num_envs 1 \
  --load_run "$PWD/model/ppo/transition_1502" \
  --checkpoint model_1502.pt \
  --keyboard
```

The policy decides when to vault from height-scan history; the keyboard sets
the velocity target. Before successful vault recovery, an obstacle episode
uses a fixed body-forward 1 m/s target. After recovery, the target can be
changed with `W/S` (forward/back), `A/D` (lateral), `Q/E` (yaw), and `Space`
(stop). Lateral/yaw control after vault was not trained broadly enough to
claim reliable performance. Playback lasts up to 30 s; the training episode
horizon is 12 s. Do not add `--env_cfg` to this command: the stored YAML is a
training snapshot, while current playback behavior comes from the live config.

## Continue PPO training

`--max_iterations` is the number of **additional** iterations when resuming.
For example, to train 1000 more iterations from this snapshot:

```bash
python scripts/instinct_rl/train.py \
  --task Instinct-Unified-Transition-PPO-G1-v0 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --logroot "$PWD/logs/instinct_rl/g1_unified_transition_ppo_continue" \
  --resume \
  --load_run "$PWD/model/ppo/transition_1502" \
  --checkpoint model_1502.pt \
  --unfreeze_actor \
  --headless
```

The transition task ends on failure or timeout. A completed vault gives one
reward and records success, then the episode continues. See
`InstinctLab/PROJECT_HANDOFF_REPORT.md` and `InstinctLab/model/README.md` for
the development history and checkpoint provenance.

## Provenance and licenses

- `InstinctLab/` is based on
  [Project Instinct's InstinctLab](https://github.com/project-instinct/InstinctLab)
  at `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057` plus local research changes.
- `instinct_rl/` is based on
  [Project Instinct's instinct_rl](https://github.com/project-instinct/instinct_rl)
  at `3a2844890387eda6d93a4465cdef9e767aba8546` plus local research changes.
- The original `LICENSE`, contributor, and attribution files remain within
  each project. The upstream repositories describe noncommercial use; review
  those license files and asset provenance before reuse.

This repository does not include the Isaac Lab or Isaac Sim source trees.

## Locomotion and Vault Videos

- [Video 1](docs/showcase/loco-vault/loco_vault_video_01.mp4)
- [Video 2](docs/showcase/loco-vault/loco_vault_video_02.webm)
