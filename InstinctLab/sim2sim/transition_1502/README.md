# Unified Transition PPO MuJoCo / C++ playback

This directory is the isolated deployment workspace for pelvis-root unified
walk/vault Transition PPO checkpoints. The current default is iteration 13507;
the source repository and checkpoint remain under `/root/gpufree-data`.

## Current compatibility facts

- The current `model_13507.pt` has a deterministic actor with 2049 inputs and 29 outputs.
- The actor input order is height scan history (527 x 3), forward velocity
  command (3), projected gravity history (3 x 5), pelvis angular velocity
  history (3 x 5), joint-position history (29 x 5), joint-velocity history
  (29 x 5), and action history (29 x 5).
- History is oldest-to-newest. The first observation after reset is replicated
  into all history slots. `height_scan` is clipped to [-1, 1] after subtracting
  0.5 m from sensor-height minus hit-height.
- The observation normalizer is part of the ONNX actor export. The IsaacLab
  action is a 29-joint position target with scale 1, default-joint offset, and
  no action clipping.
- MuJoCo's native MJCF order differs from the policy action order. Both generated
  model and controller must map by joint name, never by array position.
- The official InstinctMJ model is torso-root and uses its own MJCF asset. It is
  not the correct robot model for this pelvis-root checkpoint. The generator
  below uses the installed InstinctLab rev10 URDF and activates its commented
  floating-base joint in a generated copy.

## Setup and artifacts

The official InstinctMJ checkout and its isolated uv environment are at
`/root/gpufree-data/InstinctMJ`. Run these from that directory:

```bash
uv run python sim2sim/transition_1502/export_policy.py
uv run python sim2sim/transition_1502/generate_model.py
```

Outputs are written under `sim2sim/transition_1502/artifacts/` and
`sim2sim/transition_1502/models/`. The generated MJCF includes the rev10 meshes,
a plane, the NPZ-derived box dimensions, and the transition task's Isaac PD
gains, torque limits, and joint armature. The box is placed 1.5 m ahead of the
pelvis by default; use `--box-distance` to change that start geometry.
The generated model also includes a directional key light. Playback starts
with a tracking camera centered near the pelvis, at 3.5 m so the robot is large
enough to inspect while the near edge of the vault box remains in view.

The ONNX exporter checks the network and normalizer dimensions, exports the
observation normalizer and deterministic Actor MLP, then compares ONNX Runtime
against the PyTorch actor on randomized observations. This confirms network
export parity, not simulator observation parity.

## Two-terminal playback

The C++ ONNX Runtime server is already built. Rebuild it only if its source or
ONNX Runtime installation changes:

```bash
cmake -S sim2sim/transition_1502/cpp \
  -B sim2sim/transition_1502/cpp/build \
  -DONNXRUNTIME_ROOT=/root/gpufree-data/unitree_rl_lab_review_20261021/deploy/thirdparty/onnxruntime-linux-x64-1.22.0
cmake --build sim2sim/transition_1502/cpp/build -j2
```

Terminal 1 starts the C++ policy service:

```bash
sim2sim/transition_1502/cpp/build/transition_1502_policy_server \
  sim2sim/transition_1502/artifacts/model_13507_actor.onnx
```

Terminal 2 starts the MuJoCo viewer and simulation:

```bash
uv run python sim2sim/transition_1502/mujoco_udp_sim.py
```

The simulation has no automatic reset. Press **R** in the MuJoCo window to
return to the initial vault setup; this clears robot pose, velocity, command,
and all observation-history buffers. Closing the viewer exits. For a headless
link check, add `--headless --steps 200` to the second command.
The simulation sends one 2049-float observation at each 20 ms policy step; the
C++ process replies with 29 joint actions. The command is held at 1 m/s until the
pelvis passes the box rear plus 0.15 m. After crossing, W/up increases forward
speed by 0.1 m/s, S/down decreases it, Q/E adjusts yaw command by 0.1 rad/s up
to ±0.6 rad/s, and Space stops. Playback runs at 50 Hz when the GUI is open.
The scene uses a fixed, axis-aligned box; randomized obstacle distances and
orientations are not reproduced here.

## IsaacLab playback with manual reset

`play.py` also has a manual-reset mode. It keeps calling the transition failure
state update but returns no failure termination, disables the timeout
termination, and calls `env.reset()` only after **R** is pressed. Run this from
`/root/gpufree-data/InstinctLab` in the IsaacLab environment:

```bash
RUN=/root/gpufree-data/InstinctLab/logs/instinct_rl/g1_unified_transition_ppo/20261109_182619_g1_unified_transition_ppo_unified_vault_yaw_mix_from11506_from20261109_174214

python scripts/instinct_rl/play.py \
  --task Instinct-Unified-Transition-PPO-Play-G1-v0 \
  --num_envs 1 \
  --load_run "$RUN" \
  --checkpoint model_13507.pt \
  --keyboard \
  --manual_reset_only \
  --keyboard_linvel_step 1.0 \
  --keyboard_angvel 0.6
```

Click the Isaac Sim viewport to give it keyboard focus. **R** resets the
episode; **W/S**, **A/D**, **Q/E**, and **Space** control velocity commands.
During approach and vault the task keeps the forward command at 1 m/s. After
recovery, keyboard velocity commands are used.

## Validation status

- `model_13507.pt` exports to ONNX and passes randomized PyTorch/ONNX Runtime
  parity checks (maximum absolute error about `4.8e-6`).
- The generated rev10 model reloads with pelvis root, 29 actuators, and expected
  `nq=36`, `nv=35`, `nu=29` dimensions.
- A 200-policy-step, headless two-process run completed with `model_13507.pt`.
  The robot passed the box and remained upright at about 0.77 m pelvis height
  after 4 s. Manual reset state/history reinitialization also passed a direct
  MuJoCo check; interactive R input still needs a GUI test.
- The ray scan is analytical for this task's flat plane and axis-aligned static
  box; it matches the configured 31 x 17 grid and the checkpoint's initial scan
  mean, but has not yet been compared frame-by-frame against IsaacLab.
- The MuJoCo viewer can open, sync, and close in this container. Transfer
  quality over long rollouts and interactive keyboard control still need
  validation.
