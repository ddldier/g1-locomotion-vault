# Shadowing Task

## Basic Usage Guidelines

### BeyondMimic Shadowing

**Task ID:** `Instinct-BeyondMimic-Plane-G1-v0`

This is an exact replication of the BeyondMimic training configuration.

1. Go to `beyondmimic/config/g1/beyondmimic_plane_cfg.py` and set the `MOTION_NAME`, `_hacked_selected_file_`, `AmassMotionCfg.path` to the motion you want to use.

    - `MOTION_NAME`: A identifier for you to remember which motion you are using.
    - `AmassMotionCfg.path`: The folder path to where you store the motion files.
    - `_hacked_selected_file_`: The filename of the motion you want to use, relative to the `AmassMotionCfg.path` folder.

2. Train the policy:
```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-BeyondMimic-Plane-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):
```bash
python scripts/instinct_rl/play.py --task=Instinct-BeyondMimic-Plane-G1-v0 --load_run=<run_name>
```

### Whole Body Shadowing

**Task ID:** `Instinct-Shadowing-WholeBody-Plane-G1-v0`

1. Go to `whole_body/config/g1/plane_shadowing_cfg.py` and set the `MOTION_NAME`, `_path_`, `_hacked_selected_files_` to the motion you want to use.

    - `MOTION_NAME`: A identifier for you to remember which motion you are using.
    - `_path_`: The folder path to where you store the motion files.
    - `_hacked_selected_files_`: The filenames of the motion you want to use, relative to the `_path_` folder.

2. Train the policy:
```bash
python scripts/instinct_rl/train.py --headless --task=Instinct-Shadowing-WholeBody-Plane-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):
```bash
python scripts/instinct_rl/play.py --task=Instinct-Shadowing-WholeBody-Plane-G1-v0 --load_run=<run_name>
```

### Perceptive Shadowing

**Task IDs:**
- `Instinct-Perceptive-Shadowing-G1-v0` (Deep Whole-body Parkour)

1. Go to `perceptive/config/g1/perceptive_shadowing_cfg.py` and set the `MOTION_FOLDER` to the motion you want to use. The `motion_buffer` and corresponding terrain generator will read the `MOTION_FOLDER` and corresponding metadata.yaml file.

    - `MOTION_FOLDER`: The folder path to where you store the motion files.

2. Train the policy:
```bash
# PPO version
python scripts/instinct_rl/train.py --headless --task=Instinct-Perceptive-Shadowing-G1-v0
```

3. Play trained policy (load_run must be provided, absolute path is recommended, or use `--no_resume` to visualize untrained policy):
```bash
# PPO version
python source/instinctlab/instinctlab/tasks/shadowing/play.py --task=Instinct-Perceptive-Shadowing-G1-v0 --load_run=<run_name>
```

### XU Speed Vault Teacher（翻箱子）

已增量整合 `instinct_xu` 的第一阶段：G1 Speed Vault 数据、privileged Teacher 和 PPO。
详细改动、数据说明及云端运行步骤见 [XU_VAULT_INTEGRATION.md](XU_VAULT_INTEGRATION.md)。

- 训练：`Instinct-Perceptive-Vault-G1-v0`
- 播放：`Instinct-Perceptive-Vault-G1-Play-v0`
- Teacher 输入参考动作、本体状态和高度扫描，不使用深度相机。
- NPZ 已随任务放入 `beyondmimic/config/g1/motions/vault/`，箱体由 NPZ 参数生成；
  本任务不需要修改通用 `perceptive_shadowing_cfg.py` 的 `MOTION_FOLDER`，也不需要 `metadata.yaml`。
- 独立日志目录：`logs/instinct_rl/g1_perceptive_vault_shadowing/`。

在已配置好依赖和 GPU 的环境中，从仓库根目录运行：

```bash
python scripts/instinct_rl/train.py \
  --task Instinct-Perceptive-Vault-G1-v0 \
  --num_envs 32 --max_iterations 2 --device cuda:0 --headless
```

```bash
python scripts/instinct_rl/play.py \
  --task Instinct-Perceptive-Vault-G1-Play-v0 \
  --num_envs 1 --load_run /absolute/path/to/teacher_run
```

## Common Options

- `--num_envs`: Number of parallel environments (default varies by task)
- `--max_iterations`: Training iterations (default varies by task)
- `--load_run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
