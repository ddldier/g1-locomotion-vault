# XU Speed Vault Teacher 整合记录

本次按桌面 `InstinctLab_instinct_xu_diff_and_integration.md` 的第一阶段方案，
以当前完整 InstinctLab 为主仓库，增量引入 `instinct_xu.zip` 的 Vault Teacher。
视觉 Student、VaePpoDistill 和随包的 `instinct_rl` 留待后续阶段；OmniLoco
现已作为独立 locomotion Teacher 任务接入，但不会改变原有 Flat locomotion
任务，也不会自动进入 Vault/跑酷 Student 蒸馏。

## 新任务与数据流

```text
motions/vault/g1_speed_vault_retargeted.npz
    ├── joint_pos / base_pos_w / base_quat_w → MotionReference
    └── obstacle_size / obstacle_pos → VaultBox
                         ↓
G1PerceptiveVaultEnvCfg（平地 + 箱体 + 高度扫描，无深度相机）
                         ↓
参考动作 + 本体状态 + 高度扫描 → MLP Actor → 29 维关节目标
                         ↓
物理仿真 + 跟踪奖励 → PPO
```

| 用途 | Gym task ID |
| --- | --- |
| Teacher 训练 | `Instinct-Perceptive-Vault-G1-v0` |
| Teacher 播放 | `Instinct-Perceptive-Vault-G1-Play-v0` |

原来的 Perceptive、VAE、Box 别名、BeyondMimic、Whole Body、Parkour 和 Locomotion
注册方式保留。此前的 `Instinct-Perceptive-Box-Shadowing-G1-v0` 仍是通用感知任务别名；
现在训练学长的 Speed Vault 请使用上表的 **Vault** ID。
单独的任务 ID 本身不会隔离日志；本次新增 runner 的
`experiment_name = "g1_perceptive_vault_shadowing"` 才指定了独立日志目录。

## 文件变更

以下路径相对于 `source/instinctlab/instinctlab/`。

| 文件 | 作用 / 改动 |
| --- | --- |
| `tasks/shadowing/beyondmimic/config/g1/beyondmimic_vault_cfg.py` | 引入学长的动作加载、箱体常量和 Vault 公共配置 |
| `tasks/shadowing/beyondmimic/config/g1/motions/vault/g1_speed_vault_retargeted.npz` | 原样复制 G1 Speed Vault 动作数据 |
| `tasks/shadowing/beyondmimic/config/g1/motions/vault/g1_vault_only_motion_selection.yaml` | 单动作选择文件，改为当前 loader 支持的字符串列表与独立 `motion_weights` |
| `tasks/shadowing/perceptive/config/g1/perceptive_vault_cfg.py` | 引入 Teacher 场景、观测、接触规则、箱体随机化和 PLAY 配置 |
| `tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vault_cfg.py` | 引入标准 PPO + MLP 及独立日志名 |
| `tasks/shadowing/perceptive/config/g1/__init__.py` | 追加两个 Vault 注册；修正旧 Box 别名的日志说明 |
| `terrains/terrain_generator.py` | 补充现有代码使用的 `import torch` |
| `tasks/shadowing/README.md` | 增加任务入口、使用说明及本文链接 |

通用 motion loader、动作管理器、已有任务配置和训练脚本没有被 donor 文件覆盖。

## 相对学长代码的适配

1. YAML 从内联 `{file, weight}` 改为：

   ```yaml
   selected_files:
     - g1_speed_vault_retargeted.npz
   motion_weights:
     - 1.0
   ```

2. `perceptive_vault_cfg.py` 的 `_apply_perceptive_vault_settings()` 同时关闭
   `curriculum.beyond_adaptive_sampling` 和 `events.bin_fail_counter_smoothing`。
   桌面方案说第一阶段关闭自适应采样，但压缩包实际代码仅在 PLAY 中关闭；
   本次显式对齐第一阶段方案。训练仍按 1 秒 motion bin 采样起始状态，
   关闭自适应权重不意味着所有训练 episode 都从第 0 帧开始。

3. Gym 注册只追加 `instinct_rl` 入口，没有复制 donor 的精简任务发现逻辑。

## 已提供的数据和默认设置

- 动作：737 帧，50 Hz，29 个关节；按帧数除以采样率约 14.74 秒。
- NPZ SHA256：`5c1743295a14c5fb280a06c80f39f2fd277927a6357294121b0e097f08f85fa9`。
- 箱体尺寸约 `(0.4, 10.0, 0.6)` 米，中心位置约 `(1.7, 0.5, 0.055)` 米。
  这些是原数据坐标，箱体中心 z 不是箱顶高度，当前未自行抬高箱体。
- NPZ 中 `obstacle_quat` 为单位四元数；当前箱体配置采用默认单位旋转。
  后续更换为旋转箱体数据时，应同步处理该字段。
- Teacher：普通 MLP `[512, 256, 128]`，Actor 和 Critic 都使用特权状态；
  18 个连杆的位姿额外进入 Critic。
- 高度扫描：`3.0 × 0.8` 米、间距 `0.1` 米，共 279 点；使用
  `MultiMeshRayCasterCfg` 同时读取地面与箱体，并跟踪箱体变换。
- Actor 无深度图，相机与相机外参随机事件被关闭。
- 动作缩放沿用学长 Teacher 的 `1.0`，不同于普通 BeyondMimic 的按关节缩放。
- 手腕接触被排除在不期望接触列表之外。
- 箱体训练默认随机 x/y 各约 ±0.05 米、yaw 约 ±45°；PLAY 固定箱体。
- 训练保留学长的小幅速度扰动；PLAY 关闭推力及多项初始化随机化。
- PLAY 关闭了三项跟踪偏差终止条件，适合观察；评估严格成功率时应明确失败判据，
  不能把 PLAY 中没有重置直接当作成功。

## 云端运行

本地仅做文件与数据整合；按用户要求不创建本地环境，实际仿真在现有云端环境执行。
同步当前仓库时要包含新 Python 文件、NPZ 和 YAML，不能只同步原来已跟踪的文件。

在云端激活已能运行 InstinctLab 的环境，并进入同步后的仓库根目录。
若当前 editable 安装尚未指向这份仓库，可执行：

```bash
python -m pip install --no-deps --no-build-isolation -e ./source/instinctlab
python -m pip show instinctlab instinct_rl isaaclab
```

`instinctlab` 应指向自己的完整仓库。Teacher 使用现有标准 PPO 库，
不要为本阶段切换到 donor 随包的 `instinct_rl`。
Isaac Lab 需提供 `isaaclab.sensors.ray_caster.MultiMeshRayCasterCfg`；本地 Isaac Lab
源码中有该类，云端版本仍需一致或兼容。

首先执行小规模训练检查；该命令尚未在本地跑通：

```bash
python scripts/instinct_rl/train.py \
  --task Instinct-Perceptive-Vault-G1-v0 \
  --num_envs 32 --max_iterations 2 --device cuda:0 --headless
```

通过后再按云端显存增加 `--num_envs` 和 `--max_iterations`。
训练结果位于 `logs/instinct_rl/g1_perceptive_vault_shadowing/`。

播放已训练策略：

```bash
python scripts/instinct_rl/play.py \
  --task Instinct-Perceptive-Vault-G1-Play-v0 \
  --num_envs 1 --load_run /absolute/path/to/teacher_run
```

如需先排查固定箱体动作，在训练命令前加：

```bash
PERCEPTIVE_VAULT_BOX_RANDOMIZATION=0 python scripts/instinct_rl/train.py \
  --task Instinct-Perceptive-Vault-G1-v0 \
  --num_envs 32 --max_iterations 2 --device cuda:0 --headless
```

该开关仅关闭箱体位姿随机化，机器人自身随机化仍沿用 Teacher 配置。

## 验证与备份记录

已完成离线检查：

- 新注册的配置路径与类名存在，旧的 6 个 Perceptive 注册保持一致。
  这是静态注册检查，不是启动 Gym/Isaac Sim 的结果。
- NPZ 与 donor 字节一致；字段形状、数值有限性和四元数单位长度检查通过。
- 29 个关节名称与当前 G1 popsicle URDF 的可动关节集合一致，18 个跟踪连杆全部存在。
- 除追加注册所在文件外，整合前已修改/未跟踪的 21 个文件与备份字节一致。

训练未验证：本地 Isaac Sim Python 启动训练脚本时报
`ModuleNotFoundError: No module named 'isaaclab.app'`；没有检测到 `instinct_rl` 和
`gymnasium`，且沙箱外 `nvidia-smi` 也无法连接 NVIDIA 驱动。
用户确认使用云端环境，要求忽略本地运行验证，不在本地安装环境。

本次未创建 Git 提交。整合前修改补丁、文件副本、donor 解包和离线检查结果暂存于：

```text
/tmp/xu-vault-integration-cuk3awpx/
├── before.patch
├── backup/
├── donor/instinct_xu/
└── offline_validation.json
```

`/tmp` 是临时存储，不应作为长期唯一备份。
来源为用户提供的 `instinct_xu.zip`；其 `source/instinctlab/LICENSE`
与当前仓库根目录 `LICENSE` 字节相同，沿用原许可证。
