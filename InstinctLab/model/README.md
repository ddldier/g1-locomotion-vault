# InstinctLab 当前 Teacher / Student 模型归档

本目录保存当前双 Teacher DAgger 实验实际使用的三个 checkpoint，以及对应训练 run 的 YAML 配置。模型文件没有重新训练或转换，只做了复制归档。

## 1. 文件清单

| 目录 | 文件 | 用途 | 原始 run |
|---|---|---|---|
| `teachers/locomotion_pelvis_35000/` | `model_35000.pt` | pelvis-root locomotion Teacher | `logs/instinct_rl/g1_unified_omni_loco_pelvis_rev10/20261021_101015_G1UnifiedOmniLocoPelvisRev10_obs623_scale1_stand02_from24000_from20261020_115051/` |
| `teachers/vault_pelvis_15001/` | `model_15001.pt` | pelvis-root Vault Teacher | `logs/instinct_rl/g1_perceptive_vault_pelvis_corrected/20261019_192913_g1PerceptiveVaultPelvisPhpTeacher_concatMotionBins/` |
| `student/autonomous_dagger_vault1ms_14000/` | `model_14000.pt` | 双 Teacher DAgger 的视觉 Student | `logs/instinct_rl/g1_loco_vault_autonomous_dagger_vault1ms/20261023_230856_g1_loco_vault_autonomous_dagger/` |
| `ppo/transition_1502/` | `model_1502.pt` | 高度扫描历史观测的 Walk/Vault/Recovery PPO | `logs/instinct_rl/g1_unified_transition_ppo_xonly_no_upright/20261104_170358_g1_unified_transition_ppo_critic_warmup_from20261104_154802/` |

每个目录中的 `agent.yaml` 和 `env.yaml` 是对应 checkpoint 保存时的训练配置。`model_*.pt` 中还包含网络参数、训练状态和运行时使用的归一化器；只分析网络时优先读取 checkpoint，复现环境时同时读取同目录 YAML。

## 2. 当前模型关系

```text
locomotion Teacher (pelvis, model_35000.pt) ─┐
                                             ├─ 在线双 Teacher DAgger ─> Student (model_14000.pt)
Vault Teacher (pelvis, model_15001.pt) ──────┘
```

当前 DAgger runner 在 `source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vault_dagger_cfg.py` 中引用这两个 Teacher。代码也支持用环境变量 `LOCOMOTION_TEACHER_CHECKPOINT` 和 `VAULT_TEACHER_CHECKPOINT` 覆盖路径。

公开仓库副本默认从本目录的 `teachers/` 读取两个 Teacher，不需要原机器的绝对日志路径。PPO `transition_1502` 是后续独立实验，Actor 使用高度扫描历史；它与上表的深度相机 DAgger Student 不是同一份观测配置。PPO checkpoint 的 `env.yaml` / `agent.yaml` 保留训练时原始参数，播放时请使用当前代码配置，不要指定 `--env_cfg`。

## 3. 观测和速度约定

- 两个 Teacher 都使用各自的特权观测和高度扫描；locomotion 使用 527 点扫描，Vault 使用 279 点扫描。
- 当前 Autonomous Student 的 actor 输入是本体状态、上一动作、速度命令和 `48x32` 深度相机图 `distance_to_image_plane_noised`。Student 不直接接收高度扫描，也不直接接收 Teacher 的参考动作。
- Vault 样本的 Student 速度命令固定为 `(1.0, 0.0, 0.0)`；locomotion 样本仍使用 `base_velocity` 的命令采样。
- 当前混合环境按环境编号奇偶固定分配任务。Student actor 不含 `task_id`，任务区分依赖视觉、本体状态和命令；`task_id` 只保留在 critic / Teacher 路由侧。

## 4. 复现时还需要的文件

这些不是 checkpoint 本身，但要复现播放或继续训练：

1. `source/instinctlab/instinctlab/tasks/locomotion/config/g1/omni_loco_env_cfg.py` 及其 agent 配置；
2. `source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/` 下的 Vault、双 Teacher DAgger 环境和 agent 配置；
3. 与 `env.yaml` 中路径一致的 URDF/USD、VaultBox、相机配置和 Vault `.npz` 动作数据；
4. 若要复盘训练曲线，保留原 run 目录中的 `events.out.tfevents*`；若要审计当时的代码，保留原 run 目录中的 `git/` diff。

原始训练日志没有复制进 `model/`，因为它们体积较大且不是加载 checkpoint 的必需项。原始路径仍记录在上表中。

## 5. 校验

归档后可在 InstinctLab 根目录执行：

```bash
find model -type f -printf '%p %s bytes\\n' | sort
sha256sum model/teachers/locomotion_pelvis_35000/model_35000.pt \\
           model/teachers/vault_pelvis_15001/model_15001.pt \\
           model/student/autonomous_dagger_vault1ms_14000/model_14000.pt \\
           model/ppo/transition_1502/model_1502.pt
```
