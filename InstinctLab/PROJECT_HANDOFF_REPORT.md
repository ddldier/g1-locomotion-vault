# InstinctLab G1 行走、翻越与双 Teacher DAgger 项目交接报告

> 交接用途：帮助下一位同学快速理解项目目标、技术路线、已经完成的工作、历史问题、当前模型和下一步实验。
>
> 项目目录：`/root/gpufree-data/InstinctLab`
>
> 记录范围：截至 2026-10-25。训练和播放主要在 Isaac Lab 5.1 / Isaac Sim 5.1 云端环境完成。

## 1. 项目目标

本项目使用 Unitree G1 29 DoF 人形机器人，在 Isaac Lab 中训练两个基础能力：

1. **locomotion**：在平地上根据速度命令稳定行走、后退、横移和转向。
2. **Vault**：根据前方障碍物和参考动作完成翻越箱体动作。

最终目标是得到一个 Student policy：

- 平坦路面上自动执行 locomotion；
- 看到前方箱体后自动切换到翻越行为；
- 翻越结束后恢复行走；
- 运行时尽量只依赖机器人本体状态、速度命令和深度相机，不依赖 Teacher 的高度图、参考动作或 privileged state。

当前已完成的是“两个独立 Teacher + 混合环境在线 DAgger Student”的第一版。真正的连续行为“走近箱体 -> 翻越 -> 恢复行走”仍需要专门验证和完善。

## 2. 总体技术路线

```text
locomotion PPO Teacher                         pelvis-root Vault PPO Teacher
特权本体状态 + 527 点高度扫描                  参考动作 + 本体状态 + 279 点高度扫描
        │                                                   │
        └────────────── 双 Teacher 在线标注 ────────────────┘
                                │
                  Student 观察：本体状态 + 深度图 + 速度命令
                                │
                         DAgger / 行为蒸馏
                                │
                 一个 Student 同时学习走路和翻越
```

Teacher 使用高度扫描是为了获得稳定的 privileged policy。Student 的 actor 使用前视深度相机，目标是以后能够部署到只具备视觉和本体传感器的机器人上。

## 3. 已完成的主要工作

### 3.1 基础任务和数据整合

- 保留了原有 Flat locomotion、BeyondMimic、Whole Body Shadowing 和 Perceptive Shadowing 任务。
- 从 `instinct_xu` 方案整合了 Speed Vault Teacher：
  - Vault 动作文件：`g1_speed_vault_retargeted.npz`；
  - 箱体尺寸和位置从动作数据读取；
  - 29 个关节、约 50 Hz、约 14.7 秒参考动作；
  - Vault Teacher 使用 pelvis 作为 root。
- 增加了独立的 Vault 任务入口：
  - `Instinct-Perceptive-Vault-G1-v0`
  - `Instinct-Perceptive-Vault-Pelvis-G1-v0`
  - 对应的 `-Play-v0` 入口。
- 处理了动作选择 YAML，使其符合当前 motion loader 的格式。
- 对 URDF、关节顺序、pelvis root、参考动作和箱体坐标做过检查和修正。

相关说明：

- [shadowing README](</root/gpufree-data/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/README.md>)
- [XU Vault 整合记录](</root/gpufree-data/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/XU_VAULT_INTEGRATION.md>)

### 3.2 locomotion Teacher

locomotion 使用 OmniLoco pelvis-root 版本，保留了用于后续蒸馏的固定观测接口：

- 高度扫描：`31 x 17 = 527` 点；
- 速度命令：3 维 `(vx, vy, wz)`；
- 投影重力：3 维；
- base 角速度：3 维；
- 29 个关节位置、29 个关节速度、29 个上一动作；
- 总 actor 观测维度：`623`；
- 动作维度：`29`。

当前命令范围在 [omni_loco_env_cfg.py](</root/gpufree-data/InstinctLab/source/instinctlab/instinctlab/tasks/locomotion/config/g1/omni_loco_env_cfg.py>) 中定义，已经覆盖前进、后退、横移和转向。站立命令比例约为 `0.02`，命令保持时间为 10 秒。

最终选作 locomotion Teacher 的 checkpoint：

```text
logs/instinct_rl/g1_unified_omni_loco_pelvis_rev10/
20261021_101015_G1UnifiedOmniLocoPelvisRev10_obs623_scale1_stand02_from24000_from20261020_115051/
model_35000.pt
```

这个 Teacher 的关键问题处理包括：

- 对齐 pelvis-root URDF 和关节顺序；
- 修复站立命令下原地踏步问题；
- 增加 `feet_gait`、`foot_clearance_reward`、`feet_slide` 等步态相关奖励；
- 增加站立时的关节保持/上半身约束；
- 调整速度范围和转向范围；
- 检查并修正播放时的 checkpoint、任务 ID 和环境配置。

### 3.3 Vault Teacher

Vault Teacher 使用 PPO 训练，输入包括：

- Vault 参考动作中的关节位置和速度；
- pelvis 的参考位置和旋转；
- 本体状态；
- `3.0 x 0.8 m` 区域的 `279` 点高度扫描；
- 18 个连杆位姿等 privileged critic 信息。

它不使用深度相机。最终归档的 checkpoint 是：

```text
logs/instinct_rl/g1_perceptive_vault_pelvis_corrected/
20261019_192913_g1PerceptiveVaultPelvisPhpTeacher_concatMotionBins/
model_15001.pt
```

Vault Teacher 训练过程中遇到过：

- pelvis / torso root 版本混用；
- `.npz` 关节顺序与 URDF 不一致；
- 转换基座后 base 坐标没有同步变换；
- `AmassMotion` 内部 motion bin 属性名不匹配；
- `env.yaml` 中 Python `slice` 标签无法被普通 YAML loader 读取；
- URDF 中不被 Isaac Sim 支持的 MuJoCo 标签产生警告；
- 物理场景未创建导致 `create_articulation_view` 失败。

目前 Vault Teacher 使用的 pelvis 版本已经能够播放，后续分析必须确保使用同一个 pelvis URDF、同一份 `.npz` 和同一套关节映射。

### 3.4 双 Teacher DAgger Student

双 Teacher 环境位于：

- [dual_teacher_dagger_env_cfg.py](</root/gpufree-data/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/dual_teacher_dagger_env_cfg.py>)
- [dual_teacher_dagger_cfg.py](</root/gpufree-data/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/dual_teacher_dagger_cfg.py>)

当前实现的任务分配方式：

- 偶数环境编号：locomotion；
- 奇数环境编号：Vault；
- locomotion 环境隐藏箱体；
- Vault 环境保留箱体并使用 Vault 参考动作重置；
- critic 保留 `task_id` 和 Teacher 所需的 privileged 信息；
- Autonomous Student actor 去掉 `task_id` 和 Vault 参考动作，只保留视觉、本体和命令。

Vault Student 的速度命令固定为：

```text
(vx, vy, wz) = (1.0, 0.0, 0.0)
```

locomotion 样本仍使用 `base_velocity` 命令采样。Student 的相机观测是：

```text
distance_to_image_plane_noised
分辨率：48 x 32
相机挂载：torso_link
```

当前 Autonomous Student 的最终 checkpoint：

```text
logs/instinct_rl/g1_loco_vault_autonomous_dagger_vault1ms/
20261023_230856_g1_loco_vault_autonomous_dagger/
model_14000.pt
```

DAgger 算法当前配置为：

- `using_ppo = False`；
- 只使用 Teacher action label 的蒸馏损失；
- action loss 使用 MSE sum；
- Teacher 执行概率线性衰减，约 10000 次更新降到 0；
- `buffer_dilation_ratio = 2.0`；
- Teacher 和 Student 共享动作维度 29；
- Teacher checkpoint 的 normalizer 必须保持独立，不能对 critic 观测再次错误归一化。

## 4. 模型归档位置

为了方便后续分析，当前三个模型和对应 YAML 已归档到 [model](</root/gpufree-data/InstinctLab/model/)：

| 模型 | 文件 |
|---|---|
| locomotion Teacher | `model/teachers/locomotion_pelvis_35000/model_35000.pt` |
| Vault Teacher | `model/teachers/vault_pelvis_15001/model_15001.pt` |
| Autonomous Student | `model/student/autonomous_dagger_vault1ms_14000/model_14000.pt` |

每个模型目录都有 `agent.yaml` 和 `env.yaml`。详细说明见 [model/README.md](</root/gpufree-data/InstinctLab/model/README.md>)。

继续分析时，建议同时保留：

- 原始 run 的 TensorBoard event 文件；
- 原始 run 的 `git/` diff；
- 对应 URDF/USD；
- Vault `.npz`、动作选择 YAML 和箱体数据；
- 当前源码配置。

## 5. 已遇到的主要问题和处理结果

### 5.1 训练入口和日志路径错误

曾经在 `/root/gpufree-data/IsaacLab` 下执行 InstinctLab 的训练脚本，导致：

```text
python: can't open file .../IsaacLab/scripts/instinct_rl/train.py
```

正确入口应从 InstinctLab 根目录执行：

```bash
cd /root/gpufree-data/InstinctLab
python scripts/instinct_rl/train.py ...
```

`--logroot` 必须指向当前任务实际写入的日志根目录。播放时 `--load_run` 应使用包含 checkpoint 的具体 run 目录，不能只给上一级实验目录。

### 5.2 播放阶段找不到 checkpoint

出现过：

```text
No checkpoints in the directory ... match 'model_12000.pt'
```

原因是 `--load_run` 指向了错误的 run，或者该 run 实际只保存到更低迭代数。处理方法是先查：

```bash
find /root/gpufree-data/InstinctLab/logs/instinct_rl -name 'model_*.pt' | sort
```

然后让 `--load_run` 和 `--checkpoint` 与同一个目录中的实际文件一致。

### 5.3 物理场景和 GPU 错误

出现过：

- `Failed to create simulation view backend`；
- `no active physics scene found`；
- `Articulation object has no attribute _root_physx_view`；
- Vulkan `ERROR_OUT_OF_DEVICE_MEMORY`。

这些问题通常与 Isaac Sim 启动失败、GPU 显存不足、残留进程或环境版本不一致有关，不是策略本身的 reward 问题。排查顺序：确认旧 Isaac Sim 进程已退出、降低 `num_envs`、确认 `--headless`、检查 GPU 显存和 Isaac Lab/Isaac Sim 版本。

### 5.4 Motion reference 属性和 YAML 兼容性

曾出现 `AmassMotion` 使用 `_motion_bin_weights`，而当前类只有 `_motion_weights`。这是代码版本接口不匹配，需要统一 motion loader 与环境配置版本。

另外，某些自动生成的 `env.yaml` 含有 `!!python/object/apply:builtins.slice`，普通 `yaml.full_load` 无法解析。播放时优先使用当前代码直接生成的环境配置，或确保保存 YAML 的 loader 与读取 loader 版本一致。

### 5.5 URDF、base 和动作坐标问题

最难排查的一类问题是机器人和 `RobotReference` 同时扭曲、腿部方向不对、动作看起来像绕错根节点运动。最终定位方向包括：

- URDF 的 root / pelvis / torso 定义；
- `.npz` 的关节顺序；
- retarget 后 base position 和 base quaternion 是否同步转换；
- Teacher、Student 和 RobotReference 是否使用同一套关节顺序；
- action scale 和 default joint offset 是否一致。

后续若再次修改 URDF，必须重新检查 `.npz` 是否需要转换，不能只替换 URDF 文件。

## 6. 当前训练结果和限制

当前 Student 训练日志显示：

- locomotion action MSE 曾降到约 `0.02`，说明在训练样本标签上接近 locomotion Teacher；
- Vault action MSE 后期明显升高，约达到 `0.4` 左右；
- Teacher 执行概率在后期降到 `0`；
- termination 统计中出现 `link_pos_too_far` 和 `dataset_exhausted`；
- 混合 episode length 不能直接代表 locomotion 成功率，因为其中同时包含 locomotion 和 Vault 环境。

播放时仍观察到 Student 基本行走不稳定，甚至从开始就学不会正常走路。当前最可能的原因有：

1. locomotion Teacher 使用高度图，Student 使用深度图，存在明显输入模态差异；
2. 一个 Student 同时拟合两个动作分布，但 actor 没有显式 `task_id`，只能从视觉、本体和命令推断任务；
3. 当前环境按环境编号固定拆分任务，还没有在同一个 episode 中完整训练“接近箱体、翻越、离开箱体、恢复行走”；
4. Vault 标签误差较大，可能污染共享 Student 的训练；
5. 训练期间的 command、reset、termination、动作缩放和播放配置必须完全一致。

因此，当前模型可以作为 DAgger 研究基线，但不能直接宣称已经完成可靠的自主行走和翻越。

## 7. 推荐的后续工作顺序

### 第一阶段：先隔离验证 locomotion

1. 只加载 locomotion Teacher 和 Student 的 locomotion 子集；
2. 单独固定速度命令，例如 `(1.0, 0, 0)`；
3. 在没有箱体的环境中播放；
4. 对比 Teacher action 和 Student action、base velocity tracking、脚接触和 reset 原因；
5. 确认 Student 单独能稳定走路后，再加入 Vault 样本。

### 第二阶段：隔离验证 Vault

1. 只使用 Vault 环境和固定 `1 m/s` 命令；
2. 检查深度图中箱体位置、相机坐标系和遮挡；
3. 对比 Vault Teacher 和 Student 在接近、起跳、越障、落地四个阶段的 action MSE；
4. 统计 `link_pos_too_far`、`base_pos_too_far`、`dataset_exhausted` 的分阶段比例；
5. 确认单任务 Vault Student 稳定后再和 locomotion 合并。

### 第三阶段：做连续任务

将训练场景改成真正的连续流程：

```text
平地行走 -> 视觉发现箱体 -> 接近并翻越 -> 落地 -> 恢复平地行走
```

此时需要明确：

- 箱体出现的时间和位置；
- locomotion / Vault Teacher 的切换条件；
- Student 是否需要 task phase 或 temporal history；
- Vault 动作结束后的 locomotion reset / warm start；
- 成功率、碰撞率、摔倒率和恢复行走时间等统一指标。

### 第四阶段：再决定是否加入 PPO

当前 DAgger 是纯动作蒸馏，没有 PPO 更新。后续可以在 Student 已经能复制两个 Teacher 后，再加入小权重 PPO 或 task-level fine-tuning，用于修正 Teacher 轨迹和视觉输入之间的分布差异。加入 PPO 前必须先保留一个纯 DAgger checkpoint 作为基线。

## 8. 常用命令

### 训练双 Teacher DAgger

```bash
cd /root/gpufree-data/InstinctLab
python scripts/instinct_rl/train.py \
  --task Instinct-Perceptive-Autonomous-Vault-Dagger-G1-v0 \
  --num_envs 1024 \
  --max_iterations 12001 \
  --headless
```

`num_envs` 应根据显存调整。之前 4096 容易触发 GPU 显存不足，1024 或 2048 更适合先做验证。

### 播放当前 Student

```bash
python scripts/instinct_rl/play.py \
  --task Instinct-Perceptive-Autonomous-Vault-Play-G1-v0 \
  --num_envs 2 \
  --load_run /root/gpufree-data/InstinctLab/logs/instinct_rl/g1_loco_vault_autonomous_dagger_vault1ms/20261023_230856_g1_loco_vault_autonomous_dagger \
  --checkpoint model_14000.pt
```

### 查看 TensorBoard

```bash
tensorboard \
  --logdir /root/gpufree-data/InstinctLab/logs/instinct_rl/g1_loco_vault_autonomous_dagger_vault1ms \
  --port 6006 \
  --bind_all
```

### 查找实际 checkpoint

```bash
find /root/gpufree-data/InstinctLab/logs/instinct_rl -name 'model_*.pt' | sort
```

## 9. 交接检查表

- [ ] 确认 Isaac Sim、Isaac Lab、InstinctLab 和 instinct-rl 版本一致。
- [ ] 确认 Python editable install 指向当前 `/root/gpufree-data/InstinctLab/source/instinctlab`。
- [ ] 确认使用 pelvis-root URDF，而不是旧 torso-root 入口。
- [ ] 确认两个 Teacher checkpoint 和 Student checkpoint 来自 [model](</root/gpufree-data/InstinctLab/model/) 的归档。
- [ ] 确认 Vault `.npz`、动作选择 YAML 和 URDF 的关节顺序一致。
- [ ] 先单独验证 locomotion，再单独验证 Vault，最后验证混合 Student。
- [ ] 播放失败时记录具体 termination 原因，不能只看总 reward 或 episode length。
- [ ] 每次改 reward、观测、URDF 或动作映射后，重新保存 checkpoint、YAML 和 TensorBoard 日志路径。

## 10. 关键文件索引

| 内容 | 路径 |
|---|---|
| locomotion 环境 | `source/instinctlab/instinctlab/tasks/locomotion/config/g1/omni_loco_env_cfg.py` |
| locomotion Teacher runner | `source/instinctlab/instinctlab/tasks/locomotion/config/g1/agents/omni_loco_ppo_cfg.py` |
| Vault 环境 | `source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_vault_cfg.py` |
| 双 Teacher DAgger 环境 | `source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/dual_teacher_dagger_env_cfg.py` |
| DAgger runner | `source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/dual_teacher_dagger_cfg.py` |
| DAgger 算法和 Teacher 路径 | `source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vault_dagger_cfg.py` |
| 模型归档 | `model/` |
| Vault 数据说明 | `source/instinctlab/instinctlab/tasks/shadowing/XU_VAULT_INTEGRATION.md` |

