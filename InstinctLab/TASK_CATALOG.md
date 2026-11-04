# G1 当前任务目录（精简版）

当前只注册两个 pelvis Teacher 和一个统一高程图 Student，各自具有训练/播放入口。
注册统一维护在 source/instinctlab/instinctlab/tasks/__init__.py。

| 用途 | 训练任务 | 播放任务 | Actor 观测 |
|---|---|---|---|
| OmniLoco Teacher | Instinct-Unified-OmniLoco-Pelvis-G1-v0 | Instinct-Unified-OmniLoco-Pelvis-G1-Play-v0 | 高程扫描 + 命令 + 本体，623 维 |
| Vault Teacher | Instinct-Perceptive-Vault-Pelvis-G1-v0 | Instinct-Perceptive-Vault-Pelvis-G1-Play-v0 | 参考动作 + 高程扫描 + 状态，448 维 |
| 统一高程图 Student | Instinct-Unified-Height-Dagger-G1-v0 | Instinct-Unified-Height-Play-G1-v0 | 高程扫描 + 命令 + 本体，623 维 |

## 文件职责

| 文件（相对于 source/instinctlab/instinctlab/） | 职责 |
|---|---|
| tasks/__init__.py | 唯一注册清单 |
| tasks/locomotion/config/g1/omni_loco_env_cfg.py | locomotion Teacher 环境 |
| tasks/locomotion/config/g1/agents/omni_loco_ppo_cfg.py | locomotion Teacher PPO 配置 |
| tasks/shadowing/perceptive/config/g1/perceptive_vault_cfg.py | Vault Teacher 环境及共享基础配置 |
| tasks/shadowing/perceptive/config/g1/agents/instinct_rl_vault_cfg.py | Vault Teacher PPO 配置 |
| tasks/shadowing/perceptive/config/g1/dual_teacher_dagger_env_cfg.py | 混合环境及当前高程图 Student 观测 |
| tasks/shadowing/perceptive/config/g1/agents/dual_teacher_dagger_cfg.py | 双 Teacher 算法选择及高程图 Student MLP 配置 |

高程图配置已并入现有文件，删除上一轮新增的 unified_height_student_cfg.py。
各任务子目录 __init__.py 不再注册任务。顶层不再递归导入全部实验。

## 移除的入口类别

| 类别 | 处理 |
|---|---|
| Flat locomotion、旧非 pelvis OmniLoco | 取消注册，当前只使用选定的 pelvis OmniLoco |
| torso Vault Teacher / Student | 取消注册，避免根节点版本混用 |
| Box-Shadowing 重复别名 | 取消注册 |
| 通用 Shadowing、VAE、WholeBody、HOI、Parkour、BeyondMimic 独立任务 | 取消注册，暂不属于当前主线 |
| 单 Teacher Vault DAgger、带 task_id 的双 Teacher Student | 取消注册，当前只保留统一高程图 Student |
| 旧 Autonomous 深度图 Student | 暂停注册，源码保留供后续深度蒸馏使用 |

这里只移除上述任务的注册入口。共享环境、动作加载、算法、URDF、动作数据、
checkpoint 和原有日志没有删除。部分旧配置被当前主线继承或导入，必须保留。
旧文档和历史 agent/env YAML 中出现的旧任务名属于历史记录；以后以本表为准。
旧任务 ID 不再可用于 gym.make 或 --task；需要新的 Python 进程加载新清单。

## 当前 Student

按顺序输入 height_scan(527)、velocity_command(3)、projected_gravity(3)、
base_ang_vel(3)、joint_pos(29)、joint_vel(29)、last_action(29)，合计 623 维。
Student 扫描独立于两个 Teacher，覆盖地面和箱体并跟踪箱体变换。
Actor 不输入 task_id、参考动作、base_lin_vel 或深度图；采用单帧 MLP。
Teacher 标签仍由原始 critic 观测路由，不做额外 critic 归一化。

两个现有 Teacher 继续复用；高程图 Student 从头训练。
旧 depth model_14000.pt 的输入、归一化器和网络结构不同，不能直接续训为此策略。
该阶段仍使用按环境编号拆分的 loco / Vault 样本，尚未实现同一 episode 内的
Walk → Vault → Walk。Transition RL 和后续深度图蒸馏完成时再启用相应入口。

## 验证和使用

离线测试：

```bash
python scripts/tests/test_project_tasks.py -v
```

已有 Isaac 环境中的注册列表：

```bash
python scripts/list_envs.py
```

先检查少量环境的仿真初始化，再开始完整训练：

```bash
python scripts/instinct_rl/train.py \
  --task Instinct-Unified-Height-Dagger-G1-v0 \
  --num_envs 32 --max_iterations 2 --headless
```

正式训练可按资源调整 num_envs 和 max_iterations；当前 Student 没有 PPO 更新。
播放时使用默认均值推理，不加 --sample。

默认 Teacher checkpoint 路径沿用当前代码的原日志位置。如果仅有归档 model/，
需通过 LOCOMOTION_TEACHER_CHECKPOINT 和 VAULT_TEACHER_CHECKPOINT 指定两个 .pt。
加载器同时要求 checkpoint 同级的 params/agent.yaml、params/env.yaml；
归档 YAML 原本在同级根目录，需复制到 params/ 后使用。

## 验证范围

已检查：唯一中央注册入口、任务 ID 唯一性、全部入口文件及类名、配置替身下
训练/播放初始化、623 维观测及传感器独立性、MLP 与归一化配置、Python 语法。
本地没有 Isaac Lab、Isaac Sim、Gymnasium 或 PyTorch，未执行真实仿真或训练。


## 全部入口处置明细

| 原任务 ID | 处置 |
|---|---|
| Instinct-BeyondMimic-Plane-G1-Play-v0 | 取消注册 |
| Instinct-BeyondMimic-Plane-G1-v0 | 取消注册 |
| Instinct-Locomotion-Flat-G1-Play-v0 | 取消注册 |
| Instinct-Locomotion-Flat-G1-v0 | 取消注册 |
| Instinct-Parkour-Target-Amp-G1-Play-v0 | 取消注册 |
| Instinct-Parkour-Target-Amp-G1-v0 | 取消注册 |
| Instinct-Perceptive-Autonomous-Vault-Dagger-G1-v0 | 取消注册 |
| Instinct-Perceptive-Autonomous-Vault-Play-G1-v0 | 取消注册 |
| Instinct-Perceptive-Box-Shadowing-G1-Play-v0 | 取消注册 |
| Instinct-Perceptive-Box-Shadowing-G1-v0 | 取消注册 |
| Instinct-Perceptive-HOI-Shadowing-G1-Play-v0 | 取消注册 |
| Instinct-Perceptive-HOI-Shadowing-G1-v0 | 取消注册 |
| Instinct-Perceptive-Shadowing-G1-Play-v0 | 取消注册 |
| Instinct-Perceptive-Shadowing-G1-v0 | 取消注册 |
| Instinct-Perceptive-Vae-G1-Play-v0 | 取消注册 |
| Instinct-Perceptive-Vae-G1-v0 | 取消注册 |
| Instinct-Perceptive-Vault-Dagger-G1-v0 | 取消注册 |
| Instinct-Perceptive-Vault-Dual-Teacher-Dagger-G1-v0 | 取消注册 |
| Instinct-Perceptive-Vault-G1-Play-v0 | 取消注册 |
| Instinct-Perceptive-Vault-G1-v0 | 取消注册 |
| Instinct-Perceptive-Vault-Pelvis-Dagger-G1-v0 | 取消注册 |
| Instinct-Perceptive-Vault-Pelvis-G1-Play-v0 | 保留 |
| Instinct-Perceptive-Vault-Pelvis-G1-v0 | 保留 |
| Instinct-Perceptive-Vault-Pelvis-Student-G1-Play-v0 | 取消注册 |
| Instinct-Perceptive-Vault-Pelvis-Student-G1-v0 | 取消注册 |
| Instinct-Perceptive-Vault-Student-G1-Play-v0 | 取消注册 |
| Instinct-Perceptive-Vault-Student-G1-v0 | 取消注册 |
| Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 | 取消注册 |
| Instinct-Shadowing-WholeBody-Plane-G1-v0 | 取消注册 |
| Instinct-Unified-Height-Dagger-G1-v0 | 保留 |
| Instinct-Unified-Height-Play-G1-v0 | 保留 |
| Instinct-Unified-OmniLoco-G1-Play-v0 | 取消注册 |
| Instinct-Unified-OmniLoco-G1-v0 | 取消注册 |
| Instinct-Unified-OmniLoco-Pelvis-G1-Play-v0 | 保留 |
| Instinct-Unified-OmniLoco-Pelvis-G1-v0 | 保留 |
