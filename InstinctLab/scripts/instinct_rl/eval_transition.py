"""Fixed-seed complete-start / RSI evaluation of the unified transition task."""
# 模块级文档字符串：说明本脚本用于"固定随机种子 + 完整起点 / RSI（Random State Initialization，随机状态初始化）"
# 统一过渡任务的评测（evaluation）。

# 标准库 / 第三方库导入
import argparse                          # 命令行参数解析
import json                              # 结果序列化为 JSON
from pathlib import Path                 # 跨平台路径操作
import sys                               # 系统相关（此处未直接使用，但保留）
from isaaclab.app import AppLauncher     # Isaac Lab 应用启动器（必须先于其他 Isaac 组件启动 App）

# ---------------------------------------------------------------------------
# 1) 命令行参数定义
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# 任务注册名：默认 Instinct-Unified-Transition-PPO-Play-G1-v0
parser.add_argument("--task",default="Instinct-Unified-Transition-PPO-Play-G1-v0")
# 并行环境数量
parser.add_argument("--num_envs",type=int,default=16)
# 期望收集的完整 episode 数量（配额，达到后提前结束）
parser.add_argument("--episodes",type=int,default=100)
# 最大仿真步数（保护上限，防止死循环）
parser.add_argument("--max_steps",type=int,default=20000)
# 固定随机种子，保证评测可复现
parser.add_argument("--seed",type=int,default=42)
# RSI 触发概率：0 表示不做随机状态初始化（默认关闭）
parser.add_argument("--rsi_probability",type=float,default=0.)
# 障碍物出现概率（默认 80% 的 episode 会遇到障碍）
parser.add_argument("--obstacle_probability",type=float,default=0.8)
# 输出 JSON 报告文件路径
parser.add_argument("--report_file",default="transition_eval.json")

# 互斥且必选：要么加载 DAgger 学生模型的 actor-only 权重，要么加载完整 PPO checkpoint
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--actor_checkpoint",help="Existing DAgger Student: actor-only evaluation.")
group.add_argument("--checkpoint",help="New transition PPO checkpoint: full load.")

# 将 Isaac Lab 的通用启动参数（如 --device、--headless 等）追加到 parser
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 2) 参数合法性校验
# ---------------------------------------------------------------------------
# 三个正整数参数必须 > 0
if args.num_envs<=0 or args.episodes<=0 or args.max_steps<=0:
    parser.error("num_envs, episodes, max_steps must be positive")
# 两个概率参数必须落在 [0,1] 区间
if not 0<=args.rsi_probability<=1 or not 0<=args.obstacle_probability<=1:
    parser.error("probabilities must lie in [0,1]")

# ---------------------------------------------------------------------------
# 3) 启动 Isaac Lab 应用
# 注意：必须在导入任何依赖仿真器的模块（gym、torch 扩展、instinctlab 等）之前启动 App，
# 否则会出现底层插件未初始化的问题。
# ---------------------------------------------------------------------------
app_launcher = AppLauncher(args)
app = app_launcher.app

# ---------------------------------------------------------------------------
# 4) App 启动之后再导入依赖 Isaac 的库
# ---------------------------------------------------------------------------
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import instinctlab.tasks                                  # 注册 Instinct 的任务（必须导入才会注册 gym 环境）
from instinct_rl.runners import OnPolicyRunner            # PPO / On-Policy 训练/推理运行器
from instinct_rl.algorithms.warm_start_ppo import (
    load_actor_only,             # 只加载 actor 部分权重（用于 DAgger 学生模型评测）
    freeze_policy_normalizer,    # 冻结 policy 侧的观测归一化器（评测时不能更新统计量）
)
from instinctlab.utils.wrappers.instinct_rl import InstinctRlVecEnvWrapper  # 将 Isaac 环境包装为 instinct_rl 需要的接口

# ---------------------------------------------------------------------------
# 5) 主流程
# ---------------------------------------------------------------------------
def main():
    # 5.1) 根据任务名 + 设备 + 环境数解析环境配置
    cfg = parse_env_cfg(args.task,device=args.device,num_envs=args.num_envs)
    device = cfg.sim.device

    # 5.2) 注入评测参数到环境配置
    cfg.seed = args.seed                                   # 固定随机种子
    cfg.transition.rsi_probability = args.rsi_probability  # 设置 RSI 概率
    cfg.transition.obstacle_probability = args.obstacle_probability  # 设置障碍物概率
    # 评测时关闭 policy 观测损坏（noise / dropout 等），得到干净、可复现的观测
    cfg.observations.policy.enable_corruption = False

    # 5.3) 加载 agent（算法/网络）配置项
    agent_cfg = load_cfg_from_registry(args.task,"instinct_rl_cfg_entry_point")
    # 创建 gym 环境并用 InstinctRlVecEnvWrapper 包装
    env = InstinctRlVecEnvWrapper(gym.make(args.task,cfg=cfg))
    try:
        # 5.4) 构造 runner（此时不写日志目录）
        runner = OnPolicyRunner(env,agent_cfg.to_dict(),log_dir=None,device=device)
        if args.actor_checkpoint:
            # DAgger 学生模型：只加载 actor 网络权重
            load_actor_only(runner,args.actor_checkpoint)
        else:
            # 完整 PPO checkpoint：加载所有可训练状态（包含 critic、normalizer 等）
            runner.load(args.checkpoint)
        # 冻结 policy 侧的观测归一化器，避免评测过程中修改 running mean/var
        freeze_policy_normalizer(runner)
        # 获取推理策略（只做前向，无梯度、无探索）
        policy = runner.get_inference_policy(device=device)

        # 5.5) 观测契约自检：
        #     get_observations 读取缓存的 obs_buf，不应推进 history；
        #     连续两次调用应当返回相同内容，且最后一维大小应当为 2049。
        obs,_ = env.get_observations()
        obs2,_ = env.get_observations()
        if not torch.equal(obs,obs2) or obs.shape[-1]!=2049:
            raise RuntimeError("Cached observation/history contract failed.")

        # 5.6) 统计容器初始化
        records = {}          # 按字段名累积每个 episode 的元数据
        finished = 0          # 已完成的 episode 数（从 env 返回的 transition_episode 中取）
        steps = 0             # 实际执行的仿真步计数
        for steps in range(1,args.max_steps+1):
            if not app.is_running():       # GUI 模式下用户关窗口则提前退出
                break
            with torch.inference_mode():   # 推理模式，关闭梯度追踪以省显存
                obs,_,_,infos = env.step(policy(obs))
            # env 把每个 episode 结束时的元数据打包在 infos["transition_episode"] 中
            batch = infos.get("transition_episode")
            if batch:
                # 只接收剩余配额内的 episode，避免超过 --episodes
                take = min(len(batch["has_obstacle"]),args.episodes-finished)
                for name,value in batch.items():
                    # 逐个字段把张量转成 Python list 并累加
                    records.setdefault(name,[]).extend(value[:take].detach().cpu().tolist())
                finished += take
            if finished>=args.episodes:    # 达到配额，停止
                break

        # 5.7) 统计辅助函数
        def count(name):
            """统计 records[name] 中为真的元素个数。"""
            return sum(bool(x) for x in records.get(name,[]))

        def rate(numerator,denominator):
            """安全除法：分母为 0 时返回 None。"""
            return numerator/denominator if denominator else None

        # 5.8) 各类计数
        full = count("full_start")                                # 完整起点（从站立/标准初始状态开始）的 episode 数
        rsi = count("is_rsi")                                     # 触发 RSI 的 episode 数
        no_obstacle = finished-count("has_obstacle")              # 无障碍的 episode 数
        rsi_completed = sum(                                     # RSI episode 中"完成"的个数
            bool(a and b) for a,b in zip(records.get("is_rsi",[]),records.get("completed",[]))
        )

        # 5.9) 失败原因位掩码定义（每个 bit 代表一类失败原因；可重叠）
        #     注意：以下是一个 smoke-test 采样，不是固定的 episode 队列。
        failure_bits = {1:"low_base_height",2:"excessive_tilt",4:"low_link_net_contact",
                        8:"illegal_box_contact",16:"lane_violation"}
        codes = records.get("failure_code",[])
        # 对每条失败码，按位与判断是否包含对应失败原因
        failure_counts = {name:sum(bool(int(code)&bit) for code in codes)
                          for bit,name in failure_bits.items()}

        # 5.10) 从任务模块导入被禁止接触的刚体名单（用于报告中说明联系方式索引）
        from instinctlab.tasks.shadowing.perceptive.config.g1.unified_transition_mdp import FORBIDDEN_BODIES
        # 每个控制步对应的仿真秒数（control dt）
        dt = float(env.unwrapped.step_dt)
        # 将 episode 步数换算为秒
        episode_seconds = [float(value)*dt for value in records.get("episode_steps",[])]

        # 5.11) 组装报告字典
        report = {
            "task":args.task,"seed":args.seed,
            "checkpoint":args.actor_checkpoint or args.checkpoint,
            "deterministic":True,                               # 固定种子 + 关闭噪声 → 确定性
            "rsi_probability":args.rsi_probability,
            "control_dt":dt,"episode_seconds":episode_seconds,
            "sampling":"first_finished_episode_quota",          # 采样说明：先完成的 N 个 episode
            "failure_reason_counts":failure_counts,
            "failure_reason_bits":failure_bits,
            "failure_reasons_can_overlap":True,                 # 位掩码允许同一 episode 出现多种失败
            "contact_body_index_names":list(FORBIDDEN_BODIES),  # 违规接触刚体的名字列表
            "obstacle_probability":args.obstacle_probability,
            "episodes":finished,"requested_episodes":args.episodes,"control_steps":steps,
            "evaluation_complete":finished>=args.episodes,      # 是否在 max_steps 内完成配额
            "full_obstacle_episodes":full,"full_successes":count("full_success"),
            "full_success_rate":rate(count("full_success"),full),
            "rsi_episodes":rsi,"rsi_completions":rsi_completed,
            "rsi_completion_rate":rate(rsi_completed,rsi),
            "no_obstacle_episodes":no_obstacle,
            "no_obstacle_excursion_rate":rate(count("no_obstacle_excursion"),no_obstacle),
            "failure_rate":rate(count("failed"),finished),
            "records":records,                                  # 原始逐 episode 记录，便于离线分析
        }

        # 5.12) 写出 JSON 报告
        target = Path(args.report_file)
        target.parent.mkdir(parents=True,exist_ok=True)         # 自动创建父目录
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2))

        # 5.13) 控制台打印（不包含庞大的 records 字段）
        print(json.dumps({k:v for k,v in report.items() if k!="records"},
                         ensure_ascii=False,indent=2))

        # 5.14) 若因 max_steps 提前终止，则报告为 partial，需显式抛出异常告知调用方
        if not report["evaluation_complete"]:
            raise RuntimeError("Evaluation ended before requested episodes; report is partial.")
    finally:
        # 无论成功与否都关闭环境，释放仿真资源
        env.close()

# ---------------------------------------------------------------------------
# 6) 顶层执行 & 应用清理
# 无论 main 是否抛异常，都要关闭 Isaac Lab App。
# ---------------------------------------------------------------------------
try:
    main()
finally:
    app.close()