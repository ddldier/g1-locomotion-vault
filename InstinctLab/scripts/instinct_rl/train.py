# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 本脚本用于使用 Instinct-RL 训练强化学习智能体。
"""Script to train RL agent with Instinct-RL."""

# 先启动 Isaac Sim 仿真器。
"""Launch Isaac Sim Simulator first."""

# 导入标准库：参数解析、多进程、操作系统接口、系统模块。
import argparse
import multiprocessing as mp
import os
import sys

# 导入 Isaac Lab 应用启动器。
from isaaclab.app import AppLauncher

# local imports
# 导入本地命令行参数工具。
import cli_args  # isort: skip


# add argparse arguments
# 创建命令行参数解析器，并添加训练、视频、环境数量、任务、种子、日志等参数。
parser = argparse.ArgumentParser(description="Train an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--logroot", type=str, default=None, help="Override default log root path, typically `log/instinct_rl/`."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Enable distributed training. No need to add manually, it will be set automatically in the script.",
)
parser.add_argument(
    "--local-rank",
    type=int,
    help="Local rank for distributed training. No need to add manually, it will be set automatically in the script.",
)
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
# train.py specific arguments
parser.add_argument("--cprofile", action="store_true", default=False, help="Enable cProfile.")
parser.add_argument("--actor_checkpoint", type=str, default=None,
                    help="Transfer only a DAgger actor into a fresh WarmStartPPO run.")
parser.add_argument("--unfreeze_actor", action="store_true",
                    help="Resume an evaluated PPO checkpoint and finish critic warmup.")
parser.add_argument("--anchor_checkpoint", type=str, default=None,
                    help="Original Student reference for a fresh WalkAnchorPPO experiment.")
# append Instinct-RL cli arguments
# 添加 Instinct-RL 相关命令行参数。
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
# 添加 AppLauncher 相关命令行参数。
AppLauncher.add_app_launcher_args(parser)
# 解析已知参数，剩余参数留给 Hydra。
args_cli, hydra_args = parser.parse_known_args()
# 如果环境变量中存在 LOCAL_RANK，则启用分布式训练。
if "LOCAL_RANK" in os.environ:
    args_cli.distributed = True

# always enable cameras to record video
# 如果启用视频录制，则强制开启相机。
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
# 清空 sys.argv，只保留 Hydra 参数。
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
# 启动 Omniverse 应用。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 应用启动后的其余逻辑。
"""Rest everything follows."""

# 导入 Gymnasium、PyTorch、分布式训练和日期时间模块。
import gymnasium as gym
import torch
import torch.distributed as dist
from datetime import datetime

# 导入 Instinct-RL 的 on-policy runner。
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.algorithms.warm_start_ppo import (
    WarmStartPPO, load_actor_only, freeze_policy_normalizer,
)
from instinct_rl.algorithms.walk_anchor_ppo import WalkAnchorPPO, configure_walk_anchor

# 导入 Isaac Lab 环境配置相关类型。
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# 导入字典打印、YAML 保存、检查点路径、Hydra 任务配置等工具。
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 导入 InstinctLab 的向量环境包装器和 runner 配置。
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
# 如果处于调试模式，则等待调试器附加。
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()

# Import extensions to set up environment tasks
# 导入扩展以注册环境任务。
import instinctlab.tasks  # noqa: F401

# 启用 TF32 和 cuDNN 相关设置。
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# set affinity of in multiprocessing
# 在多进程中设置 CPU 亲和性。
def auto_affinity():
    rank = int(os.environ["RANK"])  # Get rank assigned by torch
    num_cores = mp.cpu_count() // torch.cuda.device_count()
    core_range = range(rank * num_cores, (rank + 1) * num_cores)
    core_mask = ",".join(map(str, core_range))
    os.system(f"taskset -cp {core_mask} {os.getpid()}")
    print("Affinity auto updated to:", core_mask, "for rank:", rank)


# 使用 Hydra 任务配置装饰器，指定任务名和 Instinct-RL 配置入口。
@hydra_task_config(args_cli.task, "instinct_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: InstinctRlOnPolicyRunnerCfg):
    """Train with Instinct-RL agent."""
    # override configurations with non-hydra CLI arguments
    # 使用非 Hydra 的命令行参数覆盖智能体配置。
    agent_cfg = cli_args.update_instinct_rl_cfg(agent_cfg, args_cli)
    # BEGIN transition_initialization_guard
    if args_cli.task in ("Instinct-Unified-Transition-PPO-G1-v0", "Instinct-Unified-Transition-PPO-Play-G1-v0",
                         "Instinct-Unified-Transition-WalkAnchor-PPO-G1-v0",
                         "Instinct-Unified-Transition-WalkAnchor-PPO-Play-G1-v0"):
        if not agent_cfg.resume and not args_cli.actor_checkpoint:
            raise ValueError("Transition PPO requires --actor_checkpoint or a new PPO --resume checkpoint.")
    # END transition_initialization_guard
    if args_cli.actor_checkpoint and agent_cfg.resume:
        raise ValueError("Use either actor-only transfer or PPO resume, not both.")
    if args_cli.unfreeze_actor and not agent_cfg.resume:
        raise ValueError("Unfreeze only when resuming an evaluated PPO checkpoint.")
    # 如果命令行指定了环境数量，则覆盖环境配置中的数量。
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # 如果命令行指定了最大迭代次数，则覆盖智能体配置中的值。
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    # 设置环境随机种子。
    env_cfg.seed = agent_cfg.seed
    # 设置仿真设备。
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # prepare configs for distributed training
    # 准备分布式训练配置。
    if "LOCAL_RANK" in os.environ:
        # 初始化 NCCL 进程组。
        dist.init_process_group(
            backend="nccl",
            rank=app_launcher.local_rank,
            world_size=int(os.getenv("WORLD_SIZE", 1)),
        )
        # 自动设置 CPU 亲和性。
        auto_affinity()
        # 获取当前 rank 和总进程数。
        local_rank, world_size = dist.get_rank(), dist.get_world_size()
        # 为不同 rank 设置不同的种子和设备。
        env_cfg.seed += local_rank
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        print(
            f"[INFO] Distributed training with rank: {local_rank}, world size: {world_size}, rank: {os.environ['RANK']}"
        )

    # specify directory for logging experiments
    # 指定实验日志目录。
    if args_cli.logroot is None:
        log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
    else:
        log_root_path = args_cli.logroot

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    # 构建具体运行目录，格式为时间戳加运行名。
    log_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    if getattr(env_cfg, "run_name", None):
        log_dir += f"_{env_cfg.run_name}"
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
        for h_args in hydra_args:
            log_dir += "_"
            log_dir += h_args.split("=")[0].split(".")[-1]
            log_dir += "-"
            log_dir += h_args.split("=")[1]
    log_dir = os.path.join(log_root_path, log_dir)

    # 如果配置为恢复训练，则查找检查点路径。
    if agent_cfg.resume:
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint)  # type: ignore
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Resuming experiment from directory: {resume_path}")
        resume_run_name = os.path.basename(os.path.dirname(resume_path))
        log_dir += f"_from{resume_run_name.split('_')[0]}_{resume_run_name.split('_')[1]}"

    # create isaac environment
    # 创建 Isaac 环境。
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    # 如果需要录制视频，则包装视频录制器。
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    # 如果环境是多智能体环境，则转换为单智能体环境。
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    # 用 Instinct-RL 的向量环境包装器包装环境。
    env = InstinctRlVecEnvWrapper(env)

    # create runner from instinct-rl
    # 创建 Instinct-RL 的 on-policy runner。
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    # 如果需要恢复训练，则加载检查点。
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    if args_cli.actor_checkpoint:
        if not isinstance(runner.alg, WarmStartPPO):
            raise TypeError("actor_checkpoint requires the new WarmStartPPO task.")
        load_actor_only(runner, args_cli.actor_checkpoint)
    if isinstance(runner.alg, WarmStartPPO):
        freeze_policy_normalizer(runner)
        if args_cli.unfreeze_actor:
            runner.alg.finish_critic_warmup()
    elif args_cli.unfreeze_actor:
        raise TypeError("unfreeze_actor requires WarmStartPPO.")

    # Bind training-only walking weights after actor/normalizer initialization.
    if isinstance(runner.alg, WalkAnchorPPO):
        configure_walk_anchor(runner, env.unwrapped,
                              args_cli.anchor_checkpoint or args_cli.actor_checkpoint)
    elif args_cli.anchor_checkpoint:
        raise TypeError("anchor_checkpoint requires the new WalkAnchorPPO task.")

    # dump the configuration into log-directory
    # 将配置保存到日志目录。
    if not ("LOCAL_RANK" in os.environ and dist.get_rank() > 0):
        # prevent dumping the config in non-rank-0 process
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 如果启用 cProfile，则开始性能分析。
    if args_cli.cprofile:
        import cProfile

        cprofile = cProfile.Profile()
        print(
            "Profiling enabled, a .profile file will be saved in the log directory after the program successfully"
            " finished."
        )
        cprofile.enable()

    # run training
    # 运行训练。
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=getattr(agent_cfg, "init_at_random_ep_len", False),
    )

    # 如果启用了 cProfile，则停止并保存分析结果。
    if args_cli.cprofile:
        cprofile.disable()
        cprofile.dump_stats(os.path.join(log_dir, "cprofile_stats.profile"))

    # 如果是分布式训练，则销毁进程组。
    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()
    # close the simulator
    # 关闭环境。
    env.close()


# 主程序入口。
if __name__ == "__main__":
    # run the main function
    # 运行主函数。
    main()
    # close sim app
    # 关闭仿真应用。
    simulation_app.close()