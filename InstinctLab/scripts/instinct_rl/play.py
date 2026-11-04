"""Script to play a checkpoint if an RL agent from Instinct-RL."""

# 本脚本用于加载已训练的 Instinct-RL 策略（checkpoint），在 Isaac Sim 中回放/评估。
# 支持录像、ONNX 导出、键盘控制、以及针对特定混合任务（Vault/移动）的定制化 Play 配置。

"""Launch Isaac Sim Simulator first."""

import argparse
import subprocess
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
# ============ 命令行参数定义 ============
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
# 是否录制视频
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
# 录制视频长度（步数）
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
# 开始录制视频的步数
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
# 禁用 fabric，改为 USD I/O（某些场景需要）
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# 并行环境数量（覆盖默认配置）
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# 任务名称（IsaacLab/InstinctLab 注册的任务 ID）
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# 是否将策略导出为 ONNX（仅支持单环境）
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
# 是否启用 debugpy 远程调试
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
# 强制不加载 checkpoint，使用随机初始化的模型
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# 是否启用键盘覆盖 base_velocity 命令
parser.add_argument("--keyboard", action="store_true", help="Override base velocity with keyboard input.")
# 键盘控制的前后/侧向速度步长
parser.add_argument("--keyboard_linvel_step", type=float, default=1.0, help="Keyboard forward/lateral speed in m/s.")
# 键盘控制的偏航角速度步长
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="Keyboard yaw speed in rad/s.")
# 键盘速度命令平滑过渡时间（秒）
parser.add_argument(
    "--keyboard_ramp_time", type=float, default=0.1,
    help="Seconds used to smoothly reach a new keyboard velocity command.",
)

# custom play arguments
# ============ 自定义 Play 参数 ============
# 是否从文件加载 env 配置
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
# 是否从文件加载 agent 配置
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
# 是否从策略输出分布采样动作（而不是用确定性动作）
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
# 前 N 步将动作置零（便于观察初始状态）
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
# 是否不终止（仅保留 dataset_exhausted 终止项）
parser.add_argument(
    "--no_terminate", action="store_true", default=False, help="Do not remove termination conditions in simulation."
)
# 是否给每个 reward 项分配辅助奖励（用于调试奖励项贡献）
parser.add_argument(
    "--aux_reward",
    action="store_true",
    default=False,
    help="Whether to assign auxiliary rewards to each of the env's reward term.",
)

# 注册 Instinct-RL 相关参数，以及 AppLauncher 参数
cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 如果录制视频，需要开启相机
if args_cli.video:
    args_cli.enable_cameras = True

# 启动 Isaac Sim 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from instinct_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import instinctlab.tasks  # noqa: F401  # 注册 InstinctLab 的任务
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# 如果启用键盘控制，需要 carb 键盘事件
if args_cli.keyboard:
    import carb.input
    import omni.appwindow
    from carb.input import KeyboardEventType

# 如果启用 debug，则等待调试器 attach
if args_cli.debug:
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def fix_vault_play_command(env):
    """Keep only mixed-task Vault commands at body-forward (1, 0, 0)."""
    # 说明：针对混合任务（locomotion + Vault）中的 Vault 环境，
    # 强制其 base_velocity 命令保持为 (vx, vy, wz) = (1, 0, 0)，
    # 避免随机采样导致 Vault 环境无法触发跳跃动作。

    raw_env = env.unwrapped
    observations = getattr(raw_env.cfg, "observations", None)
    critic = getattr(observations, "critic", None)
    task_term = getattr(critic, "task_id", None)
    task_func = getattr(task_term, "func", None)

    # 仅当 critic 观测里有名为 dual_task_id 的任务 ID 函数时才应用此修复
    # Only apply this fix to the existing mixed locomotion/Vault environment.
    if getattr(task_func, "__name__", None) != "dual_task_id":
        return None, None

    # 通过 task_func 获取每个环境是否是 Vault（0 列 > 0.5 视为 Vault）
    vault_mask = task_func(raw_env)[:, 0] > 0.5

    # 若无 Vault 环境，打印提示并跳过
    if not bool(vault_mask.any()):
        print("[INFO] This mixed-task playback contains only locomotion environments.")
        return vault_mask, None

    # 获取 base_velocity 命令管理器 term
    velocity_term = raw_env.command_manager.get_term("base_velocity")

    def enforce():
        # 将 Vault 环境的命令固定为前进 1 m/s，其余分量清零
        command = velocity_term.command
        command[vault_mask, 0] = 1.0
        command[vault_mask, 1:] = 0.0

        # 禁止 standing/heading 逻辑对 Vault 命令的覆盖
        # Prevent standing/heading overrides from changing Vault commands.
        for name in ("is_standing_env", "is_heading_env"):
            flags = getattr(velocity_term, name, None)
            if flags is not None:
                flags[vault_mask] = False

    # 保存原始方法以便包装
    original_resample = velocity_term._resample_command
    original_update = velocity_term._update_command

    def resample(env_ids):
        # 先执行原采样逻辑，然后对 Vault 环境强制命令
        original_resample(env_ids)
        enforce()

    def update():
        # 先执行原更新逻辑，然后对 Vault 环境强制命令
        original_update()
        enforce()

    # 用包装函数替换原始方法
    # Preserve the original locomotion sampling and updates, then fix only
    # Vault rows before observations and debug arrows read the command buffer.
    velocity_term._resample_command = resample
    velocity_term._update_command = update

    # 立即执行一次以保证首个 step 前命令已正确
    enforce()

    print("[INFO] Vault playback command: body-frame (vx, vy, wz) = (1, 0, 0).")
    print("[INFO] Locomotion commands remain unchanged; keyboard control targets locomotion only.")

    return vault_mask, enforce


def main():
    """Play with Instinct-RL agent."""
    # 解析 env 配置（根据任务和 CLI 参数）
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # 解析 agent 配置（Instinct-RL runner 相关）
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(
        args_cli.task, args_cli
    )

    # 日志根目录：logs/instinct_rl/<experiment_name>
    log_root_path = os.path.abspath(
        os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    )
    agent_cfg.load_run = args_cli.load_run

    # ============ 确定 checkpoint 路径 ============
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        if os.path.isabs(agent_cfg.load_run):
            # 绝对路径：直接拆分目录与文件名
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run),
                os.path.basename(agent_cfg.load_run),
                agent_cfg.load_checkpoint,
            )
        else:
            # 相对路径：以 log_root_path 为基准
            resume_path = get_checkpoint_path(
                log_root_path,
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )

        # log_dir 为 checkpoint 所在目录
        log_dir = os.path.dirname(resume_path)

    elif not args_cli.no_resume:
        # 未提供 checkpoint 也没用 --no_resume，报错
        raise RuntimeError(
            "[ERROR] No checkpoint specified. "
            "Specify --load_run, or use --no_resume to disable loading."
        )

    else:
        # --no_resume：使用 scratch 模型，记录到一个默认 play 目录
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    # 若指定 --env_cfg，则从日志目录加载 env.yaml 覆盖当前配置
    if args_cli.env_cfg:
        env_cfg = load_yaml(os.path.join(log_dir, "params", "env.yaml"))

    # 若指定 --agent_cfg，则从日志目录加载 agent.yaml；否则使用 CLI 解析得到的 dict
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    # 仅针对 "Instinct-Unified-Height-Play-G1-v0" 任务做专门的回放设置
    # Diagnostic settings for the height Student playback task only.
    if args_cli.task == "Instinct-Unified-Height-Play-G1-v0":
        import instinctlab.envs.mdp as instinct_mdp

        # 保留原 reset_robot 函数
        original_reset_robot = env_cfg.events.reset_robot.func

        def reset_robot_for_fixed_play(env, env_ids, motion_ref_cfg, asset_cfg):
            # 偶数环境视为 locomotion，奇数环境视为 Vault
            loco_ids = env_ids[env_ids % 2 == 0]
            vault_ids = env_ids[env_ids % 2 == 1]

            if loco_ids.numel():
                # locomotion 环境使用原始 reset
                original_reset_robot(
                    env, loco_ids,
                    motion_ref_cfg=motion_ref_cfg, asset_cfg=asset_cfg,
                )

            if vault_ids.numel():
                # Vault 环境使用固定的参考状态 reset（无噪声）
                instinct_mdp.reset_robot_state_by_reference(
                    env, vault_ids,
                    motion_ref_cfg=motion_ref_cfg, asset_cfg=asset_cfg,
                    position_offset=[0.0, 0.0, 0.0],
                    dof_vel_ratio=1.0,
                    base_lin_vel_ratio=1.0,
                    base_ang_vel_ratio=1.0,
                    randomize_pose_range={},
                    randomize_velocity_range={},
                    randomize_joint_pos_range=(0.0, 0.0),
                )

        # 注意：下面一行实际并未把 env_cfg.events.reset_robot.func 替换成 reset_robot_for_fixed_play
        # （可能是有意保留原行为，或在别处替换）。此处保持原样。
        env_cfg.events.reset_robot.func = original_reset_robot
        #noise_test = "joint_vel"

        # 下面是曾经用于测试不同噪声项效果的代码（已注释）：
        # noise_terms = (
        #     "projected_gravity",                #cause!!! zou ji bu jiu shuai
        #     "base_ang_vel",
        #     "joint_pos",                        #cause!!! bu hui shuai dan shi zou bu wan
        #     "joint_vel",                        #cause!!! bu yi ding neng fan guo qu,ke neng hui fan
        # )
        # if noise_test not in (*noise_terms, "none", "all"):
        #     raise ValueError(f"Unknown noise_test: {noise_test}")ssssssss

        policy_cfg = env_cfg.observations.policy
        policy_cfg.enable_corruption = False #noise_test != "none"

        # for term_name in noise_terms:
        #     term_cfg = getattr(policy_cfg, term_name)
        #     if noise_test not in (term_name, "all"):
        #         term_cfg.noise = None

        # print(f"[INFO] PLAY policy noise: {noise_test}")

        # 关闭 policy 观测的噪声
        policy_cfg = env_cfg.observations.policy
        policy_cfg.enable_corruption = None

        # 修改 motion_reference 配置以固定起始帧、关闭中间起点随机化
        for motion in env_cfg.scene.motion_reference.motion_buffers.values():
            motion.motion_bin_length_s = None
            motion.env_starting_stub_sampling_strategy = "independent"
            motion.motion_start_from_middle_range = [0.0, 0.0]

        # 关闭一些与位置相关的终止条件，避免 Vault 回放时被误判终止
        env_cfg.terminations.link_pos_too_far = None
        env_cfg.terminations.base_pos_too_far = None
        env_cfg.terminations.base_pg_too_far = None
        print("[INFO] Fixed Vault PLAY: start=0, reset_noise=off, policy_noise=off")

        #zhi xiu gai play de shi hou de pei zhi
        # 仅修改 play 时的配置（从 vault_play 读取原始箱体位置）
        from instinctlab.tasks.shadowing.perceptive.config.g1 import perceptive_vault_cfg as vault_play

        # 负值：箱体沿 -x 移动，靠近当前从 +x 方向接近它的机器人。
        play_box_dx = -1

        box_x, box_y, box_z = vault_play.vault_cfg.VAULT_BOX_POS
        original_reset_box = env_cfg.events.reset_vault_box.func

        def reset_box_for_play(env, env_ids, asset_cfg):
            # 区分 locomotion 与 Vault 环境
            loco_ids = env_ids[env_ids % 2 == 0]
            vault_ids = env_ids[env_ids % 2 == 1]

            if loco_ids.numel():
                # locomotion 环境用原始箱体 reset（实际不涉及箱体，但保留原有行为）
                original_reset_box(env, loco_ids, asset_cfg=asset_cfg)

            if vault_ids.numel():
                # Vault 环境：使用带位置偏移的箱体 reset，偏移量由 play_box_dx 决定
                vault_play.reset_vault_box_pose_by_motion_randomized(
                    env, vault_ids,
                    asset_cfg=asset_cfg,
                    active_pos=(box_x + play_box_dx, box_y, box_z),
                    pose_range={},
                )

        # 替换 box reset 函数
        env_cfg.events.reset_vault_box.func = reset_box_for_play
        print(f"[INFO] PLAY box x offset: {play_box_dx:+.2f} m")

    # 创建环境（如需录像则使用 rgb_array 渲染模式）
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # ============ 视频录制包装 ============
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{resume_path.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 若为多智能体 DirectMARL 环境，转换为单智能体
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 在 wrapper 和 runner reset 环境之前安装 Vault 命令修正
    # Install before the wrapper and runner reset the environment.
    vault_mask, enforce_vault_command = fix_vault_play_command(env)

    # --no_terminate：只保留 dataset_exhausted 终止项，其他终止条件全部去除
    if args_cli.no_terminate:
        manager = env.unwrapped.termination_manager
        index = manager._term_names.index("dataset_exhausted")
        manager._term_cfgs = [manager._term_cfgs[index]]
        manager._term_names = ["dataset_exhausted"]

    # 使用 Instinct-RL 的 vec env wrapper
    env = InstinctRlVecEnvWrapper(env)

    # 创建 OnPolicyRunner（不写日志目录）
    ppo_runner = OnPolicyRunner(
        env,
        agent_cfg_dict,
        log_dir=None,
        device=agent_cfg.device,
    )

    # 如有 checkpoint 则加载模型
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # --sample：从分布采样动作；否则使用确定性推理策略
    if args_cli.sample:
        ppo_runner.eval_mode()
        normalizer = ppo_runner.normalizers.get("policy")

        def policy(observations):
            # 先对观测做 normalization，再从 actor-critic 的动作分布中采样
            if normalizer is not None:
                observations = normalizer(observations)
            return ppo_runner.alg.actor_critic.act(observations)

    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ============ 导出 ONNX ============
    if agent_cfg.load_run is not None and args_cli.exportonnx:
        assert env.unwrapped.num_envs == 1, (
            "Exporting to ONNX is only supported for single environment."
        )
        export_model_dir = os.path.join(log_dir, "exported")
        os.makedirs(export_model_dir, exist_ok=True)
        obs, _ = env.get_observations()
        ppo_runner.export_as_onnx(obs, export_model_dir)

    # 若有 Vault 命令修正函数则再执行一次，确保初始状态正确
    if enforce_vault_command is not None:
        enforce_vault_command()

    # 获取初始观测
    obs, _ = env.get_observations()

    # ============ 键盘控制相关状态 ============
    # 当前键盘命令（平滑后的值）与目标值
    keyboard_command = torch.zeros(
        (env.unwrapped.num_envs, 3),
        device=env.unwrapped.device,
    )
    keyboard_command_target = torch.zeros_like(keyboard_command)

    if args_cli.keyboard:
        # 键盘事件回调：按下/重复按下时更新目标命令
        def on_keyboard_input(event):
            if event.type not in (
                KeyboardEventType.KEY_PRESS,
                KeyboardEventType.KEY_REPEAT,
            ):
                return

            key = event.input

            if key == carb.input.KeyboardInput.W:
                keyboard_command_target[:, 0] = args_cli.keyboard_linvel_step
            elif key == carb.input.KeyboardInput.S:
                keyboard_command_target[:, 0] = -args_cli.keyboard_linvel_step
            elif key == carb.input.KeyboardInput.A:
                keyboard_command_target[:, 1] = args_cli.keyboard_linvel_step
            elif key == carb.input.KeyboardInput.D:
                keyboard_command_target[:, 1] = -args_cli.keyboard_linvel_step
            elif key == carb.input.KeyboardInput.Q:
                keyboard_command_target[:, 2] = args_cli.keyboard_angvel
            elif key == carb.input.KeyboardInput.E:
                keyboard_command_target[:, 2] = -args_cli.keyboard_angvel
            elif key == carb.input.KeyboardInput.SPACE:
                # 空格清零（急停）
                keyboard_command_target.zero_()

        # 获取窗口并订阅键盘事件
        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        input_interface = carb.input.acquire_input_interface()
        input_interface.subscribe_to_keyboard_events(
            keyboard,
            on_keyboard_input,
        )

        print(
            "[INFO] Keyboard control: W/S forward/back, A/D left/right, "
            "Q/E yaw, Space stop."
        )

    # ============ 主循环 ============
    timestep = 0
    # 每个环境当前 episode 已经过的步数（用于统计 Vault 环境终止时机）
    episode_steps = torch.zeros(
    env.unwrapped.num_envs,
    dtype=torch.long,
    device=env.unwrapped.device,
    )
    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.keyboard:
                # 键盘命令平滑：alpha 由 step_dt / ramp_time 决定
                alpha = min(
                    1.0,
                    env.unwrapped.step_dt
                    / max(args_cli.keyboard_ramp_time, 1e-6),
                )
                keyboard_command += alpha * (
                    keyboard_command_target - keyboard_command
                )

                # 获取 base_velocity 命令缓冲区
                command = env.unwrapped.command_manager.get_command(
                    "base_velocity"
                )

                if vault_mask is None:
                    # 无 Vault 环境：全部用键盘命令
                    command[:] = keyboard_command
                else:
                    # 有 Vault 环境：仅对 locomotion 环境写入键盘命令
                    command[~vault_mask] = keyboard_command[~vault_mask]

                # Vault 环境命令保持为 (1,0,0)
                if enforce_vault_command is not None:
                    enforce_vault_command()

                # 注意：这一行代码存在语法错误（缺少冒号），
                # 但保持原样不修改。
                # Read observations after writing this step's keyboard command.
                if args_cli.task == "Instinct-Unified-Height-Play-G1-v0":
                    raw_env = env.unwrapped
                    term_cfg = raw_env.cfg.observations.policy.velocity_command
                    raw_env.obs_buf["policy"]["velocity_command"] = term_cfg.func(raw_env, **term_cfg.params)

                # 写入命令后重新获取观测，使速度命令观测与最新命令一致
                obs, _ = env.get_observations()

            # 策略推理得到动作
            actions = policy(obs)

            # 前 N 步将动作置零（用于观察初始状态）
            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0

            # 环境步进
            obs, rewards, dones, infos = env.step(actions)
            episode_steps += 1
            manager = env.unwrapped.termination_manager

            # 对于 Vault 环境（奇数 id）发生 done 时，打印终止原因和步数
            for env_id in dones.nonzero(as_tuple=True)[0].tolist():
                # 当前混合任务：奇数环境为 Vault
                if env_id % 2 == 1:
                    reasons = [
                        name
                        for name in manager.active_terms
                        if bool(manager.get_term(name)[env_id].item())
                    ]
                    print(
                        f"[Vault reset] env={env_id}, "
                        f"steps={episode_steps[env_id].item()}, "
                        f"seconds={episode_steps[env_id].item() * env.unwrapped.step_dt:.3f}, "
                        f"reasons={reasons}"
                    )

            # 已 done 的环境步数清零
            episode_steps[dones.bool()] = 0

        timestep += 1

        # ============ 辅助奖励：用于调试各奖励项贡献 ============
        if args_cli.aux_reward:
            aux_rewards = ppo_runner.alg.compute_auxiliary_reward(
                infos["observations"]
            )
            for aux_reward_name, aux_reward in aux_rewards.items():
                aux_term_cfg = (
                    env.unwrapped.reward_manager.get_term_cfg(aux_reward_name)
                )
                # 把辅助奖励写入对应 reward term 的 buffer，乘以系数
                aux_term_cfg.func.reward[:] = aux_reward * getattr(
                    ppo_runner.alg,
                    aux_reward_name + "_coef",
                    1.0,
                )

        # 达到设定视频长度则退出循环
        if args_cli.video and timestep == args_cli.video_length:
            break

    # 关闭环境
    env.close()

    # 录制视频结束后用 VS Code 打开视频文件
    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(
                    log_dir,
                    "videos",
                    "play",
                    f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4",
                ),
            ]
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        # 无论 main 是否异常，都关闭仿真 app
        simulation_app.close()
