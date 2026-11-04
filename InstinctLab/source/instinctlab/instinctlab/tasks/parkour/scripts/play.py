"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--useonnx", action="store_true", default=False, help="Use the exported ONNX model for inference.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# 自定义回放参数
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument("--keyboard_control", action="store_true", default=False, help="Enable keyboard control.")
parser.add_argument(
    "--keyboard_terrain",
    type=str,
    default="pyramid_stairs",
    choices=[
        "square_gaps",
        "pyramid_stairs",
        "pyramid_stairs_high",
        "pyramid_stairs_inv",
        "pyramid_stairs_inv_high",
        "boxes",
        "mesh_boxes",
        "hf_pyramid_slope_inv",
    ],
    help="Single terrain type used with --keyboard_control.",
)
parser.add_argument(
    "--keyboard_linvel_step",
    type=float,
    default=0.5,
    help="Forward/backward and lateral velocity magnitude for keyboard control.",
)
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="Yaw velocity magnitude for keyboard control.")
parser.add_argument(
    "--visual_ablation",
    type=str,
    choices=["normal", "blind"],
    default=None,
    help=(
        # 正常深度和全零深度都在同一时刻计算，用于比较视觉对动作的影响。
        "Run a paired visual ablation. Both normal-depth and zero-depth actions are evaluated from the same "
        "observation; this option selects which action controls the robot."
    ),
)
parser.add_argument(
    "--ablation_log_interval",
    type=int,
    default=10,
    help="Write one visual-ablation CSV row every N simulation steps.",
)
parser.add_argument(
    "--ablation_output_dir",
    type=str,
    default="/root/gpufree-data/visual_ablation",
    help="Directory for visual-ablation CSV and metadata files.",
)
parser.add_argument(
    "--ablation_steps",
    type=int,
    default=0,
    help="Stop after this many steps during visual ablation. Zero runs until the simulator closes.",
)

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import carb.input
import omni.appwindow
from carb.input import KeyboardEventType
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components, get_subobs_size

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def main():
    """Play with Instinct-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # 指定实验日志目录，并解析要加载的 checkpoint。
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.env_cfg:
        raise RuntimeError(
            "--env_cfg is not supported with this IsaacLab version; use the registered task configuration instead."
        )
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    if args_cli.keyboard_control:
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1e10
        terrain_cfg = env_cfg.scene.terrain.terrain_generator
        if terrain_cfg is None or args_cli.keyboard_terrain not in terrain_cfg.sub_terrains:
            raise RuntimeError(f"Terrain type '{args_cli.keyboard_terrain}' is not available in the play config.")
        terrain_cfg.sub_terrains = {
            args_cli.keyboard_terrain: copy.deepcopy(terrain_cfg.sub_terrains[args_cli.keyboard_terrain])
        }
        terrain_cfg.num_cols = 1
        env_cfg.scene.terrain.max_init_terrain_level = terrain_cfg.num_rows - 1
        command_cfg = env_cfg.commands.base_velocity
        if command_cfg.velocity_ranges is not None:
            command_cfg.velocity_ranges = {
                args_cli.keyboard_terrain: command_cfg.velocity_ranges[args_cli.keyboard_terrain]
            }
        if command_cfg.random_velocity_terrain is not None:
            command_cfg.random_velocity_terrain = [
                name for name in command_cfg.random_velocity_terrain if name == args_cli.keyboard_terrain
            ]

    # 创建 Isaac Sim 环境；键盘控制模式会在前面将环境缩小为单个实例。
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # 可选：用 Gym wrapper 录制回放视频。
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

    # 如果任务是多智能体环境，则转换成算法使用的单智能体接口。
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 使用 Instinct-RL 的 VecEnv wrapper，统一观测、动作接口。
    env = InstinctRlVecEnvWrapper(env)

    # 构建 Actor-Critic，并加载训练好的模型参数。
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # 获取推理策略；推理时会先经过视觉编码器，再经过 MoE Actor。
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 可选：把当前策略导出为 ONNX。
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    # 可选：使用已经导出的 ONNX 策略进行推理对照。
    if args_cli.useonnx:
        from onnxer import load_parkour_onnx_model

        # NOTE: This is only applicable with parkour task
        onnx_policy = load_parkour_onnx_model(
            model_dir=os.path.join(log_dir, "exported"),
            get_subobs_func=lambda obs: get_subobs_by_components(
                obs,
                agent_cfg.policy.encoder_configs.depth_encoder.component_names,
                env.get_obs_segments(),
                temporal=True,
            ),
            depth_shape=env.get_obs_segments()["depth_image"],
            proprio_slice=slice(
                0,
                get_subobs_size(
                    env.get_obs_segments(),
                    [
                        "base_lin_vel",
                        "base_ang_vel",
                        "projected_gravity",
                        "velocity_commands",
                        "joint_pos",
                        "joint_vel",
                        "actions",
                    ],
                ),
            ),
        )

    # 键盘速度指令：[前后速度、左右速度、偏航角速度]。
    override_command = torch.zeros(env.num_envs, 3, device=env.device)
    command_obs_slice = get_obs_slice(env.get_obs_segments(), "velocity_commands")
    # 深度图在扁平观测中的位置和原始形状，供消融实验使用。
    depth_obs_slice, depth_obs_shape = get_obs_slice(env.get_obs_segments(), "depth_image")

    ablation_file = None
    ablation_writer = None
    ablation_csv_path = None
    # 视觉消融模式：正常深度与全零深度同时推理，并把统计量写入 CSV。
    if args_cli.visual_ablation is not None:
        if args_cli.ablation_log_interval < 1:
            raise ValueError("--ablation_log_interval must be at least 1.")
        os.makedirs(args_cli.ablation_output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        terrain_name = args_cli.keyboard_terrain if args_cli.keyboard_control else "mixed_terrain"
        experiment_name = f"{timestamp}_{terrain_name}_{args_cli.visual_ablation}"
        ablation_csv_path = os.path.join(args_cli.ablation_output_dir, experiment_name + ".csv")
        metadata_path = os.path.join(args_cli.ablation_output_dir, experiment_name + ".json")
        metadata = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "rollout_mode": args_cli.visual_ablation,
            "terrain": terrain_name,
            "num_envs": env.num_envs,
            "depth_shape": list(depth_obs_shape),
            "keyboard_control": args_cli.keyboard_control,
            "keyboard_linvel_step": args_cli.keyboard_linvel_step,
            "keyboard_angvel": args_cli.keyboard_angvel,
            "log_interval": args_cli.ablation_log_interval,
            "max_steps": args_cli.ablation_steps,
        }
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        ablation_file = open(ablation_csv_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "step",
            "rollout_mode",
            "command_x",
            "command_y",
            "command_yaw",
            "depth_min",
            "depth_max",
            "depth_mean",
            "depth_std",
            "depth_nonzero_fraction",
            "depth_input_diff_max",
            "depth_latent_diff_mae",
            "depth_latent_diff_max",
            "normal_action_abs_mean",
            "blind_action_abs_mean",
            "action_diff_mae",
            "action_diff_rmse",
            "action_diff_max",
            "gate_diff_mae",
        ]
        num_experts = agent_cfg.policy.num_moe_experts
        fieldnames.extend(f"normal_gate_{index}" for index in range(num_experts))
        fieldnames.extend(f"blind_gate_{index}" for index in range(num_experts))
        ablation_writer = csv.DictWriter(ablation_file, fieldnames=fieldnames)
        ablation_writer.writeheader()
        ablation_file.flush()
        print(f"[INFO] Visual ablation mode: {args_cli.visual_ablation}")
        print(f"[INFO] Visual ablation CSV: {ablation_csv_path}")
        print(f"[INFO] Visual ablation metadata: {metadata_path}")

    # 将键盘按键转换成速度命令，随后覆盖策略观测中的 velocity_commands 历史。
    def on_keyboard_input(e):
        if e.type not in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
            return

        if e.input == carb.input.KeyboardInput.W:
            override_command[:, 0] = args_cli.keyboard_linvel_step
        elif e.input == carb.input.KeyboardInput.S:
            override_command[:, 0] = -args_cli.keyboard_linvel_step
        elif e.input == carb.input.KeyboardInput.A:
            override_command[:, 1] = args_cli.keyboard_linvel_step
        elif e.input == carb.input.KeyboardInput.D:
            override_command[:, 1] = -args_cli.keyboard_linvel_step
        elif e.input == carb.input.KeyboardInput.Q:
            override_command[:, 2] = args_cli.keyboard_angvel
        elif e.input == carb.input.KeyboardInput.E:
            override_command[:, 2] = -args_cli.keyboard_angvel
        elif e.input == carb.input.KeyboardInput.SPACE:
            override_command[:] = 0.0

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input = carb.input.acquire_input_interface()
    input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    # 获取环境初始观测；后续每一步由 env.step() 返回下一帧观测。
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    try:
        while simulation_app.is_running():
            # 推理阶段不计算梯度，减少显存和计算开销。
            with torch.inference_mode():
                # 键盘只改变策略看到的速度指令，不会删除深度、本体和历史动作观测。
                if args_cli.keyboard_control:
                    obs[:, command_obs_slice[0]] = override_command.repeat(1, command_obs_slice[1][0] // 3)

                if args_cli.visual_ablation is not None:
                    # 两份输入共享同一时刻的本体观测；blind 仅把 depth_image 切片置零。
                    # 使用两个独立副本，避免策略或观测归一化器复用同一个张量。
                    normal_obs = obs.clone()
                    blind_obs = obs.clone()
                    blind_obs[:, depth_obs_slice] = 0.0
                    # 显式使用和推理策略相同的归一化器，再分别调用 Actor，避免比较时复用动作结果。
                    policy_normalizer = ppo_runner.normalizers.get("policy")
                    normal_policy_obs = (
                        policy_normalizer(normal_obs) if policy_normalizer is not None else normal_obs
                    )
                    blind_policy_obs = policy_normalizer(blind_obs) if policy_normalizer is not None else blind_obs
                    normal_actions = ppo_runner.alg.actor_critic.act_inference(normal_policy_obs)
                    blind_actions = ppo_runner.alg.actor_critic.act_inference(blind_policy_obs)
                    actions = normal_actions if args_cli.visual_ablation == "normal" else blind_actions

                    if timestep % args_cli.ablation_log_interval == 0:
                        # 记录深度图统计、动作差异和 MoE Gate 概率，便于定量判断视觉是否生效。
                        depth = normal_obs[:, depth_obs_slice]
                        action_diff = normal_actions - blind_actions
                        actor_critic = ppo_runner.alg.actor_critic
                        normal_encoded = actor_critic.encoders(normal_policy_obs)
                        blind_encoded = actor_critic.encoders(blind_policy_obs)
                        depth_latent_start = actor_critic.encoders.output_segment.keys()
                        depth_latent_name = next(
                            name for name in depth_latent_start if name.endswith("depth_encoder")
                        )
                        depth_latent_slice, _ = get_obs_slice(
                            actor_critic.encoders.output_segment, depth_latent_name
                        )
                        normal_gate = torch.softmax(actor_critic.actor.gate(normal_encoded), dim=-1)
                        blind_gate = torch.softmax(actor_critic.actor.gate(blind_encoded), dim=-1)
                        row = {
                            "step": timestep,
                            "rollout_mode": args_cli.visual_ablation,
                            "command_x": override_command[:, 0].mean().item(),
                            "command_y": override_command[:, 1].mean().item(),
                            "command_yaw": override_command[:, 2].mean().item(),
                            "depth_min": depth.min().item(),
                            "depth_max": depth.max().item(),
                            "depth_mean": depth.mean().item(),
                            "depth_std": depth.std(unbiased=False).item(),
                            "depth_nonzero_fraction": (depth != 0).float().mean().item(),
                            "depth_input_diff_max": (normal_obs[:, depth_obs_slice] - blind_obs[:, depth_obs_slice])
                            .abs()
                            .max()
                            .item(),
                            "depth_latent_diff_mae": (
                                normal_encoded[:, depth_latent_slice] - blind_encoded[:, depth_latent_slice]
                            )
                            .abs()
                            .mean()
                            .item(),
                            "depth_latent_diff_max": (
                                normal_encoded[:, depth_latent_slice] - blind_encoded[:, depth_latent_slice]
                            )
                            .abs()
                            .max()
                            .item(),
                            "normal_action_abs_mean": normal_actions.abs().mean().item(),
                            "blind_action_abs_mean": blind_actions.abs().mean().item(),
                            "action_diff_mae": action_diff.abs().mean().item(),
                            "action_diff_rmse": action_diff.square().mean().sqrt().item(),
                            "action_diff_max": action_diff.abs().max().item(),
                            "gate_diff_mae": (normal_gate - blind_gate).abs().mean().item(),
                        }
                        for index in range(normal_gate.shape[-1]):
                            row[f"normal_gate_{index}"] = normal_gate[:, index].mean().item()
                            row[f"blind_gate_{index}"] = blind_gate[:, index].mean().item()
                        ablation_writer.writerow(row)
                        ablation_file.flush()
                else:
                    # 普通回放只执行一次策略推理。
                    actions = policy(obs)

                if args_cli.useonnx:
                    torch_actions = actions
                    actions = onnx_policy(obs)
                    if (actions - torch_actions).abs().max() > 1e-5:
                        print(
                            "[INFO]: ONNX model and PyTorch model have a difference of"
                            f" {(actions - torch_actions).abs().max()} in actions at joint"
                            f" {((actions - torch_actions).abs() > 1e-5).nonzero(as_tuple=True)[0]}"
                        )
                if timestep < args_cli.zero_act_until:
                    actions[:] = 0.0
                # 将关节动作送入仿真，获得下一步观测和奖励信息。
                obs, rewards, dones, infos = env.step(actions)
            timestep += 1

            if args_cli.visual_ablation is not None and 0 < args_cli.ablation_steps <= timestep:
                break

            # exit the loop if video_length is meet
            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
    finally:
        # 无论窗口正常关闭还是提前退出，都关闭 CSV 文件，避免记录损坏。
        if ablation_file is not None:
            ablation_file.close()
            print(f"[INFO] Visual ablation results saved to: {ablation_csv_path}")

    # close the simulator
    env.close()

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
