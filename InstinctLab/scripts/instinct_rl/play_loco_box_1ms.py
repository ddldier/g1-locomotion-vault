"""Play-only test: can the 2049D Unified Height Student vault from locomotion?

Scene: G1 starts facing +X on level ground; a fixed vault box is 3 m ahead.
Keys: W forward, S backward, A left, D right, all at 1 m/s; Space stops.
Only env 0 (locomotion) is the target; odd-numbered envs keep their Vault setup.
Training configuration and checkpoints are never modified.
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(
    description="Test locomotion -> vault with 1 m/s WASD control."
)
parser.add_argument(
    "--task",
    default="Instinct-Unified-Height-Play-G1-v0",
)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument(
    "--box_front_x",
    type=float,
    default=3.0,
    help="Position of the near face of the box along +X, relative to env origin (m).",
)

cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.task != "Instinct-Unified-Height-Play-G1-v0":
    parser.error("This test only supports Instinct-Unified-Height-Play-G1-v0.")
if args_cli.num_envs < 1:
    parser.error("--num_envs must be >= 1.")
if not math.isfinite(args_cli.box_front_x) or args_cli.box_front_x < 1.5:
    parser.error("--box_front_x must be finite and >= 1.5 m.")
if args_cli.headless:
    parser.error("Keyboard control needs an Isaac Sim GUI.")
if not args_cli.load_run or not args_cli.checkpoint:
    parser.error("Specify both --load_run and --checkpoint.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import carb.input
import omni.appwindow

from carb.input import KeyboardEventType
from instinct_rl.runners import OnPolicyRunner
import isaaclab.envs.mdp as mdp
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import instinctlab.tasks
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.tasks.shadowing.perceptive.config.g1 import (
    perceptive_vault_cfg as vault_play,
)


class CachedObsWrapper(InstinctRlVecEnvWrapper):
    """读取缓存中的多帧观测，不额外推进历史。"""

    def get_observations(self):
        pack = self._flatten_all_obs_groups(self.unwrapped.obs_buf)
        return pack["policy"], {"observations": pack}


def configure_test_scene(env_cfg, box_front_x: float):
    """只修改 Play 环境中的箱子和机器人重置配置。"""

    original_robot_reset = env_cfg.events.reset_robot.func
    original_box_reset = env_cfg.events.reset_vault_box.func

    box_pos = vault_play.vault_cfg.VAULT_BOX_POS
    box_size = vault_play.vault_cfg.VAULT_BOX_SIZE
    box_center = (
        box_front_x + box_size[0] / 2,
        box_pos[1],
        box_pos[2],
    )

    def reset_robot_for_test(env, env_ids, motion_ref_cfg, asset_cfg):
        loco_ids = env_ids[env_ids % 2 == 0]
        vault_ids = env_ids[env_ids % 2 == 1]

        if loco_ids.numel():
            mdp.reset_root_state_uniform(
                env,
                loco_ids,
                pose_range={
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                velocity_range={
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                asset_cfg=asset_cfg,
            )

            mdp.reset_joints_by_scale(
                env,
                loco_ids,
                position_range=(1.0, 1.0),
                velocity_range=(0.0, 0.0),
                asset_cfg=asset_cfg,
            )

        if vault_ids.numel():
            original_robot_reset(
                env,
                vault_ids,
                motion_ref_cfg=motion_ref_cfg,
                asset_cfg=asset_cfg,
            )

    def reset_box_for_test(env, env_ids, asset_cfg):
        loco_ids = env_ids[env_ids % 2 == 0]
        vault_ids = env_ids[env_ids % 2 == 1]

        if loco_ids.numel():
            # env 0 中保留箱子。
            vault_play.reset_vault_box_pose_by_motion_randomized(
                env,
                loco_ids,
                asset_cfg=asset_cfg,
                active_pos=box_center,
                pose_range={},
            )

        if vault_ids.numel():
            original_box_reset(
                env,
                vault_ids,
                asset_cfg=asset_cfg,
            )

    env_cfg.events.reset_robot.func = reset_robot_for_test
    env_cfg.events.reset_vault_box.func = reset_box_for_test
    env_cfg.observations.policy.enable_corruption = False

    if getattr(env_cfg.terminations, "base_contact", None) is not None:
        env_cfg.terminations.base_contact.params[
            "sensor_cfg"
        ].body_names = ["torso_link"]

    return box_center


def install_wasd_command(raw_env):
    """为 locomotion 设置 WASD 四方向 1m/s 速度命令。"""

    term = raw_env.command_manager.get_term("base_velocity")

    ids = torch.arange(
        raw_env.num_envs,
        device=raw_env.device,
    )
    loco_ids = ids[ids % 2 == 0]
    vault_ids = ids[ids % 2 == 1]

    # 机器人自身坐标系：
    # +X 前进，-X 后退，+Y 左移，-Y 右移。
    key_to_velocity = {
        "W": (1.0, 0.0, 0.0),
        "S": (-1.0, 0.0, 0.0),
        "A": (0.0, 1.0, 0.0),
        "D": (0.0, -1.0, 0.0),
        "SPACE": (0.0, 0.0, 0.0),
    }

    state = {
        "velocity": key_to_velocity["SPACE"]
    }

    def enforce():
        command = term.command

        # Locomotion 使用键盘速度。
        if loco_ids.numel():
            command[loco_ids, 0] = state["velocity"][0]
            command[loco_ids, 1] = state["velocity"][1]
            command[loco_ids, 2] = 0.0

        # Vault 环境仍然固定向前 1m/s。
        if vault_ids.numel():
            command[vault_ids, :] = 0.0
            command[vault_ids, 0] = 1.0

        # 避免 standing / heading 模式覆盖键盘命令。
        for name in ("is_standing_env", "is_heading_env"):
            flags = getattr(term, name, None)
            if flags is not None:
                flags[ids] = False

    # 保存环境原始速度命令更新方法。
    original_resample = term._resample_command
    original_update = term._update_command

    # 重新采样后恢复键盘命令。
    def resample(env_ids):
        original_resample(env_ids)
        enforce()

    def update():
        original_update()
        enforce()

    term._resample_command = resample
    term._update_command = update

    enforce()

    def set_direction(key: str):
        if key not in key_to_velocity:
            raise ValueError(f"Unsupported keyboard direction: {key}")

        target = key_to_velocity[key]

        if target != state["velocity"]:
            state["velocity"] = target
            print(
                f"[COMMAND] env 0: {key} -> "
                f"(vx, vy, wz) = {target}"
            )

        enforce()

    return set_direction, enforce


def refresh_current_command_observation(raw_env):
    """只刷新速度命令，不影响高度图和本体历史。"""

    policy_pack = raw_env.obs_buf["policy"]

    if not isinstance(policy_pack, dict):
        raise RuntimeError(
            "Expected concatenate_terms=False."
        )

    term_cfg = raw_env.cfg.observations.policy.velocity_command

    policy_pack["velocity_command"] = term_cfg.func(
        raw_env,
        **(term_cfg.params or {}),
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    agent_cfg = cli_args.parse_instinct_rl_cfg(
        args_cli.task,
        args_cli,
    )
    agent_cfg.load_run = args_cli.load_run

    if os.path.isabs(agent_cfg.load_run):
        checkpoint_path = get_checkpoint_path(
            os.path.dirname(agent_cfg.load_run),
            os.path.basename(agent_cfg.load_run),
            agent_cfg.load_checkpoint,
        )
    else:
        log_root = os.path.abspath(
            os.path.join(
                "logs",
                "instinct_rl",
                agent_cfg.experiment_name,
            )
        )
        checkpoint_path = get_checkpoint_path(
            log_root,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    box_center = configure_test_scene(
        env_cfg,
        args_cli.box_front_x,
    )

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    raw_env = env.unwrapped

    set_direction, enforce = install_wasd_command(raw_env)

    env = CachedObsWrapper(env)

    # 确认加载的是 2049D 多帧 Student。
    if env.num_obs != 2049:
        raise RuntimeError(
            f"Expected 2049D observations, got {env.num_obs}D."
        )

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )

    print("[LOAD]", checkpoint_path)
    runner.load(checkpoint_path)

    policy = runner.get_inference_policy(
        device=raw_env.device
    )

    obs, _ = env.get_observations()

    # ========== 键盘控制 ==========
    app_window = omni.appwindow.get_default_app_window()

    if app_window is None:
        raise RuntimeError("Isaac Sim window is unavailable.")

    keyboard = app_window.get_keyboard()

    if keyboard is None:
        raise RuntimeError("Keyboard device is unavailable.")

    input_interface = carb.input.acquire_input_interface()

    def on_key(event):
        if event.type not in (
            KeyboardEventType.KEY_PRESS,
            KeyboardEventType.KEY_REPEAT,
        ):
            return

        if event.input == carb.input.KeyboardInput.W:
            set_direction("W")

        elif event.input == carb.input.KeyboardInput.S:
            set_direction("S")

        elif event.input == carb.input.KeyboardInput.A:
            set_direction("A")

        elif event.input == carb.input.KeyboardInput.D:
            set_direction("D")

        elif event.input == carb.input.KeyboardInput.SPACE:
            set_direction("SPACE")

        # Q/E 和其他键不生效。

    keyboard_subscription = (
        input_interface.subscribe_to_keyboard_events(
            keyboard,
            on_key,
        )
    )

    print(
        f"[TEST] env 0 box center = {box_center}, "
        f"near face X={args_cli.box_front_x:.2f} m"
    )

    print("[TEST] W: forward +1 m/s | S: backward -1 m/s")
    print("[TEST] A: left +1 m/s | D: right -1 m/s")
    print("[TEST] SPACE: stop")
    print("[TEST] Press once to maintain the command.")
    print("[TEST] Click the Isaac Sim viewport before pressing keys.")

    # ========== 主循环 ==========
    crossed = False

    box_back_x = (
        box_center[0]
        + vault_play.vault_cfg.VAULT_BOX_SIZE[0] / 2
    )

    try:
        while simulation_app.is_running():
            with torch.inference_mode():

                # 确保键盘速度持续有效。
                enforce()

                # 刷新当前速度观测。
                refresh_current_command_observation(raw_env)

                obs, _ = env.get_observations()

                # Student 推理。
                actions = policy(obs)

                # 仿真。
                obs, rewards, dones, info = env.step(actions)

                # 检查 env 0 是否越过箱子。
                robot_x = (
                    raw_env.scene["robot"].data.root_pos_w[0, 0]
                    - raw_env.scene.env_origins[0, 0]
                ).item()

                if not crossed and robot_x > box_back_x + 0.5:
                    crossed = True
                    print(
                        f"[CROSSED X] env 0 pelvis x={robot_x:.2f} m"
                    )

                if bool(dones[0].item()):
                    manager = raw_env.termination_manager

                    reasons = [
                        name
                        for name in manager.active_terms
                        if bool(
                            manager.get_term(name)[0].item()
                        )
                    ]

                    print(
                        f"[RESET] env 0 x={robot_x:.2f} m, "
                        f"reasons={reasons}"
                    )

                    crossed = False

    finally:
        del keyboard_subscription
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()