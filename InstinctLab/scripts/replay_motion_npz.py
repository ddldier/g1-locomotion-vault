"""Replay retargeted G1 motion files in the Isaac Sim viewport.

This is a kinematic data viewer: poses are written directly to the simulated
robot. It does not run a policy, advance physics, or record video.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay G1 retargeted NPZ motions in Isaac Sim.")
parser.add_argument(
    "--motion",
    required=True,
    help="A retargeted .npz file, a selection .yaml file, or a directory of .npz files.",
)
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
parser.add_argument("--pause", type=float, default=0.5, help="Pause between motion clips in seconds.")
parser.add_argument("--height_offset", type=float, default=0.0, help="Extra root height in meters.")
parser.add_argument("--once", action="store_true", help="Play the selected clips once and exit.")
parser.add_argument(
    "--fixed_camera",
    action="store_true",
    help="Keep the initial camera fixed instead of following the robot.",
)
parser.add_argument(
    "--asset",
    choices=("legacy", "pelvis"),
    default="legacy",
    help="Robot asset matching the motion root: legacy torso-root or official pelvis-root.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.speed <= 0.0:
    parser.error("--speed must be greater than zero")
if args_cli.pause < 0.0:
    parser.error("--pause must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from instinctlab.assets.unitree_g1 import G1_29DOF_REV10_PELVIS_CFG, G1_29DOF_TORSOBASE_POPSICLE_CFG


REQUIRED_KEYS = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}


def resolve_motion_files(input_path: str) -> list[Path]:
    """Resolve one NPZ, every NPZ in a directory, or a YAML selection list."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Motion input does not exist: {path}")

    if path.is_dir():
        files = sorted(path.rglob("*.npz"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        with path.open(encoding="utf-8") as stream:
            selection = yaml.safe_load(stream)
        if not isinstance(selection, dict) or not isinstance(selection.get("selected_files"), list):
            raise ValueError(f"{path} must contain a 'selected_files' list")
        files = [(path.parent / item).resolve() for item in selection["selected_files"]]
    elif path.suffix.lower() == ".npz":
        files = [path]
    else:
        raise ValueError("--motion must point to an .npz file, YAML selection, or directory")

    if not files:
        raise FileNotFoundError(f"No .npz motion files found from: {path}")
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError("Motion files listed but not found:\n" + "\n".join(missing))
    return files


def load_motion(path: Path) -> dict[str, np.ndarray | float | list[str]]:
    """Load and validate the fields required by InstinctLab retargeted motions."""
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        motion = {key: archive[key].copy() for key in REQUIRED_KEYS}

    joint_names = [str(name) for name in motion["joint_names"].tolist()]
    joint_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
    base_pos = np.asarray(motion["base_pos_w"], dtype=np.float32)
    base_quat = np.asarray(motion["base_quat_w"], dtype=np.float32)
    framerate = float(np.asarray(motion["framerate"]).item())
    frame_count = joint_pos.shape[0]

    if framerate <= 0.0:
        raise ValueError(f"{path}: framerate must be greater than zero")
    if joint_pos.shape != (frame_count, len(joint_names)):
        raise ValueError(f"{path}: inconsistent joint_pos and joint_names shapes")
    if base_pos.shape != (frame_count, 3) or base_quat.shape != (frame_count, 4):
        raise ValueError(f"{path}: base_pos_w/base_quat_w have invalid shapes")
    if frame_count == 0:
        raise ValueError(f"{path}: motion has no frames")
    if not all(np.isfinite(array).all() for array in (joint_pos, base_pos, base_quat)):
        raise ValueError(f"{path}: motion contains NaN or infinite values")

    quat_norm = np.linalg.norm(base_quat, axis=1, keepdims=True)
    if np.any(quat_norm < 1.0e-6):
        raise ValueError(f"{path}: motion contains an invalid zero quaternion")

    return {
        "framerate": framerate,
        "joint_names": joint_names,
        "joint_pos": joint_pos,
        "base_pos_w": base_pos,
        "base_quat_w": base_quat / quat_norm,
    }


def create_scene(sim: SimulationContext) -> Articulation:
    """Create a ground plane, light, and the G1 used by the Parkour task."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = copy.deepcopy(
        G1_29DOF_REV10_PELVIS_CFG if args_cli.asset == "pelvis" else G1_29DOF_TORSOBASE_POPSICLE_CFG
    )
    robot_cfg.prim_path = "/World/Robot"
    if args_cli.asset == "legacy":
        shoe_urdf = (
            Path(__file__).resolve().parents[1]
            / "source/instinctlab/instinctlab/tasks/parkour/urdf/"
            "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
        )
        robot_cfg.spawn.asset_path = str(shoe_urdf)
    robot = Articulation(robot_cfg)

    sim.set_camera_view(eye=[3.0, -3.0, 2.2], target=[0.0, 0.0, 0.9])
    return robot


def replay_clip(sim: SimulationContext, robot: Articulation, path: Path) -> bool:
    """Replay one clip. Return False when the application has been closed."""
    motion = load_motion(path)
    joint_names = motion["joint_names"]
    robot_joint_names = list(robot.joint_names)
    missing_joints = [name for name in robot_joint_names if name not in joint_names]
    if missing_joints:
        raise ValueError(f"{path}: missing robot joints {missing_joints}")

    source_indices = [joint_names.index(name) for name in robot_joint_names]
    joint_pos_np = motion["joint_pos"][:, source_indices]
    base_pos_np = motion["base_pos_w"].copy()
    base_quat_np = motion["base_quat_w"]

    # Put every clip near the origin while retaining its vertical trajectory.
    base_pos_np[:, :2] -= base_pos_np[0, :2]
    base_pos_np[:, 2] += args_cli.height_offset

    joint_pos = torch.as_tensor(joint_pos_np, dtype=torch.float32, device=sim.device)
    base_pos = torch.as_tensor(base_pos_np, dtype=torch.float32, device=sim.device)
    base_quat = torch.as_tensor(base_quat_np, dtype=torch.float32, device=sim.device)
    zero_joint_vel = torch.zeros((1, robot.num_joints), dtype=torch.float32, device=sim.device)
    zero_root_vel = torch.zeros((1, 6), dtype=torch.float32, device=sim.device)

    frame_period = 1.0 / (motion["framerate"] * args_cli.speed)
    duration = len(joint_pos) / motion["framerate"]
    print(
        f"[INFO] Playing {path.name}: {len(joint_pos)} frames, "
        f"{motion['framerate']:.1f} Hz, {duration:.2f} s"
    )

    deadline = time.perf_counter()
    for frame_idx in range(len(joint_pos)):
        if not simulation_app.is_running():
            return False

        root_pose = torch.cat((base_pos[frame_idx], base_quat[frame_idx])).unsqueeze(0)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(zero_root_vel)
        robot.write_joint_state_to_sim(joint_pos[frame_idx].unsqueeze(0), zero_joint_vel)

        if not args_cli.fixed_camera and frame_idx % 3 == 0:
            x, y, z = base_pos_np[frame_idx]
            sim.set_camera_view(
                eye=[float(x + 3.0), float(y - 3.0), float(z + 1.4)],
                target=[float(x), float(y), float(z + 0.1)],
            )

        sim.render()
        robot.update(frame_period)
        deadline += frame_period
        time.sleep(max(0.0, deadline - time.perf_counter()))

    return True


def main() -> None:
    motion_files = resolve_motion_files(args_cli.motion)
    print(f"[INFO] Resolved {len(motion_files)} motion clip(s)")
    for index, path in enumerate(motion_files, start=1):
        print(f"[INFO]   {index}: {path}")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    robot = create_scene(sim)
    sim.reset()
    print(f"[INFO] Isaac Sim viewer ready ({args_cli.asset} asset). Close the window to stop playback.")

    keep_running = True
    while keep_running and simulation_app.is_running():
        for path in motion_files:
            keep_running = replay_clip(sim, robot, path)
            if not keep_running:
                break
            if args_cli.pause:
                pause_until = time.perf_counter() + args_cli.pause
                while simulation_app.is_running() and time.perf_counter() < pause_until:
                    sim.render()
                    time.sleep(0.01)
        if args_cli.once:
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(skip_cleanup=args_cli.headless and args_cli.once)
