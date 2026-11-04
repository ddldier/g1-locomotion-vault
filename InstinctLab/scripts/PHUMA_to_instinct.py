"""Convert PHUMA G1 ``.npy`` motions to InstinctLab retargeted ``.npz`` files.

PHUMA stores the floating base at ``pelvis`` and quaternions as ``xyzw``.
InstinctLab's motion reference stores the base at ``torso_link`` and uses
``wxyz`` quaternions.  Joint positions are kept in PHUMA's named 29-DOF order.
"""

from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
import tqdm


PHUMA_G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

_CHAIN = None
_SRC_TO_CHAIN = None


def _init_worker(urdf_path: str) -> None:
    global _CHAIN, _SRC_TO_CHAIN
    with open(urdf_path, encoding="utf-8") as f:
        _CHAIN = pk.build_chain_from_urdf(f.read())
    chain_names = _CHAIN.get_joint_parameter_names()
    missing = [name for name in chain_names if name not in PHUMA_G1_JOINT_NAMES]
    if missing:
        raise ValueError(f"URDF joints missing from PHUMA order: {missing}")
    _SRC_TO_CHAIN = torch.tensor(
        [PHUMA_G1_JOINT_NAMES.index(name) for name in chain_names], dtype=torch.long
    )


def _convert_file(src_tgt: tuple[str, str], invert_waist: bool) -> str:
    src_file, tgt_file = src_tgt
    motion = np.load(src_file, allow_pickle=True).item()
    required = {"root_trans", "root_ori", "dof_pos", "fps"}
    missing = required.difference(motion)
    if missing:
        raise ValueError(f"{src_file}: missing PHUMA fields {sorted(missing)}")

    root_pos = np.asarray(motion["root_trans"], dtype=np.float32)
    root_quat_xyzw = np.asarray(motion["root_ori"], dtype=np.float32)
    joint_pos_np = np.asarray(motion["dof_pos"], dtype=np.float32)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{src_file}: root_trans must have shape (T, 3)")
    if root_quat_xyzw.shape != (len(root_pos), 4):
        raise ValueError(f"{src_file}: root_ori must have shape (T, 4)")
    if joint_pos_np.shape != (len(root_pos), 29):
        raise ValueError(f"{src_file}: dof_pos must have shape (T, 29), got {joint_pos_np.shape}")

    output_joint_pos = joint_pos_np.copy()
    if invert_waist:
        for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
            output_joint_pos[:, PHUMA_G1_JOINT_NAMES.index(name)] *= -1.0

    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    joint_pos_chain = torch.from_numpy(output_joint_pos[:, _SRC_TO_CHAIN.numpy()])
    src_transform = pk.Transform3d(
        rot=torch.from_numpy(root_quat_wxyz),
        pos=torch.from_numpy(root_pos),
    )
    frame_indices = _CHAIN.get_frame_indices("pelvis", "torso_link")
    frame_poses = _CHAIN.forward_kinematics(joint_pos_chain, frame_indices)
    pelvis_poses = frame_poses["pelvis"]
    torso_poses = frame_poses["torso_link"]
    torso_in_pelvis = pelvis_poses.inverse().compose(torso_poses)
    target_transform = src_transform.compose(torso_in_pelvis)
    target_matrix = target_transform.get_matrix()
    target_quat_wxyz = pk.matrix_to_quaternion(target_matrix[:, :3, :3]).numpy()
    target_pos = target_matrix[:, :3, 3].numpy()

    os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
    np.savez(
        tgt_file,
        framerate=np.asarray(motion["fps"], dtype=np.float32),
        joint_names=np.asarray(PHUMA_G1_JOINT_NAMES),
        joint_pos=output_joint_pos,
        base_pos_w=target_pos.astype(np.float32),
        base_quat_w=target_quat_wxyz.astype(np.float32),
    )
    return tgt_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="PHUMA data/g1 file or directory")
    parser.add_argument("--tgt", required=True, help="Output directory or file")
    parser.add_argument(
        "--urdf",
        default=str(
            Path(__file__).resolve().parents[1]
            / "source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
        ),
    )
    parser.add_argument("--num_cpus", type=int, default=max(1, min(8, mp.cpu_count())))
    parser.add_argument(
        "--invert-waist",
        action="store_true",
        help="Invert waist joints if using a source with the GMR waist convention.",
    )
    args = parser.parse_args()

    src = Path(args.src).resolve()
    tgt = Path(args.tgt).resolve()
    if src.is_file():
        pairs = [(str(src), str(tgt))]
    else:
        pairs = []
        for src_file in sorted(src.rglob("*.npy")):
            rel_src = src_file.relative_to(src)
            rel = rel_src.with_name(f"{rel_src.stem}_retargeted.npz")
            pairs.append((str(src_file), str(tgt / rel)))
    if not pairs:
        raise FileNotFoundError(f"No .npy files found under {src}")

    worker = functools.partial(_convert_file, invert_waist=args.invert_waist)
    with mp.Pool(args.num_cpus, initializer=_init_worker, initargs=(args.urdf,)) as pool:
        list(tqdm.tqdm(pool.imap(worker, pairs), total=len(pairs), desc="Converting PHUMA"))
    print(f"Wrote {len(pairs)} files to {tgt}")


if __name__ == "__main__":
    main()
