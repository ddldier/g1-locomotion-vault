#!/usr/bin/env python3
"""Convert the Vault NPZ root pose from torso_link to pelvis.

The PHUMA/GMR export used by the original Vault file stores the floating base
at torso_link.  The pelvis-root Isaac asset needs the equivalent pelvis pose.
Joint values stay unchanged and are matched by name through the same official
rev1.0 URDF used by the pelvis Shadow task.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument(
        "--source-urdf",
        type=Path,
        required=True,
        help="The torso-root URDF used when the source NPZ was exported.",
    )
    args = parser.parse_args()

    src = np.load(args.src, allow_pickle=True)
    names = src["joint_names"].tolist()
    joint_pos = torch.as_tensor(src["joint_pos"], dtype=torch.float32)
    with args.source_urdf.open(encoding="utf-8") as f:
        chain = pk.build_chain_from_urdf(f.read())

    chain_names = chain.get_joint_parameter_names()
    joint_order = [names.index(name) for name in chain_names]
    frames = chain.forward_kinematics(
        joint_pos[:, joint_order], chain.get_frame_indices("pelvis", "torso_link")
    )
    pelvis_to_torso = frames["pelvis"].inverse().compose(frames["torso_link"])
    torso_world = pk.Transform3d(
        rot=torch.as_tensor(src["base_quat_w"], dtype=torch.float32),
        pos=torch.as_tensor(src["base_pos_w"], dtype=torch.float32),
    )
    pelvis_world = torso_world.compose(pelvis_to_torso.inverse())
    pelvis_matrix = pelvis_world.get_matrix()
    pelvis_pos = pelvis_matrix[:, :3, 3].numpy().astype(np.float32)
    pelvis_quat = pk.matrix_to_quaternion(pelvis_matrix[:, :3, :3]).numpy().astype(np.float32)

    # The legacy torso-root URDF reverses the parent/child direction of the
    # three waist joints relative to the official pelvis-root URDF.  Preserve
    # the same physical pose by changing their angle convention in the output.
    output_joint_pos = np.asarray(src["joint_pos"], dtype=np.float32).copy()
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        output_joint_pos[:, names.index(name)] *= -1.0

    output = {key: src[key] for key in src.files}
    output["joint_pos"] = output_joint_pos
    output["base_pos_w"] = pelvis_pos
    output["base_quat_w"] = pelvis_quat
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.dst, **output)

    print(f"wrote: {args.dst}")
    print(f"frames={len(pelvis_pos)} root_z={pelvis_pos[:, 2].min():.6f}..{pelvis_pos[:, 2].max():.6f}")


if __name__ == "__main__":
    main()
