#!/usr/bin/env python3
"""Build a standalone pelvis-root rev10 MJCF for the transition_1502 policy."""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = Path(
    "/root/gpufree-data/InstinctLab/source/instinctlab/instinctlab/assets/resources/"
    "unitree_g1/official_rev10/g1_29dof_rev_1_0.urdf"
)
DEFAULT_MOTION = Path(
    "/root/gpufree-data/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/"
    "beyondmimic/config/g1/motions/vault/g1_speed_vault_pelvis_retargeted.npz"
)
JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)


def gains(joint: str) -> tuple[float, float, float, float]:
    if "hip_pitch" in joint or "hip_yaw" in joint:
        return 40.17923847137318, 2.5578897650279457, 88.0, 0.01017752
    if "hip_roll" in joint or "knee" in joint:
        return 99.09842777666113, 6.3088018534966395, 139.0, 0.025101925
    if "ankle_" in joint:
        return 28.50124619574858, 1.814445686584846, 50.0, 0.00721945
    if joint.startswith("waist_yaw"):
        return 40.17923847137318, 2.5578897650279457, 88.0, 0.01017752
    if joint.startswith("waist_"):
        return 28.50124619574858, 1.814445686584846, 50.0, 0.00721945
    if "wrist_pitch" in joint or "wrist_yaw" in joint:
        return 16.77832748089279, 1.06814150219, 5.0, 0.00425
    if any(part in joint for part in ("shoulder_", "elbow", "wrist_roll")):
        return 14.25062309787429, 0.907222843292423, 25.0, 0.003609725
    raise ValueError(f"No actuator settings for joint {joint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "models/g1_rev10_transition.xml",
    )
    parser.add_argument("--box-distance", type=float, default=1.5)
    args = parser.parse_args()
    if args.box_distance <= 0:
        parser.error("--box-distance must be positive")

    assets_dir = args.output.parent / "g1_rev10_meshes"
    assets_dir.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(args.urdf.read_text())
    if root.tag != "robot":
        raise ValueError("Expected a URDF robot root element")
    compiler = root.find("mujoco/compiler")
    if compiler is None:
        raise ValueError("URDF must contain its MuJoCo compiler settings")
    compiler.set("meshdir", str(assets_dir.resolve()))
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None:
            continue
        mesh_path = (args.urdf.parent / filename).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        shutil.copy2(mesh_path, assets_dir / mesh_path.name)
        mesh.set("filename", mesh_path.name)

    # The supplied rev10 URDF comments out this free joint for IsaacLab imports.
    world = ET.Element("link", {"name": "world"})
    free_joint = ET.Element("joint", {"name": "floating_base_joint", "type": "floating"})
    ET.SubElement(free_joint, "parent", {"link": "world"})
    ET.SubElement(free_joint, "child", {"link": "pelvis"})
    root.insert(2, world)
    root.insert(3, free_joint)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.option.timestep = 0.005
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0.0, 0.0, 0.1),
        friction=(1.0, 0.005, 0.0001), rgba=(0.35, 0.36, 0.38, 1.0),
    )
    spec.worldbody.add_light(
        name="key_light",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
        pos=(0.0, -3.0, 6.0),
        dir=(0.0, 0.3, -1.0),
        active=True,
        castshadow=True,
        ambient=(0.35, 0.35, 0.35),
        diffuse=(0.8, 0.8, 0.8),
        specular=(0.2, 0.2, 0.2),
    )

    with np.load(args.motion, allow_pickle=False) as motion:
        size = motion["obstacle_size"].astype(float)
        position = motion["obstacle_pos"].astype(float)
    spec.worldbody.add_geom(
        name="vault_box", type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(args.box_distance + size[0] / 2, 0.0, position[2]),
        size=tuple(size / 2), friction=(1.0, 0.005, 0.0001),
        rgba=(0.12, 0.55, 0.65, 1.0),
    )

    for joint in JOINTS:
        kp, kd, effort, _armature = gains(joint)
        actuator = spec.add_actuator(
            name=f"{joint}_position", target=joint,
            trntype=mujoco.mjtTrn.mjTRN_JOINT, forcelimited=True,
            forcerange=(-effort, effort),
        )
        actuator.set_to_position(kp=kp, kv=kd)

    model = spec.compile()
    if model.nq != 36 or model.nv != 35 or model.nu != 29:
        raise RuntimeError(f"Unexpected MuJoCo dimensions nq={model.nq}, nv={model.nv}, nu={model.nu}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis") < 0:
        raise RuntimeError("Generated model is not pelvis-root")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link") < 0:
        raise RuntimeError("Generated model is missing torso_link")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_xml = ET.fromstring(spec.to_xml())
    for joint_cfg in generated_xml.findall(".//joint"):
        name = joint_cfg.get("name")
        if name in JOINTS:
            joint_cfg.set("armature", str(gains(name)[3]))
    args.output.write_text(ET.tostring(generated_xml, encoding="unicode"))
    # Re-read the generated artifact so path, mesh, joint, and actuator errors fail here.
    check = mujoco.MjModel.from_xml_path(str(args.output))
    if (check.nq, check.nv, check.nu) != (36, 35, 29):
        raise RuntimeError("Saved MJCF did not reload with the expected dimensions")
    print(f"MJCF:       {args.output}")
    print(f"Meshes:     {assets_dir} ({len(list(assets_dir.glob('*.STL')))} STL files)")
    print(f"Root body:  pelvis; torso body id {mujoco.mj_name2id(check, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')}")
    print(f"Dimensions: nq={check.nq}, nv={check.nv}, nu={check.nu}")
    print(f"Box:        size={size.tolist()}, center=({args.box_distance + size[0] / 2:.3f}, 0, {position[2]:.3f})")


if __name__ == "__main__":
    main()
