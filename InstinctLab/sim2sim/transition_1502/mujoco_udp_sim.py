#!/usr/bin/env python3
"""Pelvis-root MuJoCo playback that delegates actor inference to the C++ UDP server."""

from __future__ import annotations

import argparse
import math
import socket
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models/g1_rev10_transition.xml"
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
DEFAULTS = {"hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6}
DEFAULTS_BY_NAME = {
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
POLICY_DT = 0.02
RECOVERY_HOLD_S = 0.6
ACTION_OFFSET = 7
HEIGHT_POINTS = tuple(
    (float(x), float(y))
    for y in np.arange(-0.8, 0.8 + 0.05, 0.1)
    for x in np.arange(-1.5, 1.5 + 0.05, 0.1)
)
assert len(HEIGHT_POINTS) == 527


def joint_default(name: str) -> float:
    if name in DEFAULTS_BY_NAME:
        return DEFAULTS_BY_NAME[name]
    for suffix, value in DEFAULTS.items():
        if suffix in name:
            return value
    return 0.0


class Playback:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "vault_box")
        free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self.root_qadr = int(model.jnt_qposadr[free_id])
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS]
        self.actuator_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_position") for name in JOINTS
        ]
        if (self.pelvis_id < 0 or self.box_geom_id < 0 or self.root_qadr < 0
                or min(self.joint_ids + self.actuator_ids) < 0):
            raise RuntimeError("Generated MJCF does not match the pelvis Student joint contract")

        qadr = self.model.jnt_qposadr[self.joint_ids]
        vadr = self.model.jnt_dofadr[self.joint_ids]
        self.qadr = qadr.astype(int)
        self.vadr = vadr.astype(int)
        self.defaults = np.asarray([joint_default(name) for name in JOINTS], dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """Return to the initial vault setup; called only at startup or on R."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.root_qadr:self.root_qadr + 7] = (
            0.0, 0.0, 0.7585, 1.0, 0.0, 0.0, 0.0
        )
        self.data.qpos[self.qadr] = self.defaults
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.actuator_ids] = self.defaults
        mujoco.mj_forward(self.model, self.data)
        self.command = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        self.requested_speed = 1.0
        self.requested_yaw = 0.0
        self.crossed_box = False
        self.crossed_at = None
        self.recovery_complete = False
        box_half = self.model.geom_size[self.box_geom_id]
        self.box_rear = self.data.geom_xpos[self.box_geom_id, 0] + box_half[0] + 0.15
        first = self.measure()
        self.history = {
            "height": deque([first[0]] * 3, maxlen=3),
            "gravity": deque([first[1]] * 5, maxlen=5),
            "ang_vel": deque([first[2]] * 5, maxlen=5),
            "joint_pos": deque([first[3]] * 5, maxlen=5),
            "joint_vel": deque([first[4]] * 5, maxlen=5),
            "last_action": deque([np.zeros(29, dtype=np.float32)] * 5, maxlen=5),
        }

    def measure(self):
        pos = self.data.xpos[self.pelvis_id]
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        scan = np.empty(527, dtype=np.float32)
        point = np.empty(3, dtype=np.float64)
        box_center = self.data.geom_xpos[self.box_geom_id]
        box_half = self.model.geom_size[self.box_geom_id]
        box_top = box_center[2] + box_half[2]
        for i, (x, y) in enumerate(HEIGHT_POINTS):
            point[:] = (pos[0] + c * x - s * y, pos[1] + s * x + c * y, pos[2])
            in_box = (abs(point[0] - box_center[0]) <= box_half[0]
                      and abs(point[1] - box_center[1]) <= box_half[1])
            hit_z = box_top if in_box else 0.0
            scan[i] = np.clip(pos[2] - hit_z - 0.5, -1.0, 1.0)

        gravity_world = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
        gravity = np.empty(3, dtype=np.float64)
        mujoco.mju_mulMatTVec3(gravity, rotation.reshape(-1), gravity_world)
        spatial = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.pelvis_id, spatial, 1)
        return (
            scan,
            gravity.astype(np.float32),
            spatial[:3].astype(np.float32),
            (self.data.qpos[self.qadr] - self.defaults).astype(np.float32),
            self.data.qvel[self.vadr].astype(np.float32),
        )

    def observation(self) -> np.ndarray:
        pos = self.data.xpos[self.pelvis_id]
        just_crossed = (not self.crossed_box) and (
            pos[0] > self.box_rear and abs(pos[1]) < 1.0
        )
        if just_crossed:
            self.crossed_box = True
            self.crossed_at = float(self.data.time)
            print("[SIM2SIM] Box rear crossed; holding vault command during recovery.", flush=True)
        self.recovery_complete = (
            self.crossed_at is not None
            and self.data.time - self.crossed_at >= RECOVERY_HOLD_S
        )
        self.command[0] = self.requested_speed if self.recovery_complete else 1.0
        self.command[1] = 0.0
        self.command[2] = self.requested_yaw if self.recovery_complete else 0.0
        h = self.history
        obs = np.concatenate((
            np.concatenate(tuple(h["height"])),
            self.command,
            np.concatenate(tuple(h["gravity"])),
            np.concatenate(tuple(h["ang_vel"])),
            np.concatenate(tuple(h["joint_pos"])),
            np.concatenate(tuple(h["joint_vel"])),
            np.concatenate(tuple(h["last_action"])),
        )).astype(np.float32, copy=False)
        if obs.shape != (2049,):
            raise RuntimeError(f"Observation shape changed: {obs.shape}")
        return obs

    def apply_action(self, action: np.ndarray) -> None:
        self.history["last_action"].append(action.astype(np.float32, copy=True))
        self.data.ctrl[self.actuator_ids] = self.defaults + action
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        sample = self.measure()
        for key, value in zip(("height", "gravity", "ang_vel", "joint_pos", "joint_vel"), sample):
            self.history[key].append(value)


def run(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    playback = Playback(model, data)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(args.timeout)
    sock.connect((args.host, args.port))
    viewer = None
    manual_reset_requested = False
    if not args.headless:
        import glfw
        from mujoco import viewer as mujoco_viewer

        def keyboard(keycode: int) -> None:
            nonlocal manual_reset_requested
            if keycode == glfw.KEY_R:
                manual_reset_requested = True
                return
            if not playback.recovery_complete:
                return
            if keycode in (glfw.KEY_W, glfw.KEY_UP):
                playback.requested_speed = min(1.5, playback.requested_speed + 0.1)
            elif keycode in (glfw.KEY_S, glfw.KEY_DOWN):
                playback.requested_speed = max(-1.0, playback.requested_speed - 0.1)
            elif keycode == glfw.KEY_SPACE:
                playback.requested_speed = 0.0
                playback.requested_yaw = 0.0
            elif keycode == glfw.KEY_Q:
                playback.requested_yaw = min(0.6, playback.requested_yaw + 0.1)
            elif keycode == glfw.KEY_E:
                playback.requested_yaw = max(-0.6, playback.requested_yaw - 0.1)
            print(
                f"post-vault command: vx={playback.requested_speed:+.1f} m/s, "
                f"wz={playback.requested_yaw:+.1f} rad/s",
                flush=True,
            )

        viewer = mujoco_viewer.launch_passive(model, data, key_callback=keyboard)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = playback.pelvis_id
        viewer.cam.lookat[:] = (0.6, 0.0, 0.78)
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -12.0
    print("MuJoCo ready: root=pelvis, vault cmd_x=1.0 m/s")
    print(f"Policy peer:  {args.host}:{args.port}; close viewer or Ctrl+C to stop")
    if viewer is not None:
        print(
            "After vault recovery: W/up +0.1 m/s, S/down -0.1 m/s, "
            "Q/E yaw +/-0.1 rad/s, Space stop; R manual reset."
        )
    count = 0
    next_tick = time.perf_counter()
    try:
        while (viewer is None or viewer.is_running()) and (args.steps <= 0 or count < args.steps):
            if manual_reset_requested:
                manual_reset_requested = False
                playback.reset()
                next_tick = time.perf_counter()
                print("[SIM2SIM] Manual reset by R", flush=True)
            observation = playback.observation()
            sock.send(observation.tobytes())
            response = sock.recv(29 * np.dtype(np.float32).itemsize)
            if len(response) != 29 * np.dtype(np.float32).itemsize:
                raise RuntimeError(f"Bad C++ action packet length: {len(response)}")
            action = np.frombuffer(response, dtype=np.float32).copy()
            playback.apply_action(action)
            count += 1
            if viewer is not None:
                viewer.sync()
                next_tick += POLICY_DT
                delay = next_tick - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
            if count % 100 == 0:
                pid = playback.pelvis_id
                print(f"step={count:6d} sim_t={data.time:6.2f}s pelvis_x={data.xpos[pid,0]:+.2f} "
                      f"pelvis_z={data.xpos[pid,2]:.2f} action_max={np.abs(action).max():.2f}", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
            time.sleep(0.2)
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="0 runs until the viewer closes")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
