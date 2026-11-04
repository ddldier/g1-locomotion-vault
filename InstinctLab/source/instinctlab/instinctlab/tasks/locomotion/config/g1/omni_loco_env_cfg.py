"""Plane-only unified locomotion teacher for the G1 29-DoF robot.

The policy interface in this task is the frozen 623-D interface intended for
later multi-expert distillation.  The existing G1 flat task remains unchanged.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.locomotion.mdp as locomotion_mdp
from instinctlab.tasks.shadowing.beyondmimic.config.g1 import beyondmimic_vault_cfg
from instinctlab.assets.unitree_g1 import G1_29DOF_REV10_PELVIS_CFG, beyondmimic_g1_29dof_actuators
from instinctlab.envs import InstinctRlEnv
from instinctlab.monitors import MonitorTermCfg

from .flat_env_cfg import (
    G1FlatActionsCfg,
    G1FlatCommandsCfg,
    G1FlatCriticObsCfg,
    G1FlatEnvCfg,
    G1FlatObservationsCfg,
    G1FlatRewardsCfg,
    G1FlatSceneCfg,
)


OMNI_LOCO_TASK_ID = "Instinct-Unified-OmniLoco-G1-v0"
OMNI_LOCO_OBSERVATION_TERMS = (
    "height_scan",
    "velocity_command",
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel",
    "last_action",
)
OMNI_LOCO_OBSERVATION_DIMS = (527, 3, 3, 3, 29, 29, 29)
# 继续扩大速度范围：后退最低 -1.0 m/s，前进最高 1.5 m/s，转向最高 ±1.3 rad/s。
# 暂时不启用速度课程，先验证策略在目标速度范围内是否能够稳定行走。
OMNI_LOCO_COMMAND_RANGES = ((-1.5, 1.8), (-0.8, 0.8), (-1.5, 1.5))
# 对齐 Unitree RL Lab G1 29DOF 的 2% 站立采样比例，配置和运行时检查共用。
OMNI_LOCO_STANDING_RATIO = 0.02
OMNI_LOCO_JOINT_ORDER = (
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)
OMNI_LOCO_PELVIS_JOINT_ORDER = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)


@configclass
class G1OmniLocoSceneCfg(G1FlatSceneCfg):
    """The original plane scene plus the frozen 31 x 17 height scan."""

    height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        update_period=0.02,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(3.0, 1.6)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class G1OmniLocoPelvisSceneCfg(G1OmniLocoSceneCfg):
    """OmniLoco scene using the official pelvis-root rev1.0 asset."""

    robot = G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        update_period=0.02,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(3.0, 1.6)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class G1OmniLocoActionsCfg(G1FlatActionsCfg):
    def __post_init__(self):
        self.joint_pos.scale = 1.0
        self.joint_pos.use_default_offset = True


@configclass
class G1OmniLocoCommandsCfg(G1FlatCommandsCfg):
    def __post_init__(self):
        self.base_velocity.resampling_time_range = (10.0, 10.0)
        self.base_velocity.rel_standing_envs = OMNI_LOCO_STANDING_RATIO
        self.base_velocity.rel_heading_envs = 0.0
        self.base_velocity.heading_command = False
        self.base_velocity.ranges.lin_vel_x = OMNI_LOCO_COMMAND_RANGES[0]
        self.base_velocity.ranges.lin_vel_y = OMNI_LOCO_COMMAND_RANGES[1]
        self.base_velocity.ranges.ang_vel_z = OMNI_LOCO_COMMAND_RANGES[2]


@configclass
class LocoTeacherPolicyObsCfg(ObsGroup):
    """Frozen actor observation contract: 527 + 3 + 3 + 3 + 29 + 29 + 29."""

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )
    velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    projected_gravity = ObsTerm(
        func=instinct_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    base_ang_vel = ObsTerm(func=instinct_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
    last_action = ObsTerm(func=instinct_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class G1OmniLocoObservationsCfg(G1FlatObservationsCfg):
    policy: LocoTeacherPolicyObsCfg = LocoTeacherPolicyObsCfg()
    critic: G1FlatCriticObsCfg = G1FlatCriticObsCfg()


@configclass
class G1OmniLocoMonitorCfg:
    tracking = MonitorTermCfg(func=locomotion_mdp.LocoTrackingMonitorTerm)


@configclass
class G1OmniLocoRewardsCfg(G1FlatRewardsCfg):
    """OmniLoco foot rewards aligned with Unitree RL Lab's G1 recipe."""

    # Disable the inherited reward that keeps paying for prolonged single stance.
    feet_air_time = None

    # 官方 G1 配置没有这两项额外站立惩罚，关闭以避免重复约束腿部。
    # 本轮仅对齐站立采样和步态门控；仍继承本项目原有 stand_still=-0.8，
    # 并非完整复制官方奖励配方，便于从正常的 24000 代模型做最小改动续训。
    stand_still_leg_pos = None
    stand_still_leg_vel = None

    # 仅覆盖 OmniLoco：加强竖直速度惩罚，降低上下弹跳的收益。
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # 上身关节速度惩罚：抑制腰部、肩部、肘部和腕部的快速来回摆动。
    # 权重先设为较小值，避免把手臂动作限制得过于僵硬。
    upper_body_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-3.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_.*",
                    ".*_shoulder_.*",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )

    # 略微加强腰部偏离默认姿态的惩罚，帮助减少上身侧向晃动。
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
    )

    # 肩部位置偏差惩罚：比父类的 -0.1 更强，减少肩膀随行走左右乱摆。
    # 仍保留一定摆臂空间，避免把手臂完全锁死。
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
            )
        },
    )

    # 接触历史中两脚都没有超过 1 N 的接触力时，施加腾空惩罚。
    both_feet_airborne = RewTerm(
        func=locomotion_mdp.both_feet_airborne_penalty,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "contact_threshold": 1.0,
        },
    )

    # Unitree's periodic contact target: 0.8 s cycle, half-cycle left/right
    # phase offset, and 55% stance portion.
    # command_name 启用官方零命令门控：命令范数 <= 0.1 时不给交替踏步奖励。
    feet_gait = RewTerm(
        func=locomotion_mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
        },
    )

    # Reward swing feet passing near 10 cm clearance while moving.
    foot_clearance = RewTerm(
        func=locomotion_mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "target_height": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )

    # Replace InstinctLab's generic contact_slide with Isaac Lab's official
    # feet-only sliding penalty used by Unitree RL Lab.
    feet_slide = RewTerm(
        func=velocity_mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
        },
    )

    # Unitree RL Lab's official G1 recipe does not use an explicit
    # left/right completed-phase symmetry penalty.
    feet_air_contact_symmetry = None


@configclass
class G1OmniLocoEnvCfg(G1FlatEnvCfg):
    """A3 training configuration; all baseline rewards/events/PPO settings are inherited."""

    scene: G1OmniLocoSceneCfg = G1OmniLocoSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: G1OmniLocoActionsCfg = G1OmniLocoActionsCfg()
    commands: G1OmniLocoCommandsCfg = G1OmniLocoCommandsCfg()
    observations: G1OmniLocoObservationsCfg = G1OmniLocoObservationsCfg()
    rewards: G1OmniLocoRewardsCfg = G1OmniLocoRewardsCfg()
    monitors: G1OmniLocoMonitorCfg = G1OmniLocoMonitorCfg()

    def __post_init__(self):
        super().__post_init__()
        # The parent flat task replaces this with a per-joint scale.  The unified
        # teacher contract deliberately overrides it after the parent hook.
        self.actions.joint_pos.scale = 1.0
        self.actions.joint_pos.use_default_offset = True
        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = OMNI_LOCO_STANDING_RATIO
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = OMNI_LOCO_COMMAND_RANGES[0]
        self.commands.base_velocity.ranges.lin_vel_y = OMNI_LOCO_COMMAND_RANGES[1]
        self.commands.base_velocity.ranges.ang_vel_z = OMNI_LOCO_COMMAND_RANGES[2]
        self.run_name = "G1UnifiedOmniLocoPlane_obs623_scale1"


@configclass
class G1OmniLocoEnvCfg_PLAY(G1OmniLocoEnvCfg):
    scene: G1OmniLocoSceneCfg = G1OmniLocoSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        # 仅在 Play 场景加入与 Shadow NPZ 完全一致的固定 VaultBox；训练环境不加，
        # 因而不会改变原 Omni Teacher 的观测/奖励契约。
        self.scene.vault_box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/VaultBox",
            spawn=sim_utils.MeshCuboidCfg(
                size=beyondmimic_vault_cfg.VAULT_BOX_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.6, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=beyondmimic_vault_cfg.VAULT_BOX_POS),
        )
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1OmniLocoPelvisEnvCfg(G1OmniLocoEnvCfg):
    """Pelvis-root OmniLoco teacher; the torso-root task remains unchanged."""

    scene: G1OmniLocoPelvisSceneCfg = G1OmniLocoPelvisSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "G1UnifiedOmniLocoPelvisRev10_obs623_scale1"


@configclass
class G1OmniLocoPelvisEnvCfg_PLAY(G1OmniLocoPelvisEnvCfg):
    scene: G1OmniLocoPelvisSceneCfg = G1OmniLocoPelvisSceneCfg(num_envs=1, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.events.base_external_force_torque = None
        self.events.push_robot = None


class G1OmniLocoTeacherEnv(InstinctRlEnv):
    """Environment with fail-fast checks for the teacher interface contract."""

    expected_standing_ratio = OMNI_LOCO_STANDING_RATIO
    expected_command_ranges = OMNI_LOCO_COMMAND_RANGES
    expected_policy_terms = OMNI_LOCO_OBSERVATION_TERMS
    expected_policy_dims = OMNI_LOCO_OBSERVATION_DIMS
    expected_scan_rays = 527
    allow_curriculum = False
    expected_joint_order = OMNI_LOCO_JOINT_ORDER

    def load_managers(self):
        super().load_managers()
        self._validate_teacher_contract()

    def _validate_teacher_contract(self):
        errors: list[str] = []

        policy_terms = tuple(self.observation_manager.active_terms["policy"])
        policy_dims = tuple(math.prod(dim) for dim in self.observation_manager.group_obs_term_dim["policy"])
        if policy_terms != self.expected_policy_terms:
            errors.append(f"policy terms {policy_terms} != {self.expected_policy_terms}")
        if policy_dims != self.expected_policy_dims or sum(policy_dims) != sum(self.expected_policy_dims):
            errors.append(f"policy dims {policy_dims} (sum={sum(policy_dims)}) != {self.expected_policy_dims}")

        if self.expected_scan_rays is not None:
            scanner = self.scene.sensors["height_scanner"]
            ray_count = int(scanner.data.ray_hits_w.shape[1])
            if ray_count != self.expected_scan_rays:
                errors.append(f"height scan has {ray_count} rays, expected {self.expected_scan_rays}")

        action_term = self.action_manager.get_term("joint_pos")
        if self.action_manager.total_action_dim != 29:
            errors.append(f"action dimension is {self.action_manager.total_action_dim}, expected 29")
        if tuple(action_term._joint_names) != self.expected_joint_order:
            errors.append(f"resolved joint order differs: {tuple(action_term._joint_names)}")
        if not action_term.cfg.use_default_offset:
            errors.append("JointPositionAction use_default_offset is false")
        if action_term.cfg.clip is not None:
            errors.append(f"JointPositionAction clip differs from the unclipped Vault action contract: {action_term.cfg.clip}")
        scale = action_term._scale
        if isinstance(scale, torch.Tensor):
            scale_ok = bool(torch.allclose(scale, torch.ones_like(scale)))
        else:
            scale_ok = float(scale) == 1.0
        if not scale_ok:
            errors.append(f"action scale is {scale}, expected 1.0")

        if self.cfg.sim.dt != 0.005 or self.cfg.decimation != 4 or not math.isclose(self.step_dt, 0.02):
            errors.append(
                f"timing is dt={self.cfg.sim.dt}, decimation={self.cfg.decimation}, step_dt={self.step_dt}; "
                "expected 0.005, 4, 0.02"
            )

        command_cfg = self.cfg.commands.base_velocity
        resolved_ranges = (
            tuple(command_cfg.ranges.lin_vel_x),
            tuple(command_cfg.ranges.lin_vel_y),
            tuple(command_cfg.ranges.ang_vel_z),
        )
        if resolved_ranges != self.expected_command_ranges:
            errors.append(f"command ranges {resolved_ranges} != {self.expected_command_ranges}")
        if command_cfg.heading_command or command_cfg.rel_heading_envs != 0.0:
            errors.append("heading command is enabled")
        if not math.isclose(command_cfg.rel_standing_envs, self.expected_standing_ratio):
            errors.append(
                f"standing-command ratio is {command_cfg.rel_standing_envs}, expected {self.expected_standing_ratio}"
            )
        if tuple(command_cfg.resampling_time_range) != (10.0, 10.0):
            errors.append(f"command hold is {command_cfg.resampling_time_range}, expected 10 seconds")
        if self.curriculum_manager.active_terms and not self.allow_curriculum:
            errors.append(f"curriculum terms are active: {self.curriculum_manager.active_terms}")

        actual_actuators = self.cfg.scene.robot.actuators
        if tuple(actual_actuators) != tuple(beyondmimic_g1_29dof_actuators):
            errors.append(f"actuator groups {tuple(actual_actuators)} differ from the Vault/BeyondMimic contract")
        else:
            for name, expected in beyondmimic_g1_29dof_actuators.items():
                actual = actual_actuators[name]
                if actual.to_dict() != expected.to_dict():
                    errors.append(f"actuator group '{name}' differs from the Vault/BeyondMimic contract")

        default_joint_pos = self.scene["robot"].data.default_joint_pos[:, action_term._joint_ids]
        expected_offset = default_joint_pos
        if isinstance(action_term._offset, torch.Tensor):
            if not torch.allclose(action_term._offset, expected_offset):
                errors.append("action offset differs from the default pose")
        else:
            errors.append("action offset is not the per-joint default pose")

        # Functional semantics: zero maps to default q, and +/-0.1 maps to
        # default q +/-0.1 for every resolved joint.
        zero = torch.zeros_like(action_term.raw_actions)
        action_term.process_actions(zero)
        if not torch.allclose(action_term.processed_actions, expected_offset, atol=1.0e-6, rtol=0.0):
            errors.append("zero action does not map to the default pose")
        if self.num_envs >= 29:
            probe = torch.zeros_like(zero)
            probe[:29, :] = 0.1 * torch.eye(29, device=self.device)
            action_term.process_actions(probe)
            if not torch.allclose(
                action_term.processed_actions[:29], expected_offset[:29] + probe[:29], atol=1.0e-6, rtol=0.0
            ):
                errors.append("+0.1 action probe does not map to default pose +0.1")
            action_term.process_actions(-probe)
            if not torch.allclose(
                action_term.processed_actions[:29], expected_offset[:29] - probe[:29], atol=1.0e-6, rtol=0.0
            ):
                errors.append("-0.1 action probe does not map to default pose -0.1")
        else:
            for joint_index in range(29):
                probe = torch.zeros_like(zero)
                probe[:, joint_index] = 0.1
                action_term.process_actions(probe)
                if not torch.allclose(action_term.processed_actions, expected_offset + probe, atol=1.0e-6, rtol=0.0):
                    errors.append(f"+0.1 action probe failed for joint {joint_index}")
                action_term.process_actions(-probe)
                if not torch.allclose(action_term.processed_actions, expected_offset - probe, atol=1.0e-6, rtol=0.0):
                    errors.append(f"-0.1 action probe failed for joint {joint_index}")
        action_term.process_actions(zero)

        if errors:
            raise RuntimeError("Unified OmniLoco teacher contract failed:\n  - " + "\n  - ".join(errors))
        print(
            f"[A1 PASS] Unified OmniLoco contract: policy={sum(self.expected_policy_dims)} "
            f"(height_scan={self.expected_scan_rays}), action=29, "
            f"scale=1.0, dt=0.005, decimation=4, command_hold=10s, stand={self.expected_standing_ratio:.2f}"
        )


class G1OmniLocoPelvisTeacherEnv(G1OmniLocoTeacherEnv):
    """Contract validator for the official pelvis-root asset."""

    expected_joint_order = OMNI_LOCO_PELVIS_JOINT_ORDER

    def _validate_teacher_contract(self):
        super()._validate_teacher_contract()
