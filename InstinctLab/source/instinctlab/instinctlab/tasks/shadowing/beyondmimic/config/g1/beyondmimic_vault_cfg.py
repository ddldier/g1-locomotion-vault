import os

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.tasks.shadowing.beyondmimic.beyondmimic_env_cfg as beyondmimic_cfg
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.monitors import (
    ActuatorMonitorTerm,
    MonitorTermCfg,
    RewardSumMonitorTerm,
    ShadowingJointReferenceMonitorTerm,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

# The legacy Vault configuration remains torso-root so old torso checkpoints
# and play tasks keep their original contract.  The pelvis-root variant below
# explicitly overrides this path with ``VAULT_FK_URDF``.
_G1_RESOURCES_DIR = os.path.dirname(os.path.dirname(G1_CFG.spawn.asset_path))
VAULT_FK_URDF = os.path.join(_G1_RESOURCES_DIR, "official_rev10", "g1_29dof_rev_1_0.urdf")
MOTION_NAME = "MimicKitSpeedVault"
MOTION_DIR = os.path.join(os.path.dirname(__file__), "motions", "vault")

VAULT_MOTION_FILE = "g1_speed_vault_retargeted.npz"
VAULT_PELVIS_MOTION_FILE = "g1_speed_vault_pelvis_retargeted.npz"
MOTION_SELECTION_FILE = os.path.join(MOTION_DIR, "g1_vault_only_motion_selection.yaml")
PELVIS_MOTION_SELECTION_FILE = os.path.join(MOTION_DIR, "g1_vault_pelvis_motion_selection.yaml")
VAULT_BOX_HIDDEN_POS = (1000.0, 1000.0, -100.0)

with np.load(os.path.join(MOTION_DIR, VAULT_MOTION_FILE)) as _vault_npz:
    _size = _vault_npz["obstacle_size"]
    _pos = _vault_npz["obstacle_pos"]
    VAULT_BOX_SIZE: tuple[float, float, float] = (float(_size[0]), float(_size[1]), float(_size[2]))
    VAULT_BOX_POS: tuple[float, float, float] = (float(_pos[0]), float(_pos[1]), float(_pos[2]))


@configclass
class VaultMotionCfg(AmassMotionCfgBase):
    """Converted MimicKit G1 vault and flat locomotion motions."""

    path = MOTION_DIR
    retargetting_func = None
    filtered_motion_selection_filepath = MOTION_SELECTION_FILE
    supported_file_endings = ["retargeted.npz"]
    motion_target_framerate = 50.0
    motion_start_from_middle_range = [0.0, 0.8]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    motion_bin_length_s = 1.0
    env_starting_stub_sampling_strategy = "concat_motion_bins"
    _hacked_selected_file = VAULT_MOTION_FILE


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.0,
    update_period=0.02,
    num_frames=1,
    visualizing_robot_offset=(0.0, 2.0, 0.0),
    visualizing_robot_from="reference_frame",
    motion_buffers={
        MOTION_NAME: VaultMotionCfg(),
    },
    # This task contains one selected motion file; every process must see the
    # complete clip instead of splitting it as if it were a multi-clip set.
    mp_split_method="None",
)


def _vault_box_cfg(debug_vis: bool = False) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/VaultBox",
        spawn=sim_utils.MeshCuboidCfg(
            size=VAULT_BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3) if not debug_vis else (0.0, 0.6, 0.9)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=VAULT_BOX_POS),
    )


def reset_vault_box_by_motion(
    env,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("vault_box"),
    motion_buffer_name: str = MOTION_NAME,
    vault_motion_file: str = VAULT_MOTION_FILE,
    active_pos: tuple[float, float, float] = VAULT_BOX_POS,
    hidden_pos: tuple[float, float, float] = VAULT_BOX_HIDDEN_POS,
):
    """Show the vault box only for envs that sampled the vault motion."""

    if asset_cfg.name not in env.scene.rigid_objects:
        return

    vault_box = env.scene[asset_cfg.name]
    motion_reference = env.scene[motion_ref_cfg.name]
    motion_buffer = motion_reference.motion_buffers[motion_buffer_name]
    assigned_ids = motion_buffer.env_ids_to_assigned_ids(env_ids).to(motion_buffer.buffer_device)
    motion_ids = motion_buffer._assigned_env_motion_selection[assigned_ids].detach().cpu().tolist()
    current_motion_files = [motion_buffer._all_motion_files[motion_id] for motion_id in motion_ids]

    is_vault_motion = torch.tensor(
        [os.path.basename(motion_file) == vault_motion_file for motion_file in current_motion_files],
        dtype=torch.bool,
        device=env.device,
    )

    root_state = vault_box.data.default_root_state[env_ids].clone()
    root_state[:, :3] = torch.tensor(hidden_pos, dtype=root_state.dtype, device=env.device)
    root_state[:, :3] += env.scene.env_origins[env_ids]
    if is_vault_motion.any():
        root_state[is_vault_motion, :3] = torch.tensor(active_pos, dtype=root_state.dtype, device=env.device)
        root_state[is_vault_motion, :3] += env.scene.env_origins[env_ids[is_vault_motion]]

    root_state[:, 7:] = 0.0
    vault_box.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    vault_box.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def _sync_link_observation_terms(env_cfg):
    if env_cfg.observations.policy.__dict__.get("link_pos", None) is not None:
        env_cfg.observations.policy.link_pos.params["asset_cfg"].body_names = (
            env_cfg.scene.motion_reference.link_of_interests
        )
    if env_cfg.observations.policy.__dict__.get("link_rot", None) is not None:
        env_cfg.observations.policy.link_rot.params["asset_cfg"].body_names = (
            env_cfg.scene.motion_reference.link_of_interests
        )
    if hasattr(env_cfg.observations, "critic") and env_cfg.observations.critic is not None:
        if env_cfg.observations.critic.__dict__.get("link_pos", None) is not None:
            env_cfg.observations.critic.link_pos.params["asset_cfg"].body_names = (
                env_cfg.scene.motion_reference.link_of_interests
            )
        if env_cfg.observations.critic.__dict__.get("link_rot", None) is not None:
            env_cfg.observations.critic.link_rot.params["asset_cfg"].body_names = (
                env_cfg.scene.motion_reference.link_of_interests
            )


def _apply_vault_task_settings(env_cfg, debug_vis: bool = False):
    env_cfg.scene.vault_box = _vault_box_cfg(debug_vis=debug_vis)
    env_cfg.scene.robot.actuators = beyondmimic_g1_29dof_actuators
    env_cfg.actions.joint_pos.scale = beyondmimic_action_scale
    env_cfg.events.push_robot = None
    env_cfg.events.bin_fail_counter_smoothing = None
    env_cfg.events.reset_vault_box = EventTermCfg(
        func=reset_vault_box_by_motion,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("vault_box"),
            "motion_buffer_name": MOTION_NAME,
            "vault_motion_file": VAULT_MOTION_FILE,
            "active_pos": VAULT_BOX_POS,
            "hidden_pos": VAULT_BOX_HIDDEN_POS,
        },
    )
    env_cfg.curriculum.beyond_adaptive_sampling = None

    env_cfg.terminations.base_pos_too_far.params["distance_threshold"] = 0.5
    env_cfg.terminations.base_pos_too_far.params["height_only"] = False
    env_cfg.terminations.link_pos_too_far.params["distance_threshold"] = 0.5
    env_cfg.terminations.link_pos_too_far.params["height_only"] = False
    env_cfg.terminations.link_pos_too_far.params["reference_cfg"].body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]

    # Vaulting may legitimately touch the obstacle with the hands/wrists.
    env_cfg.rewards.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        (
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
            r"(?!left_wrist_roll_link$)(?!left_wrist_pitch_link$)(?!left_wrist_yaw_link$)"
            r"(?!right_wrist_roll_link$)(?!right_wrist_pitch_link$)(?!right_wrist_yaw_link$).+$"
        )
    ]

    _sync_link_observation_terms(env_cfg)

    assert len(list(env_cfg.scene.motion_reference.motion_buffers.keys())) == 1


@configclass
class G1BeyondMimicVaultEnvCfg(beyondmimic_cfg.BeyondMimicEnvCfg):
    """G1 BeyondMimic vault environment using the MimicKit speed-vault clip."""

    scene: beyondmimic_cfg.BeyondMimicSceneCfg = beyondmimic_cfg.BeyondMimicSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_vault_task_settings(self, debug_vis=False)

        self.run_name = "G1BeyondMimic_MimicKitSpeedVault_noPush"


@configclass
class G1BeyondMimicVaultEnvCfg_PLAY(G1BeyondMimicVaultEnvCfg):
    """G1 BeyondMimic vault environment for playing/evaluation."""

    scene: beyondmimic_cfg.BeyondMimicSceneCfg = beyondmimic_cfg.BeyondMimicSceneCfg(
        num_envs=1,
        env_spacing=5.0,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        robot_reference=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference"),
        motion_reference=motion_reference_cfg.replace(
            debug_vis=True,
        ),
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=[4.0, 1.5, 1.2],
        lookat=[1.0, 0.5, 0.4],
        origin_type="asset_root",
        asset_name="robot",
    )

    def __post_init__(self):
        super().__post_init__()
        _apply_vault_task_settings(self, debug_vis=True)

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 3
            self.scene.terrain.terrain_generator.num_cols = 3
            self.scene.terrain.terrain_generator.curriculum = False

        self.scene.motion_reference.symmetric_augmentation_joint_mapping = None
        self.scene.motion_reference.visualizing_marker_types = ["relative_links"]
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None

        self.episode_length_s = 6000.0
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_bin_length_s = None

        for term in self.terminations.__dict__.values():
            if term is None:
                continue
            if "print_reason" in term.params:
                term.params["print_reason"] = True

        for cmd in self.commands.__dict__.values():
            cmd.debug_vis = True

        self.monitors.shoulder_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_shoulder_roll.*"),
                "torque_plot_scale": 1e-2,
                "joint_power_plot_scale": 1e-1,
            },
        )
        self.monitors.waist_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names="waist_roll.*"),
                "torque_plot_scale": 1e-2,
                "joint_power_plot_scale": 1e-1,
            },
        )
        self.monitors.knee_actuator = MonitorTermCfg(
            func=ActuatorMonitorTerm,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_knee.*"),
                "torque_plot_scale": 1e-2,
                "joint_power_plot_scale": 1e-1,
            },
        )
        self.monitors.reward_sum = MonitorTermCfg(
            func=RewardSumMonitorTerm,
        )
        self.monitors.reference_stat_case = MonitorTermCfg(
            func=ShadowingJointReferenceMonitorTerm,
            params=dict(
                reference_cfg=SceneEntityCfg(
                    "motion_reference",
                    joint_names=[
                        "left_hip_pitch.*",
                    ],
                ),
            ),
        )
