"""Reference-free G1 height-history walk/vault/recovery task and task entry point."""
import math
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ViewerCfg, mdp
from isaaclab.managers import EventTermCfg as Event, ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from instinctlab.assets.unitree_g1 import G1_29DOF_REV10_PELVIS_CFG
from instinctlab.envs import InstinctRlEnv, InstinctLabRLEnvCfg
from .unified_transition_core import TransitionSettings
from . import unified_transition_mdp as task_mdp

with np.load(task_mdp.default_motion_file(),allow_pickle=False) as _z:
    BOX_SIZE = tuple(float(x) for x in _z["obstacle_size"])
    BOX_POS = tuple(float(x) for x in _z["obstacle_pos"])


@configclass
class EmptyCfg:
    pass


@configclass
class TransitionSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",terrain_type="plane",collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.,restitution=0.),
        debug_vis=False)
    robot = G1_29DOF_REV10_PELVIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    vault_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/VaultBox",
        spawn=sim_utils.MeshCuboidCfg(
            size=BOX_SIZE,activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12,0.55,0.65))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_POS))
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*",history_length=3,track_air_time=True)
    # One source body with multiple partner filters (PhysX one-to-many contract).
    box_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/VaultBox",history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/"+name for name in task_mdp.BOX_FILTER_BODIES])
    student_height_scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.,0.,20.)),ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1,size=(3.,1.6)),
        mesh_prim_paths=["/World/ground",MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_.*/VaultBox",track_mesh_transforms=True)],debug_vis=False)
    sky_light = AssetBaseCfg(prim_path="/World/skyLight",spawn=sim_utils.DomeLightCfg(intensity=850.))


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",joint_names=[".*"],
                                          scale=1.,use_default_offset=True)


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",resampling_time_range=(10.,10.),rel_standing_envs=0.0,
        rel_heading_envs=0.,heading_command=False,debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.,1.5),lin_vel_y=(0.,0.),ang_vel_z=(0.,0.)))


@configclass
class PolicyObsCfg(ObsGroup):
    # Keep the trained Student's term order, scaling and noise ranges.
    height_scan = ObsTerm(func=mdp.height_scan,params={"sensor_cfg":SceneEntityCfg("student_height_scanner")},clip=[-1.,1.])
    velocity_command = ObsTerm(func=task_mdp.velocity_command)
    projected_gravity = ObsTerm(func=mdp.projected_gravity,noise=UniformNoiseCfg(n_min=-0.05,n_max=0.05))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel,noise=UniformNoiseCfg(n_min=-0.2,n_max=0.2))
    joint_pos = ObsTerm(func=mdp.joint_pos_rel,noise=UniformNoiseCfg(n_min=-0.01,n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel,noise=UniformNoiseCfg(n_min=-0.5,n_max=0.5))
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False
        self.history_length = None
        self.height_scan.history_length = 3
        self.height_scan.flatten_history_dim = True
        self.velocity_command.history_length = 0
        for name in ("projected_gravity","base_ang_vel","joint_pos","joint_vel","last_action"):
            getattr(self,name).history_length = 5
            getattr(self,name).flatten_history_dim = True


@configclass
class CriticObsCfg(ObsGroup):
    state = ObsTerm(func=task_mdp.privileged_observation)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 0


@configclass
class ObservationsCfg:
    policy: PolicyObsCfg = PolicyObsCfg()
    critic: CriticObsCfg = CriticObsCfg()


@configclass
class EventsCfg:
    reset_scene = Event(func=task_mdp.reset_transition,mode="reset")


@configclass
class RewardsCfg:
    cross = RewTerm(func=task_mdp.event_reward,weight=20.,params={"event":"cross_event"})
    complete = RewTerm(func=task_mdp.event_reward,weight=30.,params={"event":"complete_event"})
    failure = RewTerm(func=task_mdp.event_reward,weight=-20.,params={"event":"failure_event"})
    approach = RewTerm(func=task_mdp.dense_reward,weight=2.,params={"name":"approach_rate"})
    progress = RewTerm(func=task_mdp.dense_reward,weight=1.,params={"name":"progress_rate"})
    speed = RewTerm(func=task_mdp.dense_reward,weight=0.5,params={"name":"speed_reward"})
    action_rate = RewTerm(func=mdp.action_rate_l2,weight=-0.01)
    torques = RewTerm(func=mdp.joint_torques_l2,weight=-4e-6)
    joint_limits = RewTerm(func=mdp.joint_pos_limits,weight=-0.2)


@configclass
class TerminationsCfg:
    # Failure updates the shared task state before rewards read it.
    # Completing the vault is a one-time reward/metric, not an episode reset:
    # the policy must keep walking after recovery until failure or timeout.
    failure = DoneTerm(func=task_mdp.failure)
    time_out = DoneTerm(func=mdp.time_out,time_out=True)


@configclass
class G1UnifiedTransitionEnvCfg(InstinctLabRLEnvCfg):
    scene: TransitionSceneCfg = TransitionSceneCfg(num_envs=4096,env_spacing=20.)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: EmptyCfg = EmptyCfg()
    monitors: EmptyCfg = EmptyCfg()
    transition: TransitionSettings = TransitionSettings()
    viewer: ViewerCfg = ViewerCfg(eye=(4.,-5.,2.5),lookat=(1.,0.,0.8),
                                  origin_type="asset_root",asset_name="robot")

    def __post_init__(self):
        self.transition.validate()
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.episode_length_s = self.transition.max_episode_s
        # Explicit finite-horizon MDP; critic sees time remaining.
        # Avoid the old PPO's approximate timeout bootstrap for this new task.
        self.is_finite_horizon = True
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.box_contacts.update_period = self.sim.dt
        self.run_name = "g1_unified_transition_ppo"


@configclass
class G1UnifiedTransitionEnvCfg_PLAY(G1UnifiedTransitionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        # Give interactive playback enough time to walk after a vault.
        # Keep training episodes at the original 12 s horizon.
        self.transition.max_episode_s = 30.0
        self.episode_length_s = self.transition.max_episode_s
        self.observations.policy.enable_corruption = False
        self.transition.rsi_probability = 0.  # Evaluate complete starts by default.


class G1UnifiedTransitionEnv(InstinctRlEnv):
    use_cached_policy_observations = True

    def load_managers(self):
        super().load_managers()
        expected_terms = ("height_scan","velocity_command","projected_gravity","base_ang_vel",
                          "joint_pos","joint_vel","last_action")
        dims = tuple(math.prod(d) for d in self.observation_manager.group_obs_term_dim["policy"])
        if tuple(self.observation_manager.active_terms["policy"]) != expected_terms or dims != (1581,3,15,15,145,145,145):
            raise RuntimeError(f"Student observation contract differs: {dims}")
        action = self.action_manager.get_term("joint_pos")
        if tuple(action._joint_names) != task_mdp.EXPECTED_JOINT_ORDER:
            raise RuntimeError("Resolved action joint order differs from the trained pelvis Student.")
        scale = torch.as_tensor(action._scale,device=self.device)
        if not torch.all(scale==1.) or not action.cfg.use_default_offset or action.cfg.clip is not None:
            raise RuntimeError("Student requires action scale=1, default offset, no clip.")
        if self.action_manager.total_action_dim != 29:
            raise RuntimeError("Student requires 29 actions.")

    def step(self, action):
        self.extras.pop("transition_episode",None)
        return super().step(action)

    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        records = None
        if hasattr(self,"_transition_state"):
            valid = ids[self.episode_length_buf[ids]>0]
            if valid.numel():
                records = self._transition_state.records(valid)
                records["episode_steps"] = self.episode_length_buf[valid].detach().clone()
        result = super()._reset_idx(env_ids)
        if records is not None:
            self.extras["transition_episode"] = records
            # Counts permit correct denominators; full and RSI outcomes stay separate.
            logs = self.extras.setdefault("log",{})
            for name in ("has_obstacle","is_rsi","full_start","crossed","completed","full_success","failed","no_obstacle_excursion"):
                logs["Transition/count_"+name] = records[name].float().sum()
            full = records["full_start"]
            rsi = records["is_rsi"]
            logs["Transition/full_success_rate"] = records["full_success"][full].float().mean() if full.any() else 0.
            logs["Transition/rsi_completion_rate"] = records["completed"][rsi].float().mean() if rsi.any() else 0.
        return result
