"""Isaac adapters, NPZ reset-only sampling, task rewards and privileged state."""
from pathlib import Path
import numpy as np
import torch
from isaaclab.envs import mdp as isaac_mdp
from .unified_transition_core import TransitionState, box_local, quat_multiply
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
import instinctlab.tasks.locomotion.mdp as locomotion_mdp
LEGAL_BODIES = (
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
)
FORBIDDEN_BODIES = (
    "pelvis", "torso_link", "left_knee_link", "right_knee_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link", "left_elbow_link", "right_elbow_link",
)
BOX_FILTER_BODIES = LEGAL_BODIES + FORBIDDEN_BODIES
EXPECTED_JOINT_ORDER = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)


def default_motion_file():
    return str(Path(__file__).resolve().parents[3] / "beyondmimic/config/g1/motions/vault/g1_speed_vault_pelvis_retargeted.npz")


class ResetMotion:
    """Load motion once; no running reference, phase clock, or tracking loss."""
    def __init__(self, path, joint_names, device, frame_range):
        with np.load(path, allow_pickle=False) as z:
            names = list(z["joint_names"].astype(str))
            if set(names) != set(joint_names):
                raise ValueError("RSI joint names differ from the robot.")
            order = [names.index(name) for name in joint_names]
            fps = float(z["framerate"])
            pos = torch.tensor(z["base_pos_w"],device=device)
            quat = torch.tensor(z["base_quat_w"],device=device)
            joints = torch.tensor(z["joint_pos"][:,order],device=device)
            self.box_size = torch.tensor(z["obstacle_size"],device=device)
            self.box_pos = torch.tensor(z["obstacle_pos"],device=device)
        if fps <= 0 or not all(torch.isfinite(x).all() for x in (pos,quat,joints,self.box_size,self.box_pos)):
            raise ValueError("Invalid RSI motion.")
        self.pos = pos
        self.quat = torch.nn.functional.normalize(quat,dim=-1)
        self.joints = joints
        # Central differences internally, one-sided endpoints, world-frame angular velocity.
        self.lin_vel = torch.gradient(pos,spacing=(1/fps,),dim=0)[0]
        self.joint_vel = torch.gradient(joints,spacing=(1/fps,),dim=0)[0]
        idx = torch.arange(len(pos),device=device)
        lo, hi = (idx-1).clamp_min(0),(idx+1).clamp_max(len(pos)-1)
        conj = self.quat[lo].clone()
        conj[:,1:] *= -1
        dq = quat_multiply(self.quat[hi],conj)
        dq = torch.where(dq[:,:1]<0,-dq,dq)
        norm = torch.linalg.vector_norm(dq[:,1:],dim=-1,keepdim=True)
        angle = 2*torch.atan2(norm,dq[:,:1].clamp_min(1e-8))
        self.ang_vel = dq[:,1:] * (angle/norm.clamp_min(1e-8)) / ((hi-lo)/fps)[:,None]
        start,stop = frame_range
        if not 0 <= start < stop <= len(pos):
            raise ValueError("Invalid RSI frame interval.")
        valid = idx[(idx>=start)&(idx<stop)]
        x = pos[valid,0]-self.box_pos[0]
        half = self.box_size[0]/2
        self.buckets = [valid[(x < -half)&(x >= -2.5)],
                        valid[(x >= -half)&(x <= half+0.8)],
                        valid[(x > half+0.8)&(x <= half+2.5)]]
        if any(len(b)==0 for b in self.buckets):
            raise ValueError("RSI approach/vault/recovery buckets must all be nonempty.")

    def sample(self, count):
        phase = torch.randint(3,(count,),device=self.pos.device)
        frames = torch.empty(count,dtype=torch.long,device=self.pos.device)
        for i,bucket in enumerate(self.buckets):
            mask = phase==i
            frames[mask] = bucket[torch.randint(len(bucket),(int(mask.sum()),),device=phase.device)]
        return frames


def state(env):
    if not hasattr(env,"_transition_state"):
        env._transition_state = TransitionState(env.num_envs,env.device,env.cfg.transition)
    return env._transition_state


def motion(env):
    if not hasattr(env,"_transition_motion"):
        cfg = env.cfg.transition
        env._transition_motion = ResetMotion(cfg.motion_file or default_motion_file(),
                                              env.scene["robot"].joint_names,env.device,cfg.frame_range)
    return env._transition_motion
class _FrozenDaggerNormalizer(torch.nn.Module):
    """Frozen EmpiricalNormalization loaded from DAgger checkpoint."""
    def __init__(self):
        super().__init__()
        self.eps = 1e-2
        self.register_buffer(
            "_mean", torch.zeros(1, 2049)
        )
        self.register_buffer(
            "_var", torch.ones(1, 2049)
        )
        self.register_buffer(
            "_std", torch.ones(1, 2049)
        )
        self.register_buffer(
            "count", torch.tensor(0, dtype=torch.long)
        )
    def forward(self, x):
        return (x - self._mean) / (self._std + self.eps)
    
def load_vault_dagger_reference(env, checkpoint_path):
    """Load and cache the frozen multi-frame DAgger Student."""
    path = str(Path(checkpoint_path).expanduser())
    cached = getattr(env, "_vault_dagger_reference", None)
    if cached is not None and cached["path"] == path:
        return cached["actor"], cached["normalizer"]
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    actor = torch.nn.Sequential(
        torch.nn.Linear(2049, 512),
        torch.nn.ELU(),
        torch.nn.Linear(512, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 29),
    )

    actor_state = {
        key.removeprefix("actor."): value
        for key, value in ckpt["model_state_dict"].items()
        if key.startswith("actor.")
    }
    actor.load_state_dict(actor_state, strict=True)
    normalizer = _FrozenDaggerNormalizer()
    normalizer.load_state_dict(
        ckpt["policy_normalizer_state_dict"],
        strict=True,
    )
    actor = actor.to(env.device).eval()
    normalizer = normalizer.to(env.device).eval()
    actor.requires_grad_(False)
    normalizer.requires_grad_(False)
    env._vault_dagger_reference = {
        "path": path,
        "actor": actor,
        "normalizer": normalizer,
    }
    print(f"[Vault Imitation] Frozen DAgger Student loaded: {path}")
    return actor, normalizer

def vault_dagger_imitation(env, checkpoint_path, sigma=0.5):
    """Reward the PPO action for matching the frozen DAgger Student."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    # 更新状态机，确保使用当前 step 的状态。
    s = update(env)
    # 只在已经进入 Vault、尚未跨越、没有失败的环境启用。
    vault_mask = (
        s.has_obstacle
        & s.vault_started
        & ~s.crossed
        & ~s.failed
    )
    reward = torch.zeros(
        env.num_envs,
        device=env.device,
        dtype=torch.float32,
    )
    if not vault_mask.any():
        return reward
    # 获取冻结的 DAgger Actor 和它自己的 Normalizer。
    teacher_actor, teacher_normalizer = load_vault_dagger_reference(
        env,
        checkpoint_path,
    )
    # 获取当前已经缓存的原始 policy observation。
    # 不重新调用 observation_manager，避免重复推进 history。
    policy_terms = env.obs_buf["policy"]
    term_names = env.observation_manager.active_terms["policy"]
    policy_obs = torch.cat(
        [
            policy_terms[name].reshape(env.num_envs, -1)
            for name in term_names
        ],
        dim=-1,
    )
    # 严格检查观测和动作维度。
    if policy_obs.shape[-1] != 2049:
        raise RuntimeError(
            f"DAgger observation mismatch: {policy_obs.shape[-1]} != 2049"
        )
    student_action = env.action_manager.action
    if student_action.shape[-1] != 29:
        raise RuntimeError(
            f"DAgger action mismatch: {student_action.shape[-1]} != 29"
        )
    # Teacher 只推理，不参与梯度更新。
    with torch.no_grad():
        teacher_action = teacher_actor(
            teacher_normalizer(policy_obs)
        )
        # 对 29 个关节的动作误差取均值。
        mse = (
            student_action - teacher_action
        ).square().mean(dim=-1)
        # 平滑的模仿奖励，范围为 (0, 1]。
        imitation = torch.exp(
            -mse / (2.0 * sigma * sigma)
        )
    return imitation * vault_mask.float()

class YawCurriculumVelocityCommand(
    isaac_mdp.UniformVelocityCommand
):
    """G1 yaw curriculum: stand, turn, arc walking, straight walking."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        n = len(env_ids)
        if n == 0:
            return
        device = self.device
        # 四种命令的采样概率
        mode = torch.rand(n, device=device)
        # 10% 静止站立
        stand = mode < 0.10
        # 30% 原地转向
        turn_in_place = (
            (mode >= 0.10)
            & (mode < 0.60)
        )
        # 30% 行进转弯
        turn_while_walking = (
            (mode >= 0.60)
            & (mode < 0.80)
        )
        # 30% 直线行走
        straight = mode >= 0.80
        command = torch.zeros(
            n, 3, device=device
        )

        # 正负方向均衡采样
        yaw_magnitude = torch.empty(
            n, device=device
        ).uniform_(0.20, 0.60)

        yaw_sign = torch.where(
            torch.rand(n, device=device) < 0.5,
            -1.0,
            1.0,
        )

        yaw = yaw_sign * yaw_magnitude

        # 1. 原地转向
        command[turn_in_place, 0] = 0.0
        command[turn_in_place, 1] = 0.0
        command[turn_in_place, 2] = yaw[turn_in_place]

        # 2. 行进中转弯
        forward_speed = torch.empty(
            n, device=device
        ).uniform_(0.30, 1.00)

        command[turn_while_walking, 0] = (
            forward_speed[turn_while_walking]
        )
        command[turn_while_walking, 2] = (
            yaw[turn_while_walking]
        )

        # 3. 直线行走，包括前进和后退
        straight_speed = torch.empty(
            n, device=device
        ).uniform_(-0.80, 1.20)

        command[straight, 0] = (
            straight_speed[straight]
        )

        # stand 保持三维全零
        self.vel_command_b[env_ids] = command

def velocity_command(env):
    """
    Unified locomotion + vault command.
    No obstacle:
        use yaw curriculum commands.
    Obstacle before stable completion:
        hold forward command for approach,
        vault and landing recovery.
    After completion:
        allow normal velocity commands again.
    """
    command = env.command_manager.get_command(
        "base_velocity"
    )
    s = state(env)
    # 有箱子、尚未完成稳定恢复时，
    # 保持正常前进指令，避免箱前原地旋转。
    obstacle_forward = (
        s.has_obstacle
        & ~s.completed
        & ~s.failed
    )
    # 这里必须写回 CommandManager 的命令缓冲区，
    # 使 Actor、速度跟踪奖励和 Recovery 状态机
    # 读取到一致的目标命令。
    command[obstacle_forward, 0] = (
        env.cfg.transition.forward_command
    )
    command[obstacle_forward, 1] = 0.0
    command[obstacle_forward, 2] = 0.0
    return command.clone()


def reset_transition(env, env_ids):
    ids = torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    if not ids.numel():
        return
    cfg = env.cfg.transition
    cfg.validate()
    m = motion(env)
    robot,box = env.scene["robot"],env.scene["vault_box"]
    n = len(ids)
    has = torch.rand(n,device=env.device)<cfg.obstacle_probability
    rsi = has & (torch.rand(n,device=env.device)<cfg.rsi_probability)
    root = robot.data.default_root_state[ids].clone()
    root[:,:2] = 0.
    root[:,3:7] = root.new_tensor((1.,0.,0.,0.))
    root[:,7:] = 0.
    joints = robot.data.default_joint_pos[ids].clone()
    joint_vel = torch.zeros_like(joints)
    distances = root.new_tensor(cfg.distances)
    distance = distances[torch.randint(len(distances),(n,),device=env.device)]
    box_pose = root[:,:7].clone()
    box_pose[:,:3] = m.box_pos
    box_pose[:,0] = distance+m.box_size[0]/2
    box_pose[:,1] = 0.
    if rsi.any():
        frames = m.sample(int(rsi.sum()))
        root[rsi,:3] = m.pos[frames]
        root[rsi,3:7] = m.quat[frames]
        root[rsi,7:10] = m.lin_vel[frames]
        root[rsi,10:13] = m.ang_vel[frames]
        joints[rsi] = m.joints[frames]
        joint_vel[rsi] = m.joint_vel[frames]
        # Translate robot and box together: preserve every RSI relative pose.
        box_pose[rsi,:3] = m.box_pos
        shift = root[rsi,:2].clone()
        root[rsi,:2] -= shift
        box_pose[rsi,:2] -= shift
    hidden = ~has
    box_pose[hidden,:3] = box_pose.new_tensor((0.,0.,-50.))
    initial_x = root[:,0]-box_pose[:,0]
    origins = env.scene.env_origins[ids]
    root[:,:3] += origins
    box_pose[:,:3] += origins
    robot.write_root_pose_to_sim(root[:,:7],env_ids=ids)
    robot.write_root_velocity_to_sim(root[:,7:],env_ids=ids)
    robot.write_joint_state_to_sim(joints,joint_vel,env_ids=ids)
    box.write_root_pose_to_sim(box_pose,env_ids=ids)
    box.write_root_velocity_to_sim(torch.zeros(n,6,device=env.device),env_ids=ids)
    state(env).reset(ids,has,rsi,initial_x,m.box_size[0]/2,env.common_step_counter)


def update(env):
    s = state(env)
    if torch.all(s.stamp==env.common_step_counter):
        return s
    pending = s.stamp != env.common_step_counter
    robot,box = env.scene["robot"],env.scene["vault_box"]
    data = robot.data
    m = motion(env)
    if not hasattr(env,"_transition_body_ids"):
        by_name = {name:i for i,name in enumerate(robot.body_names)}
        missing = set(BOX_FILTER_BODIES)-set(by_name)
        if missing:
            raise ValueError(f"Missing robot links: {missing}")
        contact_names = env.scene["contact_forces"].body_names
        by_contact = {name:i for i,name in enumerate(contact_names)}
        env._transition_body_ids = (
            [by_name[x] for x in LEGAL_BODIES[:2]],
            [by_contact[x] for x in LEGAL_BODIES[:2]],
            [by_name[x] for x in FORBIDDEN_BODIES],
            [by_contact[x] for x in FORBIDDEN_BODIES])
    foot_ids,foot_force_ids,bad_ids,bad_force_ids = env._transition_body_ids
    rel = box_local(data.root_pos_w,box.data.root_pos_w,box.data.root_quat_w)
    links = box_local(data.body_pos_w,box.data.root_pos_w,box.data.root_quat_w)
    force_data = env.scene["box_contacts"].data
    forces = force_data.force_matrix_w
    if forces is None or forces.shape[1:3] != (1,len(BOX_FILTER_BODIES)):
        raise RuntimeError("Box filtered contact matrix must be (N,1,filters,3). Check contact sensor support.")
    magnitudes = torch.linalg.vector_norm(forces[:,0],dim=-1)
    legal = magnitudes[:,:len(LEGAL_BODIES)].amax(-1)>5.
    illegal_box = magnitudes[:,len(LEGAL_BODIES):].amax(-1)>s.cfg.box_force_threshold
    contacts = env.scene["contact_forces"].data.net_forces_w_history
    peaks = torch.linalg.vector_norm(contacts,dim=-1).amax(dim=1)
    heights = data.body_pos_w[:,:,2]-env.scene.env_origins[:,None,2]
    feet_ground = ((heights[:,foot_ids]<0.18)&(peaks[:,foot_force_ids]>10.)).any(-1)
    illegal_ground = ((heights[:,bad_ids]<0.25)&(peaks[:,bad_force_ids]>s.cfg.ground_force_threshold)).any(-1)
    box_top = box.data.root_pos_w[:,2]-env.scene.env_origins[:,2]+m.box_size[2]/2
    s.update(env.common_step_counter,env.step_dt,relative_base=rel,
             half_length=m.box_size[0]/2,box_top=box_top,
             base_height=data.root_pos_w[:,2]-env.scene.env_origins[:,2],
             gravity_z=data.projected_gravity_b[:,2],lin_vel=data.root_lin_vel_b,
             ang_vel=data.root_ang_vel_b,command=velocity_command(env),
             feet_ground=feet_ground,all_links_clear=links[:,:,0].amin(-1)>m.box_size[0]/2+s.cfg.rear_margin,
             legal_box_contact=legal,illegal_box_contact=illegal_box,illegal_ground_contact=illegal_ground)
    # Record actual filter/low-link forces without changing either contact rule.
    box_force,box_index = magnitudes[:,len(LEGAL_BODIES):].max(-1)
    low_link_forces = torch.where(heights[:,bad_ids]<0.25,peaks[:,bad_force_ids],0.)
    ground_force,ground_index = low_link_forces.max(-1)
    s.terminal_illegal_box_force[pending] = box_force[pending]
    s.terminal_illegal_ground_force[pending] = ground_force[pending]
    s.terminal_box_body_index[pending] = torch.where(box_force>0.,box_index,-1)[pending]
    s.terminal_ground_body_index[pending] = torch.where(ground_force>0.,ground_index,-1)[pending]
    return s


def failure(env):
    return update(env).failed


def success(env):
    return update(env).completed


def event_reward(env, event):
    # RewardManager multiplies by dt; these are one-time event magnitudes.
    return getattr(update(env),event).float()/env.step_dt


def dense_reward(env, name):
    return getattr(update(env),name)

def locomotion_reward_mask(env):
    """Enable normal locomotion rewards outside the active vault phase."""
    s = update(env)
    # 平地、接近阶段、跨越后的恢复阶段都开启。
    # 进入 Vault 后关闭，直到成功跨越。
    enabled = (
        ~s.has_obstacle
        | ~s.vault_started
        | s.crossed
        | s.recovery_seed
    )
    # 失败状态不再产生普通 locomotion 奖励。
    enabled &= ~s.failed
    return enabled.float()

def locomotion_track_lin_vel(env, command_name, std):
    reward = locomotion_mdp.track_lin_vel_xy_yaw_frame_exp(
        env,
        command_name=command_name,
        std=std,
    )
    return reward * locomotion_reward_mask(env)

def locomotion_track_ang_vel(env, command_name, std):
    reward = locomotion_mdp.track_ang_vel_z_world_exp(
        env,
        command_name=command_name,
        std=std,
    )
    return reward * locomotion_reward_mask(env)

def locomotion_stand_still(env, command_name, asset_cfg):
    """Penalize drift from the default pose only during locomotion phases."""
    reward = locomotion_mdp.stand_still(
        env,
        command_name=command_name,
        asset_cfg=asset_cfg,
    )
    return reward * locomotion_reward_mask(env)


def locomotion_feet_air_time(
    env,
    command_name,
    threshold,
    sensor_cfg,
):
    reward = locomotion_mdp.feet_air_time_positive_biped(
        env,
        command_name=command_name,
        threshold=threshold,
        sensor_cfg=sensor_cfg,
    )
    return reward * locomotion_reward_mask(env)

# -------------------------------------------------
# Unitree OmniLoco upper-body stabilization rewards
# -------------------------------------------------

def locomotion_upper_body_joint_vel(env, asset_cfg):
    """Penalize excessive upper-body joint velocity."""

    reward = isaac_mdp.joint_vel_l2(
        env,
        asset_cfg=asset_cfg,
    )

    return reward * locomotion_reward_mask(env)


def locomotion_joint_deviation_torso(env, asset_cfg):
    """Keep waist joints close to their default positions."""

    reward = isaac_mdp.joint_deviation_l1(
        env,
        asset_cfg=asset_cfg,
    )

    return reward * locomotion_reward_mask(env)



def gait_correction_mask(env):
    """平地、正常接近、稳定恢复后启用步态修正。"""
    s = update(env)

    active = (
        ~s.has_obstacle
        | (
            s.has_obstacle
            & ~s.vault_started
            & ~s.is_rsi
        )
        | s.completed
    )

    return (active & ~s.failed).float()


# def locomotion_feet_gait_masked(
#     env,
#     period,
#     offset,
#     threshold,
#     sensor_cfg,
#     command_name,
# ):
#     reward = locomotion_mdp.feet_gait(
#         env,
#         period=period,
#         offset=offset,
#         threshold=threshold,
#         sensor_cfg=sensor_cfg,
#         command_name=command_name,
#     )

#     s = update(env)

#     # 正常行走、Vault 前接近阶段开启；
#     # Vault 及其后的恢复阶段暂不使用固定时钟奖励。
#     mask = (
#         gait_correction_mask(env)
#         * (~s.vault_started).float()
#     )

#     return reward * mask
def locomotion_feet_gait_masked(
    env,
    period,
    offset,
    threshold,
    sensor_cfg,
    command_name,
):
    reward = locomotion_mdp.feet_gait(
        env,
        period=period,
        offset=offset,
        threshold=threshold,
        sensor_cfg=sensor_cfg,
        command_name=command_name,
    )
    s = update(env)
    command = env.command_manager.get_command(command_name)
    linear_speed = torch.linalg.vector_norm(
        command[:, :2],
        dim=-1,
    )
    yaw_speed = torch.abs(command[:, 2])
    # 判断是否为原地转向
    turn_in_place = (
        (linear_speed < 0.1)
        & (yaw_speed > 0.15)
    )
    # 原有 Gait Correction Mask 保持不变
    mask = gait_correction_mask(env)
    # Vault 阶段关闭固定步态周期
    mask = mask * (~s.vault_started).float()
    # 原地转向不强制匹配直线行走的 0.8s 周期
    mask = mask * (~turn_in_place).float()
    return reward * mask

def loco_lin_vel_z_penalty(env):
    reward = isaac_mdp.lin_vel_z_l2(env)
    return reward * gait_correction_mask(env)


def loco_flat_orientation_penalty(env):
    reward = isaac_mdp.flat_orientation_l2(env)
    return reward * gait_correction_mask(env)


def loco_feet_slide_penalty(env, sensor_cfg, asset_cfg):
    reward = velocity_mdp.feet_slide(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )
    return reward * gait_correction_mask(env)

def locomotion_joint_deviation_arms(env, asset_cfg):
    """Discourage excessive arm deviation from natural default posture."""

    reward = isaac_mdp.joint_deviation_l1(
        env,
        asset_cfg=asset_cfg,
    )

    return reward * locomotion_reward_mask(env)

def locomotion_feet_symmetry_masked(
    env, command_name, sensor_cfg,
    max_time=1.0,
):
    reward = locomotion_mdp.feet_air_contact_symmetry_penalty(
        env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        max_time=max_time,
    )
    return reward * gait_correction_mask(env)

def turn_in_place_progress_reward(
    env,
    command_name="base_velocity",
):
    """Encourage actual yaw rotation when commanded to turn in place."""
    command = env.command_manager.get_command(
        command_name
    )
    linear_speed = torch.linalg.vector_norm(
        command[:, :2],
        dim=-1,
    )
    target_wz = command[:, 2]
    # 只用于真正的原地转向指令
    pivot = (
        (linear_speed < 0.08)
        & (torch.abs(target_wz) > 0.15)
    )
    robot = env.scene["robot"]
    actual_wz = robot.data.root_ang_vel_w[:, 2]
    # 正确方向的实际旋转速度
    signed_wz = (
        actual_wz * torch.sign(target_wz)
    )
    # 达到目标时得 1；
    # 完全不旋转得 0；
    # 反向旋转为负值。
    progress = signed_wz / torch.abs(
        target_wz
    ).clamp_min(0.2)
    progress = torch.clamp(
        progress,
        min=-1.0,
        max=1.0,
    )
    return (
        progress
        * pivot.float()
        * gait_correction_mask(env)
    )


def privileged_observation(env):
    s = state(env)  # Never advance state from observation reads.
    robot,box = env.scene["robot"],env.scene["vault_box"]
    d = robot.data
    h = s.has_obstacle.float()[:,None]
    m = motion(env)
    rel = box_local(d.root_pos_w,box.data.root_pos_w,box.data.root_quat_w)*h
    flags = torch.stack((s.has_obstacle,s.is_rsi,s.full_start,s.recovery_seed,s.ever_front,
                         s.seen_over,s.touched_box,s.crossed,s.lane_violation),-1).float()
    remaining = (1-env.episode_length_buf.float()/env.max_episode_length).clamp(0.,1.)[:,None]
    return torch.cat((d.root_lin_vel_b,d.root_ang_vel_b,d.projected_gravity_b,velocity_command(env),
                      d.joint_pos-d.default_joint_pos,d.joint_vel,env.action_manager.action,
                      rel,m.box_size.expand(env.num_envs,-1)*h,
                      torch.nn.functional.one_hot(s.stage,4).float(),flags,
                      (s.best_x*h[:,0])[:,None],s.recovery_timer[:,None],remaining),-1)
