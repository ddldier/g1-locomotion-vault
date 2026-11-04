"""Isaac adapters, NPZ reset-only sampling, task rewards and privileged state."""
from pathlib import Path
import numpy as np
import torch
from .unified_transition_core import TransitionState, box_local, quat_multiply

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


def velocity_command(env):
    command = env.command_manager.get_command("base_velocity")
    transition = state(env)
    # Before a vault is completed, keep the approach/vault target fixed at
    # the teacher's forward speed.  After recovery, preserve the command
    # written by play.py (including keyboard control).
    h = transition.has_obstacle & ~transition.completed
    command[h] = command.new_tensor((env.cfg.transition.forward_command,0.,0.))
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
