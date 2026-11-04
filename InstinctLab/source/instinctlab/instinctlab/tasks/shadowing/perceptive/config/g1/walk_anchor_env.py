"""Training-only original-Student anchor weights; no Actor observation changes."""
import torch
from . import unified_transition_mdp as mdp


def select_walk_anchor_weight(has_obstacle, is_rsi, crossed, failed, front_gap):
    """Pure elementwise rule; accepts Torch tensors (and NumPy arrays in offline tests)."""
    far_full_approach = has_obstacle & ~is_rsi & ~crossed & (front_gap > 1.0)
    return ((~has_obstacle) * 1.0 + far_full_approach * 0.25) * (~failed)


def walk_anchor_weight(env):
    s = mdp.state(env)  # Read only: never advance transition timers from act().
    robot, box = env.scene["robot"], env.scene["vault_box"]
    rel = mdp.box_local(robot.data.root_pos_w, box.data.root_pos_w, box.data.root_quat_w)
    front_gap = -mdp.motion(env).box_size[0] / 2 - rel[:, 0]
    weight = select_walk_anchor_weight(s.has_obstacle, s.is_rsi, s.crossed, s.failed, front_gap)
    return weight.to(dtype=torch.float32).reshape(-1, 1)
