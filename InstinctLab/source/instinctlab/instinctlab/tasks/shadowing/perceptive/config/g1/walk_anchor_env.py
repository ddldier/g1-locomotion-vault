"""Training-only original-Student anchor weights; no Actor observation changes."""
import torch
from . import unified_transition_mdp as mdp


def select_walk_anchor_weight(has_obstacle, is_rsi, crossed, recovery_seed,failed, front_gap):
    """Pure elementwise rule; accepts Torch tensors (and NumPy arrays in offline tests)."""
    far_full_approach = has_obstacle & ~is_rsi & ~crossed & (front_gap > 1.0)
    recovery = crossed | recovery_seed
    weight = (
        (~has_obstacle).float()
        + 0.25 * far_full_approach.float()
        + recovery.float()
    )
    # 失败状态不再施加 Anchor
    return weight.clamp(0.0, 1.0) * (~failed).float()


def walk_anchor_weight(env):
    s = mdp.state(env)  # Read only: never advance transition timers from act().
    robot, box = env.scene["robot"], env.scene["vault_box"]
    rel = mdp.box_local(robot.data.root_pos_w, box.data.root_pos_w, box.data.root_quat_w)
    front_gap = -mdp.motion(env).box_size[0] / 2 - rel[:, 0]
    weight = select_walk_anchor_weight(s.has_obstacle, s.is_rsi, s.crossed, s.recovery_seed, s.failed, front_gap)
    return weight.to(dtype=torch.float32).reshape(-1, 1)
