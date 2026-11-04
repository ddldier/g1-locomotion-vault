"""Original-Student anchoring on walking samples, on top of WarmStartPPO."""
from collections import namedtuple
from copy import deepcopy
import math
import torch

from instinct_rl.algorithms.warm_start_ppo import WarmStartPPO
from instinct_rl.storage.rollout_storage import RolloutStorage
from instinct_rl.utils.utils import get_subobs_size


AnchorMiniBatch = namedtuple("AnchorMiniBatch", RolloutStorage.MiniBatch._fields + ("walk_anchor_weight",))


class WalkAnchorRollout(RolloutStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.walk_anchor_weights = torch.zeros(
            self.num_transitions_per_env, self.num_envs, 1, device=self.device)

    def add_transitions(self, transition):
        weight = transition.walk_anchor_weight
        if weight.shape != (self.num_envs, 1):
            raise ValueError("Walk anchor weight must be (num_envs,1).")
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("Walk anchor rollout overflow.")
        self.walk_anchor_weights[self.step].copy_(weight)
        super().add_transitions(transition)

    def get_minibatch_from_selection(self, T_select, B_select, padded_B_slice=None, prev_done_mask=None):
        if padded_B_slice is not None:
            raise TypeError("WalkAnchorPPO supports only the current non-recurrent Student.")
        batch = super().get_minibatch_from_selection(T_select, B_select)
        return AnchorMiniBatch(*batch, self.walk_anchor_weights[T_select, B_select])


def masked_anchor_loss(mu, reference_mu, sigma, weight):
    """Mean per-joint Gaussian mean penalty; zero-weight samples have zero gradient.

    Average over the FULL minibatch, not over selected samples. Thus approach
    weight=0.25 really is one quarter of the flat constraint, including when a
    minibatch contains only approach samples.
    """
    per_sample = (0.5 * ((mu - reference_mu.detach()) / sigma.detach()).square()).mean(-1)
    return (per_sample * weight.reshape(-1).detach()).mean()


class WalkAnchorPPO(WarmStartPPO):
    def __init__(self, actor_critic, walk_anchor_loss_coef=0.1, **kwargs):
        if not math.isfinite(walk_anchor_loss_coef) or walk_anchor_loss_coef < 0:
            raise ValueError("walk_anchor_loss_coef must be finite and nonnegative.")
        super().__init__(actor_critic, **kwargs)
        self.walk_anchor_loss_coef = float(walk_anchor_loss_coef)
        # This is separate from actor_critic and never enters its optimizer.
        self.anchor_actor = deepcopy(actor_critic.actor).to(self.device).eval()
        self.anchor_actor.requires_grad_(False)
        self._anchor_normalizer_state = None
        self._anchor_source = None
        self._anchor_ready = False
        self._anchor_weight_provider = None

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        if num_rewards != 1:
            raise TypeError("WalkAnchorPPO expects one critic/reward stream.")
        self.transition = RolloutStorage.Transition()
        critic_size = get_subobs_size(obs_format["critic"]) if "critic" in obs_format else None
        self.storage = WalkAnchorRollout(
            num_envs, num_transitions_per_env, [get_subobs_size(obs_format["policy"])],
            [critic_size], [num_actions], num_rewards=num_rewards, device=self.device)

    def act(self, obs, critic_obs):
        if not self._anchor_ready or self._anchor_weight_provider is None:
            raise RuntimeError("Call configure_walk_anchor before collecting a training rollout.")
        # Capture PRE-action phase, paired with obs_t, not the post-step/reset state.
        weight = self._anchor_weight_provider().detach().to(self.device)
        if weight.shape != (obs.shape[0], 1):
            raise ValueError("Walk anchor provider returned the wrong batch shape.")
        if not torch.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
            raise ValueError("Walk anchor weights must be finite and in [0,1].")
        self.transition.walk_anchor_weight = weight
        return super().act(obs, critic_obs)

    def compute_losses(self, minibatch):
        losses, variables, stats = super().compute_losses(minibatch)
        weight = minibatch.walk_anchor_weight
        stats["walk_anchor_fraction"] = (weight > 0).float().mean()
        stats["walk_anchor_mean_weight"] = weight.mean()
        if self.critic_only:
            stats["walk_anchor_rms_rad"] = weight.new_zeros(())
            return losses, variables, stats
        if not self._anchor_ready:
            raise RuntimeError("Frozen Student reference has not been initialized.")
        # minibatch.obs already uses the frozen original policy normalizer.
        # Applying normalization again here would be a policy-interface error.
        with torch.no_grad():
            reference_mu = self.anchor_actor(minibatch.obs)
        mu, sigma = self.actor_critic.action_mean, self.actor_critic.action_std
        losses["walk_anchor_loss"] = masked_anchor_loss(mu, reference_mu, sigma, weight)
        with torch.no_grad():
            selected = (weight.reshape(-1) > 0).float()
            squared = (mu - reference_mu).square().mean(-1)
            stats["walk_anchor_rms_rad"] = ((squared * selected).sum() / selected.sum().clamp_min(1)).sqrt()
        return losses, variables, stats

    def state_dict(self):
        state = super().state_dict()
        if not self._anchor_ready:
            raise RuntimeError("Refuse to save a training checkpoint without its frozen reference.")
        state["walk_anchor"] = {
            "version": 1,
            "actor_state_dict": self.anchor_actor.state_dict(),
            "policy_normalizer_state_dict": self._anchor_normalizer_state,
            "source_checkpoint": self._anchor_source,
            "loss_coef": self.walk_anchor_loss_coef,
        }
        return state

    def load_state_dict(self, state_dict):
        reference = state_dict.get("walk_anchor")
        if reference is None:
            raise ValueError("This is not a walk-anchor checkpoint. Start a new experiment from the original Student.")
        if reference.get("version") != 1:
            raise ValueError("Unsupported walk-anchor checkpoint version.")
        super().load_state_dict(state_dict)
        self.anchor_actor.load_state_dict(reference["actor_state_dict"], strict=True)
        self.anchor_actor.eval().requires_grad_(False)
        self._anchor_normalizer_state = reference["policy_normalizer_state_dict"]
        self._anchor_source = reference["source_checkpoint"]
        self.walk_anchor_loss_coef = float(reference["loss_coef"])
        self._anchor_ready = True


def configure_walk_anchor(runner, env, checkpoint=None):
    """Bind simulator-only mask and load/check the frozen original Student."""
    alg = runner.alg
    if not isinstance(alg, WalkAnchorPPO):
        raise TypeError("configure_walk_anchor requires WalkAnchorPPO.")
    policy_norm = runner.normalizers["policy"]
    if policy_norm.until != 0:
        raise ValueError("Freeze the policy normalizer before configuring the reference.")
    current_norm = policy_norm.state_dict()
    if checkpoint:
        source = torch.load(checkpoint, map_location="cpu", weights_only=True)
        actor = {k[6:]: v for k, v in source["model_state_dict"].items() if k.startswith("actor.")}
        original_norm = source["policy_normalizer_state_dict"]
        # In this version the reference is the same original Student used for initialization.
        if alg._anchor_ready:
            if any(not torch.equal(v.cpu(), alg.anchor_actor.state_dict()[k].cpu()) for k, v in actor.items()):
                raise ValueError("Refuse to replace a checkpoint's frozen reference on resume.")
        alg.anchor_actor.load_state_dict(actor, strict=True)
        alg.anchor_actor.eval().requires_grad_(False)
        alg._anchor_normalizer_state = {k: v.detach().cpu().clone() for k, v in original_norm.items()}
        alg._anchor_source = str(checkpoint)
        alg._anchor_ready = True
    if not alg._anchor_ready:
        raise ValueError("A fresh walk-anchor run needs an original Student --actor_checkpoint.")
    original_norm = alg._anchor_normalizer_state
    if set(original_norm) != set(current_norm):
        raise ValueError("Original Student normalizer fields differ.")
    if any(not torch.equal(original_norm[k].cpu(), current_norm[k].cpu()) for k in current_norm):
        raise ValueError("Policy normalizer differs from the frozen original Student.")
    if any(not torch.isfinite(v).all() for v in alg.anchor_actor.state_dict().values()):
        raise ValueError("Frozen reference has nonfinite parameters.")
    from instinctlab.tasks.shadowing.perceptive.config.g1.walk_anchor_env import walk_anchor_weight
    alg._anchor_weight_provider = lambda: walk_anchor_weight(env)
    print(f"[WalkAnchor] original={alg._anchor_source}, coefficient={alg.walk_anchor_loss_coef}")
    print("[WalkAnchor] flat=1, full-start approach farther than 1.0m=0.25, vault/RSI/early recovery=0")
