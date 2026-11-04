# 启用 PEP 563 延迟注解求值：类型注解在运行时不会立即求值，
# 因此可以在 TYPE_CHECKING 块中安全引用仅用于类型检查的类。
from __future__ import annotations

# 导入 PyTorch，用于张量运算，例如 sqrt、clamp、mean、std 等。
import torch
# 导入 Sequence 抽象基类，用于类型注解中表示环境 id 序列。
from collections.abc import Sequence
# 导入 TYPE_CHECKING 与 Literal：
# - TYPE_CHECKING 用于避免运行时导入仅类型检查需要的类；
# - Literal 用于限制 ratio_option 只能取特定字符串值。
from typing import TYPE_CHECKING, Literal

# 导入 Isaac Lab 的管理器项基类，供下方课程项类继承。
from isaaclab.managers.manager_base import ManagerTermBase

# 仅在类型检查时导入这些类，避免运行时产生不必要的依赖或循环导入。
if TYPE_CHECKING:
    # 环境类型：ManagerBasedRLEnv 是 Isaac Lab 的基于管理器的强化学习环境。
    from isaaclab.envs import ManagerBasedRLEnv
    # 课程项配置类型：用于描述课程项的参数。
    from isaaclab.managers import CurriculumTermCfg
    # AMASS 动作文件类型：此处仅用于类型提示，代码中未直接使用。
    from isaaclab.motion_reference.motion_files.amass_motion import AmassMotion

    # 动作参考管理器类型：负责管理动作片段、权重与采样。
    from instinctlab.motion_reference import MotionReferenceManager


# 根据环境重置时的“超时/终止”状态更新动作参考权重。
# 设计意图：
# - 超时（timeout）通常代表成功存活或完成，降低其对应动作的采样权重；
# - 终止（terminated）通常代表失败，提高其对应动作的采样权重。
def update_motion_reference_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reference_name: str = "motion_reference",
    success_weight_ratio: float = 0.6,
) -> dict[str, float] | None:
    """Update the motion reference weights based on the timeouts and terminated states."""
    # 从场景中按名称获取动作参考管理器。
    motion_reference: MotionReferenceManager = env.scene[reference_name]
    # 如果环境对象没有记录重置原因（超时/终止）的标记，则无法进行权重更新。
    if not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        return None
    # 从传入的 env_ids 中筛选出本次重置原因为“超时”的环境 id。
    timeout_env_ids = env_ids[env.reset_time_outs[env_ids]]
    # 从传入的 env_ids 中筛选出本次重置原因为“终止”的环境 id。
    terminated_env_ids = env_ids[env.reset_terminated[env_ids]]
    # 对超时环境对应动作降低权重，降低幅度由 success_weight_ratio 控制。
    motion_reference.update_motion_weights(
        timeout_env_ids,
        weight_ratio=success_weight_ratio,
    )
    # 对终止环境对应动作提高权重，倍率为 success_weight_ratio 的倒数。
    motion_reference.update_motion_weights(
        terminated_env_ids,
        weight_ratio=1.0 / success_weight_ratio,
    )
    # 返回本次更新中不同重置状态所占比例，便于日志记录或调试。
    return {
        "timeout_ratio": len(timeout_env_ids) / len(env_ids) if len(env_ids) > 0 else 0.0,
        "terminated_ratio": len(terminated_env_ids) / len(env_ids) if len(env_ids) > 0 else 0.0,
    }


# 根据每个环境在已分配动作上的完成进度更新动作参考权重。
# 核心思想：进度越低的环境，其动作越“欠训练”，应赋予更高权重；
# 同时保持本批次的总权重不变，避免整体采样规模漂移。
def update_motion_reference_weights_by_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reference_name: str = "motion_reference",
) -> dict[str, float] | None:
    """Update the motion reference weights based on (1-progress) of the environment."""
    # 本批次需要处理的环境数量。
    num_envs = len(env_ids)
    # 若没有环境需要处理，直接返回 None。
    if num_envs == 0:
        return None
    # 获取动作参考管理器。
    motion_reference: MotionReferenceManager = env.scene[reference_name]
    # 读取这些环境当前对应的动作权重。
    current_weights = motion_reference.get_current_motion_weights(env_ids)
    # 记录当前总权重，后续用于保持权重和不变。
    weight_sum = current_weights.sum()
    # 计算进度：环境已运行时长 / 分配给该环境的动作总长度。
    progress = (env.episode_length_buf * env.step_dt)[env_ids] / motion_reference.assigned_motion_lengths[env_ids]

    # 将 (1 - 进度) 作为原始权重比例：进度越低，比例越高。
    # normalize the ratio to make the weight sum remain the same
    # make the env with less progress have higher weight
    weight_ratio = 1.0 - progress
    # 先归一化到和为 1，再乘回原来的总权重，从而保持权重总和不变。
    weight_ratio /= weight_ratio.sum()
    weight_ratio *= weight_sum
    # 将计算出的权重比例写回动作参考管理器。
    motion_reference.update_motion_weights(
        env_ids,
        weight_ratio=weight_ratio,
    )
    # 返回调试统计信息：原权重和、新比例和、进度均值与标准差。
    return {
        "weight_sum": weight_sum,
        "weight_ratio_sum": weight_ratio.sum(),
        "progress_mean": progress.mean(),
        "progress_std": progress.std(),
    }


# 基于延迟统计量（进度、经验长度等）更新动作参考权重的课程项。
# 使用指数滑动平均（EMA）平滑统计量，避免单步波动导致权重剧烈变化。
class update_motion_reference_weights_by_delayed_stats(ManagerTermBase):
    """A manager term that updates the motion reference weights based on the delayed statistics of the environment,
    such as progress, shadowing error ...
    """

    # 初始化：从配置参数读取延迟进度/经验长度初值，并获取动作参考管理器。
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        # 调用 ManagerTermBase 初始化。
        super().__init__(cfg, env)
        # 延迟进度，使用 EMA 平滑后的平均进度；初值由 cfg.params 提供。
        self.delayed_progress = cfg.params.get("init_delayed_progress", 0.0)
        # 延迟经验长度，使用 EMA 平滑后的平均经验长度（秒）；初值由 cfg.params 提供。
        self.experience_length = cfg.params.get("init_experience_length", 0.0)
        # 获取动作参考管理器，后续在该管理器上更新动作权重。
        self.motion_reference: MotionReferenceManager = env.scene[cfg.params.get("reference_name", "motion_reference")]

    # 每次课程更新时调用，根据当前批次环境的延迟统计信息更新动作参考权重。
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reference_name: str = "motion_reference",
        progress_refresh_epsilon: float = 0.01,
        length_refresh_epsilon: float = 0.01,
        ratio_option: Literal["div_mean", "div_sum"] = "div_mean",
        init_delayed_progress: float = 0.0,
        init_experience_length: float = 0.02,
    ) -> dict[str, float] | None:
        """Update the motion reference weights based on the delayed statistics of the environment."""
        # 若本批次没有环境，或环境尚未执行任何 step，则跳过更新。
        if len(env_ids) == 0 or env.common_step_counter < 1:
            # if no env_ids or the environment has not stepped yet, return None
            return None

        # 本批次环境的经验长度，等于已执行步数乘以每步时间。
        experience_length = (env.episode_length_buf * env.step_dt)[env_ids]

        # 使用 EMA 更新全局平均经验长度，使其平滑变化。
        self.experience_length = (
            1 - length_refresh_epsilon
        ) * self.experience_length + length_refresh_epsilon * experience_length.mean()
        # 计算每个环境在当前分配动作上的进度。
        progress = experience_length / self.motion_reference.assigned_motion_lengths[env_ids]
        # 使用 EMA 更新全局平均进度，得到延迟进度统计量。
        self.delayed_progress = (
            1 - progress_refresh_epsilon
        ) * self.delayed_progress + progress_refresh_epsilon * progress.mean()

        # compute and normalize the weight ratio
        # 计算权重比例：进度越低权重越高；同时除以 sqrt(经验长度) 做尺度平衡。
        weight_ratio = (1.0 - progress) / torch.sqrt(torch.clamp(experience_length, min=1e-6))
        # 根据 ratio_option 选择不同的归一化策略：
        # - "div_mean"：除以延迟平均基准，使权重围绕平均进度缩放；
        # - "div_sum"：直接除以本批次权重和，使权重和为 1。
        if ratio_option == "div_mean":
            weight_ratio /= (1 - self.delayed_progress) / torch.sqrt(self.experience_length)
        elif ratio_option == "div_sum":
            weight_ratio /= weight_ratio.sum()
        else:
            # 未知选项直接报错，避免静默使用错误配置。
            raise ValueError(f"Unknown ratio option: {ratio_option}")

        # update the motion reference weights
        # 确认环境 id 数量与权重数量一致，避免错误对齐。
        assert len(env_ids) == len(weight_ratio), "env_ids and weight_ratio must have the same length"
        # 将计算出的权重比例写回动作参考管理器。
        self.motion_reference.update_motion_weights(
            env_ids,
            weight_ratio=weight_ratio,
        )
        # 返回调试统计信息，包括延迟统计量、权重和、进度均值与标准差。
        return {
            "delayed_progress": self.delayed_progress,
            "delayed_experience_length": self.experience_length,
            "weight_ratio_sum": weight_ratio.sum(),
            "progress_mean": progress.mean(),
            "progress_std": progress.std(),
        }


# 根据环境经验更新动作参考权重：
# - 超时（通常成功）的动作降低权重；
# - 已覆盖大部分轨迹且超时的动作额外降低权重；
# - 被采样过的动作也略微降低权重，以减少重复采样。
def update_motion_reference_weights_by_experience(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reference_name: str = "motion_reference",
    success_weight_ratio: float = 1.0,
    full_trajectory_weight_ratio: float = 0.6,
    full_trajectory_threshold: float = 0.8,
    sampled_weight_ratio: float = 0.99,
):
    """Update the motion reference weights based on the experience of the environment.
    If the motion is successful or sampled, the weight of this motion shall be decreased.

    Args:
        - success_weight_ratio: The weight ratio for succeeded motion env.
        - full_trajectory_weight_ratio: The weight ratio the env that experience the most of the trajectory and succeeded.
        - sampled_weight_ratio: The weight ratio for the motion that is sampled once.

    """
    # 获取动作参考管理器。
    motion_reference: MotionReferenceManager = env.scene[reference_name]
    # 若环境没有重置状态标记，无法判断成功/失败，直接返回。
    if not hasattr(env, "reset_time_outs") or not hasattr(env, "reset_terminated"):
        return None
    # 计算本批次环境的经验长度。
    experience_length = (env.episode_length_buf * env.step_dt)[env_ids]
    # 判断哪些环境已经走完超过阈值比例的动作轨迹。
    full_trajectory_env_ids_mask = (
        experience_length / motion_reference.complete_motion_lengths[env_ids] > full_trajectory_threshold
    )

    # 筛选出超时环境。
    timeout_env_ids = env_ids[env.reset_time_outs[env_ids]]

    # 筛选出“既覆盖大部分轨迹又超时”的环境。
    full_trajectory_env_ids = env_ids[full_trajectory_env_ids_mask & env.reset_time_outs[env_ids]]

    # 如果存在超时环境且配置要求降低成功权重，则降低这些动作的权重。
    if len(timeout_env_ids) > 0 and success_weight_ratio < 1.0:
        motion_reference.update_motion_weights(
            timeout_env_ids,
            weight_ratio=success_weight_ratio,
        )
    # 如果存在完整轨迹环境且需要降低“被采样过”的权重，
    # 则对本批次所有传入 env_ids 对应的动作应用 sampled_weight_ratio。
    if len(full_trajectory_env_ids) > 0 and sampled_weight_ratio < 1.0:
        motion_reference.update_motion_weights(
            env_ids,
            weight_ratio=sampled_weight_ratio,
        )
    # 对完成大部分轨迹且超时的动作进一步降低权重。
    if len(full_trajectory_env_ids) > 0 and full_trajectory_weight_ratio < 1.0:
        motion_reference.update_motion_weights(
            full_trajectory_env_ids,
            weight_ratio=full_trajectory_weight_ratio,
        )

    # 返回超时比例和采样比例统计。
    return {
        "timeout_ratio": len(timeout_env_ids) / len(env_ids) if len(env_ids) > 0 else 0.0,
        "sampled_ratio": (len(env_ids) - len(timeout_env_ids)) / len(env_ids) if len(env_ids) > 0 else 0.0,
    }