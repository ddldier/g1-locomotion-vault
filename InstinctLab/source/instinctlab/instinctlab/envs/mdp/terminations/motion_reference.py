from __future__ import annotations

# 允许在类型注解中直接使用尚未导入/尚未定义的类名，
# 从而避免运行时的循环导入问题。

import torch
from typing import TYPE_CHECKING

# Isaac Lab 中的数学工具，包含四元数乘法、求逆、轴角转换、重力向量投影等函数。
import isaaclab.utils.math as math_utils

# Isaac Lab 的管理器基类和场景实体配置。
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

# 运动参考数据相关的工具函数，例如计算位置距离、关节差异、匹配参考时间等。
import instinctlab.motion_reference.utils as motion_reference_utils

# TYPE_CHECKING 为 True 时才会导入这些类型，仅用于静态类型检查，
# 不会在运行时真正导入，避免不必要的依赖或循环导入。
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from instinctlab.motion_reference import MotionReferenceManager


def pos_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    distance_threshold: float = 0.5,
    check_at_keyframe_threshold: float = 0.01,
    print_reason: bool = False,
    height_only: bool = False,
) -> torch.Tensor:
    """Check if the position is far from the reference position.

    Args:
        env: The environment object.
        distance_threshold: The distance threshold.
        check_at_keyframe: only check when the time is close to the keyframe.
        height_only: only check the height (z-axis) difference.

    Returns:
        True if the position is far from the reference position, False otherwise.
    """
    # 计算机器人基座位置与运动参考中基座位置之间的距离。
    # masking=True 表示根据参考数据的有效掩码处理无效帧。
    # squared=False 表示返回欧氏距离，而不是距离平方。
    # return_diff=height_only 表示当 height_only=True 时返回坐标差值，
    # 以便后面只取 z 轴差值进行高度比较。
    distance = motion_reference_utils.get_base_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        masking=True,
        squared=False,
        return_diff=height_only,
    )

    # 如果只检查高度方向，则从距离/差值中取 z 轴并取绝对值。
    if height_only:
        distance = distance[..., 2].abs()  # only check z-axis

    # 将当前距离与参考时间进行匹配。
    # 该函数用于处理参考运动的时间对齐，例如只在接近关键帧时检查。
    # multiply_by_frame_interval=False 表示不额外乘以帧间隔。
    distance = motion_reference_utils.matching_reference_timing(
        env,
        distance,
        env.scene[reference_cfg.name],
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=False,
    )

    # 判断距离是否超过阈值。返回布尔张量，每个环境一个结果。
    return_ = distance > distance_threshold

    # 如果需要打印原因，并且至少有一个环境满足“远离参考位置”的条件，
    # 则输出这些环境对应的运动标识符，便于调试。
    if print_reason and return_.any():
        motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"pos_far_from_ref: {return_.sum()} at {motion_identifiers}")

    # 返回每个环境是否“位置远离参考位置”的布尔结果。
    return return_


class joint_pos_far_from_ref(ManagerTermBase):
    # 该类是一个管理器项，用于判断关节位置是否远离参考关节位置。
    def __init__(self, cfg, env):
        # 调用父类构造函数，保存配置和环境。
        super().__init__(cfg, env)

        # 保存需要排除的关节索引列表。
        # 这些关节在计算关节差异时会被忽略，例如某些不重要的关节或装饰性关节。
        self.exclude_joint_indices = []

        # 如果配置参数中提供了 exclude_joint_names，则查找这些关节名对应的索引。
        if cfg.params.get("exclude_joint_names", []):
            # 从场景中获取机器人资产。
            robot_asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

            # 遍历需要排除的关节名称，并通过 find_joints 找到对应索引。
            for joint_name in cfg.params["exclude_joint_names"]:
                self.exclude_joint_indices += robot_asset.find_joints(joint_name)[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
        difference_threshold: float = 0.5,
        check_at_keyframe_threshold: float = 0.01,
        exclude_joint_names=[],  # list of joint names to exclude
        print_reason: bool = False,
    ) -> torch.Tensor:
        """Check if the joint position is far from the reference joint position."""
        # 计算当前机器人关节位置与参考关节位置之间的差异。
        # masking=True 表示使用参考数据的有效掩码。
        difference_ = motion_reference_utils.get_joint_position_difference(
            env,
            asset_cfg,
            reference_cfg,
            masking=True,
        )

        # 将需要排除的关节差异置零，使其不参与后续最大差异判断。
        # set exclude joints to zero
        if self.exclude_joint_indices:
            difference_[:, self.exclude_joint_indices] = 0.0

        # 对每个环境取所有关节差异绝对值的最大值，
        # 表示该环境中差异最大的关节。
        # 然后与参考时间进行匹配，只在接近关键帧时进行检查。
        difference = motion_reference_utils.matching_reference_timing(
            env,
            torch.abs(difference_).max(-1).values,
            env.scene[reference_cfg.name],
            check_at_keyframe_threshold=check_at_keyframe_threshold,
            multiply_by_frame_interval=False,
        )

        # 判断最大关节差异是否超过阈值。
        return_ = torch.abs(difference) > difference_threshold

        # 如果需要打印原因，并且存在超过阈值的情况，则输出调试信息：
        # 满足条件的环境数量、最大差异值、以及差异最大的关节索引。
        if print_reason and return_.any():
            print(
                "joint_pos_far_from_ref: ",
                return_.sum(),
                "max diff: ",
                torch.abs(difference_).max(),
                "at joint:",
                difference_.argmax(),
            )

        # 返回每个环境是否“关节位置远离参考”的布尔结果。
        return return_


def link_pos_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    distance_threshold: float = 0.5,
    in_base_frame: bool = False,
    check_at_keyframe_threshold: float = 0.01,
    height_only: bool = False,  # recommend using with `in_base_frame=False`
    print_reason: bool = False,
) -> torch.Tensor:
    # 计算机器人各 link 位置与参考 link 位置之间的距离。
    # in_base_frame=True 表示在机器人基座坐标系下计算距离。
    # return_diff=height_only 表示当 height_only=True 时返回坐标差值，
    # 便于后续只比较 z 轴高度。
    distance = motion_reference_utils.get_link_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        masking=True,
        squared=False,
        in_base_frame=in_base_frame,
        return_diff=height_only,
    )  # (batch_size, num_links)

    # 如果只检查高度方向，则取每个 link 的 z 轴差值绝对值。
    if height_only:
        distance = distance[..., 2].abs()  # only check z-axis

    # 对每个环境，在所有 link 中取最大距离/最大高度差，
    # 表示该环境中最远离参考的 link。
    distance = distance.max(-1).values

    # 将距离与参考时间进行匹配，支持只在接近关键帧时检查。
    distance = motion_reference_utils.matching_reference_timing(
        env,
        distance,
        env.scene[reference_cfg.name],
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=False,
    )

    # 判断最大 link 距离是否超过阈值。
    return_ = distance > distance_threshold

    # 如果需要打印原因，并且存在超过阈值的情况，
    # 则输出满足条件的环境数量以及对应的运动标识符。
    if print_reason and return_.any():
        motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"link_pos_far_from_ref: {return_.sum()} at {motion_identifiers}")

    # 返回每个环境是否“link 位置远离参考”的布尔结果。
    return return_


def rot_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    rot_axis_threshold: float = 1.5714 / 2,
    check_at_keyframe_threshold: float = 0.01,
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the rotation is far from the reference rotation.

    Args:
        env: The environment object.
        rot_axis_threshold: difference threshold in terms of the rotation axis norm.
        check_at_keyframe: only check when the timt is close to the keyframe.

    Returns:
        True if the rotation is far from the reference rotation, False otherwise.
    """
    # 获取机器人和运动参考管理器。
    robot: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]

    # 获取机器人根节点当前的世界坐标系四元数旋转。
    # robot.data.root_state_w[:, 3:7] 对应四元数部分 [w, x, y, z]。
    # get the current rotation
    rot = robot.data.root_state_w[:, 3:7]  # shape: [N, 4]

    # 获取参考运动中的基座四元数。
    # base_quat_w 形状为 [N, n_frames, 4]，表示每个环境、每一帧的参考基座旋转。
    ref_rot = motion_reference.data.base_quat_w  # shape: [N, n_frames, 4]

    # 根据当前瞄准帧索引，从所有参考帧中取出每个环境对应的参考旋转。
    ref_rot = ref_rot[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    # 计算当前旋转与参考旋转之间的相对旋转：
    # rot_diff = inv(ref_rot) * rot。
    # calculate the axis-angle norm as the difference
    rot_diff = math_utils.quat_mul(math_utils.quat_inv(ref_rot), rot)  # shape: [N, 4]

    # 将相对旋转四元数转换为轴角表示，得到旋转轴乘以旋转角度。
    rot_diff_axis = math_utils.axis_angle_from_quat(rot_diff)  # shape: [N, 3]

    # 计算轴角向量的范数，即旋转角度大小，作为旋转差异度量。
    rot_diff_norm = torch.norm(rot_diff_axis, dim=-1)  # shape: [N,]

    # 如果启用关键帧检查，则只在当前时间接近关键帧时保留旋转差异，
    # 否则将差异乘以 0，从而不触发判断。
    if check_at_keyframe_threshold >= 0:
        near_keyframe = (motion_reference.time_to_aiming_frame <= check_at_keyframe_threshold).float()
        rot_diff_norm = rot_diff_norm * near_keyframe

    # 判断旋转差异是否超过阈值。
    return_ = rot_diff_norm > rot_axis_threshold

    # 如果需要打印原因，并且存在超过阈值的情况，则输出满足条件的环境数量。
    if print_reason and return_.any():
        print("rot_far_from_ref: ", return_.sum())

    # 返回每个环境是否“旋转远离参考”的布尔结果。
    return return_


def projected_gravity_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    projected_gravity_threshold: float = 0.5,
    check_at_keyframe_threshold: float = 0.01,
    z_only: bool = True,
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the projected gravity is far from the reference projected gravity."""

    # 获取机器人和运动参考管理器。
    robot: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]

    # 获取机器人根节点当前的世界坐标系四元数旋转。
    # get the current rotation
    rot = robot.data.root_state_w[:, 3:7]  # shape: [N, 4]

    # 获取参考运动中的基座四元数，形状为 [N, n_frames, 4]。
    ref_rot = motion_reference.data.base_quat_w  # shape: [N, n_frames, 4]

    # 根据当前瞄准帧索引，取出每个环境对应的参考旋转。
    ref_rot = ref_rot[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    # 将世界坐标系下的重力向量投影到机器人当前基座坐标系中。
    # quat_apply_inverse(rot, gravity) 等价于用 rot 的逆旋转重力向量。
    pg = math_utils.quat_apply_inverse(rot, robot.data.GRAVITY_VEC_W)

    # 将世界坐标系下的重力向量投影到参考基座坐标系中。
    ref_pg = math_utils.quat_apply_inverse(ref_rot, robot.data.GRAVITY_VEC_W)

    # 如果 z_only=True，则只比较投影重力向量的 z 分量差异；
    # 否则比较完整三维投影重力向量的欧氏距离。
    if z_only:
        diff = (pg[:, 2] - ref_pg[:, 2]).abs()
    else:
        diff = torch.norm(pg - ref_pg, dim=-1)

    # 如果启用关键帧检查，则只在当前时间接近关键帧时保留差异，
    # 否则将差异乘以 0，从而不触发判断。
    if check_at_keyframe_threshold >= 0:
        near_keyframe = (motion_reference.time_to_aiming_frame <= check_at_keyframe_threshold).float()
        diff = diff * near_keyframe

    # 判断投影重力差异是否超过阈值。
    return_ = diff > projected_gravity_threshold

    # 如果需要打印原因，并且存在超过阈值的情况，
    # 则输出满足条件的环境数量以及对应的运动标识符。
    if print_reason and return_.any():
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"projected_gravity_far_from_ref: {return_.sum()} at {motion_identifiers}")

    # 返回每个环境是否“投影重力远离参考”的布尔结果。
    return return_