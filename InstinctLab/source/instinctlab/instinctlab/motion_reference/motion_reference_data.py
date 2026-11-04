# 启用延迟注解，使得类型注解在运行时不会被求值，便于前向引用。
from __future__ import annotations

# 导入 PyTorch 库。
import torch
# 导入抽象基类 Sequence，用于类型提示。
from collections.abc import Sequence
# 导入 dataclass 和 fields，用于定义数据类和获取字段。
from dataclasses import dataclass, fields

# 导入 instinctlab 的 torch 工具。
import instinctlab.utils.torch as torch_utils


# 定义 MotionSequence 数据类，表示标准的运动参考缓冲区，可以加载到内存中。
# 预计算额外的链接姿态以便在仿真期间更快地访问。
# 每个运动可以具有前导维度 [N, ...] 或没有批次维度的 [...]。
@dataclass
class MotionSequence:
    """A standard motion reference buffer that can be used to load into memory. Additional link poses are precomputed
    for faster access during simulation.
    Each motion can have leading dims as [N, ...] or [...] with no batch dimension.
    """

    # 关节位置。形状：[N, num_frames, num_joints]，N 是批次大小。
    joint_pos: torch.Tensor = None  # type: ignore
    """ Positions of the joints of the given robot model.

    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    # 关节速度。形状：[N, num_frames, num_joints]，N 是批次大小。
    joint_vel: torch.Tensor = None  # type: ignore
    """ Velocities of the joints of the given robot model.
    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    # 基座在世界坐标系中的位置。形状：[N, num_frames, 3]，N 是批次大小。
    base_pos_w: torch.Tensor = None  # type: ignore
    """ Positions of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    # 基座在世界坐标系中的线速度。形状：[N, num_frames, 3]，N 是批次大小。
    base_lin_vel_w: torch.Tensor = None  # type: ignore
    """ Linear velocities of the base of the robot model.
    Shape: [N, num_frames, 3], N is the batch size.
    """

    # 基座在世界坐标系中的四元数。形状：[N, num_frames, 4]，N 是批次大小。
    base_quat_w: torch.Tensor = None  # type: ignore
    """ Quaternions of the base of the robot model.

    Shape: [N, num_frames, 4], N is the batch size.
    """

    # 基座在世界坐标系中的角速度。形状：[N, num_frames, 3]，N 是批次大小。
    base_ang_vel_w: torch.Tensor = None  # type: ignore
    """ Angular velocities of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    # 运动数据的帧率。形状：[N,]，N 是批次大小。
    framerate: torch.Tensor = None  # type: ignore
    """ The framerate of the motion data.

    Shape: [N,], N is the batch size.
    """

    # 每个运动数据在批次中的帧数。形状：[N,]，N 是批次大小。
    buffer_length: torch.Tensor = None  # type: ignore
    """ The number of frames of each motion data across batch

    Shape: [N,], N is the batch size.
    """

    # 感兴趣链接相对于运动参考基座坐标系的位置。
    # 注意：这是从 joint_pos 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_pos_b: torch.Tensor = None  # type: ignore
    """ The positions of the interested links (w.r.t the base frame of the motion reference).
    NOTE: This is a passive parameter computed from joint_poss.

    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接相对于基座坐标系的四元数。
    # 注意：这是从 joint_pos 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 4]，N 是批次大小。
    link_quat_b: torch.Tensor = None  # type: ignore
    """ The quaternions of the interested links (w.r.t the base).
    NOTE: This is a passive parameter computed from joint_poss.

    Shape: [N, num_frames, num_links, 4], N is the batch size.
    """

    # 感兴趣链接在世界坐标系中的位置。
    # 注意：这是从 base_pos、base_quat 和 joint_poss 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_pos_w: torch.Tensor = None  # type: ignore
    """ The positions of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_pos, base_quat, and joint_poss.

    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接在世界坐标系中的四元数。
    # 形状：[N, num_frames, num_links, 4]，格式为 (w, x, y, z)，N 是批次大小。
    link_quat_w: torch.Tensor = None  # type: ignore
    """ The quaternions of the interested links (w.r.t the world).

    Shape: [N, num_frames, num_links, 4], in (w, x, y, z), N is the batch size.
    """

    # 感兴趣链接相对于运动参考基座坐标系的线速度。
    # 注意：这是从 joint_vels 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_lin_vel_b: torch.Tensor = None  # type: ignore
    """ The linear velocities of the interested links (w.r.t the base frame of the motion reference).
    NOTE: This is a passive parameter computed from joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接相对于运动参考基座坐标系的角速度。
    # 注意：这是从 joint_vels 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_ang_vel_b: torch.Tensor = None  # type: ignore
    """ The angular velocities of the interested links (w.r.t the base frame of the motion reference).
    NOTE: This is a passive parameter computed from joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接在世界坐标系中的线速度。
    # 注意：这是从 base_lin_vel、base_quat 和 joint_vels 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_lin_vel_w: torch.Tensor = None  # type: ignore
    """ The linear velocities of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_lin_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接在世界坐标系中的角速度。
    # 注意：这是从 base_ang_vel、base_quat 和 joint_vels 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_ang_vel_w: torch.Tensor = None  # type: ignore
    """ The angular velocities of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_ang_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 具有帧维度的属性元组。
    attrs_with_frame_dim: tuple = (
        "joint_pos",
        "joint_vel",
        "base_pos_w",
        "base_lin_vel_w",
        "base_quat_w",
        "base_ang_vel_w",
        "link_pos_b",
        "link_quat_b",
        "link_pos_w",
        "link_quat_w",
        "link_lin_vel_b",
        "link_ang_vel_b",
        "link_lin_vel_w",
        "link_ang_vel_w",
    )
    # 仅具有批次维度的属性元组。
    attrs_only_batch_dim: tuple = ("framerate", "buffer_length")

    # 静态方法：创建一个空的 MotionSequence 实例。
    @staticmethod
    def make_empty(
        batch_size: int,
        num_frames: int,
        num_joints: int,
        num_links: int,
        device=torch.device("cpu"),
    ) -> MotionSequence:
        # 创建并初始化所有张量为零。
        return_ = MotionSequence(
            joint_pos=torch.zeros(batch_size, num_frames, num_joints, device=device),
            joint_vel=torch.zeros(batch_size, num_frames, num_joints, device=device),
            base_pos_w=torch.zeros(batch_size, num_frames, 3, device=device),
            base_lin_vel_w=torch.zeros(batch_size, num_frames, 3, device=device),
            base_quat_w=torch.zeros(batch_size, num_frames, 4, device=device),
            base_ang_vel_w=torch.zeros(batch_size, num_frames, 3, device=device),
            framerate=torch.zeros(batch_size, dtype=torch.int, device=device),
            buffer_length=torch.zeros(batch_size, dtype=torch.int, device=device),
            link_pos_b=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_quat_b=torch.zeros(batch_size, num_frames, num_links, 4, device=device),
            link_pos_w=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_quat_w=torch.zeros(batch_size, num_frames, num_links, 4, device=device),
            link_lin_vel_b=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_ang_vel_b=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_lin_vel_w=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
            link_ang_vel_w=torch.zeros(batch_size, num_frames, num_links, 3, device=device),
        )
        # 将四元数的 w 分量设置为 1.0。
        return_.base_quat_w[:, :, 0] = 1.0  # Set the w component of the quaternion to 1.0
        return_.link_quat_b[:, :, :, 0] = 1.0  # Set the w component of the link quaternions to 1.0
        return_.link_quat_w[:, :, :, 0] = 1.0  # Set the w component of the link quaternions to 1.0
        return return_

    # 静态方法：创建一个空的、连接批次的 MotionSequence 实例。
    @staticmethod
    def make_emtpy_concat_batch(
        buffer_lengths: Sequence[int],
        num_joints: int,
        num_links: int,
        device=torch.device("cpu"),
    ) -> MotionSequence:
        # 使用 ConcatBatchTensor 创建连接批次的张量。
        return_ = MotionSequence(
            joint_pos=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_joints,), device=device
            ),  # type: ignore
            joint_vel=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_joints,), device=device
            ),  # type: ignore
            base_pos_w=torch_utils.ConcatBatchTensor(batch_sizes=buffer_lengths, data_shape=(3,), device=device),  # type: ignore
            base_lin_vel_w=torch_utils.ConcatBatchTensor(batch_sizes=buffer_lengths, data_shape=(3,), device=device),  # type: ignore
            base_quat_w=torch_utils.ConcatBatchTensor(batch_sizes=buffer_lengths, data_shape=(4,), device=device),  # type: ignore
            base_ang_vel_w=torch_utils.ConcatBatchTensor(batch_sizes=buffer_lengths, data_shape=(3,), device=device),  # type: ignore
            framerate=torch.zeros((len(buffer_lengths),), device=device),
            buffer_length=torch.tensor(buffer_lengths, device=device),
            link_pos_b=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_quat_b=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 4), device=device
            ),  # type: ignore
            link_pos_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_quat_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 4), device=device
            ),  # type: ignore
            link_lin_vel_b=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_ang_vel_b=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_lin_vel_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
            link_ang_vel_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_links, 3), device=device
            ),  # type: ignore
        )
        # 将四元数的 w 分量设置为 1.0。
        # Set the w component of the quaternion to 1.0
        return_.base_quat_w.fill_data(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))  # type: ignore
        return_.link_quat_b.fill_data(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_links, -1))  # type: ignore
        return_.link_quat_w.fill_data(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_links, -1))  # type: ignore
        return return_

    # 后初始化方法，用于验证属性的一致性。
    def __post_init__(self):
        if not self.joint_pos is None:
            # 所有具有帧维度的属性应该具有相同的前导维度。
            # all attributes should have the same leading dimensions
            assert all(
                getattr(self, attr).shape[0] == self.joint_pos.shape[0] for attr in self.attrs_with_frame_dim
            ), "All attributes should have the same leading dimension."
        else:
            # 所有属性应该为 None。
            # all attributes should be None
            assert all(
                getattr(self, attr) is None for attr in (self.attrs_with_frame_dim + self.attrs_only_batch_dim)
            ), "All attributes should be None or not None."


# 定义 MotionReferenceData 数据类，表示运动参考数据。
# 每个运动可以具有前导维度 [N, ...] 或没有批次维度的 [...]。
@dataclass
class MotionReferenceData:
    """Each motion can have leading dims as [N, ...] or [...] with no batch dimension."""

    # 关节位置（在给定机器人模型的 urdf 坐标系中）。
    # 形状：[N, num_frames, num_joints]，N 是批次大小。
    joint_pos: torch.Tensor = None  # type: ignore
    """ Positions of the joints in urdf coordinate of the given robot model.

    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    # 关节速度。
    # 形状：[N, num_frames, num_joints]，N 是批次大小。
    joint_vel: torch.Tensor = None  # type: ignore
    """ Velocities of the joints of the given robot model.

    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    # 关节位置掩码。
    # 形状：[N, num_frames, num_joints]，N 是批次大小。
    joint_pos_mask: torch.Tensor = None  # type: ignore
    """
    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    # 关节速度掩码。
    # 形状：[N, num_frames, num_joints]，N 是批次大小。
    joint_vel_mask: torch.Tensor = None  # type: ignore
    """
    Shape: [N, num_frames, num_joints], N is the batch size.
    """

    # 基座在世界坐标系中的位置。
    # 形状：[N, num_frames, 3]，N 是批次大小。
    base_pos_w: torch.Tensor = None  # type: ignore
    """ Positions of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    # 基座在世界坐标系中的线速度。
    # 形状：[N, num_frames, 3]，N 是批次大小。
    base_lin_vel_w: torch.Tensor = None  # type: ignore
    """ Linear velocities of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    # 基座在世界坐标系中的四元数。
    # 形状：[N, num_frames, 4]，格式为 (w, x, y, z)，N 是批次大小。
    base_quat_w: torch.Tensor = None  # type: ignore
    """ Quaternions of the base of the robot model.

    Shape: [N, num_frames, 4], in (w, x, y, z), N is the batch size.
    """

    # 基座在世界坐标系中的角速度。
    # 形状：[N, num_frames, 3]，N 是批次大小。
    base_ang_vel_w: torch.Tensor = None  # type: ignore
    """ Angular velocities of the base of the robot model.

    Shape: [N, num_frames, 3], N is the batch size.
    """

    # 基座位置平面掩码。形状：[N, num_frames]，N 是每个项的批次大小。
    base_pos_plane_mask: torch.Tensor = None  # type: ignore
    """ Shape: [N, num_frames], N is the batch size for each term. """
    # 基座位置高度掩码。形状：[N, num_frames]，N 是每个项的批次大小。
    base_pos_height_mask: torch.Tensor = None  # type: ignore
    """ Shape: [N, num_frames], N is the batch size for each term. """
    # 基座朝向掩码。形状：[N, num_frames]，N 是每个项的批次大小。
    base_orientation_mask: torch.Tensor = None  # type: ignore
    """ Shape: [N, num_frames], N is the batch size for each term. """
    # 基座航向掩码。形状：[N, num_frames]，N 是每个项的批次大小。
    base_heading_mask: torch.Tensor = None  # type: ignore
    """ The masks for the base position and orientation.

    Shape: [N, num_frames], N is the batch size for each term.
    """

    # 感兴趣链接相对于基座坐标系（传感器刷新时）的位置。
    # 注意：这是从 base_pos、base_quat 和 joint_poss 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_pos_b: torch.Tensor = None  # type: ignore
    """ The positions of the interested links (w.r.t the base when sensor refreshes).
    NOTE: This is a passive parameter computed from base_pos, base_quat, and joint_poss.

    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接相对于基座坐标系（传感器刷新时）的四元数。
    # 注意：这是从 base_pos、base_quat 和 joint_poss 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 4]，格式为 (w, x, y, z)，N 是批次大小。
    link_quat_b: torch.Tensor = None  # type: ignore
    """ The quaternions of the interested links (w.r.t the base when sensor refreshes).
    NOTE: This is a passive parameter computed from base_pos, base_quat, and joint_poss.

    Shape: [N, num_frames, num_links, 4], in (w, x, y, z), N is the batch size.
    """

    # 感兴趣链接在世界坐标系中的位置。
    # 注意：这是从 base_pos、base_quat 和 joint_poss 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_pos_w: torch.Tensor = None  # type: ignore
    """ The positions of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_pos, base_quat, and joint_poss.

    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接在世界坐标系中的四元数。
    # 形状：[N, num_frames, num_links, 4]，格式为 (w, x, y, z)，N 是批次大小。
    link_quat_w: torch.Tensor = None  # type: ignore
    """ The quaternions of the interested links (w.r.t the world).

    Shape: [N, num_frames, num_links, 4], in (w, x, y, z), N is the batch size.
    """

    # 感兴趣链接相对于基座坐标系（传感器刷新时）的线速度。
    # 注意：这是从 base_lin_vel、base_quat 和 joint_vels 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_lin_vel_b: torch.Tensor = None  # type: ignore
    """ The linear velocities of the interested links (w.r.t the base when sensor refreshes).
    NOTE: This is a passive parameter computed from base_lin_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接相对于基座坐标系（传感器刷新时）的角速度。
    # 注意：这是从 base_ang_vel、base_quat 和 joint_vels 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_ang_vel_b: torch.Tensor = None  # type: ignore
    """ The angular velocities of the interested links (w.r.t the base when sensor refreshes).
    NOTE: This is a passive parameter computed from base_ang_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接在世界坐标系中的线速度。
    # 注意：这是从 base_lin_vel、base_quat 和 joint_vels 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_lin_vel_w: torch.Tensor = None  # type: ignore
    """ The linear velocities of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_lin_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 感兴趣链接在世界坐标系中的角速度。
    # 注意：这是从 base_ang_vel、base_quat 和 joint_vels 计算得到的被动参数。
    # 形状：[N, num_frames, num_links, 3]，N 是批次大小。
    link_ang_vel_w: torch.Tensor = None  # type: ignore
    """ The angular velocities of the interested links (w.r.t the world).
    NOTE: This is a passive parameter computed from base_ang_vel, base_quat, and joint_vels.
    Shape: [N, num_frames, num_links, 3], N is the batch size.
    """

    # 链接位置掩码。
    # 形状：[N, num_frames, num_links]，N 是批次大小。
    link_pos_mask: torch.Tensor = None  # type: ignore
    """ The mask for the link positions.

    Shape: [N, num_frames, num_links], N is the batch size.
    """

    # 链接旋转掩码。
    # 形状：[N, num_frames, num_links]，N 是批次大小。
    link_rot_mask: torch.Tensor = None  # type: ignore
    """ The mask for the link rotations.

    Shape: [N, num_frames, num_links], N is the batch size.
    """

    # 到达特定帧的预期时间。
    # 形状：[N, num_frames]，N 是批次大小。
    time_to_target_frame: torch.Tensor = None  # type: ignore
    """ The time expected to reach the specific frame.

    Shape: [N, num_frames], N is the batch size.
    """

    # 布尔张量，指示当前数据是否有效。
    # 通常表示数据缓冲区是否耗尽。
    # 形状：[N, num_frames]，N 是批次大小。
    validity: torch.Tensor = None  # type: ignore
    """ A boolean tensor indicating whether the current data is valid or not.
    Usually indicating the data buffer is exhausted or not.

    Shape: [N, num_frames], N is the batch size.
    """

    # 静态方法：创建一个空的 MotionReferenceData 实例。
    @staticmethod
    def make_empty(
        num_envs: int,
        num_frames: int,
        num_joints: int,
        num_links: int,
        device=torch.device("cpu"),
        **kwargs,
    ) -> MotionReferenceData:
        # 创建并初始化所有张量。
        return_ = MotionReferenceData(
            joint_pos=torch.zeros(num_envs, num_frames, num_joints, device=device),
            joint_vel=torch.zeros(num_envs, num_frames, num_joints, device=device),
            joint_pos_mask=torch.ones(num_envs, num_frames, num_joints, device=device, dtype=torch.bool),
            joint_vel_mask=torch.ones(num_envs, num_frames, num_joints, device=device, dtype=torch.bool),
            base_pos_w=torch.zeros(num_envs, num_frames, 3, device=device),
            base_lin_vel_w=torch.zeros(num_envs, num_frames, 3, device=device),
            base_quat_w=torch.zeros(num_envs, num_frames, 4, device=device),
            base_ang_vel_w=torch.zeros(num_envs, num_frames, 3, device=device),
            base_pos_plane_mask=torch.ones(num_envs, num_frames, device=device, dtype=torch.bool),
            base_pos_height_mask=torch.ones(num_envs, num_frames, device=device, dtype=torch.bool),
            base_orientation_mask=torch.ones(num_envs, num_frames, device=device, dtype=torch.bool),
            base_heading_mask=torch.ones(num_envs, num_frames, device=device, dtype=torch.bool),
            link_pos_b=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            link_quat_b=torch.zeros(num_envs, num_frames, num_links, 4, device=device),
            link_pos_w=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            link_quat_w=torch.zeros(num_envs, num_frames, num_links, 4, device=device),
            link_lin_vel_b=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            link_ang_vel_b=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            link_lin_vel_w=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            link_ang_vel_w=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            link_pos_mask=torch.ones(num_envs, num_frames, num_links, device=device, dtype=torch.bool),
            link_rot_mask=torch.ones(num_envs, num_frames, num_links, device=device, dtype=torch.bool),
            time_to_target_frame=torch.zeros(num_envs, num_frames, device=device),
            validity=torch.zeros(num_envs, num_frames, dtype=torch.bool, device=device),
        )
        # 将四元数的 w 分量设置为 1.0。
        return_.base_quat_w[:, :, 0] = 1.0  # Set the w component of the quaternion to 1.0
        return_.link_quat_b[:, :, :, 0] = 1.0  # Set the w component of the link quaternions to 1.0
        return_.link_quat_w[:, :, :, 0] = 1.0  # Set the w component of the link quaternions to 1.0
        return return_

    # 重置方法：为给定的 env_ids 重置运动参考数据。将张量置零并标记为无效。
    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Reset the motion reference data for the given env_ids. Zeros tensors and marks as invalid."""
        # 将 env_ids 转换为张量。
        env_ids = torch.as_tensor(env_ids, device=self.validity.device)
        # 遍历所有字段。
        for field in fields(self):
            tensor = getattr(self, field.name)
            # 跳过非张量或形状不匹配的字段。
            if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != self.validity.shape[0]:
                continue
            # 根据字段名称进行不同的重置。
            if field.name == "validity":
                tensor[env_ids] = False
            elif field.name == "time_to_target_frame":
                tensor[env_ids] = -1.0
            elif field.name.endswith("_mask"):
                tensor[env_ids] = True
            else:
                tensor[env_ids] = 0
                # 如果是四元数，将 w 分量设为 1.0。
                if "quat" in field.name:
                    tensor[env_ids, ..., 0] = 1.0


# 定义 MotionReferenceState 数据类，表示运动参考开始时的机器人状态。用于重置。
@dataclass
class MotionReferenceState:
    """The robot state at the beginning of the motion reference. For the purpose of reset."""

    # 关节位置。
    joint_pos: torch.Tensor = None  # type: ignore
    # 关节速度。
    joint_vel: torch.Tensor = None  # type: ignore
    # 基座在世界坐标系中的位置。
    base_pos_w: torch.Tensor = None  # type: ignore
    # 基座在世界坐标系中的四元数。
    base_quat_w: torch.Tensor = None  # type: ignore
    # 基座在世界坐标系中的线速度。
    base_lin_vel_w: torch.Tensor = None  # type: ignore
    # 基座在世界坐标系中的角速度。
    base_ang_vel_w: torch.Tensor = None  # type: ignore

    # 由于链接位置可以直接从正向运动学获得，因此不必在此存储。

    # 后初始化方法，验证属性的一致性。
    def __post_init__(self):
        if not self.joint_pos is None:
            # 检查 joint_pos 和 joint_vel 形状相同。
            assert (
                self.joint_pos.shape == self.joint_vel.shape
            ), "The shape of joint_pos and joint_vel should be the same."
            # 检查第一维大小相同。
            assert (
                self.joint_pos.shape[0] == self.base_pos_w.shape[0] == self.base_quat_w.shape[0]
            ), "The first dimension of joint_pos, base_pos, base_quat, and base_lin_vel should be the same."
        else:
            # 所有属性应该为 None。
            assert (
                self.joint_pos is None
                and self.joint_vel is None
                and self.base_pos_w is None
                and self.base_quat_w is None
                and self.base_lin_vel_w is None
                and self.base_ang_vel_w is None
            ), "All attributes should be None or not None."

    # 静态方法：创建一个空的 MotionReferenceState 实例。
    @staticmethod
    def make_empty(
        num_envs: int,
        num_joints: int,
        device=torch.device("cpu"),
        **kwargs,
    ) -> MotionReferenceState:
        # 创建并初始化所有张量。
        return_ = MotionReferenceState(
            joint_pos=torch.zeros(num_envs, num_joints, device=device),
            joint_vel=torch.zeros(num_envs, num_joints, device=device),
            base_pos_w=torch.zeros(num_envs, 3, device=device),
            base_quat_w=torch.zeros(num_envs, 4, device=device),
            base_lin_vel_w=torch.zeros(num_envs, 3, device=device),
            base_ang_vel_w=torch.zeros(num_envs, 3, device=device),
        )
        # 将四元数的 w 分量设置为 1.0。
        return_.base_quat_w[:, 0] = 1.0
        return return_

    # 索引方法，用于获取指定环境的状态。
    def __getitem__(self, idx: Sequence[int] | torch.Tensor | None):
        if idx is None:
            return self

        # 返回新的 MotionReferenceState，其中张量被索引。
        return MotionReferenceState(
            joint_pos=self.joint_pos[idx],
            joint_vel=self.joint_vel[idx],
            base_pos_w=self.base_pos_w[idx],
            base_quat_w=self.base_quat_w[idx],
            base_lin_vel_w=self.base_lin_vel_w[idx],
            base_ang_vel_w=self.base_ang_vel_w[idx],
        )