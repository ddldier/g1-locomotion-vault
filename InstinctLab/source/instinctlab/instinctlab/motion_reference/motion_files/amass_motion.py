from __future__ import annotations

# 导入标准库和第三方库
import inspect          # 用于检查对象类型（如判断是否为类）
import os               # 操作系统路径操作
import pickle as pkl    # 序列化工具（此处未直接使用）
import yaml             # 解析 YAML 配置文件
from collections.abc import Sequence  # 抽象序列类型，用于类型注解
from typing import TYPE_CHECKING      # 用于类型检查时的条件导入

# 导入 Isaac Lab 的数学工具
import isaaclab.utils.math as math_utils

# 从 instinctlab 导入运动参考相关模块
from instinctlab.motion_reference import MotionReferenceData, MotionReferenceState, MotionSequence
from instinctlab.motion_reference.motion_buffer import MotionBuffer
from instinctlab.motion_reference.utils import estimate_angular_velocity, estimate_velocity
from instinctlab.utils.torch import ConcatBatchTensor

# 仅在类型检查时导入配置类，避免运行时循环导入
if TYPE_CHECKING:
    from .amass_motion_cfg import AmassMotionCfg

# 导入数值计算和深度学习库
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Sequence

import pytorch_kinematics as pk  # 用于运动学计算（如正向运动学）
import warp as wp                # NVIDIA Warp，用于 GPU 加速计算


# 定义一个 Warp 内核：在给定运动片段的 CDF 上采样一个时间点
@wp.kernel
def sample_bin_kernel(
    motion_ids: wp.array(dtype=wp.int32),   # 每个待采样环境对应的运动 ID
    batch_starts: wp.array(dtype=wp.int32), # 每个运动在拼接 CDF 中的起始偏移
    batch_sizes: wp.array(dtype=wp.int32),  # 每个运动的 bin 数量
    cdf: wp.array(dtype=wp.float32),        # 拼接后的累积分布函数（每个运动内部归一化）
    output: wp.array(dtype=wp.float32),     # 输出：采样得到的时间（秒）
    bin_length_s: wp.float32,               # 每个 bin 的时间长度（秒）
    seed: wp.int32,                         # 随机种子
):
    tid = wp.tid()                          # 当前线程 ID（对应一个待采样环境）
    state = wp.rand_init(seed + tid)        # 初始化该线程的随机状态
    motion = motion_ids[tid]                # 获取该环境对应的运动 ID
    start = batch_starts[motion]            # 该运动在拼接 CDF 中的起始索引
    size = batch_sizes[motion]              # 该运动的 bin 数量

    total = cdf[start + size - 1]           # 该运动的总概率（CDF 最后一个值）
    if total == wp.float(0.0):              # 如果总概率为 0，则直接输出 0
        output[tid] = wp.float(0.0)
        return
    u = wp.randf(state, wp.float(0.0), total)  # 在 [0, total) 内均匀采样

    # 二分查找：找到第一个 CDF 值 >= u 的 bin
    low = int(0)
    high = size - 1
    while low < high:
        mid = (low + high) // 2
        if cdf[start + mid] < u:
            low = mid + 1
        else:
            high = mid
    selected = low                          # 选中的 bin 索引

    # 在选中的 bin 内部再均匀采样一个小数偏移，得到连续时间
    bin_idx = wp.float(selected) + wp.randf(state, wp.float(0.0), wp.float(1.0))
    output[tid] = bin_idx * bin_length_s    # 转换为秒


# 使用 Warp 内核采样起始时间
def sample_start_times(
    motion_ids: torch.Tensor,               # 每个待采样环境对应的运动 ID
    motion_bin_weights: ConcatBatchTensor,  # 每个运动的 bin 权重（拼接形式）
    buffer_device: torch.device,            # 缓冲区所在设备
    motion_bin_length_s: float | None,      # 每个 bin 的时间长度
) -> torch.Tensor:
    if motion_bin_length_s is None:
        raise ValueError("motion_bin_length_s must be provided")

    num_assign = len(motion_ids)            # 待采样的环境数量
    temp_starts = torch.empty(num_assign, dtype=torch.float32, device=buffer_device)

    # 计算全局累积分布（跨所有运动拼接）
    global_cum = torch.cumsum(motion_bin_weights._concatenated_tensor, dim=0)
    # 每个运动的前一个运动的结束索引
    prev_ends = motion_bin_weights._batch_starts[1:] - 1
    # 每个运动在全局 CDF 中的起始偏移值
    segment_offsets_values = torch.cat([torch.tensor([0.0], device=global_cum.device), global_cum[prev_ends]])
    segment_lengths = motion_bin_weights._batch_sizes
    # 将偏移值重复扩展到每个 bin
    offsets = torch.repeat_interleave(segment_offsets_values, segment_lengths)
    # 得到每个运动内部归一化的 CDF（减去该运动的起始偏移）
    flat_cdf = global_cum - offsets

    # 启动 Warp 内核进行采样
    wp.launch(
        kernel=sample_bin_kernel,
        dim=num_assign,
        inputs=[
            wp.from_torch(motion_ids.to(torch.int32), dtype=wp.int32),
            wp.from_torch(motion_bin_weights._batch_starts.to(torch.int32), dtype=wp.int32),
            wp.from_torch(motion_bin_weights._batch_sizes.to(torch.int32), dtype=wp.int32),
            wp.from_torch(flat_cdf, dtype=wp.float32),
            wp.from_torch(temp_starts, dtype=wp.float32),
            motion_bin_length_s,
            torch.randint(0, (1 << 31) - 1, (1,), device=buffer_device).item(),
        ],
        device=buffer_device,
    )

    return temp_starts


# AmassMotion 类：处理 AMASS 格式的运动文件，继承自 MotionBuffer
class AmassMotion(MotionBuffer):
    """Processing AMASS formatted file structure.
    NOTE: the indexing logic is as follows:
        self._env_buffer_selection: corresponding _motion_buffer index of each env
        self._buffer_motion_selection: the indices of the motion files to use in each motion buffer
    """

    cfg: AmassMotionCfg  # 类型注解：配置对象

    def __init__(
        self,
        cfg: AmassMotionCfg,
        *args,
        **kwargs,
    ):
        """Initialize the motion dataset.

        Args:
            cfg: The configuration parameters for the motion dataset.
            robot_chain: The pytorch_kinematics.Chain for computing positions for link_of_interests.
        """
        # 初始化基类
        super().__init__(cfg, *args, **kwargs)
        self.output_device = self.device                # 输出数据所在设备
        if self.cfg.buffer_device == "output_device":
            self.buffer_device = self.output_device     # 缓冲区设备与输出设备相同
        else:
            self.buffer_device = torch.device("cpu")    # 否则缓冲区在 CPU
        self._joint_limits = self._joint_limits.to(self.buffer_device)  # 关节限位移到缓冲区设备
        self._refresh_motion_file_list()                # 刷新运动文件列表
        self._prepare_retargetting_func()               # 准备重定向函数

    """
    Properties.
    """

    @property
    def num_trajectories(self) -> int:
        """返回运动轨迹的总数（即运动文件数量）"""
        return len(self._all_motion_files)

    @property
    def complete_motion_lengths(self) -> torch.Tensor:
        """返回每个已分配运动的完整时长（秒）"""
        return (
            self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[:]]
            / self._all_motion_sequences.framerate[self._assigned_env_motion_selection[:]]
        )

    @property
    def assigned_motion_lengths(self) -> torch.Tensor:
        """返回每个已分配运动从起始时间到结束的剩余时长（秒）"""
        return (
            (
                self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection]
                / self._all_motion_sequences.framerate[self._assigned_env_motion_selection]
            )
            - self._motion_buffer_start_time_s
        ).to(self.output_device)

    """
    Operations.
    """

    def enable_trajectories(self, traj_ids: torch.Tensor | slice | None = None) -> None:
        """启用指定的运动轨迹（根据 ID 过滤运动文件），并加载对应的运动序列。"""
        if not traj_ids is None:
            if isinstance(traj_ids, torch.Tensor):
                # 按张量索引选择文件
                self._all_motion_files = [self._all_motion_files[i] for i in traj_ids]
            elif isinstance(traj_ids, slice):
                # 按切片选择文件
                self._all_motion_files = self._all_motion_files[traj_ids]
            self._motion_weights = self._motion_weights[traj_ids].to(self.buffer_device)
        self._load_motion_sequences()                   # 加载运动序列
        if self.cfg.motion_bin_length_s is not None:
            self._init_motion_bin_weights()             # 如果配置了 bin 长度，则初始化 bin 权重

    def set_env_ids_assignments(self, env_ids):
        """设置环境 ID 到运动缓冲区的分配，并采样起始信息。"""
        return_ = super().set_env_ids_assignments(env_ids)
        # 设置环境-运动选择缓冲区
        self._sample_assigned_env_starting_stub()
        return return_

    def reset(self, env_ids: Sequence[int] | torch.Tensor, symmetric_augmentation_mask_buffer: torch.Tensor) -> None:
        """重置指定环境：重新采样起始信息，并随机生成对称增强掩码。"""
        self._sample_assigned_env_starting_stub(env_ids)
        # 采样对称增强掩码（AMASS 中所有运动都支持增强）
        symmetric_augmentation_mask_buffer[env_ids] = torch.randint(
            0,
            2,
            (len(env_ids),),
            device=self.output_device,
            dtype=torch.bool,
        )

    def fill_init_reference_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor,
        state_buffer: MotionReferenceState,
    ) -> None:
        """填充初始参考状态（关节位置、速度、基座位姿等）到 state_buffer。"""
        assert self.env_ids_is_assigned(env_ids).all(), "The env_ids should be assigned to this motion buffer."
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        # 根据起始时间计算帧索引
        frame_selection = torch.floor(
            self._motion_buffer_start_time_s[assigned_ids]
            * self._all_motion_sequences.framerate[self._assigned_env_motion_selection[assigned_ids]]
        ).to(torch.long)

        # 从运动序列中提取基座位置、四元数、线速度、角速度
        base_pos_w = self._all_motion_sequences.base_pos_w[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]
        base_quat_w = self._all_motion_sequences.base_quat_w[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]

        base_lin_vel_w = self._all_motion_sequences.base_lin_vel_w[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]
        base_ang_vel_w = self._all_motion_sequences.base_ang_vel_w[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]

        # 提取关节位置和速度
        joint_pos = self._all_motion_sequences.joint_pos[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]

        joint_vel = self._all_motion_sequences.joint_vel[
            self._assigned_env_motion_selection[assigned_ids], frame_selection
        ]

        # 将基座位置加上环境原点偏移
        base_pos_w += self._get_motion_based_origin(env_origins, env_ids)

        # 检查是否有任何连杆低于地面
        if self.cfg.ensure_link_below_zero_ground:
            link_pos_w = self._all_motion_sequences.link_pos_w[
                self._assigned_env_motion_selection[assigned_ids], frame_selection
            ].to(self.output_device)
            min_link_height = link_pos_w[..., 2].min(dim=-1).values
            if (min_link_height < 0).any():
                # 如果有连杆低于地面，则抬高基座
                base_pos_w[..., 2] += -min_link_height
        base_pos_w[..., 2] += self.cfg.motion_start_height_offset

        # 将数据写入 state_buffer
        state_buffer.joint_pos[env_ids] = joint_pos.to(self.output_device)
        state_buffer.joint_vel[env_ids] = joint_vel.to(self.output_device)
        state_buffer.base_pos_w[env_ids] = base_pos_w.to(self.output_device)
        state_buffer.base_quat_w[env_ids] = base_quat_w.to(self.output_device)
        state_buffer.base_lin_vel_w[env_ids] = base_lin_vel_w.to(self.output_device)
        state_buffer.base_ang_vel_w[env_ids] = base_ang_vel_w.to(self.output_device)

    def fill_motion_data(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: MotionReferenceData,
    ) -> None:
        """填充指定时间戳的运动数据到 data_buffer。"""
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        # 计算每个环境对应的帧索引（可能包含多个时间戳）
        frame_selections = torch.round(
            (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + sample_timestamp.to(self.buffer_device))
            * self._all_motion_sequences.framerate[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
        )  # 形状 [len(env_ids), num_frames]

        # 检查采样帧是否越界，并记录有效性掩码
        data_buffer.validity[env_ids] = (
            (
                frame_selections
                < self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(
                    -1
                )
            )
            .to(torch.bool)
            .to(self.output_device)
        )
        # 将越界的帧索引截断到最后一帧
        frame_selections = torch.where(
            data_buffer.validity[env_ids].to(self.buffer_device),
            frame_selections,
            self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
            - 1,
        ).to(torch.long)

        # 扁平化索引以便批量提取数据
        assigned_ids_across_frame = assigned_ids.unsqueeze(-1).repeat(
            1, frame_selections.shape[1]
        )  # 形状 [len(env_ids), num_frames]
        assigned_ids_across_frame_ = assigned_ids_across_frame.flatten()
        frame_selections_ = frame_selections.flatten()

        # 提取关节位置和速度
        data_buffer.joint_pos[env_ids] = (
            self._all_motion_sequences.joint_pos[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.joint_pos.shape[1:])
            .to(self.output_device)
        )
        data_buffer.joint_vel[env_ids] = (
            self._all_motion_sequences.joint_vel[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.joint_vel.shape[1:])
            .to(self.output_device)
        )
        # 提取连杆在基座坐标系下的位置和四元数
        data_buffer.link_pos_b[env_ids] = (
            self._all_motion_sequences.link_pos_b[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_pos_b.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_quat_b[env_ids] = (
            self._all_motion_sequences.link_quat_b[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_quat_b.shape[1:])
            .to(self.output_device)
        )
        # 提取连杆在世界坐标系下的位置和四元数
        data_buffer.link_pos_w[env_ids] = (
            self._all_motion_sequences.link_pos_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_pos_w.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_quat_w[env_ids] = (
            self._all_motion_sequences.link_quat_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_quat_w.shape[1:])
            .to(self.output_device)
        )
        # 提取连杆在基座坐标系下的线速度和角速度
        data_buffer.link_lin_vel_b[env_ids] = (
            self._all_motion_sequences.link_lin_vel_b[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_lin_vel_b.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_ang_vel_b[env_ids] = (
            self._all_motion_sequences.link_ang_vel_b[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_ang_vel_b.shape[1:])
            .to(self.output_device)
        )
        # 提取连杆在世界坐标系下的线速度和角速度
        data_buffer.link_lin_vel_w[env_ids] = (
            self._all_motion_sequences.link_lin_vel_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_lin_vel_w.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_ang_vel_w[env_ids] = (
            self._all_motion_sequences.link_ang_vel_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(len(env_ids), *data_buffer.link_ang_vel_w.shape[1:])
            .to(self.output_device)
        )

        # 提取基座位置、线速度、四元数、角速度
        base_pos_w = self._all_motion_sequences.base_pos_w[
            self._assigned_env_motion_selection[assigned_ids_across_frame_],
            frame_selections_,
        ].reshape(len(env_ids), frame_selections.shape[1], 3)
        base_lin_vel_w = self._all_motion_sequences.base_lin_vel_w[
            self._assigned_env_motion_selection[assigned_ids_across_frame_],
            frame_selections_,
        ].reshape(len(env_ids), frame_selections.shape[1], 3)
        base_quat_w = self._all_motion_sequences.base_quat_w[
            self._assigned_env_motion_selection[assigned_ids_across_frame_],
            frame_selections_,
        ].reshape(len(env_ids), frame_selections.shape[1], 4)
        base_ang_vel_w = self._all_motion_sequences.base_ang_vel_w[
            self._assigned_env_motion_selection[assigned_ids_across_frame_],
            frame_selections_,
        ].reshape(len(env_ids), frame_selections.shape[1], 3)
        data_buffer.base_pos_w[env_ids] = base_pos_w.to(self.output_device)
        data_buffer.base_lin_vel_w[env_ids] = base_lin_vel_w.to(self.output_device)
        data_buffer.base_quat_w[env_ids] = base_quat_w.to(self.output_device)
        data_buffer.base_ang_vel_w[env_ids] = base_ang_vel_w.to(self.output_device)

        # 加上环境原点偏移
        data_buffer.base_pos_w[env_ids] += self._get_motion_based_origin(env_origins, env_ids).unsqueeze(
            1
        )  # 避免原地操作
        data_buffer.link_pos_w[env_ids] += (
            self._get_motion_based_origin(env_origins, env_ids).unsqueeze(1).unsqueeze(1)
        )  # 避免原地操作

        # AMASS 提供了所有关节/连杆，因此所有掩码都有效
        data_buffer.joint_pos_mask[env_ids] = True
        data_buffer.joint_vel_mask[env_ids] = True
        data_buffer.base_pos_plane_mask[env_ids] = True
        data_buffer.base_pos_height_mask[env_ids] = True
        data_buffer.base_orientation_mask[env_ids] = True
        data_buffer.base_heading_mask[env_ids] = True
        data_buffer.link_pos_mask[env_ids] = True
        data_buffer.link_rot_mask[env_ids] = True

    def get_current_motion_identifiers(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> list[str]:
        """获取每个环境当前分配的运动文件标识符。
        Args:
            env_ids: 要获取标识符的环境 ID。
        Returns:
            每个环境的运动标识符列表。
        """
        if env_ids is None:
            return [self._all_motion_files[i] for i in self._assigned_env_motion_selection]
        else:
            assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
            return [self._all_motion_files[i] for i in self._assigned_env_motion_selection[assigned_ids]]

    def get_current_motion_weights(self, env_ids=None):
        """获取当前运动权重。"""
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device) if env_ids is not None else None
        if not hasattr(self, "_motion_weights"):
            return super().get_current_motion_weights(env_ids)
        else:
            return self._motion_weights[self._assigned_env_motion_selection[assigned_ids]]

    def update_motion_weights(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        weight_ratio: float | torch.Tensor = 1.0,
    ) -> None:
        """更新运动采样权重。
        ## Args:
            - env_ids: 要更新的环境 ID。如果为 None，则更新所有环境。
        """
        if not hasattr(self, "_motion_weights"):
            return super().update_motion_weights(env_ids, weight_ratio)
        if env_ids is not None:
            env_ids = self.env_ids_to_assigned_ids(env_ids)
        self._motion_weights[self._assigned_env_motion_selection[env_ids]] *= weight_ratio
        # 归一化运动权重
        self._motion_weights = torch.clamp(self._motion_weights, min=0.0)
        self._motion_weights /= self._motion_weights.sum()

    """
    Internal helper functions (for loading motion files and calling retargetting functions).
    """

    def _init_motion_weights(self, motion_weights: torch.Tensor | None = None) -> None:
        """初始化每个运动文件及其时间段的采样权重。"""
        # 运动权重
        if motion_weights is not None:
            self._motion_weights = torch.tensor(
                motion_weights,
                device=self.buffer_device,
                dtype=torch.float,
            )
        else:
            self._motion_weights = torch.ones(
                len(self._all_motion_files),
                device=self.buffer_device,
                dtype=torch.float,
            )

        # 安全检查
        assert len(self._all_motion_files) == len(self._motion_weights), (
            "The number of motion files and motion weights should be the same."
            f"Got {len(self._all_motion_files)} motion files and {len(self._motion_weights)} motion weights."
        )

    def _init_motion_bin_weights(self) -> None:
        """初始化每个运动的时间 bin 权重（用于按时间采样）。"""
        print("Initializing motion bin weights motion_start_from_middle_range is disabled.")
        # 计算每个运动的 bin 数量
        num_motion_bins = torch.floor(
            (self._all_motion_sequences.buffer_length / self._all_motion_sequences.framerate).to(self.buffer_device)
            / self.cfg.motion_bin_length_s
        ).to(torch.long)
        # 创建拼接的 bin 权重张量
        self._motion_bin_weights = ConcatBatchTensor(
            batch_sizes=num_motion_bins,
            data_shape=tuple(),
            device=self.buffer_device,
        )
        for motion_idx in range(len(self._all_motion_files)):
            # 每个运动的 bin 概率初始均匀
            motion_bin_probability = torch.ones(
                num_motion_bins[motion_idx],
                device=self.buffer_device,
                dtype=torch.float,
            )
            motion_bin_probability /= motion_bin_probability.sum()
            self._motion_bin_weights[motion_idx] = motion_bin_probability

    def _refresh_motion_file_list(self):
        """从文件系统刷新运动文件列表和运动权重。"""
        # 搜索目录下所有以 "_poses.npz" 结尾的文件
        if self.cfg.filtered_motion_selection_filepath is not None:
            if self.cfg.subset_selection is not None:
                print("Warning: subset_selection is ignored when filtered_motion_files is provided.")
            with open(self.cfg.filtered_motion_selection_filepath) as f:
                yaml_data = yaml.safe_load(f)
            all_motion_files = yaml_data["selected_files"]
            for i, motion_file in enumerate(all_motion_files):
                all_motion_files[i] = os.path.join(self.cfg.path, motion_file)
        elif self.cfg.subset_selection is not None:
            all_motion_files = []
            for subset, subjects in self.cfg.subset_selection.items():
                if subjects is None:
                    # 该子集下的所有受试者
                    subjects = os.listdir(os.path.join(self.cfg.path, subset))
                for subject in subjects:
                    for root, _, files in os.walk(os.path.join(self.cfg.path, subset, subject)):
                        for file in files:
                            for endings in self.cfg.supported_file_endings:
                                if file.endswith(endings):
                                    all_motion_files.append(os.path.join(root, file))
        else:
            all_motion_files = []
            for root, _, files in os.walk(self.cfg.path, followlinks=True):
                for file in files:
                    for endings in self.cfg.supported_file_endings:
                        if file.endswith(endings):
                            all_motion_files.append(os.path.join(root, file))
        self._all_motion_files: list[str] = all_motion_files

        # 用于调试时覆盖运动文件列表的 hack
        # 注意：需注意 _motion_weights 的值。
        # self._all_motion_files = [
        #     # "/home/leo/Datasets/AMASS/CMU/140/140_04_poses.npz",
        #     # "/home/leo/Datasets/AMASS/CMU/140/140_07_poses.npz",
        #     # "/home/leo/Datasets/AMASS/CMU/140/140_08_poses.npz",
        #     # "/home/leo/Datasets/AMASS/KIT/1721/RecoveryStepping_30_45_02_poses.npz",
        #     "/home/leo/Datasets/AMASS/BioMotionLab_NTroje/rub072/0014_knocking2_poses.npz",
        #
        print(f"[{self.__class__.__name__} Motion] {len(all_motion_files)} files found")
        assert len(self._all_motion_files) > 0, (
            "No motion files found. Please check the path."
            "The supported file are shown in the source code of this function."
        )

        # 初始化运动权重（如果 YAML 中提供了 motion_weights）
        self._init_motion_weights(yaml_data.get("motion_weights", None) if "yaml_data" in locals() else None)

    def _read_motion_file(self, motion_file_idx: int) -> MotionSequence:
        """根据文件扩展名读取运动文件，并返回 MotionSequence。"""
        filepath = self._all_motion_files[motion_file_idx]
        if filepath.endswith("poses.npz") or filepath.endswith("stageii.npz"):
            return self._read_amass_motion_file(filepath)
        elif filepath.endswith("retargetted.npz") or filepath.endswith("retargeted.npz"):
            return self._read_retargetted_motion_file(filepath)
        elif filepath.endswith("soma.csv"):
            return self._read_retargeted_soma_motion_file(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")

    def _read_amass_motion_file(self, filepath: str) -> MotionSequence:
        """读取 AMASS 格式的 .npz 文件，并返回重定向后的 MotionSequence。（涉及 IO）"""
        raw_data = np.load(filepath, mmap_mode="r")
        try:
            framerate = raw_data["mocap_framerate"]
        except KeyError:
            framerate = raw_data["mocap_frame_rate"]
        poses = raw_data["poses"]
        poses = torch.from_numpy(poses).to(dtype=torch.float, device=self.buffer_device)
        root_trans = torch.from_numpy(raw_data["trans"]).to(dtype=torch.float, device=self.buffer_device)
        framerate = framerate.item()
        return self._build_motion_sequence_from_smpl(poses, root_trans, framerate)

    def _read_retargetted_motion_file(self, filepath: str) -> MotionSequence:
        """读取已重定向的 .npz 文件，并返回 MotionSequence。"""
        raw_data = np.load(filepath, mmap_mode="r", allow_pickle=True)
        framerate = raw_data["framerate"].item()
        joint_names = (
            raw_data["joint_names"] if isinstance(raw_data["joint_names"], list) else raw_data["joint_names"].tolist()
        )
        joint_pos = torch.as_tensor(raw_data["joint_pos"], device=self.buffer_device, dtype=torch.float)
        root_trans = torch.as_tensor(raw_data["base_pos_w"], device=self.buffer_device, dtype=torch.float)
        root_quat = torch.as_tensor(raw_data["base_quat_w"], device=self.buffer_device, dtype=torch.float)

        frame_range = getattr(self.cfg, "retargeted_frame_range", None)
        if frame_range is not None:
            start, stop = frame_range
            if (
                not isinstance(start, int)
                or not isinstance(stop, int)
                or not 0 <= start < stop <= joint_pos.shape[0]
            ):
                raise ValueError(f"Invalid retargeted_frame_range {frame_range} for {filepath}")
            print(
                f"[Motion] {os.path.basename(filepath)}: raw frames "
                f"[{start}, {stop}) / {joint_pos.shape[0]} before interpolation"
            )
            # Slice all kinematic inputs together. Velocities and FK are then
            # rebuilt from this same window; the original NPZ is never changed.
            joint_pos = joint_pos[start:stop]
            root_trans = root_trans[start:stop]
            root_quat = root_quat[start:stop]

        # 将关节顺序映射到 Isaac 关节名称
        retargetted_joints_to_output_joints_ids = [joint_names.index(j_name) for j_name in self.isaac_joint_names]
        joint_pos = joint_pos[:, retargetted_joints_to_output_joints_ids]

        return self._pack_retargetted_motion_sequence(
            root_trans,
            root_quat,
            joint_pos,
            framerate,
        )

    def _read_retargeted_soma_motion_file(self, filepath: str) -> MotionSequence:
        """读取 SOMA 格式的 CSV 文件，并返回 MotionSequence。"""
        with open(filepath, encoding="utf-8-sig") as f:
            header = f.readline().strip().split(",")
        if len(header) < 8:
            raise ValueError(f"Invalid soma CSV format in {filepath}: expected at least 8 columns, got {len(header)}.")

        raw_values = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)
        if raw_values.ndim == 1:
            raw_values = raw_values[None, :]
        if raw_values.shape[1] != len(header):
            raise ValueError(
                "Soma CSV header/data column mismatch: "
                f"{filepath} has {len(header)} header columns but {raw_values.shape[1]} data columns."
            )
        framerate = self.cfg.assumed_file_framerate

        joint_names = header[7:]

        # 平移：厘米转米
        root_trans = torch.as_tensor(raw_values[:, 1:4] * 0.01, dtype=torch.float, device=self.buffer_device)

        # 旋转：欧拉角 XYZ（度）转四元数 (w, x, y, z)
        euler_xyz = torch.as_tensor(np.deg2rad(raw_values[:, 4:7]), dtype=torch.float, device=self.buffer_device)
        root_quat = math_utils.quat_from_euler_xyz(euler_xyz[:, 0], euler_xyz[:, 1], euler_xyz[:, 2])

        # 关节：度转弧度
        joint_pos = torch.as_tensor(np.deg2rad(raw_values[:, 7:]), dtype=torch.float, device=self.buffer_device)

        # 将关节映射到 isaac_joint_names
        # SOMA 重定向器通常在关节名后附加 "_dof"
        joint_names_clean = [j.replace("_dof", "") for j in joint_names]

        missing_joints = [j_name for j_name in self.isaac_joint_names if j_name not in joint_names_clean]
        if missing_joints:
            raise ValueError(
                "Joint mapping failed for soma CSV. Missing joints required by the articulation: "
                f"{missing_joints}. File: {filepath}"
            )

        retargetted_joints_to_output_joints_ids = [joint_names_clean.index(j_name) for j_name in self.isaac_joint_names]

        joint_pos = joint_pos[:, retargetted_joints_to_output_joints_ids]

        return self._pack_retargetted_motion_sequence(
            root_trans,
            root_quat,
            joint_pos,
            framerate,
        )

    def _build_motion_sequence_from_smpl(
        self, poses: torch.Tensor, root_trans: torch.Tensor, framerate: float
    ) -> MotionSequence:
        """从 SMPL 姿态和根平移构建 MotionSequence（包含重定向）。"""
        # 如果配置了跳过帧，则减少数据量
        if self.cfg.skip_frames >= 1:
            nframes = poses.shape[0]
            sampling_frame_idxs = np.arange(0, nframes, int(self.cfg.skip_frames + 1)).astype(np.int32)
            framerate /= self.cfg.skip_frames + 1
            poses = poses[sampling_frame_idxs]
            root_trans = root_trans[sampling_frame_idxs]

        # 如果需要重定向
        assert self.retargetting_func is not None, "Retargetting function is not provided."
        # 根据 SMPL-H 定义，仅使用前 22 个关节，但 SMPL-X 使用 24 个关节
        smpl_poses = poses[:, : 24 * 3].reshape(-1, 24, 3)

        joint_pos, root_poses = self.retargetting_func(
            smpl_poses,
            root_trans,
            joint_names=self.isaac_joint_names,
            **self.cfg.retargetting_func_kwargs,
        )
        root_trans, root_quat = root_poses[..., :3], root_poses[..., 3:]  # 形状 [num_frames, 7]

        return self._pack_retargetted_motion_sequence(
            root_trans,
            root_quat,
            joint_pos,
            framerate,
        )

    def _pack_retargetted_motion_sequence(
        self, root_trans: torch.Tensor, root_quat: torch.Tensor, joint_pos: torch.Tensor, framerate: float | int
    ):
        """将重定向后的运动序列打包为 MotionSequence 对象。
        Args:
            - root_trans: 形状 [num_frames, 3]
            - root_quat: 形状 [num_frames, 4]
            - joint_pos: 形状 [num_frames, num_joints]
            - framerate: 运动序列的帧率
        """

        if self.cfg.motion_interpolate_func:
            # 应用运动插值以匹配目标帧率
            root_trans, root_quat, joint_pos = self.cfg.motion_interpolate_func(
                root_trans, root_quat, joint_pos, framerate, self.cfg.motion_target_framerate
            )
            framerate = self.cfg.motion_target_framerate

        # 估计关节速度、基座线速度、基座角速度
        if self.cfg.velocity_estimation_method is not None:
            joint_vel = estimate_velocity(
                joint_pos.unsqueeze(0), 1 / framerate, self.cfg.velocity_estimation_method
            ).squeeze(0)
            base_lin_vel_w = estimate_velocity(
                root_trans.unsqueeze(0), 1 / framerate, self.cfg.velocity_estimation_method
            ).squeeze(0)
            base_ang_vel_w = estimate_angular_velocity(
                root_quat.unsqueeze(0), 1 / framerate, self.cfg.velocity_estimation_method
            ).squeeze(0)
        else:
            joint_vel = torch.zeros_like(joint_pos)
            base_lin_vel_w = torch.zeros_like(root_trans)
            base_ang_vel_w = torch.zeros_like(root_trans)

        # 注意：如果感兴趣的连杆为空，可能会导致 bug。
        link_pos_quat_b = self.forward_kinematics_func(joint_pos)  # [num_frames, num_links, 7]
        link_pos_b = link_pos_quat_b[..., :3]
        link_quat_b = link_pos_quat_b[..., 3:]
        # 将连杆从基座坐标系转换到世界坐标系
        link_pos_w, link_quat_w = math_utils.combine_frame_transforms(
            root_trans.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            root_quat.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            link_pos_b,
            link_quat_b,
        )

        if self.cfg.velocity_estimation_method is not None:
            # 估计连杆在基座坐标系下的线速度
            link_lin_vel_b = estimate_velocity(
                link_pos_b.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            # 估计连杆在基座坐标系下的角速度
            link_ang_vel_b = estimate_angular_velocity(
                link_quat_b.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            # 估计连杆在世界坐标系下的线速度
            link_lin_vel_w = estimate_velocity(
                link_pos_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
            # 估计连杆在世界坐标系下的角速度
            link_ang_vel_w = estimate_angular_velocity(
                link_quat_w.permute(1, 0, 2), 1 / framerate, self.cfg.velocity_estimation_method
            ).permute(1, 0, 2)
        else:
            link_lin_vel_b = torch.zeros_like(link_pos_b)
            link_ang_vel_b = torch.zeros_like(link_quat_b)
            link_lin_vel_w = torch.zeros_like(link_pos_w)
            link_ang_vel_w = torch.zeros_like(link_quat_w)

        return MotionSequence(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            base_pos_w=root_trans,
            base_lin_vel_w=base_lin_vel_w,
            base_quat_w=root_quat,
            base_ang_vel_w=base_ang_vel_w,
            link_pos_b=link_pos_b,
            link_quat_b=link_quat_b,
            link_lin_vel_b=link_lin_vel_b,
            link_ang_vel_b=link_ang_vel_b,
            link_lin_vel_w=link_lin_vel_w,
            link_ang_vel_w=link_ang_vel_w,
            link_pos_w=link_pos_w,
            link_quat_w=link_quat_w,
            framerate=torch.as_tensor(framerate),
            buffer_length=torch.as_tensor(joint_pos.shape[0]),
        )

    def _prepare_retargetting_func(self):
        """准备重定向函数（可能是类或函数）。"""
        # print(f"[AMASS Motion] Retargetting to {self.isaac_joint_names}")
        if inspect.isclass(self.cfg.retargetting_func):
            self.retargetting_func = self.cfg.retargetting_func(
                joint_names=self.isaac_joint_names,
                device=self.buffer_device,
                **self.cfg.retargetting_func_kwargs,
            )
        else:
            self.retargetting_func = self.cfg.retargetting_func

    def _load_motion_sequences(self):
        """将所有运动序列文件加载到内存中（已重定向形式）。"""
        # 首次全部加载
        # TODO: 使用多进程加载运动文件
        print(f"[AMASS Motion] Loading motion files, should be {len(self._all_motion_files)} in total...")
        all_motion_sequences = list(map(self._read_motion_file, range(len(self._all_motion_files))))
        print(f"[AMASS Motion] All {len(all_motion_sequences)} motion files loaded.")

        # 将所有运动序列填充到单个 PyTorch 张量缓冲区，以加速访问
        # max_buffer_length = max([int(motion.buffer_length) for motion in all_motion_sequences])
        # self._all_motion_sequences = MotionSequence.make_empty(
        #     len(all_motion_sequences),
        #     max_buffer_length,
        #     num_joints=self.articulation_view.max_dofs,
        #     num_links=self.num_link_to_ref,
        #     device=self.buffer_device,
        # )
        print(
            "[AMASS Motion] buffer lengths statistics:"
            f" mean: {np.array([motion.buffer_length for motion in all_motion_sequences]).mean()},"
            f" max: {np.array([motion.buffer_length for motion in all_motion_sequences]).max()},"
            f" min: {np.array([motion.buffer_length for motion in all_motion_sequences]).min()},"
        )
        # 创建拼接批处理张量
        self._all_motion_sequences = MotionSequence.make_emtpy_concat_batch(
            buffer_lengths=[int(motion.buffer_length) for motion in all_motion_sequences],
            num_joints=self.articulation_view.max_dofs,
            num_links=self.num_link_to_ref,
            device=self.buffer_device,
        )
        for i, motion in enumerate(all_motion_sequences):
            for attr in self._all_motion_sequences.attrs_with_frame_dim:
                getattr(self._all_motion_sequences, attr)[i, : motion.buffer_length] = getattr(motion, attr)
            for attr in self._all_motion_sequences.attrs_only_batch_dim:
                getattr(self._all_motion_sequences, attr)[i] = getattr(motion, attr)

    """
    Internal helper functions (for sampling motion files).
    """

    def _sample_assigned_env_starting_stub(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """为分配的环境采样运动文件 ID 和起始时间。
        这里的 stub 指的是为分配的环境采样运动数据所需的必要信息。
        """

        # 如果尚未构建必要的 stub，则构建。
        if not hasattr(self, "_assigned_env_motion_selection"):
            self._assigned_env_motion_selection = torch.zeros(
                self.num_assigned_envs,
                dtype=torch.long,
                device=self.buffer_device,
            )
        if not hasattr(self, "_motion_buffer_start_time_s"):
            self._motion_buffer_start_time_s = torch.zeros(
                self.num_assigned_envs,
                dtype=torch.float,
                device=self.buffer_device,
            )

        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        if self.cfg.env_starting_stub_sampling_strategy == "independent":
            self._sample_assigned_env_motion_selection(assigned_ids)
            self._sample_env_motion_start_time(assigned_ids)
        elif self.cfg.env_starting_stub_sampling_strategy == "concat_motion_bins":
            self._sample_as_concat_motion_bins(assigned_ids)
        else:
            raise ValueError(
                f"Unsupported env starting stub sampling method: {self.cfg.env_starting_stub_sampling_strategy}"
            )

    def _sample_assigned_env_motion_selection(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """为缓冲区重新采样运动选择，如果尚未构建则构建。
        这里采样的是环境-运动选择，与起始时间独立。
        """
        if len(assigned_ids) == 0:
            return
        self._assigned_env_motion_selection[assigned_ids] = torch.multinomial(
            self._motion_weights,
            len(assigned_ids),
            replacement=True,
        ).to(self.buffer_device)

    def _sample_env_motion_start_time(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """为分配的环境采样起始时间。（在重置时）
        这里采样的是环境起始时间，与运动选择独立。
        """
        assert assigned_ids is not None

        if self.cfg.motion_start_from_middle_range[1] > 0.0:
            assert (
                self.cfg.motion_start_from_middle_range[0] >= 0
                and self.cfg.motion_start_from_middle_range[1] >= self.cfg.motion_start_from_middle_range[0]
            ), (
                "motion_start_from_middle_range should be non-negative and the second value should be larger than the"
                " first one."
            )
            # 在指定范围内随机采样起始时间比例
            random_start_time = (
                torch.rand(
                    len(assigned_ids),
                    device=self.buffer_device,
                )
                * (self.cfg.motion_start_from_middle_range[1] - self.cfg.motion_start_from_middle_range[0])
                + self.cfg.motion_start_from_middle_range[0]
            )
            # 转换为绝对时间（秒）
            random_start_time *= self._all_motion_sequences.buffer_length[
                self._assigned_env_motion_selection[assigned_ids]
            ].to(torch.float)
            random_start_time /= self._all_motion_sequences.framerate[
                self._assigned_env_motion_selection[assigned_ids]
            ].to(torch.float)
            self._motion_buffer_start_time_s[assigned_ids] = random_start_time

        if hasattr(self, "_motion_bin_weights"):
            motion_ids = self._assigned_env_motion_selection[assigned_ids]
            # 使用 Warp 内核基于 bin 权重采样起始时间
            self._motion_buffer_start_time_s[assigned_ids] = sample_start_times(
                motion_ids, self._motion_bin_weights, self.buffer_device, self.cfg.motion_bin_length_s
            )

    def _sample_as_concat_motion_bins(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """将所有运动视为拼接成单个运动，从中采样运动文件 ID 和起始时间。"""
        # 从拼接的 bin 权重中采样扁平化的 bin ID
        flattened_bin_ids = torch.multinomial(
            self._motion_bin_weights._concatenated_tensor,
            len(assigned_ids),
            replacement=True,
        ).to(self.buffer_device)
        # 识别运动 ID 和实际的 bin ID
        motion_ids, bin_ids = self._motion_bin_weights.unwarp_flattened_idx(flattened_bin_ids)
        self._assigned_env_motion_selection[assigned_ids] = motion_ids
        # 计算起始时间：bin 索引 * bin 长度 + 随机偏移
        self._motion_buffer_start_time_s[assigned_ids] = (
            bin_ids * self.cfg.motion_bin_length_s
            + torch.rand(len(assigned_ids), device=self.buffer_device) * self.cfg.motion_bin_length_s
        )

    """
    Internal helper functions (Others).
    """

    def _get_motion_based_origin(self, env_origins: torch.Tensor, env_ids: Sequence[int] | torch.Tensor):
        """获取基于运动的环境原点，或直接使用调用者提供的 env_origins。"""
        return env_origins[env_ids]