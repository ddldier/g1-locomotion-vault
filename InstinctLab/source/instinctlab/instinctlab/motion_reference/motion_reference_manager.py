# 启用延迟注解，使得类型注解在运行时不会被求值，便于前向引用。
from __future__ import annotations
# 导入 inspect，用于检查对象的签名。
import inspect
# 导入 os，用于操作系统相关功能。
import os
# 导入 Sequence，用于类型提示。
from collections.abc import Sequence
# 导入 copy，用于浅拷贝。
from copy import copy
# 导入 PrettyTable，用于生成表格字符串。
from prettytable import PrettyTable
# 导入 TYPE_CHECKING，用于类型检查时导入。
from typing import TYPE_CHECKING
# 导入 Omni PhysX 的 tensors API。
import omni.physics.tensors.impl.api as physx
# 导入 Isaac Lab 的数学工具和字符串工具。
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
# 导入可视化标记类。
from isaaclab.markers import VisualizationMarkers
# 导入交互式场景类。
from isaaclab.scene import InteractiveScene
# 导入传感器基类。
from isaaclab.sensors import SensorBase
# 导入带时间戳的缓冲区。
from isaaclab.utils.buffers import TimestampedBuffer

# 导入 instinctlab 的关节视图获取函数。
from instinctlab.utils.prims import get_articulation_view

# 导入运动参考数据和状态类。
from .motion_reference_data import MotionReferenceData, MotionReferenceState

# 仅在类型检查时导入以下模块。
if TYPE_CHECKING:
    from .motion_reference_cfg import MotionReferenceManagerCfg
    from .motion_buffer import MotionBuffer

# 导入 NumPy、PyTorch 和分布式训练模块。
import numpy as np
import torch
import torch.distributed as dist

# 导入 pytorch_kinematics 用于正向运动学。
import pytorch_kinematics as pk


# 定义 MotionReferenceManager 类，继承自 SensorBase。
class MotionReferenceManager(SensorBase):
    """运动参考管理器，支持多种类型的运动参考：
        1. PlaneDataset：来自预收集的数据集，机器人基座姿态轨迹在世界坐标系中。
        2. InteractiveDataset：来自预收集的数据集，机器人基座姿态轨迹在世界坐标系中。
           数据集还标注了交互对象，可以告诉我们在仿真初始化时如何生成交互对象。
        3. TerrainDataset：来自预收集的数据集，机器人基座姿态轨迹在世界坐标系中。
           数据集还标注了地形信息，可以告诉我们在仿真初始化时如何生成地形。
        4. GenerativeMotion：例如，运动命令可以直接从机器人当前状态生成。
           但这类运动参考只包含机器人基座姿态轨迹，不需要运动参考缓冲区。
    """

    # 类型注解：配置类为 MotionReferenceManagerCfg。
    cfg: MotionReferenceManagerCfg

    def __init__(
        self,
        cfg: MotionReferenceManagerCfg,
    ):
        """初始化运动参考管理器。"""
        # 调用父类 SensorBase 的初始化。
        super().__init__(cfg)
        # 再次声明 cfg 的类型。
        self.cfg: MotionReferenceManagerCfg

    def __str__(self):
        """获取运动参考管理缓冲区的表格信息。"""
        # 构造消息头部，包含运动缓冲区数量。
        msg = f"<MotionReferenceManager> contains {len(self._motion_buffers)} motion buffers to use.\n"

        # 创建表格用于显示术语信息。
        table = PrettyTable()
        table.title = f"Active Motion Buffers"
        table.field_names = ["Name", "Num Trajectories"]
        # 设置表格列的对齐方式。
        table.align["Name"] = "l"
        table.align["Num Trajectories"] = "r"
        # 为每个术语添加信息。
        for name, buffer in self._motion_buffers.items():
            table.add_row([name, buffer.num_trajectories])
        # 将表格转换为字符串。
        msg += table.get_string()
        msg += "\n"

        # 返回消息。
        return msg

    """
    属性
    """

    @property
    def data(self) -> MotionReferenceData:
        # 如果需要，更新传感器。
        self._update_outdated_buffers()
        # 返回数据。
        return self._data

    @property
    def reference_frame(self) -> MotionReferenceData:
        """当前时间步场景中所有关节的参考状态（被动获取）。"""
        # 检查参考帧时间戳是否过时。
        outdated_mask = torch.logical_or(
            (self._reference_frame_timestamp - self._timestamp).abs() > 1e-6, self._reference_frame_timestamp < 1e-6
        )
        # 如果有过时的，则更新。
        if outdated_mask.any():
            env_ids_to_update = self._ALL_INDICES[outdated_mask]
            for name, buffer in self._motion_buffers.items():
                env_ids_assignment = self._motion_buffer_assignment[name]
                # 目前只支持连续的 env_ids 分配。
                assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
                # 计算属于当前缓冲区的环境 ID 掩码。
                env_ids_this_buffer_mask = torch.logical_and(
                    env_ids_to_update >= env_ids_assignment.start, env_ids_to_update < env_ids_assignment.stop
                )
                env_ids_this_buffer = env_ids_to_update[env_ids_this_buffer_mask]
                # 填充运动数据到参考帧。
                buffer.fill_motion_data(
                    env_ids_this_buffer,
                    self._timestamp[env_ids_this_buffer].unsqueeze(-1),  # (N, 1)
                    self.env_origins,
                    self._reference_frame,
                )
                # 如果需要对称增强初始化参考状态。
                if self._symmetric_augmentation_conditions_met:
                    env_ids_to_augment = env_ids_this_buffer[self._env_symmetric_augmentation_mask[env_ids_this_buffer]]
                    self._symmetric_augment_reference_data(self._reference_frame, env_ids_to_augment)
            # 更新参考帧时间戳。
            self._reference_frame_timestamp[outdated_mask] = self._timestamp[outdated_mask]
        # 返回参考帧。
        return self._reference_frame

    @property
    def reference_relative_base_pos(self) -> torch.Tensor:
        """计算参考链接在世界坐标系中的位置（但相对于当前机器人的位置），
        我们计算一个参考基座位置，匹配机器人的 x-y 位置但不匹配 z 位置。
        形状：(num_envs, 3)
        """
        # 检查时间戳是否过时。
        if ((self._reference_relative_base_pos.timestamp - self._timestamp).abs() > 1e-6).any():
            # 获取机器人根变换并克隆 x-y-z。
            self._reference_relative_base_pos.data = self._view.get_root_transforms().clone()[:, :3]
            # 将 z 替换为参考帧基座的 z。
            self._reference_relative_base_pos.data[:, 2] = self.reference_frame.base_pos_w[:, 0, 2]
            # 更新时间戳。
            self._reference_relative_base_pos.timestamp = self._timestamp.clone()
        return self._reference_relative_base_pos.data

    @property
    def reference_relative_delta_quat(self) -> torch.Tensor:
        """为了旋转链接的 pos_w/quat_w 并匹配机器人的旋转，以便与机器人实际的链接 pos_w/quat_w 进行比较，
        我们需要修正其偏航角。任何世界坐标系中的旋转在左乘此四元数后，将得到机器人相对坐标系的位置。
        形状：(num_envs, 4)
        """
        # 检查时间戳是否过时。
        if ((self._reference_relative_delta_quat.timestamp - self._timestamp).abs() > 1e-6).any():
            # 获取机器人根四元数并转换为 wxyz 格式。
            _view_quat = math_utils.convert_quat(self._view.get_root_transforms().clone()[:, 3:7], to="wxyz")
            # 计算偏航四元数。
            self._reference_relative_delta_quat.data = math_utils.yaw_quat(
                math_utils.quat_mul(
                    _view_quat,
                    math_utils.quat_inv(self.reference_frame.base_quat_w[:, 0]),
                )
            )
            # 更新时间戳。
            self._reference_relative_delta_quat.timestamp = self._timestamp.clone()
        return self._reference_relative_delta_quat.data

    @property
    def reference_link_pos_relative_w(self) -> torch.Tensor:
        """假设运动基座框架位于机器人的 x-y 位置、朝向；关节位置/横滚-俯仰保持不变，
        则链接在世界坐标系中的位置。
        形状：(num_envs, num_links, 3)
        """
        # 检查时间戳是否过时。
        if ((self._reference_link_pos_relative_w.timestamp - self._timestamp).abs() > 1e-6).any():
            # 扩展增量四元数到所有链接。
            delta_quat = self.reference_relative_delta_quat.unsqueeze(1).expand(
                -1,
                self.num_link_of_interests,
                -1,
            )
            # 扩展基座位置到所有链接。
            base_pos = self.reference_relative_base_pos.unsqueeze(1).expand(
                -1,
                self.num_link_of_interests,
                -1,
            )
            # 计算相对世界坐标系中的链接位置。
            self._reference_link_pos_relative_w.data = base_pos + math_utils.quat_apply(
                delta_quat,
                self.reference_frame.link_pos_w[:, 0]
                - self.reference_frame.base_pos_w.expand(
                    -1,
                    self.num_link_of_interests,
                    -1,
                ),
            )
            # 更新时间戳。
            self._reference_link_pos_relative_w.timestamp = self._timestamp.clone()
        return self._reference_link_pos_relative_w.data

    @property
    def reference_link_quat_relative_w(self) -> torch.Tensor:
        """假设运动基座框架位于机器人的 x-y 位置、朝向；关节位置/横滚-俯仰保持不变，
        则链接在世界坐标系中的四元数（w, x, y, z 格式）。
        形状：(num_envs, num_links, 4)
        """
        # 检查时间戳是否过时。
        if ((self._reference_link_quat_relative_w.timestamp - self._timestamp).abs() > 1e-6).any():
            # 计算相对世界坐标系中的链接四元数。
            self._reference_link_quat_relative_w.data = math_utils.quat_mul(
                self.reference_relative_delta_quat.unsqueeze(1).expand(
                    -1,
                    self.num_link_of_interests,
                    -1,
                ),
                self.reference_frame.link_quat_w[:, 0],
            )
            # 更新时间戳。
            self._reference_link_quat_relative_w.timestamp = self._timestamp.clone()
        return self._reference_link_quat_relative_w.data

    @property
    def num_frames(self) -> int:
        """返回帧数。"""
        return self.cfg.num_frames

    @property
    def num_link_of_interests(self) -> int:
        """返回感兴趣的链接数量。"""
        return len(self.cfg.link_of_interests)

    @property
    def num_joints(self):
        """返回关节数量。"""
        return self._view.max_dofs

    @property
    def joint_names(self) -> list[str]:
        """获取关节名称列表。"""
        return self.isaac_joint_names

    @property
    def num_bodies(self):
        """返回刚体数量（感兴趣的链接数量）。"""
        return len(self.cfg.link_of_interests)

    @property
    def body_names(self) -> list[str]:
        """获取链接名称列表。"""
        return self.cfg.link_of_interests

    @property
    def num_link_to_ref(self) -> int:
        """返回需要参考的链接数量。如果机器人模型路径或感兴趣链接未提供，则返回 0。"""
        if self.cfg.robot_model_path is None or self.cfg.link_of_interests is None:
            return 0
        else:
            return len(self.cfg.link_of_interests)

    @property
    def frame_interval_s(self) -> torch.Tensor:
        """帧间隔（秒）。形状：(num_envs,)"""
        return self._frame_interval_s

    @property
    def ALL_INDICES(self) -> torch.Tensor:
        """返回所有环境索引。"""
        return self._ALL_INDICES

    @property
    def time_passed_from_update(self) -> torch.Tensor:
        """返回自上次更新以来经过的时间。"""
        return self._timestamp - self._timestamp_last_update

    @property
    def is_at_keyframe(self) -> torch.Tensor:
        """判断经过的时间是否是 frame_interval_s 的整数倍。
        注意：目前假设给定环境中的所有帧间隔相同。
        """
        return torch.where(
            self._timestamp > self._timestamp_last_update,
            (torch.clip(self.time_passed_from_update, min=0.0) % self.frame_interval_s) < 1e-6,
            torch.zeros_like(self._timestamp, dtype=torch.bool),
        )

    @property
    def aiming_frame_idx(self) -> torch.Tensor:
        """当前机器人应该瞄准的帧索引（相对于参考数据）。
        例如，第 0 帧是 `_timestamp_last_update + frame_interval_s` 时刻的帧。
        当 `_timestamp - _timestamp_last_update` == frame_interval_s 时，瞄准帧仍应为 0。
        """
        # 检查时间戳是否过时。
        if ((self._aiming_frame_idx.timestamp - self._timestamp).abs() > 1e-6).any():
            time_passed_from_update = self.time_passed_from_update  # (N,)
            time_to_target_frame = self.data.time_to_target_frame  # (N, num_frames)
            # 计算瞄准帧索引。
            aiming_frame_idx = torch.sum(
                torch.logical_and(
                    time_passed_from_update.unsqueeze(-1) > time_to_target_frame,
                    time_to_target_frame > 0.0,
                ),
                dim=-1,
            )  # (N,)
            # 如果索引超出帧数，设为 -1。
            aiming_frame_idx[aiming_frame_idx >= self.num_frames] = -1
            # 更新数据和时间戳。
            self._aiming_frame_idx.data = aiming_frame_idx
            self._aiming_frame_idx.timestamp = self._timestamp.clone()
        return self._aiming_frame_idx.data

    @property
    def time_to_aiming_frame(self) -> torch.Tensor:
        """到达瞄准帧剩余的时间。"""
        return (self.aiming_frame_idx + 1) * self.frame_interval_s - (self.time_passed_from_update)

    @property
    def env_origins(self) -> torch.Tensor:
        """返回环境原点。"""
        return (
            self._env_origins if hasattr(self, "_env_origins") else torch.zeros(self._view.count, 3, device=self.device)
        )

    @property
    def motion_buffers(self) -> dict[str, MotionBuffer]:
        """获取活动的运动缓冲区。"""
        return self._motion_buffers

    @property
    def _motion_buffer_num_trajectories(self) -> dict[str, int]:
        """获取每个运动缓冲区的轨迹数量（私有、在线、类属性）。"""
        return {name: buffer.num_trajectories for name, buffer in self._motion_buffers.items()}

    @property
    def complete_motion_lengths(self) -> torch.Tensor:
        """对于 ALL_INDICES，运动参考的长度（秒）。
        ## 注意：该属性返回所有环境的完整轨迹长度，即使环境从运动文件的中间开始。
        长度基于运动缓冲区的 complete_motion_lengths 计算。
        """
        # 初始化长度为 1。
        lengths = torch.ones(self._view.count, device=self.device)
        for buffer_name, motion_buffer in self.motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[buffer_name]
            env_ids_this_buffer = self.ALL_INDICES[env_ids_assignment]
            # 赋值完整运动长度。
            lengths[env_ids_this_buffer] = motion_buffer.complete_motion_lengths

        return lengths

    @property
    def assigned_motion_lengths(self) -> torch.Tensor:
        """对于 ALL_INDICES，运动参考的长度（秒），但仅针对分配的环境 ID。
        ## 注意：该属性返回所有环境的轨迹长度。如果环境从运动文件的中间开始，
        长度计算为从起始点到运动文件结尾的长度。
        """
        # 初始化长度为 1。
        lengths = torch.ones(self._view.count, device=self.device)
        for buffer_name, motion_buffer in self.motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[buffer_name]
            env_ids_this_buffer = self.ALL_INDICES[env_ids_assignment]
            # 赋值分配的运动长度。
            lengths[env_ids_this_buffer] = motion_buffer.assigned_motion_lengths

        return lengths

    """
    操作。
    """

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """将运动参考管理器作为传感器重置，同时重置运动参考组件。"""
        # 断言已初始化。
        assert (
            self.is_initialized
        ), "Motion reference manager is not initialized successfully. Please check the error message above."
        # 调用父类重置。
        super().reset(env_ids)
        # 如果 env_ids 为 None，则使用所有索引。
        if env_ids is None:
            env_ids = self.ALL_INDICES

        # 重置运动缓冲区选择和重采样选择。
        self._reset_motion_buffers(env_ids)
        self._reset_data(env_ids)
        self._resample_buffer_collate_params(env_ids)
        self._resample_update_period(env_ids)

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """返回给定关节名称的 joint_ids。为了与 isaaclab SceneEntity 接口保持一致。

        有关名称匹配的更多信息，请参阅 :func:`isaaclab.utils.string.resolve_matching_names` 函数。

        Args:
            name_keys: 用于匹配关节名称的正则表达式或正则表达式列表。
            joint_subset: 要搜索的关节子集。默认为 None，表示搜索关节中的所有关节。
            preserve_order: 是否在输出中保留名称键的顺序。默认为 False。

        Returns:
            包含关节索引和名称的元组列表。
        """
        # 如果未指定关节子集，则使用所有关节名称。
        if joint_subset is None:
            joint_subset = self.isaac_joint_names
        # 查找关节。
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """返回给定链接名称的 link_ids。为了与 isaaclab SceneEntity 接口保持一致。
        注意：这些 ID 是在 self.cfg.link_of_interests 中的选择，而不是关节中的所有链接 ID。
        """
        # 查找刚体。
        return string_utils.resolve_matching_names(name_keys, self.cfg.link_of_interests, preserve_order)

    def get_init_reference_state(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> MotionReferenceState:
        """获取给定 env_ids 的初始参考状态。"""
        # 将 env_ids 转换为张量。
        env_ids = torch.as_tensor(env_ids, device=self.device)
        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            # 目前只支持连续的 env_ids 分配。
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            # 计算属于当前缓冲区的环境 ID 掩码。
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            env_ids_this_buffer = (
                self.ALL_INDICES[env_ids_assignment] if env_ids is None else env_ids[env_ids_this_buffer_mask]
            )
            # 填充初始参考状态。
            buffer.fill_init_reference_state(
                env_ids_this_buffer,
                self.env_origins,
                self._init_reference_state,
            )
            # 如果需要对称增强初始参考状态。
            if self._symmetric_augmentation_conditions_met:
                env_ids_to_augment = env_ids_this_buffer[self._env_symmetric_augmentation_mask[env_ids_this_buffer]]
                # 对关节位置和速度进行对称增强。
                self._init_reference_state.joint_pos[env_ids_to_augment] = self._symmetric_augment_joint_buffer(
                    self._init_reference_state.joint_pos[env_ids_to_augment],
                )
                self._init_reference_state.joint_vel[env_ids_to_augment] = self._symmetric_augment_joint_buffer(
                    self._init_reference_state.joint_vel[env_ids_to_augment],
                )
                # 对基座位置、四元数、线速度和角速度进行对称增强。
                self._init_reference_state.base_pos_w[env_ids_to_augment, 1] *= -1
                self._init_reference_state.base_quat_w[env_ids_to_augment] = self._symmetric_augment_quat_buffer(
                    self._init_reference_state.base_quat_w[env_ids_to_augment],
                )
                self._init_reference_state.base_lin_vel_w[env_ids_to_augment, 1] *= -1
                self._init_reference_state.base_ang_vel_w[env_ids_to_augment] = self._symmetric_augment_ang_vel_buffer(
                    self._init_reference_state.base_ang_vel_w[env_ids_to_augment],
                )
        # 返回指定 env_ids 的初始参考状态。
        return self._init_reference_state[env_ids]

    def target_link_pose_forward_kinematics(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """考虑到运动参考中感兴趣的链接已知，正向运动学输出可以固定为一系列链接姿态。
        该函数用于从关节位置（按 IsaacSim 顺序）计算目标链接姿态。
        注意：此方法作为可调用对象供运动缓冲区计算目标链接姿态。后续应优化以加快计算速度。
        ## 参数：
            - joint_pos: (N, num_dofs) 张量，按 IsaacSim 顺序的关节位置。
        ## 返回：
            目标链接在 base_link 坐标系中的姿态。(N, num_link_to_ref, 7) 张量，顺序为 `self.cfg.link_of_interests`。
        """
        # 记录输入设备。
        input_device = joint_pos.device
        # 将关节位置移到当前设备。
        joint_pos = joint_pos.to(self.device)
        # 使用正向运动学计算所有链接姿态。
        all_link_poses = self._robot_kinematics_chain.forward_kinematics(joint_pos[:, self._joint_order_isaac_to_pk])
        # 初始化输出张量。
        link_pos_quat_b = torch.zeros(joint_pos.shape[0], self.num_link_to_ref, 7, device=self.device)
        # 遍历感兴趣的链接。
        for link_idx, link_name in enumerate(self.cfg.link_of_interests):
            pose_mat = all_link_poses[link_name].get_matrix().reshape(-1, 4, 4)
            link_pos_quat_b[:, link_idx, :3] = pose_mat[:, :3, 3]
            link_pos_quat_b[:, link_idx, 3:] = math_utils.quat_from_matrix(pose_mat[:, :3, :3])
        # 返回结果并移回输入设备。
        return link_pos_quat_b.to(input_device)

    def get_current_motion_identifiers(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> list[str]:
        """获取给定 env_ids 的当前运动标识符。"""
        # 如果 env_ids 为 None，则使用所有索引。
        if env_ids is None:
            env_ids = self.ALL_INDICES
        motion_identifiers = []
        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            # 目前只支持连续的 env_ids 分配。
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            # 计算属于当前缓冲区的环境 ID 掩码。
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            # 获取当前运动标识符。
            motion_identifiers.extend(buffer.get_current_motion_identifiers(env_ids[env_ids_this_buffer_mask]))  # type: ignore
        return motion_identifiers

    def get_current_motion_weights(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> torch.Tensor:
        """获取给定 env_ids 的当前运动权重。"""
        # 如果 env_ids 为 None，则使用所有索引。
        if env_ids is None:
            env_ids = self.ALL_INDICES
        # 初始化权重张量。
        motion_weights = torch.empty(len(env_ids), device=self.device, dtype=torch.float32)
        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            # 目前只支持连续的 env_ids 分配。
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            # 计算属于当前缓冲区的环境 ID 掩码。
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            # 获取当前运动权重。
            motion_weights[env_ids_this_buffer_mask] = buffer.get_current_motion_weights(
                env_ids[env_ids_this_buffer_mask]
            )  # type: ignore
        return motion_weights

    def update_motion_weights(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        weight_ratio: float | torch.Tensor = 1.0,
    ):
        """更新给定 env_ids 的运动权重，以便在重置时更新运动采样权重。
        ## 注意：
            - 不建议使用多个运动缓冲区。
            - 不建议使用多进程。
        """
        # 如果 weight_ratio 是张量，检查长度是否匹配。
        if isinstance(weight_ratio, torch.Tensor):
            assert len(weight_ratio) == len(
                env_ids
            ), "weight_ratio must be a scalar or a tensor with the same length as env_ids."
        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            # 目前只支持连续的 env_ids 分配。
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            # 计算属于当前缓冲区的环境 ID 掩码。
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            env_ids_this_buffer = env_ids[env_ids_this_buffer_mask]  # type: ignore
            # 更新运动权重。
            buffer.update_motion_weights(
                env_ids_this_buffer,
                weight_ratio=(
                    weight_ratio[env_ids_this_buffer_mask] if isinstance(weight_ratio, torch.Tensor) else weight_ratio
                ),
            )

    """
    将运动参考与场景中其他组件连接的操作。必须通过事件（'reset' 模式或 'startup' 模式）触发。
    """

    def match_scene(self, scene: InteractiveScene):
        """将运动参考与场景匹配，包括对象和地形。考虑到某些运动参考需要利用未展平的地形，
        必须将此信息提供给运动参考缓冲区和运动参考管理器。
        """
        # 存储环境原点作为场景管理的精确 Python 对象（注意：按引用传递，不是拷贝）。
        self._env_origins = scene.env_origins
        for buffer in self.motion_buffers.values():
            # 将场景与运动缓冲区匹配。
            buffer.match_scene(scene)

    """
    实现。
    """

    def _initialize_impl(self):
        """初始化实现。"""
        # 调用父类初始化。
        super()._initialize_impl()
        # 初始化数据。
        self._initialize_data()
        # 如果提供了机器人模型路径，则初始化运动学链和关节顺序映射。
        if self.cfg.robot_model_path is not None:
            self._initialize_robot_kinematics()
        # 初始化运动缓冲区。
        self._initialize_motion_buffers()
        # 重采样缓冲区整理参数。
        self._resample_buffer_collate_params()
        # 重采样更新周期。
        self._resample_update_period()
        # 打印运动参考管理缓冲区的表格信息。
        print(self)

    def _update_buffers_impl(self, env_ids: Sequence[int] | torch.Tensor):
        """为给定 env_ids 更新运动参考缓冲区。"""
        # 将 env_ids 转换为张量。
        env_ids = torch.as_tensor(env_ids, device=self.device)
        # 计算到达特定帧的剩余时间。
        time_to_target_frame = torch.arange(self.cfg.num_frames, device=self.device, dtype=torch.float32)
        if self.cfg.data_start_from == "one_frame_interval":
            time_to_target_frame += 1
        time_to_target_frame = time_to_target_frame.unsqueeze(0).repeat(len(env_ids), 1)
        time_to_target_frame *= self.frame_interval_s[env_ids].unsqueeze(-1)

        for name, buffer in self._motion_buffers.items():
            env_ids_assignment = self._motion_buffer_assignment[name]
            # 目前只支持连续的 env_ids 分配。
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            # 计算属于当前缓冲区的环境 ID 掩码。
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            env_ids_this_buffer = env_ids[env_ids_this_buffer_mask]
            # 填充运动数据。
            buffer.fill_motion_data(
                env_ids_this_buffer,
                self._timestamp[env_ids_this_buffer].unsqueeze(-1) + time_to_target_frame[env_ids_this_buffer_mask],
                self.env_origins,
                self._data,
            )
            # 如果需要对称增强初始参考状态。
            if self._symmetric_augmentation_conditions_met:
                env_ids_to_augment = env_ids_this_buffer[self._env_symmetric_augmentation_mask[env_ids_this_buffer]]
                self._symmetric_augment_reference_data(self._data, env_ids_to_augment)
        # 设置到达目标帧的时间。
        self._data.time_to_target_frame[env_ids] = time_to_target_frame

        # 将无效时间戳数据设置为无效值。
        self._data.time_to_target_frame[self._data.validity == 0] = -1.0

    """
    管理器内部操作。
    """

    def _prepare_data_class_kwargs(self) -> dict:
        """准备数据类 `make_empty` 方法的 kwargs。"""
        make_empty_kwargs: dict = {
            "device": self.device,
            "num_joints": self._view.max_dofs,
            "num_links": self.num_link_to_ref,
        }
        # 检查 make_empty 的签名。
        sig = inspect.signature(self.cfg.data_class_type.make_empty)
        if "num_objects" in sig.parameters:
            scene_object_names = getattr(self.cfg, "scene_object_names", [])
            make_empty_kwargs["num_objects"] = len(scene_object_names)
            make_empty_kwargs["scene_object_names"] = scene_object_names
        return make_empty_kwargs

    def _prepare_state_class_kwargs(self) -> dict:
        """准备状态类 `make_empty` 方法的 kwargs。"""
        make_empty_kwargs: dict = {
            "device": self.device,
            "num_joints": self._view.max_dofs,
        }
        # 检查 make_empty 的签名。
        sig = inspect.signature(self.cfg.state_class_type.make_empty)
        if "num_objects" in sig.parameters:
            scene_object_names = getattr(self.cfg, "scene_object_names", [])
            make_empty_kwargs["num_objects"] = len(scene_object_names)
            make_empty_kwargs["scene_object_names"] = scene_object_names
        return make_empty_kwargs

    def _initialize_data(self):
        """初始化 _data 作为 '传感器' 输出。"""
        # 创建物理仿真视图。
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # 获取关节视图。
        self._view: physx.ArticulationView = get_articulation_view(
            self.cfg.prim_path,
            self._physics_sim_view,
        )
        # 初始化所有索引。
        self._ALL_INDICES = torch.arange(self._view.count, device=self.device)
        # 获取 Isaac 关节名称。
        self.isaac_joint_names = self._view.shared_metatype.dof_names

        # 准备数据类参数。
        make_empty_data_kwargs = self._prepare_data_class_kwargs()

        # 创建空数据。
        self._data = self.cfg.data_class_type.make_empty(
            self._view.count,
            self.cfg.num_frames,
            **make_empty_data_kwargs,
        )

        # 创建参考帧数据。
        self._reference_frame = self.cfg.data_class_type.make_empty(
            self._view.count,
            1,  # num_frames
            **make_empty_data_kwargs,
        )
        # 为参考帧数据添加额外的时间戳，用于延迟更新参考帧。
        self._reference_frame_timestamp = torch.zeros(self._view.count, device=self.device)

        # 准备状态类参数。
        make_empty_state_kwargs = self._prepare_state_class_kwargs()

        # 创建初始参考状态。
        self._init_reference_state = self.cfg.state_class_type.make_empty(
            self._view.count,
            **make_empty_state_kwargs,
        )

        # 创建带时间戳的缓冲区。
        self._aiming_frame_idx = TimestampedBuffer()

        self._reference_relative_base_pos = TimestampedBuffer()
        self._reference_relative_delta_quat = TimestampedBuffer()
        self._reference_link_pos_relative_w = TimestampedBuffer()
        self._reference_link_quat_relative_w = TimestampedBuffer()

    def _initialize_motion_buffers(self):
        """初始化所有运动参考缓冲区，计算当前进程的特定数据范围（以防多进程）。"""
        self._motion_buffers: dict[str, MotionBuffer] = dict()
        # 断言至少指定一个运动缓冲区。
        assert len(self.cfg.motion_buffers) > 0, "At least one motion buffer should be specified."
        for motion_buffer_name, motion_buffer_cfg in self.cfg.motion_buffers.items():
            motion_buffer_cls: type[MotionBuffer] = motion_buffer_cfg.class_type
            # 实例化运动缓冲区。
            self._motion_buffers[motion_buffer_name] = motion_buffer_cls(
                motion_buffer_cfg,
                articulation_view=self._view,
                link_of_interests=self.cfg.link_of_interests,
                forward_kinematics_func=self.target_link_pose_forward_kinematics,
                device=self.device,
            )
        # 步骤：根据 rank_id/world_size 计算当前进程的运动轨迹分配（如果多进程）。
        # 参考 PyTorch 分布式启动的设计：https://pytorch.org/docs/stable/elastic/run.html
        if dist.is_initialized():
            # 多进程，计算轨迹分配。
            local_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            local_rank = 0
            world_size = None
        # 为当前进程分配运动轨迹。
        self._assign_motion_trajectories_for_this_process(local_rank, world_size)
        # 为环境分配运动缓冲区。
        self._assign_motion_buffer_for_envs()
        # TODO: 步骤：如果需要，存储交互信息（与地形或刚体对象）。
        # 等待事件与场景中的实体匹配。

    def _initialize_robot_kinematics(self):
        """初始化机器人运动学链。"""
        # 读取 URDF 文件并构建运动学链。
        with open(self.cfg.robot_model_path) as f:
            self._robot_kinematics_chain = pk.build_chain_from_urdf(f.read()).to(dtype=torch.float, device=self.device)
        # joint_pos_pk = joint_pos_isaac[_joint_order_isaac_to_pk]
        # 初始化关节顺序映射。
        self._joint_order_isaac_to_pk = torch.ones(self._view.max_dofs, device=self.device, dtype=torch.long) * -1
        for joint_i, joint_name in enumerate(self._robot_kinematics_chain.get_joint_parameter_names()):
            if not joint_name in self.isaac_joint_names:
                raise RuntimeError(
                    f"Joint name {joint_name} in the robot kinematics chain is not found in the physics simulation"
                    " view."
                )

            joint_idx_isaac = self.isaac_joint_names.index(joint_name)
            self._joint_order_isaac_to_pk[joint_i] = joint_idx_isaac
        # TODO: 考虑到跟踪链接已从配置中已知，正向运动学可以通过 ONNX 或 JIT 加速。

    def _reset_motion_buffers(self, env_ids: Sequence[int] | torch.Tensor):
        """如果给定 env_ids 选择了运动缓冲区，则重置每个运动缓冲区。"""
        for name in self._motion_buffers.keys():
            env_ids_assignment = self._motion_buffer_assignment[name]
            # 目前只支持连续的 env_ids 分配。
            assert env_ids_assignment.step is None, "Only support continuous env_ids assignment for now."
            # 计算属于当前缓冲区的环境 ID 掩码。
            env_ids_this_buffer_mask = torch.logical_and(
                env_ids >= env_ids_assignment.start, env_ids < env_ids_assignment.stop
            )
            # 重置运动缓冲区。
            self._motion_buffers[name].reset(
                env_ids=env_ids[env_ids_this_buffer_mask],  # type: ignore
                symmetric_augmentation_mask_buffer=self._env_symmetric_augmentation_mask,
            )

    def _reset_data(self, env_ids: Sequence[int] | torch.Tensor):
        """为给定 env_ids 重置参考数据/参考帧。"""
        self._data.reset(env_ids)
        self._reference_frame.reset(env_ids)

    def _resample_buffer_collate_params(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """为给定 env_ids 重采样运动整理行为的参数。例如 frame_interval_s、symmetric_augmentation_mask。"""
        # 重采样帧间隔。
        if env_ids is None and isinstance(self.cfg.frame_interval_s, Sequence):
            self._frame_interval_s = (
                torch.rand(self._view.count, device=self.device)
                * (self.cfg.frame_interval_s[1] - self.cfg.frame_interval_s[0])
                + self.cfg.frame_interval_s[0]
            )
        elif isinstance(self.cfg.frame_interval_s, Sequence):
            self._frame_interval_s[env_ids] = (
                torch.rand(len(env_ids), device=self.device)  # type: ignore
                * (self.cfg.frame_interval_s[1] - self.cfg.frame_interval_s[0])
                + self.cfg.frame_interval_s[0]
            )
        elif (not hasattr(self, "_frame_interval_s")) and (not isinstance(self.cfg.frame_interval_s, Sequence)):
            self._frame_interval_s = torch.ones(self._view.count, device=self.device) * self.cfg.frame_interval_s
        elif hasattr(self, "_frame_interval_s") and (not isinstance(self.cfg.frame_interval_s, Sequence)):
            pass
        else:
            raise ValueError("Invalid frame_interval_s configuration.")

        # 重采样对称增强掩码。
        if env_ids is None or not hasattr(self, "_env_symmetric_augmentation_mask"):
            # 指定是否应对给定 env_ids 应用对称增强。
            # self._data 中的值应已经对称增强。
            self._env_symmetric_augmentation_mask = torch.zeros(self._view.count, device=self.device, dtype=torch.bool)

    def _resample_update_period(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """为给定 env_ids 重采样更新周期。"""
        if env_ids is None and isinstance(self.cfg.update_period, (list, tuple)):
            # 初始化更新周期，使传感器管理器在给定周期更新传感器。
            self.cfg = copy(self.cfg)
            self.cfg.update_period_range = copy(self.cfg.update_period)
            self.cfg.update_period = (
                torch.ones(
                    self._view.count,
                    device=self.device,
                )
                * self.cfg.update_period[0]
            )
            env_ids = self._ALL_INDICES  # type: ignore

        if isinstance(self.cfg.update_period, torch.Tensor) and self.cfg.update_period_sample_strategy is not None:
            if self.cfg.update_period_sample_strategy == "uniform":
                self.cfg.update_period[env_ids] = (
                    torch.rand(
                        len(env_ids),  # type: ignore
                        device=self.device,
                    )
                    * (self.cfg.update_period_range[1] - self.cfg.update_period_range[0])
                    + self.cfg.update_period_range[0]
                )  # type: ignore
            elif self.cfg.update_period_sample_strategy == "uniform_frame_limits":
                # 目前假设采样帧索引仍然线性增加。
                max_motion_ref_length_s = self.frame_interval_s * self.cfg.num_frames  # (len(env_ids),)
                self.cfg.update_period[env_ids] = (
                    torch.rand(
                        len(env_ids),  # type: ignore
                        device=self.device,
                    )
                    * (max_motion_ref_length_s[env_ids] - self.frame_interval_s[env_ids])
                    + self.frame_interval_s[env_ids]
                )
            else:
                raise ValueError(f"Unknown update_period_sample_strategy: {self.cfg.update_period_sample_strategy}")

    def _assign_motion_trajectories_for_this_process(self, local_rank: int = 0, world_size: int | None = None):
        """根据 rank_id/world_size 为当前进程分配运动轨迹。
        `self._motion_buffer_num_trajectories` 将被更新。对于当前进程，只有一部分轨迹将被启用。
        未使用的运动缓冲区（Python 对象）将被删除。
        """
        if world_size is None or self.cfg.mp_split_method == "None":
            # 单进程，为所有运动缓冲区启用所有轨迹。
            # 无分割方法，为所有运动缓冲区启用所有轨迹。
            for buffer in self._motion_buffers.values():
                buffer.enable_trajectories()
        elif self.cfg.mp_split_method == "Even":
            # 为每个进程均匀分割轨迹（轮询）。
            # |----motion1-----|----motion2-----|----motion3-----|----motion4-----|----motion5-----|
            # [123412341234123412341234123412341234123412341234123412341234123412341234123412341234]
            idxs = torch.arange(
                sum(self._motion_buffer_num_trajectories.values()),
                device=self.device,
                dtype=torch.long,
            )
            if len(idxs) % world_size != 0:
                idxs = idxs[: -(len(idxs) % world_size)]
            idxs = idxs.reshape(-1, world_size)
            idxs = idxs[:, local_rank]
            buffer_traj_start = 0
            for buffer_name in list(self._motion_buffers.keys()):
                buffer = self._motion_buffers[buffer_name]
                buffer_traj_end = buffer.num_trajectories + buffer_traj_start
                idxs_this_buffer = idxs[(idxs >= buffer_traj_start) & (idxs < buffer_traj_end)] - buffer_traj_start
                if len(idxs_this_buffer) > 0:
                    buffer.enable_trajectories(idxs_this_buffer)
                else:
                    buffer = self._motion_buffers.pop(buffer_name)
                    del buffer
                buffer_traj_start = buffer_traj_end
        elif self.cfg.mp_split_method == "Segment":
            # 根据总轨迹数分割轨迹。
            # 假设所有轨迹放在单个序列中，每个进程将分配一个段。
            # |----motion1-----|----motion2-----|----motion3-----|----motion4-----|----motion5-----|
            # |-------------------process1-------------------|-------------------process2----------|
            total_traj_global = sum(self._motion_buffer_num_trajectories.values())
            num_traj_per_process = total_traj_global // world_size
            process_traj_start = num_traj_per_process * local_rank
            process_traj_end = (
                num_traj_per_process * (local_rank + 1) if local_rank < world_size - 1 else total_traj_global
            )
            buffer_traj_start = 0
            for buffer_name in list(self._motion_buffers.keys()):
                buffer = self._motion_buffers[buffer_name]
                buffer_traj_end = buffer.num_trajectories + buffer_traj_start
                buffer_slice_start = np.clip(process_traj_start, buffer_traj_start, buffer_traj_end) - buffer_traj_start
                buffer_slice_end = np.clip(process_traj_end, buffer_traj_start, buffer_traj_end) - buffer_traj_start
                if buffer_slice_start < buffer_slice_end:
                    buffer.enable_trajectories(slice(buffer_slice_start, buffer_slice_end))
                else:
                    buffer = self._motion_buffers.pop(buffer_name)
                    del buffer
                buffer_traj_start = buffer_traj_end
        else:
            raise ValueError(f"Unknown mp_split_method: {self.cfg.mp_split_method}")

    def _assign_motion_buffer_for_envs(self):
        """为每个环境分配使用哪个运动缓冲区。但在启动阶段定义这一点会导致更稳定的缓冲区数据行为。
        这就像基于课程的运动训练设置中固定的地形类型。
        """
        # 计算每个运动缓冲区的轨迹比例以及应分配给每个缓冲区的环境数量。
        total_traj = sum(self._motion_buffer_num_trajectories.values())
        traj_ratio_d = {name: num_traj / total_traj for name, num_traj in self._motion_buffer_num_trajectories.items()}
        num_env_assignment_d = {name: int(ratio * self._view.count) for name, ratio in traj_ratio_d.items()}

        # 检查分配是否匹配所有环境，否则从最大/最小处稍作增减。
        if sum(num_env_assignment_d.values()) > self._view.count:
            overflow_num = sum(num_env_assignment_d.values()) - self._view.count
            max_num_env_name = max(num_env_assignment_d.keys(), key=num_env_assignment_d.get)  # type: ignore
            num_env_assignment_d[max_num_env_name] -= overflow_num
        elif sum(num_env_assignment_d.values()) < self._view.count:
            underflow_num = self._view.count - sum(num_env_assignment_d.values())
            min_num_env_name = min(num_env_assignment_d.keys(), key=num_env_assignment_d.get)  # type: ignore
            num_env_assignment_d[min_num_env_name] += underflow_num

        # 为每个环境分配运动缓冲区。
        self._motion_buffer_assignment: dict[str, slice] = dict()
        start_idx = 0
        for name, num_env in num_env_assignment_d.items():
            self._motion_buffer_assignment[name] = slice(start_idx, start_idx + num_env)
            start_idx += num_env

        # 通知运动缓冲区，以便它们准备缓冲区。
        for name, buffer in self._motion_buffers.items():
            buffer.set_env_ids_assignments(self._motion_buffer_assignment[name])

    """
    数据增强实现。
    """

    @property
    def _symmetric_augmentation_conditions_met(self) -> bool | torch.Tensor:
        """检查是否满足对称增强条件。"""
        return (
            (self.cfg.symmetric_augmentation_joint_mapping is not None)
            and (self.cfg.symmetric_augmentation_joint_reverse_buf is not None)
            and (self.num_link_to_ref == 0 or self.cfg.symmetric_augmentation_link_mapping is not None)
            and self._env_symmetric_augmentation_mask.any()
        )

    def _symmetric_augment_joint_buffer(self, joint_pos_buf: torch.Tensor):
        """对称增强关节位置/速度缓冲区，原地修改值。
        假设最后一个维度是关节维度。
        """
        num_dims = len(joint_pos_buf.shape)
        joint_reverse_buf = torch.tensor(self.cfg.symmetric_augmentation_joint_reverse_buf, device=self.device)
        if num_dims > 1:
            for _ in range(num_dims - 1):
                joint_reverse_buf = joint_reverse_buf.unsqueeze(0)
        joint_pos_buf[..., :] = joint_pos_buf[..., self.cfg.symmetric_augmentation_joint_mapping] * joint_reverse_buf
        return joint_pos_buf

    def _symmetric_augment_ang_vel_buffer(self, ang_vel_buf: torch.Tensor):
        """对称增强角速度缓冲区，原地修改值。
        假设最后一个维度是角速度维度。
        通过关于 x-z 平面镜像，角速度的 y 分量和 z 分量应取反。
        """
        ang_vel_buf[..., 0] *= -1
        ang_vel_buf[..., 2] *= -1
        return ang_vel_buf

    def _symmetric_augment_link_pos_buffer(self, link_pos_buf: torch.Tensor):
        """关于 x-z 平面对称增强链接位置/旋转缓冲区，原地修改值。
        假设最后一个维度是位置维度，倒数第二个维度是链接维度。
        """
        link_pos_buf[:] = link_pos_buf[..., self.cfg.symmetric_augmentation_link_mapping, :]
        link_pos_buf[..., 1] *= -1
        return link_pos_buf

    def _symmetric_augment_link_quat_buffer(self, link_quat_buf: torch.Tensor):
        """关于 x-z 平面对称增强链接四元数缓冲区，原地修改值。
        假设最后一个维度是四元数 (w, x, y, z) 维度，倒数第二个维度是链接维度。
        """
        # 关于 x-z 平面镜像四元数。
        # 来自 https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation
        link_quat_buf[:] = link_quat_buf[..., self.cfg.symmetric_augmentation_link_mapping, :]
        link_quat_buf[..., 1] *= -1
        link_quat_buf[..., 3] *= -1
        return link_quat_buf

    def _symmetric_augment_quat_buffer(self, quat_buf: torch.Tensor):
        """关于 x-z 平面对称增强四元数缓冲区，原地修改值。
        假设最后一个维度是四元数 (w, x, y, z) 维度。
        """
        # 来自 https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation
        quat_buf[..., 1] *= -1
        quat_buf[..., 3] *= -1
        return quat_buf

    def _symmetric_augment_reference_data(self, data_buf: MotionReferenceData, env_ids: torch.Tensor | slice):
        """关于 x-z 平面对称增强参考数据缓冲区的所有部分，原地修改值。"""
        data_buf.joint_pos[env_ids] = self._symmetric_augment_joint_buffer(
            data_buf.joint_pos[env_ids],
        )
        data_buf.joint_vel[env_ids] = self._symmetric_augment_joint_buffer(
            data_buf.joint_vel[env_ids],
        )
        data_buf.base_pos_w[env_ids, :, 1] *= -1
        data_buf.base_quat_w[env_ids] = self._symmetric_augment_quat_buffer(
            data_buf.base_quat_w[env_ids],
        )
        data_buf.base_lin_vel_w[env_ids, :, 1] *= -1
        data_buf.base_ang_vel_w[env_ids] = self._symmetric_augment_ang_vel_buffer(
            data_buf.base_ang_vel_w[env_ids],
        )
        data_buf.link_pos_w[env_ids] = self._symmetric_augment_link_pos_buffer(
            data_buf.link_pos_w[env_ids],
        )
        data_buf.link_quat_w[env_ids] = self._symmetric_augment_link_quat_buffer(
            data_buf.link_quat_w[env_ids],
        )
        data_buf.link_pos_b[env_ids] = self._symmetric_augment_link_pos_buffer(
            data_buf.link_pos_b[env_ids],
        )
        data_buf.link_quat_b[env_ids] = self._symmetric_augment_link_quat_buffer(
            data_buf.link_quat_b[env_ids],
        )
        data_buf.link_lin_vel_b[env_ids, :, 1] *= -1
        data_buf.link_ang_vel_b[env_ids] = self._symmetric_augment_ang_vel_buffer(
            data_buf.link_ang_vel_b[env_ids],
        )
        data_buf.link_lin_vel_w[env_ids, :, 1] *= -1
        data_buf.link_ang_vel_w[env_ids] = self._symmetric_augment_ang_vel_buffer(
            data_buf.link_ang_vel_w[env_ids],
        )

    """
    可视化。
    """

    def _find_reference_view(self):
        """查找 ArticulationView 以作为运动参考可视化。"""
        if (
            hasattr(self, "_backend")
            and hasattr(self, "_physics_sim_view")
            and (not self.cfg.reference_prim_path is None)
        ):
            self._reference_view: physx.ArticulationView = get_articulation_view(
                self.cfg.reference_prim_path, self._physics_sim_view
            )
            if self._reference_view._backend is None:
                delattr(self, "_reference_view")
                raise RuntimeError("Failed to find the reference view.")

            # The simulated robot and the visual reference are separate
            # articulation views.  After changing the floating root to
            # pelvis, Isaac may assign a different runtime DOF order to the
            # two views even though they contain the same named joints.  The
            # motion buffer is stored in the main robot's Isaac order, so
            # build an explicit name-based permutation for the reference view.
            reference_joint_names = list(self._reference_view.shared_metatype.dof_names)
            motion_joint_names = list(self.isaac_joint_names)
            if set(reference_joint_names) != set(motion_joint_names):
                missing = sorted(set(motion_joint_names) - set(reference_joint_names))
                extra = sorted(set(reference_joint_names) - set(motion_joint_names))
                raise RuntimeError(
                    "Motion reference DOF names do not match the robot DOF names: "
                    f"missing={missing}, extra={extra}"
                )
            self._reference_dof_from_motion = torch.tensor(
                [motion_joint_names.index(name) for name in reference_joint_names],
                device=self.device,
                dtype=torch.long,
            )
            print(
                "[MotionReference] reference DOF order mapped by name: "
                f"robot={motion_joint_names[:5]}... reference={reference_joint_names[:5]}..."
            )

    def _set_reference_view_state(self):
        """将关节视图设置为参考状态，用于运动可视化。"""
        if self.cfg.visualizing_robot_from == "aiming_frame":
            aiming_frame_idx = self.aiming_frame_idx
            robot_pos_w = self.data.base_pos_w[self.ALL_INDICES, aiming_frame_idx].clone()
            robot_quat_w_ = self.data.base_quat_w[self.ALL_INDICES, aiming_frame_idx]
            robot_quat_w = math_utils.convert_quat(robot_quat_w_, to="xyzw")
            robot_joint_pos = self.data.joint_pos[self.ALL_INDICES, aiming_frame_idx]
        elif self.cfg.visualizing_robot_from == "reference_frame":
            robot_pos_w = self.reference_frame.base_pos_w[self.ALL_INDICES, 0].clone()
            robot_quat_w_ = self.reference_frame.base_quat_w[self.ALL_INDICES, 0]
            robot_quat_w = math_utils.convert_quat(robot_quat_w_, to="xyzw")
            robot_joint_pos = self.data.joint_pos[self.ALL_INDICES, 0]
        else:
            raise ValueError(f"Unsupported cfg.visualizing_robot_from: {self.cfg.visualizing_robot_from}")

        if hasattr(self, "_reference_dof_from_motion"):
            robot_joint_pos = robot_joint_pos[:, self._reference_dof_from_motion]

        # 添加位置偏移，以防参考与真实机器人重叠。
        robot_pos_w[:, 0] += self.cfg.visualizing_robot_offset[0]
        robot_pos_w[:, 1] += self.cfg.visualizing_robot_offset[1]
        robot_pos_w[:, 2] += self.cfg.visualizing_robot_offset[2]

        # 设置根变换。
        root_pose_w = torch.cat([robot_pos_w, robot_quat_w], dim=-1)
        self._reference_view.set_root_transforms(
            root_pose_w,
            indices=self.ALL_INDICES,
        )
        self._reference_view.set_root_velocities(
            torch.zeros_like(root_pose_w[..., :6]),
            indices=self.ALL_INDICES,
        )

        # 设置关节位置。
        self._reference_view.set_dof_positions(
            robot_joint_pos,
            indices=self.ALL_INDICES,
        )
        self._reference_view.set_dof_velocities(
            torch.zeros_like(robot_joint_pos),
            indices=self.ALL_INDICES,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置调试可视化的可见性。"""
        # 设置标记的可见性
        # 注意：父类只处理回调，不处理可见性。
        if debug_vis and self.cfg.visualizing_marker_types:
            if not hasattr(self, "_visualizer"):
                self._visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # 设置可见性为 True。
            self._visualizer.set_visibility(True)
        else:
            if hasattr(self, "_visualizer"):
                self._visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if self.cfg.visualizing_marker_types:
            if not hasattr(self, "_view") or not hasattr(self, "_visualizer"):
                return
            aiming_frame_idx = self.aiming_frame_idx

            pos_list = []
            quat_list = []
            index_list = []

            if "root" in self.cfg.visualizing_marker_types:
                root_pos_w = self.data.base_pos_w[self.ALL_INDICES, aiming_frame_idx]
                root_quat_w = self.data.base_quat_w[self.ALL_INDICES, aiming_frame_idx]
                root_indices = torch.ones(self._view.count, device=self.device, dtype=torch.long) * 0

                pos_list.append(root_pos_w.reshape(-1, 3))
                quat_list.append(root_quat_w.reshape(-1, 4))
                index_list.append(root_indices.reshape(-1))

            if "links" in self.cfg.visualizing_marker_types:
                link_pos_w = self.data.link_pos_w[self.ALL_INDICES, aiming_frame_idx]
                link_quat_w = self.data.link_quat_w[self.ALL_INDICES, aiming_frame_idx]
                link_indices = torch.ones(link_pos_w.shape[:2], device=self.device, dtype=torch.long) * 1

                pos_list.append(link_pos_w.reshape(-1, 3))
                quat_list.append(link_quat_w.reshape(-1, 4))
                index_list.append(link_indices.reshape(-1))

            if "relative_links" in self.cfg.visualizing_marker_types:
                link_pos_w = self.reference_link_pos_relative_w
                link_quat_w = self.reference_link_quat_relative_w
                link_indices = torch.ones(link_pos_w.shape[:2], device=self.device, dtype=torch.long) * 2

                pos_list.append(link_pos_w.reshape(-1, 3))
                quat_list.append(link_quat_w.reshape(-1, 4))
                index_list.append(link_indices.reshape(-1))

            if pos_list:
                pos_w = torch.cat(pos_list, dim=0)
                quat_w = torch.cat(quat_list, dim=0)
                indices = torch.cat(index_list, dim=0)
                self._visualizer.visualize(
                    translations=pos_w,
                    orientations=quat_w,
                    marker_indices=indices,
                )

        if not hasattr(self, "_reference_view"):
            self._find_reference_view()
        if hasattr(self, "_reference_view"):
            self._set_reference_view_state()

    def _invalidate_initialize_callback(self, event):
        """使场景元素无效。"""
        # 调用父类。
        super()._invalidate_initialize_callback(event)
        # 将所有现有视图设置为 None 以使其无效。
        delattr(self, "_physics_sim_view")
        delattr(self, "_view")
        if hasattr(self, "_reference_view"):
            delattr(self, "_reference_view")
