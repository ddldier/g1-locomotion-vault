# 启用延迟注解，使得类型注解在运行时不会被求值，便于前向引用。
from __future__ import annotations

# 导入 PyTorch 库。
import torch
# 导入抽象基类 Sequence，用于类型提示。
from collections.abc import Sequence
# 导入 TYPE_CHECKING，用于在类型检查时导入模块，避免运行时循环导入。
from typing import TYPE_CHECKING

# 导入 Omni PhysX 的 tensors API，用于获取物理仿真视图（如重力）。
import omni.physics.tensors.impl.api as physx

# 导入 Isaac Lab 的仿真工具和数学工具。
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
# 导入命令项基类 CommandTerm。
from isaaclab.managers import CommandTerm
# 导入可视化标记类。
from isaaclab.markers.visualization_markers import VisualizationMarkers

# 导入 instinctlab 的数学工具。
import instinctlab.utils.math as instinct_math_utils

# 仅在类型检查时导入以下模块，避免运行时依赖。
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from instinctlab.motion_reference import MotionReferenceManager

    # 导入各种命令配置类，用于类型提示。
    from .commands_cfg import (
        BaseHeightRefCommandCfg,
        BaseLinVelRefCommandCfg,
        HeadingErrorRefCommandCfg,
        HeadingRefCommandCfg,
        JointPosErrRefCommandCfg,
        JointPosRefCommandCfg,
        JointVelRefCommandCfg,
        LinkPosErrRefCommandCfg,
        LinkPosRefCommandCfg,
        LinkRefCommandCfg,
        LinkRotErrRefCommandCfg,
        LinkRotRefCommandCfg,
        PoseRefCommandCfg,
        PositionRefCommandCfg,
        ProjectedGravityRefCommandCfg,
        RotationRefCommandCfg,
        ShadowingCommandBaseCfg,
        TimeToTargetCommandCfg,
    )


# 定义 Shadowing 命令的基类，继承自 CommandTerm。
class ShadowingCommandBase(CommandTerm):
    """Shadowing 命令项的基类，封装了与运动参考管理器交互的通用逻辑。"""

    # 类型注解：配置类为 ShadowingCommandBaseCfg。
    cfg: ShadowingCommandBaseCfg

    def __init__(self, cfg: ShadowingCommandBaseCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用父类 CommandTerm 的初始化。
        super().__init__(cfg, env)

        # self.robot: Articulation = env.scene[cfg.asset_name]
        # 从环境中获取运动参考管理器实例。
        self._motion_reference: MotionReferenceManager = env.scene[cfg.motion_reference.name]
        # 命令张量，在子类中重写。
        self._command: torch.Tensor = None  # override in the child class # type: ignore
        # 掩码张量，在子类中重写。
        self._mask: torch.Tensor = None  # override in the child class # type: ignore

    @property
    def command(self) -> torch.Tensor:
        """返回命令张量（未展平）。"""
        # 如果配置要求包含当前状态命令，则将当前状态拼接到命令后。
        if self.cfg.current_state_command:
            return torch.cat([self._command, self._current_state], dim=1)
        # 否则直接返回命令。
        return self._command  # type: ignore

    @property
    def mask(self) -> torch.Tensor:
        """返回命令张量的掩码。
        ## 注意：
            如果 current_state_command 为 True，掩码应在时间维度上额外返回一个 ones。
        """
        # 基类中未实现，子类必须重写。
        raise NotImplementedError(
            "The mask property is not implemented in the base class.",
        )

    @property
    def ALL_INDICES(self):
        """代理属性，用于访问 ALL_INDICES，以便读取当前时间步的目标运动/命令。"""
        return self._motion_reference.ALL_INDICES

    @property
    def aiming_frame_idx(self):
        """代理属性，用于访问运动参考的瞄准帧索引。
        ## 注意：
            一个有趣的功能：当 aiming_frame_idx 变为 -1（无效）时，
            command_term.command[ALL_INDICES, aiming_frame_idx] 会返回运动参考的最后一帧（即当前状态）。
        """
        return self._motion_reference.aiming_frame_idx

    @property
    def link_of_interests(self) -> Sequence[str] | None:
        """返回运动参考中感兴趣的链接名称。"""
        return self._motion_reference.cfg.link_of_interests

    @property
    def time_to_aiming_frame(self) -> torch.Tensor:
        """代理属性，用于访问到达运动参考瞄准帧的时间。"""
        return self._motion_reference.time_to_aiming_frame

    @property
    def _motion_reference_updated_env_ids(self) -> Sequence[int] | torch.Tensor:
        """返回需要更新的环境索引，考虑到运动参考（作为传感器）可能在仿真步中更新，
        但每个命令项在每个环境步中只更新一次。
        """
        return torch.where(self._motion_reference.time_passed_from_update < (self._env.step_dt - 1e-6))[0]

    def _update_metrics(self):
        """根据运动参考和时间更新指标。基类中为空实现。"""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """不依赖于命令项。self._command 总是依赖于运动参考。基类中为空实现。"""
        pass

    def _update_command(self):
        """根据运动参考和时间更新命令。"""
        # 触发运动参考数据的更新。
        self._motion_reference.data

        # 更新位置命令的实时模式。
        if getattr(self.cfg, "realtime_mode", False):
            # 实时模式：更新所有环境。
            env_ids = self._motion_reference.ALL_INDICES
        else:
            # 非实时模式：只更新需要更新的环境。
            env_ids = self._motion_reference_updated_env_ids
        # 如果有需要更新的环境，则调用子类的更新方法。
        if len(env_ids) > 0:
            self._update_command_by_env_ids(env_ids)
        # 如果存在可视化器，则计算调试可视化数据。
        if getattr(self, "_visualizer", None) is not None:
            self._compute_debug_vis_data()

    def reset(self, env_ids: Sequence[int] | None = None):
        """重置指定环境索引的命令项。"""
        # 根据环境索引更新命令。
        self._update_command_by_env_ids(env_ids)
        # 返回空字典，表示没有指标需要返回，子类可重写。
        return {}  # no metrics to return, override if needed.

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """根据运动参考和时间，为指定环境索引更新命令。子类必须实现。"""
        raise NotImplementedError

    def _debug_vis_callback(self, event):
        """调试可视化回调，基类中为空。"""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置调试可视化的可见性。"""
        # 设置标记的可见性
        # 注意：父类只处理回调，不处理可见性。
        if debug_vis and (getattr(self.cfg, "visualizer_cfg", None) is not None):
            # 如果可视化器尚未创建，则创建。
            if not hasattr(self, "_visualizer"):
                self._visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # 设置可见性为 True。
            self._visualizer.set_visibility(True)
        else:
            # 如果存在可视化器，则设置可见性为 False。
            if hasattr(self, "_visualizer"):
                self._visualizer.set_visibility(False)

    def _compute_debug_vis_data(self):
        """计算并更新可视化数据。基类中为空实现。"""
        pass


# 定义 PoseRefCommand 类，继承自 ShadowingCommandBase，用于位置和姿态参考命令。
class PoseRefCommand(ShadowingCommandBase):
    """位置和姿态参考命令。"""

    cfg: PoseRefCommandCfg

    def __init__(self, cfg: PoseRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 为了向后兼容，将 self.cfg.realtime_mode 从布尔值转换为整数。
        if isinstance(self.cfg.realtime_mode, bool):
            self.cfg.realtime_mode = int(self.cfg.realtime_mode)
        # 根据旋转模式确定命令的数据维度。
        if self.cfg.rotation_mode == "quaternion":
            data_dims = (3 + 4,)
        elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
            data_dims = (3 + 3,)
        elif self.cfg.rotation_mode == "tannorm":
            data_dims = (3 + 6,)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        # 初始化命令张量，形状为 (num_envs, num_frames, data_dims)。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )  # (n_envs, num_frames, 3 + (4 or 3))
        # 初始化掩码张量，形状为 (num_envs, num_frames, 4)，分别对应平面、高度、朝向、航向掩码。
        self._mask = torch.ones(
            (
                self.num_envs,
                self._motion_reference.num_frames,
                4,
            ),  # (plane mask, height mask, orientation mask, heading mask)
            device=self.device,
            dtype=torch.bool,
        )
        # 如果需要当前状态作为伪命令帧，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 4),  # (plane mask, height mask, orientation mask, heading mask)
                device=self.device,
                dtype=torch.bool,
            )
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码，结合运动参考的有效性和各类掩码。"""
        # 将基础掩码与运动参考的有效性进行逻辑与。
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        # 对每个通道分别与运动参考的对应掩码进行逻辑与。
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_pos_plane_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.base_pos_height_mask)
        mask[..., 2] = torch.logical_and(mask[..., 2], self._motion_reference.data.base_orientation_mask)
        mask[..., 3] = torch.logical_and(mask[..., 3], self._motion_reference.data.base_heading_mask)
        # 如果需要当前状态，则拼接当前状态的掩码。
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command(self):
        """根据运动参考和时间更新命令。"""
        # 触发运动参考数据更新。
        self._motion_reference.data

        # 更新位置命令的实时模式。
        if self.cfg.realtime_mode == 1:
            # 实时模式：更新所有环境。
            env_ids = self._motion_reference.ALL_INDICES
        else:
            # 非实时模式：只更新需要更新的环境。
            env_ids = self._motion_reference_updated_env_ids
        if len(env_ids) > 0:
            self._update_command_by_env_ids(env_ids)
        # 如果存在可视化器，则计算调试可视化数据。
        if getattr(self, "_visualizer", None) is not None:
            self._compute_debug_vis_data()

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新命令。"""
        # 根据锚定框架计算锚点的逆变换。
        if self.cfg.anchor_frame == "robot":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids],
                self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids],
            )
        elif self.cfg.anchor_frame == "reference":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._motion_reference.data.base_pos_w[env_ids, 0],
                self._motion_reference.data.base_quat_w[env_ids, 0],
            )
        # 将运动参考的基座位置变换到锚点坐标系。
        self._command[env_ids, :, :3] = math_utils.transform_points(
            self._motion_reference.data.base_pos_w[env_ids],
            anchor_pos_w_inv,
            anchor_quat_w_inv,
        )

        # 更新旋转命令的实时模式。
        if self.cfg.realtime_mode == -1:
            env_ids = self._motion_reference.ALL_INDICES
        # 计算基座在锚点坐标系下的四元数。
        base_quat_b = math_utils.quat_mul(
            anchor_quat_w_inv.unsqueeze(1).expand(-1, self._motion_reference.num_frames, -1),
            self._motion_reference.data.base_quat_w[env_ids],
        )
        # 根据旋转模式存储旋转命令。
        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids, :, 3:7] = base_quat_b
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids, :, 3:6] = math_utils.axis_angle_from_quat(base_quat_b)
        elif self.cfg.rotation_mode == "euler":
            base_quat_b_ = base_quat_b.flatten(0, 1)
            euler_b_ = torch.stack(math_utils.euler_xyz_from_quat(base_quat_b_), dim=-1)
            self._command[env_ids, :, 3:6] = euler_b_.view(self._command[env_ids, :, 3:6].shape)
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids, :, 3:9] = instinct_math_utils.quat_to_tan_norm(base_quat_b)

        # 应用掩码：如果高度或平面掩码有效，则保留位置命令，否则置零。
        self._command[env_ids, :, :3] *= torch.logical_or(
            self._motion_reference.data.base_pos_height_mask[env_ids],
            self._motion_reference.data.base_pos_plane_mask[env_ids],
        ).unsqueeze(-1) * self._mask[env_ids, :, 0:2].any(dim=-1, keepdim=True)
        # 应用掩码：如果朝向或航向掩码有效，则保留旋转命令，否则置零。
        self._command[env_ids, :, 3:] *= torch.logical_or(
            self._motion_reference.data.base_orientation_mask[env_ids],
            self._motion_reference.data.base_heading_mask[env_ids],
        ).unsqueeze(-1) * self._mask[env_ids, :, 2:4].any(dim=-1, keepdim=True)
        # 应用有效性掩码。
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)

        # 如果需要当前状态命令，则设置当前状态为默认值（单位姿态）。
        if self.cfg.current_state_command:
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, 3] = 1.0
            elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
                pass
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, 3:9] = 0
                self._current_state[env_ids, :, 3] = 1
                self._current_state[env_ids, :, 8] = 1

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if not hasattr(self, "_vis_pose_ref_pos_w"):
            return
        # 可视化位置和姿态参考。
        self._visualizer.visualize(
            translations=self._vis_pose_ref_pos_w,
            orientations=self._vis_pose_ref_quat_w,
        )

    def _compute_debug_vis_data(self):
        """计算调试可视化数据。"""
        root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w

        # 由于该命令定义在机器人本地坐标系中，我们需要将命令数据变换到世界坐标系。
        # 即使它没有及时刷新。
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]
        # 将位置命令变换到世界坐标系。
        pose_ref_pos_w = math_utils.transform_points(
            keyframe_command[:, :3].unsqueeze(1),
            root_pos_w,
            root_quat_w,
        ).squeeze(1)
        # 根据旋转模式将旋转命令变换到世界坐标系。
        if self.cfg.rotation_mode == "quaternion":
            pose_ref_quat_w = math_utils.quat_mul(
                root_quat_w,
                keyframe_command[:, 3:7],
            )
        elif self.cfg.rotation_mode == "axis_angle":
            pose_ref_quat_b = math_utils.quat_from_angle_axis(
                torch.norm(keyframe_command[:, 3:6], dim=-1),
                keyframe_command[:, 3:6],
            )
            pose_ref_quat_w = math_utils.quat_mul(
                root_quat_w,
                pose_ref_quat_b,
            )
        elif self.cfg.rotation_mode == "euler":
            pose_ref_quat_b = math_utils.quat_from_euler_xyz(
                keyframe_command[:, 3],
                keyframe_command[:, 4],
                keyframe_command[:, 5],
            )
            pose_ref_quat_w = math_utils.quat_mul(
                root_quat_w,
                pose_ref_quat_b,
            )
        elif self.cfg.rotation_mode == "tannorm":
            pose_ref_quat_b = instinct_math_utils.tan_norm_to_quat(keyframe_command[:, 3:9])
            pose_ref_quat_w = math_utils.quat_mul(
                root_quat_w,
                pose_ref_quat_b,
            )
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        # 保存可视化数据。
        self._vis_pose_ref_pos_w = pose_ref_pos_w
        self._vis_pose_ref_quat_w = pose_ref_quat_w


# 定义 PositionRefCommand 类，继承自 ShadowingCommandBase，仅包含位置参考。
class PositionRefCommand(ShadowingCommandBase):
    """仅包含位置参考的命令。"""

    cfg: PositionRefCommandCfg

    def __init__(self, cfg: PositionRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 为了向后兼容，将 self.cfg.realtime_mode 从布尔值转换为整数。
        if isinstance(self.cfg.realtime_mode, bool):
            self.cfg.realtime_mode = int(self.cfg.realtime_mode)
        # 初始化命令张量，形状为 (num_envs, num_frames, 3)。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 3),
            device=self.device,
        )
        # 初始化掩码张量，形状为 (num_envs, num_frames, 2)，分别对应平面和高度掩码。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 2),  # (plane mask, height mask)
            device=self.device,
            dtype=torch.bool,
        )
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, 3),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 2),  # (plane mask, height mask)
                device=self.device,
                dtype=torch.bool,
            )
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_pos_plane_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.base_pos_height_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新位置命令。"""
        # 根据锚定框架计算锚点的逆变换。
        if self.cfg.anchor_frame == "robot":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids],
                self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids],
            )
        elif self.cfg.anchor_frame == "reference":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._motion_reference.reference_frame.base_pos_w[env_ids, 0],
                self._motion_reference.reference_frame.base_quat_w[env_ids, 0],
            )
        elif self.cfg.anchor_frame == "ref_rot_robot_pos":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids],
                self._motion_reference.data.base_quat_w[env_ids, 0],
            )
        elif self.cfg.anchor_frame == "ref_pos_robot_rot":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._motion_reference.data.base_pos_w[env_ids, 0],
                self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids],
            )
        # 将参考位置变换到锚点坐标系，并应用掩码。
        self._command[env_ids] = (
            math_utils.transform_points(
                self._motion_reference.data.base_pos_w[env_ids],
                anchor_pos_w_inv,
                anchor_quat_w_inv,
            )
            * torch.logical_or(
                self._motion_reference.data.base_pos_height_mask[env_ids],
                self._motion_reference.data.base_pos_plane_mask[env_ids],
            ).unsqueeze(-1)
            * self._mask[env_ids].any(dim=-1, keepdim=True)
        )

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        robot_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w
        # 将命令位置变换到世界坐标系用于可视化。
        _vis_point_w: torch.tensor = math_utils.transform_points(
            self._command,  # (num_envs, num_frames, 3)
            robot_pos_w,
            robot_quat_w,
        )  # (num_envs, num_frames, 3)
        _scales = torch.ones_like(_vis_point_w)
        _scales[:, 1:] *= 0.5
        self._visualizer.visualize(
            translations=_vis_point_w.flatten(0, 1),
            scales=_scales.flatten(0, 1),
        )


# 定义 RotationRefCommand 类，继承自 ShadowingCommandBase，仅包含旋转参考。
class RotationRefCommand(ShadowingCommandBase):
    """仅包含姿态命令中旋转部分的序列命令。"""

    cfg: RotationRefCommandCfg

    def __init__(self, cfg: RotationRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。"""
        super().__init__(cfg, env)
        # 为了向后兼容，将 self.cfg.realtime_mode 从布尔值转换为整数。
        if isinstance(self.cfg.realtime_mode, bool):
            self.cfg.realtime_mode = int(self.cfg.realtime_mode)
        # 根据旋转模式确定数据维度。
        if self.cfg.rotation_mode == "quaternion":
            data_dims = (4,)
        elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
            data_dims = (3,)
        elif self.cfg.rotation_mode == "tannorm":
            data_dims = (6,)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        # 初始化命令张量。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )  # (n_envs, num_frames, 3 + (4 or 3))
        # 初始化掩码张量，形状为 (num_envs, num_frames, 2)，分别对应朝向和航向掩码。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 2),  # (orientation mask, heading mask)
            device=self.device,
            dtype=torch.bool,
        )
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 2),  # (orientation mask, heading mask)
                device=self.device,
                dtype=torch.bool,
            )
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_orientation_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.base_heading_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新旋转命令。"""
        # 获取机器人基座四元数的逆。
        _, robot_quat_w_inv = math_utils.subtract_frame_transforms(
            self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids],
            self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids],
        )

        # 根据是否在基座坐标系中，计算参考四元数。
        if self.cfg.in_base_frame:
            quat_reference = math_utils.quat_mul(
                robot_quat_w_inv.unsqueeze(1).expand(-1, self._motion_reference.num_frames, -1),
                self._motion_reference.data.base_quat_w[env_ids],
            )
        else:
            quat_reference = self._motion_reference.data.base_quat_w[env_ids]

        # 根据旋转模式存储命令。
        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids] = quat_reference
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids] = math_utils.axis_angle_from_quat(quat_reference)
        elif self.cfg.rotation_mode == "euler":
            quat_reference_ = quat_reference.flatten(0, 1)
            euler_ = torch.stack(math_utils.euler_xyz_from_quat(quat_reference_), dim=-1)
            self._command[env_ids] = euler_.view(self._command[env_ids].shape)
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids] = instinct_math_utils.quat_to_tan_norm(quat_reference)

        # 应用掩码和有效性。
        self._command[env_ids] *= torch.logical_or(
            self._motion_reference.data.base_orientation_mask[env_ids],
            self._motion_reference.data.base_heading_mask[env_ids],
        ).unsqueeze(-1) * self._mask[env_ids].any(dim=-1, keepdim=True)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)

        # 如果需要当前状态且处于基座坐标系，设置当前状态为单位旋转。
        if self.cfg.current_state_command and self.cfg.in_base_frame:
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, 0] = 1.0
            elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
                pass
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :] = 0
                self._current_state[env_ids, :, 0] = 1
                self._current_state[env_ids, :, 5] = 1
        # 如果需要当前状态且不在基座坐标系，设置当前状态为机器人当前姿态。
        if self.cfg.current_state_command and not self.cfg.in_base_frame:
            robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids]
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, :] = robot_quat_w.unsqueeze(1)
            elif self.cfg.rotation_mode in "axis_angle":
                self._current_state[env_ids, :, :] = math_utils.axis_angle_from_quat(robot_quat_w).unsqueeze(1)
            elif self.cfg.rotation_mode == "euler":
                self._current_state[env_ids, :, :] = math_utils.euler_xyz_from_quat(robot_quat_w).unsqueeze(1)
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :] = instinct_math_utils.quat_to_tan_norm(robot_quat_w).unsqueeze(1)

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if not hasattr(self, "_vis_quat_reference_w"):
            return
        self._visualizer.visualize(
            translations=self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w,
            orientations=self._vis_quat_reference_w,
        )

    def _compute_debug_vis_data(self):
        """计算调试可视化数据。"""
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w

        # 由于该命令定义在机器人本地坐标系中，我们需要将命令数据变换到世界坐标系。
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]
        # 根据旋转模式将命令转换为四元数。
        if self.cfg.rotation_mode == "quaternion":
            quat_reference = keyframe_command[:]
        elif self.cfg.rotation_mode == "axis_angle":
            quat_reference = math_utils.quat_from_angle_axis(
                torch.norm(keyframe_command, dim=-1),
                keyframe_command,
            )
        elif self.cfg.rotation_mode == "euler":
            quat_reference = math_utils.quat_from_euler_xyz(
                keyframe_command[..., 0],
                keyframe_command[..., 1],
                keyframe_command[..., 2],
            )
        elif self.cfg.rotation_mode == "tannorm":
            quat_reference = instinct_math_utils.tan_norm_to_quat(keyframe_command)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        # 如果在基座坐标系中，则转换到世界坐标系。
        if self.cfg.in_base_frame:
            quat_reference = math_utils.quat_mul(
                root_quat_w,
                quat_reference,
            )
        self._vis_quat_reference_w = quat_reference


# 定义 ProjectedGravityRefCommand 类，继承自 ShadowingCommandBase，用于投影重力参考。
class ProjectedGravityRefCommand(ShadowingCommandBase):
    """将重力向量投影到参考机器人基座坐标系上的命令。"""

    cfg: ProjectedGravityRefCommandCfg

    def __init__(self, cfg: ProjectedGravityRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 创建物理仿真视图以获取重力。
        physics_sim_view = physx.create_simulation_view("torch")
        physics_sim_view.set_subspace_roots("/")
        gravity = physics_sim_view.get_gravity()
        # 转换为方向向量。
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        # 保存重力向量在世界坐标系中的表示，并扩展到所有环境。
        self.GRAVITY_VEC_W = math_utils.normalize(
            torch.tensor(
                (gravity[0], gravity[1], gravity[2]),
                device=self.device,
            ).unsqueeze(0)
        ).expand(
            self.num_envs, -1
        )  # (num_envs, 3)
        # 初始化命令张量。
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, 3),
            device=self.device,
        )
        # 初始化掩码张量，仅包含朝向掩码。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (orientation mask)
            device=self.device,
            dtype=torch.bool,
        )
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, 3),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 1),  # (orientation mask)
                device=self.device,
                dtype=torch.bool,
            )
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_orientation_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新命令。"""
        # 获取参考基座在世界坐标系中的四元数。
        base_quat_w = self._motion_reference.data.base_quat_w[env_ids]
        # 将重力向量投影到基座坐标系。
        self._command[env_ids] = math_utils.quat_apply_inverse(base_quat_w, self.GRAVITY_VEC_W[env_ids].unsqueeze(1))
        # 应用掩码和有效性。
        self._command[env_ids] *= self._motion_reference.data.base_orientation_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids]
        # 如果需要当前状态，设置为机器人当前投影重力。
        if self.cfg.current_state_command:
            self._current_state[env_ids] = (
                self._env.scene[self.cfg.asset_cfg.name].data.projected_gravity_b[env_ids].unsqueeze(1)
            )
        # 如果存在可视化器，计算 roll/pitch 参考用于可视化。
        if hasattr(self, "_visualizer"):
            _quat_ref = self._motion_reference.data.base_quat_w[
                env_ids, self._motion_reference.aiming_frame_idx[env_ids]
            ]
            self.compute_rollpitch_ref(env_ids, _quat_ref)

    def compute_rollpitch_ref(self, env_ids, _quat_ref):
        """计算参考的 roll 和 pitch 角，用于可视化。"""
        _roll_ref, _pitch_ref = math_utils.euler_xyz_from_quat(_quat_ref)[:2]
        _roll_ref = math_utils.wrap_to_pi(_roll_ref)
        _pitch_ref = math_utils.wrap_to_pi(_pitch_ref)
        # 如果尚未初始化，则创建存储张量。
        if not hasattr(self, "_roll_ref"):
            self._roll_ref = torch.zeros(self._command.shape[0], device=self.device)
        if not hasattr(self, "_pitch_ref"):
            self._pitch_ref = torch.zeros(self._command.shape[0], device=self.device)
        # 存储计算得到的 roll 和 pitch。
        self._roll_ref[env_ids] = _roll_ref
        self._pitch_ref[env_ids] = _pitch_ref

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if not hasattr(self, "_roll_ref") or not hasattr(self, "_pitch_ref"):
            return
        # 获取机器人基座位置并抬高用于可视化。
        marker_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.clone()
        marker_pos_w[:, 2] += 1.2  # raise the position for visualization
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w
        robot_heading_w = math_utils.euler_xyz_from_quat(robot_quat_w)[2]  # (num_envs,)
        # 使用参考的 roll/pitch 和机器人的 heading 构建可视化朝向。
        marker_orientations = math_utils.quat_from_euler_xyz(
            self._roll_ref,  # roll
            self._pitch_ref,  # pitch
            robot_heading_w,  # yaw
        )
        scales = torch.ones_like(marker_pos_w[:, :3])  # scale for visualization
        scales[:, 0] = 0.5  # set the scale for the x-axis
        scales[:, 1] = 0.5  # set the scale for the y-axis
        scales[:, 2] = 0.1  # set the scale for the z-axis
        self._visualizer.visualize(
            translations=marker_pos_w,
            orientations=marker_orientations,
            scales=scales,
        )


# 定义 HeadingRefCommand 类，继承自 ShadowingCommandBase，仅包含参考的航向（yaw）。
class HeadingRefCommand(ShadowingCommandBase):
    """仅包含参考航向的命令，即参考旋转的 yaw 分量。"""

    cfg: HeadingRefCommandCfg

    def __init__(self, cfg: HeadingRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 初始化命令张量，形状为 (num_envs, num_frames, 1)。
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, 1),
            device=self.device,
        )
        # 初始化掩码张量，仅包含航向掩码。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (heading mask)
            device=self.device,
            dtype=torch.bool,
        )
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, 1),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 1),  # (heading mask)
                device=self.device,
                dtype=torch.bool,
            )
        # 初始化指标：航向误差。
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_heading_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_metrics(self):
        """更新航向误差指标。"""
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w
        robot_heading_w = math_utils.euler_xyz_from_quat(robot_quat_w)[2]  # (num_envs,)
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        heading_command_w = self._command[ALL_INDICES, aiming_frame_idx].squeeze(-1)  # (num_envs,)
        heading_error = torch.abs(math_utils.wrap_to_pi(heading_command_w - robot_heading_w))
        self.metrics["error_heading"] = heading_error

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新航向命令。"""
        # 获取参考基座在世界坐标系中的四元数。
        base_quat_w = self._motion_reference.data.base_quat_w[env_ids]
        # 提取航向（yaw）分量。
        self._command[env_ids] = math_utils.wrap_to_pi(
            math_utils.euler_xyz_from_quat(base_quat_w.reshape(-1, 4))[2]
        ).reshape(-1, self._motion_reference.num_frames, 1)
        # 应用掩码和有效性。
        self._command[env_ids] *= self._motion_reference.data.base_heading_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids]
        # 如果需要当前状态，设置为机器人当前航向。
        if self.cfg.current_state_command:
            self._current_state[env_ids] = (
                math_utils.wrap_to_pi(
                    math_utils.euler_xyz_from_quat(self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids])[
                        2
                    ]
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if not hasattr(self, "_vis_heading_ref"):
            return
        base_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.8
        scales = torch.ones_like(base_pos_w)  # scale for visualization
        scales[:, 0] = 4.0  # set the scale for the heading arrow
        scales[:, 1] = 1.0  # set the scale for the heading arrow
        scales[:, 2] = 0.1  # set the scale for the heading arrow
        # 可视化航向参考。
        self._visualizer.visualize(translations=base_pos_w, orientations=self._vis_heading_ref, scales=scales)

    def _compute_debug_vis_data(self):
        """计算调试可视化数据。"""
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        heading_command_w = self._command[ALL_INDICES, aiming_frame_idx].squeeze(-1)  # (num_envs,)
        # 根据航向角构建四元数。
        heading_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(heading_command_w),  # pitch
            torch.zeros_like(heading_command_w),  # roll
            heading_command_w,  # yaw
        )
        heading_quat = math_utils.normalize(heading_quat)  # normalize the quaternion
        self._vis_heading_ref = heading_quat


# 定义 HeadingErrorRefCommand 类，继承自 ShadowingCommandBase，包含参考的航向误差。
class HeadingErrorRefCommand(ShadowingCommandBase):
    """仅包含参考航向误差的命令，即参考旋转的 yaw 分量与机器人当前航向的误差。"""

    cfg: HeadingErrorRefCommandCfg

    def __init__(self, cfg: HeadingErrorRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 初始化命令张量，形状为 (num_envs, num_frames, 1)。
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, 1),
            device=self.device,
        )
        # 初始化目标航向张量。
        self._target_heading = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, 1),
            device=self.device,
        )
        # 初始化掩码张量，仅包含航向掩码。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (heading mask)
            device=self.device,
            dtype=torch.bool,
        )
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, 1),
                device=self.device,
            )
        # 初始化指标：航向误差。
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        # 更新命令。
        self._update_command()

    def _update_metrics(self):
        """更新航向误差指标。"""
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w
        robot_heading_w = math_utils.euler_xyz_from_quat(robot_quat_w)[2]  # (num_envs,)
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        heading_command_w = self._target_heading[ALL_INDICES, aiming_frame_idx].squeeze(-1)  # (num_envs,)
        heading_error = torch.abs(math_utils.wrap_to_pi(heading_command_w - robot_heading_w))
        self.metrics["error_heading"] = heading_error

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新航向误差命令。"""
        if len(env_ids) == 0:
            return
        # 获取参考基座在世界坐标系中的四元数。
        base_quat_w = self._motion_reference.data.base_quat_w[env_ids]
        # 提取目标航向（yaw）分量。
        self._target_heading[env_ids] = math_utils.wrap_to_pi(
            math_utils.euler_xyz_from_quat(base_quat_w.reshape(-1, 4))[2]
        ).reshape(-1, self._motion_reference.num_frames, 1)
        # 获取机器人当前航向。
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids]
        robot_heading_w = math_utils.euler_xyz_from_quat(robot_quat_w)[2]  # (num_envs,)
        # 计算航向误差。
        self._command[env_ids] = math_utils.wrap_to_pi(
            self._target_heading[env_ids] - robot_heading_w.unsqueeze(1).unsqueeze(1)
        )
        # 应用掩码和有效性。
        self._command[env_ids] *= self._motion_reference.data.base_heading_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids]

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if not hasattr(self, "_vis_heading_ref"):
            return
        base_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.8
        scales = torch.ones_like(base_pos_w)  # scale for visualization
        scales[:, 0] = 4.0  # set the scale for the heading arrow
        scales[:, 1] = 1.0  # set the scale for the heading arrow
        scales[:, 2] = 0.1  # set the scale for the heading arrow
        # 可视化航向参考。
        self._visualizer.visualize(translations=base_pos_w, orientations=self._vis_heading_ref, scales=scales)

    def _compute_debug_vis_data(self):
        """计算调试可视化数据。"""
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        heading_command_w = self._target_heading[ALL_INDICES, aiming_frame_idx].squeeze(-1)  # (num_envs,)
        # 根据航向角构建四元数。
        heading_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(heading_command_w),  # pitch
            torch.zeros_like(heading_command_w),  # roll
            heading_command_w,  # yaw
        )
        heading_quat = math_utils.normalize(heading_quat)  # normalize the quaternion
        self._vis_heading_ref = heading_quat


# 定义 BaseHeightRefCommand 类，继承自 ShadowingCommandBase，仅包含基座高度参考。
class BaseHeightRefCommand(ShadowingCommandBase):
    """仅包含基座高度的命令。
    注意：当前实现仅直接参考运动参考。如果地形不是零平面，可能会导致误差。
    """

    cfg: BaseHeightRefCommandCfg

    def __init__(self, cfg: BaseHeightRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 数据维度为 1（高度）。
        data_dims = (1,)
        # 初始化命令张量。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        # 初始化掩码张量，仅包含高度掩码。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (height mask)
            device=self.device,
            dtype=torch.bool,
        )
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 1),  # (height mask)
                device=self.device,
                dtype=torch.bool,
            )
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_pos_height_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新高度命令。"""
        # 直接取运动参考基座位置的 z 分量。
        self._command[env_ids, :] = self._motion_reference.data.base_pos_w[env_ids][:, :, 2:3]
        # 应用掩码和有效性。
        self._command[env_ids] *= (
            self._motion_reference.data.base_pos_height_mask[env_ids].unsqueeze(-1)
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1)
            * self._mask[env_ids]
        )
        # 如果需要当前状态，设置为机器人当前基座高度。
        if self.cfg.current_state_command:
            self._current_state[env_ids] = (
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids, 2:3].unsqueeze(1)
            )

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if not hasattr(self, "_vis_marker_pos_w"):
            return
        self._visualizer.visualize(
            translations=self._vis_marker_pos_w,
        )

    def _compute_debug_vis_data(self):
        """计算调试可视化数据。"""
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        # 获取机器人当前位置，并将 z 替换为命令高度。
        marker_translation = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.clone()
        marker_translation[:, 2] = self._command[ALL_INDICES, aiming_frame_idx, 0]
        self._vis_marker_pos_w = marker_translation


# 定义 BaseLinVelRefCommand 类，继承自 ShadowingCommandBase，仅包含基座线速度参考。
class BaseLinVelRefCommand(ShadowingCommandBase):
    """仅包含基座线速度的命令。"""

    cfg: BaseLinVelRefCommandCfg

    def __init__(self, cfg: BaseLinVelRefCommandCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        super().__init__(cfg, env)
        # 数据维度为 3（线速度）。
        data_dims = (3,)
        # 初始化命令张量。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        # 初始化掩码张量，仅包含速度掩码。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 1),  # (velocity mask)
            device=self.device,
            dtype=torch.bool,
        )
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 1),  # (velocity mask)
                device=self.device,
                dtype=torch.bool,
            )
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。目前没有来自运动参考数据的掩码。"""
        mask = self._mask  # No mask from motion reference data currently.
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新线速度命令。"""
        # 将世界坐标系下的基座线速度变换到基座坐标系。
        self._command[env_ids] = math_utils.quat_apply_inverse(
            self._motion_reference.data.base_quat_w[env_ids], self._motion_reference.data.base_lin_vel_w[env_ids]
        )
        # 应用有效性和掩码。
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids]
        # 如果需要当前状态，设置为机器人当前基座线速度（基座坐标系）。
        if self.cfg.current_state_command:
            self._current_state[env_ids] = (
                self._env.scene[self.cfg.asset_cfg.name].data.root_lin_vel_b[env_ids].unsqueeze(1)
            )


# 定义 JointPosRefCommand 类，继承自 ShadowingCommandBase，仅包含关节位置参考。
class JointPosRefCommand(ShadowingCommandBase):
    """关节位置参考命令。"""

    cfg: JointPosRefCommandCfg

    def __init__(self, cfg: JointPosRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 数据维度为关节数量。
        data_dims = (self._motion_reference.data.joint_pos.shape[2],)
        # 初始化命令张量。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        # 初始化掩码张量，形状为 (num_envs, num_frames, num_joints)。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.data.joint_pos.shape[2]),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_joints)
        # 获取并复制默认关节位置。
        # 形状 (num_envs, num_joints)
        self._default_joint_pos = self._env.scene[cfg.asset_cfg.name].data.default_joint_pos.clone()
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.data.joint_pos.shape[2]),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_joints)
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(mask, self._motion_reference.data.joint_pos_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新关节位置命令。"""
        # 参考关节位置减去默认关节位置，得到相对命令。
        self._command[env_ids, :] = self._motion_reference.data.joint_pos[env_ids] - self._default_joint_pos[
            env_ids
        ].unsqueeze(1)
        # 应用关节位置掩码、有效性和掩码。
        self._command[env_ids, :] *= self._motion_reference.data.joint_pos_mask[env_ids]
        self._command[env_ids, :] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids, :] *= self._mask[env_ids]
        # 如果需要当前状态，设置为机器人当前关节位置减去默认位置。
        if self.cfg.current_state_command:
            self._current_state[env_ids] = self._env.scene[self.cfg.asset_cfg.name].data.joint_pos[env_ids].unsqueeze(
                1
            ) - self._default_joint_pos[env_ids].unsqueeze(1)


# 定义 JointPosErrRefCommand 类，继承自 ShadowingCommandBase，包含关节位置误差。
class JointPosErrRefCommand(ShadowingCommandBase):
    """关节位置误差命令。"""

    cfg: JointPosErrRefCommandCfg

    def __init__(self, cfg: JointPosErrRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 数据维度为关节数量。
        data_dims = (self._motion_reference.data.joint_pos.shape[2],)
        # 初始化命令张量。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        # 初始化掩码张量。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.data.joint_pos.shape[2]),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_joints)
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.data.joint_pos.shape[2]),
                device=self.device,
                dtype=torch.bool,
            )
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(mask, self._motion_reference.data.joint_pos_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新关节位置误差命令。"""
        # 参考关节位置减去机器人当前关节位置。
        self._command[env_ids, :] = self._motion_reference.data.joint_pos[env_ids] - self._env.scene[
            self.cfg.asset_cfg.name
        ].data.joint_pos[env_ids].unsqueeze(1)
        # 应用掩码和有效性。
        self._command[env_ids] *= (
            self._motion_reference.data.joint_pos_mask[env_ids]
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1)
            * self._mask[env_ids]
        )


# 定义 JointVelRefCommand 类，继承自 ShadowingCommandBase，仅包含关节速度参考。
class JointVelRefCommand(ShadowingCommandBase):
    """关节速度参考命令。"""

    cfg: JointVelRefCommandCfg

    def __init__(self, cfg: JointVelRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 数据维度为关节数量。
        data_dims = (self._motion_reference.data.joint_vel.shape[2],)
        # 初始化命令张量。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        # 初始化掩码张量。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.data.joint_vel.shape[2]),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_joints)
        # 获取并复制默认关节速度。
        # 形状 (num_envs, num_joints)
        self._default_joint_vel = self._env.scene[cfg.asset_cfg.name].data.default_joint_vel.clone()
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.data.joint_vel.shape[2]),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_joints)
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(mask, self._motion_reference.data.joint_vel_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新关节速度命令。"""
        # 参考关节速度减去默认关节速度。
        self._command[env_ids, :] = self._motion_reference.data.joint_vel[env_ids] - self._default_joint_vel[
            env_ids
        ].unsqueeze(1)
        # 应用关节速度掩码、有效性和掩码。
        self._command[env_ids, :] *= (
            self._motion_reference.data.joint_vel_mask[env_ids]
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1)
            * self._mask[env_ids]
        )
        # 如果需要当前状态，设置为机器人当前关节速度减去默认速度。
        if self.cfg.current_state_command:
            self._current_state[env_ids] = self._env.scene[self.cfg.asset_cfg.name].data.joint_vel[env_ids].unsqueeze(
                1
            ) - self._default_joint_vel[env_ids].unsqueeze(1)


# 定义 LinkRefCommand 类，继承自 ShadowingCommandBase，包含链接位置和旋转参考。
class LinkRefCommand(ShadowingCommandBase):
    """链接位置和旋转参考命令。"""

    cfg: LinkRefCommandCfg

    def __init__(self, cfg: LinkRefCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 根据旋转模式确定数据维度。
        if self.cfg.rotation_mode == "quaternion":
            data_dims = (3 + 4,)
        elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
            data_dims = (3 + 3,)
        elif self.cfg.rotation_mode == "tannorm":
            data_dims = (3 + 6,)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        # 初始化命令张量，形状为 (num_envs, num_frames, num_links, data_dims)。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, *data_dims),
            device=self.device,
        )
        # 初始化掩码张量，形状为 (num_envs, num_frames, num_links, 2)，分别对应位置和旋转掩码。
        self._mask = torch.ones(
            (
                self.num_envs,
                self._motion_reference.num_frames,
                self._motion_reference.num_link_to_ref,
                2,  # (position mask, rotation mask)
            ),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_links, 2) # (position mask, rotation mask)
        # 获取感兴趣链接在机器人中的 body id。
        self._body_ids = self._env.scene[cfg.asset_cfg.name].find_bodies(
            self._motion_reference.cfg.link_of_interests, preserve_order=True
        )[0]
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref, 2),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_links, 2) # (position mask, rotation mask)
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.link_pos_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.link_rot_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新链接参考命令。"""
        # 设置链接位置命令（基座坐标系）。
        self._command[env_ids, :, :, :3] = (
            self._motion_reference.data.link_pos_b[env_ids]
            * self._motion_reference.data.link_pos_mask[env_ids].unsqueeze(-1)
            * self._mask[env_ids, :, :, 0:1]
        )
        # 根据旋转模式设置链接旋转命令。
        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids, :, :, 3:7] = (
                self._motion_reference.data.link_quat_b[env_ids]
                * self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
                * self._mask[env_ids, :, :, 1:2]
            )
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids, :, :, 3:6] = (
                math_utils.axis_angle_from_quat(
                    self._motion_reference.data.link_quat_b[env_ids],
                )
                * self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
                * self._mask[env_ids, :, :, 1:2]
            )
        elif self.cfg.rotation_mode == "euler":
            euler_ = torch.stack(
                math_utils.euler_xyz_from_quat(
                    self._motion_reference.data.link_quat_b[env_ids].reshape(-1, 4),
                ),
                dim=-1,
            )
            self._command[env_ids, :, :, 3:6] = (
                euler_.reshape(-1, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, 3)
                * self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
                * self._mask[env_ids, :, :, 1:2]
            )
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids, :, :, 3:9] = (
                instinct_math_utils.quat_to_tan_norm(self._motion_reference.data.link_quat_b[env_ids])
                * self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
                * self._mask[env_ids, :, :, 1:2]
            )
        # 应用有效性掩码。
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)

        # 如果需要当前状态，计算机器人当前链接的位置和旋转（基座坐标系）。
        if self.cfg.current_state_command:
            root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w
            root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w
            root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(
                root_pos_w[env_ids], root_quat_w[env_ids]
            )
            # 当前链接位置变换到基座坐标系。
            self._current_state[env_ids, :, :, :3] = math_utils.transform_points(
                self._env.scene[self.cfg.asset_cfg.name].data.body_link_pos_w[env_ids][:, self._body_ids],
                root_pos_w_inv,
                root_quat_w_inv,
            ).unsqueeze(1)
            # 当前链接旋转变换到基座坐标系。
            current_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.body_link_quat_w[env_ids][:, self._body_ids]
            current_quat_b = math_utils.quat_mul(
                root_quat_w_inv.unsqueeze(1).expand(-1, self._motion_reference.num_link_to_ref, -1),
                current_quat_w,
            )

            # 根据旋转模式存储当前状态。
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, :, 3:7] = current_quat_b.unsqueeze(1)
            elif self.cfg.rotation_mode == "axis_angle":
                self._current_state[env_ids, :, :, 3:6] = math_utils.axis_angle_from_quat(current_quat_b).unsqueeze(1)
            elif self.cfg.rotation_mode == "euler":
                self._current_state[env_ids, :, :, 3:6] = torch.stack(
                    math_utils.euler_xyz_from_quat(current_quat_b),
                    dim=1,
                )
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :, 3:9] = instinct_math_utils.quat_to_tan_norm(
                    current_quat_b
                ).unsqueeze(1)

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if not hasattr(self, "_vis_link_pos_w"):
            return
        self._visualizer.visualize(
            translations=self._vis_link_pos_w,
            orientations=self._vis_link_quat_w,
        )

    def _compute_debug_vis_data(self):
        """计算调试可视化数据。"""
        root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w

        # 由于该命令定义在机器人本地坐标系中，我们需要将命令数据变换到世界坐标系。
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]
        num_links = keyframe_command.shape[1]
        # 将链接位置变换到世界坐标系。
        link_pos_w = math_utils.transform_points(
            keyframe_command[:, :, :3],
            root_pos_w,
            root_quat_w,
        )
        # 根据旋转模式将链接旋转变换到世界坐标系。
        if self.cfg.rotation_mode == "quaternion":
            link_quat_w = math_utils.quat_mul(
                root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
                keyframe_command[:, :, 3:7],
            )
        elif self.cfg.rotation_mode == "axis_angle":
            link_rot_b = keyframe_command[:, :, 3:6].reshape(-1, 3)
            link_quat_b = math_utils.quat_from_angle_axis(torch.norm(link_rot_b, dim=-1), link_rot_b)
            link_quat_w = math_utils.quat_mul(
                root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
                link_quat_b.reshape(-1, num_links, 4),
            )
        elif self.cfg.rotation_mode == "euler":
            link_rot_b = keyframe_command[:, :, 3:6].reshape(-1, 3)
            link_quat_b = math_utils.quat_from_euler_xyz(
                link_rot_b[..., 0],  # pitch
                link_rot_b[..., 1],  # roll
                link_rot_b[..., 2],  # yaw
            )
            link_quat_w = math_utils.quat_mul(
                root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
                link_quat_b.reshape(-1, num_links, 4),
            )
        elif self.cfg.rotation_mode == "tannorm":
            link_rot_b = keyframe_command[:, :, 3:9].reshape(-1, 6)
            link_quat_b = instinct_math_utils.tan_norm_to_quat(link_rot_b)
            link_quat_w = math_utils.quat_mul(
                root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
                link_quat_b.reshape(-1, num_links, 4),
            )
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")

        # 保存可视化数据。
        self._vis_link_pos_w = link_pos_w.reshape(-1, 3)
        self._vis_link_quat_w = link_quat_w.reshape(-1, 4)


# 定义 LinkPosRefCommand 类，继承自 ShadowingCommandBase，仅包含链接位置参考。
class LinkPosRefCommand(ShadowingCommandBase):
    """链接位置参考命令。"""

    cfg: LinkPosRefCommandCfg

    def __init__(self, cfg: LinkPosRefCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 数据维度为 3（位置）。
        data_dims = (3,)
        # 初始化命令张量。
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, *data_dims),
            device=self.device,
        )
        # 初始化掩码张量，形状为 (num_envs, num_frames, num_links)。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_links) # (position mask)
        # 获取感兴趣链接的 body id。
        self._body_ids = self._env.scene[cfg.asset_cfg.name].find_bodies(
            self._motion_reference.cfg.link_of_interests, preserve_order=True
        )[0]
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_links, 1) # (position mask)
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(mask, self._motion_reference.data.link_pos_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新链接位置命令。"""
        # 设置链接位置命令（基座坐标系），并应用掩码和有效性。
        self._command[env_ids, :, :, :] = (
            self._motion_reference.data.link_pos_b[env_ids]
            * self._motion_reference.data.link_pos_mask[env_ids].unsqueeze(-1)
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)
            * self._mask[env_ids].unsqueeze(-1)
        )

        # 如果需要当前状态，计算机器人当前链接位置（基座坐标系）。
        if self.cfg.current_state_command:
            root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids]
            root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids]
            root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
            self._current_state[env_ids] = math_utils.transform_points(
                self._env.scene[self.cfg.asset_cfg.name].data.body_pos_w[env_ids][:, self._body_ids],
                root_pos_w_inv,
                root_quat_w_inv,
            ).unsqueeze(1)

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if not hasattr(self, "_vis_link_pos_w"):
            return
        self._visualizer.visualize(
            translations=self._vis_link_pos_w,
        )

    def _compute_debug_vis_data(self):
        """计算调试可视化数据。"""
        root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w

        # 将命令位置变换到世界坐标系用于可视化。
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]
        link_pos_w = math_utils.transform_points(
            keyframe_command[:, :, :3],
            root_pos_w,
            root_quat_w,
        )
        self._vis_link_pos_w = link_pos_w.reshape(-1, 3)


# 定义 LinkRotRefCommand 类，继承自 ShadowingCommandBase，仅包含链接旋转参考。
class LinkRotRefCommand(ShadowingCommandBase):
    """链接旋转参考命令。"""

    cfg: LinkRotRefCommandCfg

    def __init__(self, cfg: LinkRotRefCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 根据旋转模式确定数据维度。
        if self.cfg.rotation_mode == "quaternion":
            data_dims = (4,)
        elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
            data_dims = (3,)
        elif self.cfg.rotation_mode == "tannorm":
            data_dims = (6,)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        # 初始化命令张量。
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, *data_dims),
            device=self.device,
        )
        # 初始化掩码张量。
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_links, 1) # (rotation mask)
        # 获取感兴趣链接的 body id。
        self._body_ids = self._env.scene[cfg.asset_cfg.name].find_bodies(
            self._motion_reference.cfg.link_of_interests, preserve_order=True
        )[0]
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.num_link_to_ref),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_links, 1) # (rotation mask)
        # 更新命令。
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        """返回命令的掩码。"""
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(mask, self._motion_reference.data.link_rot_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新链接旋转命令。"""
        # 获取参考链接旋转（基座坐标系）。
        link_quat_ref_b = self._motion_reference.data.link_quat_b[env_ids]
        # 根据旋转模式存储命令。
        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids] = link_quat_ref_b
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids] = math_utils.axis_angle_from_quat(link_quat_ref_b)
        elif self.cfg.rotation_mode == "euler":
            link_quat_ref_b_ = link_quat_ref_b.reshape(-1, 4)
            self._command[env_ids] = torch.stack(
                math_utils.euler_xyz_from_quat(link_quat_ref_b_),
                dim=-1,
            ).reshape(-1, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, 3)
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids] = instinct_math_utils.quat_to_tan_norm(link_quat_ref_b)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        # 应用掩码和有效性。
        self._command[env_ids] *= self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids].unsqueeze(-1)

        # 如果需要当前状态，计算机器人当前链接旋转（基座坐标系）。
        if self.cfg.current_state_command:
            current_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.body_quat_w[env_ids]
            current_quat_w = current_quat_w[:, self._body_ids]
            current_quat_b = math_utils.quat_mul(
                self._env.scene[self.cfg.asset_cfg.name]
                .data.root_quat_w[env_ids]
                .unsqueeze(1)
                .expand(-1, self._motion_reference.num_link_to_ref, -1),
                current_quat_w,
            )
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, 0] = current_quat_b
            elif self.cfg.rotation_mode == "axis_angle":
                self._current_state[env_ids, 0] = math_utils.axis_angle_from_quat(current_quat_b)
            elif self.cfg.rotation_mode == "euler":
                self._current_state[env_ids, 0] = math_utils.euler_xyz_from_quat(current_quat_b)
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, 0] = instinct_math_utils.quat_to_tan_norm(current_quat_b)


# 定义 LinkPosErrRefCommand 类，继承自 LinkPosRefCommand，包含链接位置误差。
class LinkPosErrRefCommand(LinkPosRefCommand):
    """链接位置误差命令。"""

    cfg: LinkPosErrRefCommandCfg

    def __init__(self, cfg: LinkPosErrRefCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新链接位置误差命令。"""
        # 获取机器人当前链接位置（世界坐标系）。
        link_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.body_pos_w[:, self._body_ids]
        # 计算基座坐标系的逆变换。
        root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(
            self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids],
            self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids],
        )
        # 将当前链接位置变换到基座坐标系。
        link_pos_b = math_utils.transform_points(
            link_pos_w[env_ids],
            root_pos_w_inv,
            root_quat_w_inv,
        )
        # 计算参考位置与当前位置的误差。
        self._command[env_ids] = self._motion_reference.data.link_pos_b[env_ids] - link_pos_b.unsqueeze(1)
        # 应用掩码和有效性。
        self._command[env_ids] *= self._motion_reference.data.link_pos_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids].unsqueeze(-1)

    def _debug_vis_callback(self, event):
        """调试可视化回调。"""
        if not hasattr(self, "_vis_err_pos"):
            return
        # 可视化误差方向。
        self._visualizer.visualize(
            translations=self._vis_err_pos,
            orientations=self._vis_err_quat,
            scales=self._vis_err_scale,
        )

    def _compute_debug_vis_data(self):
        """计算调试可视化数据。"""
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w
        link_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.body_pos_w[:, self._body_ids]

        # 获取世界坐标系下的误差方向。
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]  # (n_envs, n_links, 3)
        num_links = keyframe_command.shape[1]
        # 将误差方向从基座坐标系变换到世界坐标系。
        link_pos_err_w = math_utils.quat_apply(
            root_quat_w.unsqueeze(1).expand(-1, num_links, -1),
            keyframe_command,
        ).reshape(-1, 3)
        # 将误差方向转换为箭头可视化。
        err_scale, err_quat, err_pos = self._resolve_direction_to_arrow(link_pos_err_w, link_pos_w.reshape(-1, 3))

        self._vis_err_pos = err_pos
        self._vis_err_quat = err_quat
        self._vis_err_scale = err_scale

    def _resolve_direction_to_arrow(
        self,
        direction: torch.Tensor,
        start_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算箭头的方向、缩放和位置。
        Args:
            direction: 箭头方向，形状 (N, 3)。
            start_point: 箭头起点，形状 (N, 3)。
        """
        # 获取标记的默认缩放。
        default_scale = self._visualizer.cfg.markers["arrow"].scale
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        # 归一化方向。
        normalized_direction = direction / torch.norm(direction, dim=-1, keepdim=True)
        # 箭头缩放。
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(direction.shape[0], 1)
        arrow_scale[:, 0] *= torch.norm(direction, dim=-1) * 10
        # 箭头方向（四元数）。
        axis = torch.cross(default_direction, normalized_direction, dim=-1)
        dot_prod_ = torch.sum(default_direction * normalized_direction, dim=-1)
        angle = torch.acos(torch.clamp(dot_prod_, -1.0, 1.0))
        arrow_quat = math_utils.quat_from_angle_axis(
            angle,
            axis,
        )
        # 箭头位置。
        # 注意：由于箭头文件的 USD 原点既不在起点也不在中间，
        # 而是在从起点到终点的箭头长度的 0.25 处。
        arrow_pos = start_point + 0.25 * direction

        return arrow_scale, arrow_quat, arrow_pos


# 定义 LinkRotErrRefCommand 类，继承自 LinkRotRefCommand，包含链接旋转误差。
class LinkRotErrRefCommand(LinkRotRefCommand):
    """链接旋转误差命令。"""

    cfg: LinkRotErrRefCommandCfg

    def __init__(self, cfg: LinkRotErrRefCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新链接旋转误差命令。"""
        # 获取机器人当前链接四元数（世界坐标系）。
        link_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.body_quat_w[env_ids]
        link_quat_w = link_quat_w[:, self._body_ids]  # (n_envs, n_links, 4)
        # 根据是否在基座坐标系中，计算参考四元数和当前四元数。
        if self.cfg.in_base_frame:
            # 将链接四元数变换到基座坐标系。
            root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids]
            link_quat_ref_ = self._motion_reference.data.link_quat_b[env_ids]
            link_quat_ = math_utils.quat_mul(
                math_utils.quat_inv(root_quat_w.unsqueeze(1).expand(-1, self._motion_reference.num_link_to_ref, -1)),
                link_quat_w,
            )
        else:
            # 使用世界坐标系下的参考和当前四元数。
            link_quat_ref_ = self._motion_reference.data.link_quat_w[env_ids]
            link_quat_ = link_quat_w
        # 计算误差四元数，即参考四元数在当前链接四元数坐标系下的表示。
        link_quat_err = math_utils.quat_mul(
            math_utils.quat_inv(link_quat_).unsqueeze(1).expand(-1, self._motion_reference.num_frames, -1, -1),
            link_quat_ref_,
        )
        # 根据旋转模式存储误差命令。
        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids] = link_quat_err
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids] = math_utils.axis_angle_from_quat(link_quat_err)
        elif self.cfg.rotation_mode == "euler":
            link_quat_err_ = link_quat_err.reshape(-1, 4)
            self._command[env_ids] = torch.stack(
                math_utils.euler_xyz_from_quat(link_quat_err_),
                dim=-1,
            ).reshape(-1, self._motion_reference.num_frames, self._motion_reference.num_link_to_ref, 3)
            # (n_envs, n_frames, n_links, 3)
            # 欧拉角按 xyz 顺序
            # (pitch, roll, yaw)
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids] = instinct_math_utils.quat_to_tan_norm(link_quat_err)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        # 应用掩码和有效性。
        self._command[env_ids] *= self._motion_reference.data.link_rot_mask[env_ids].unsqueeze(-1)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)
        self._command[env_ids] *= self._mask[env_ids].unsqueeze(-1)

        # 如果需要当前状态，误差为零，设置单位旋转。
        if self.cfg.current_state_command:
            # 所有误差为零。
            self._current_state[env_ids] = 0.0
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, :, -1] = 1
            elif self.cfg.rotation_mode in ("axis_angle", "euler"):
                pass  # 已经设置为零
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :, 0] = 1
                self._current_state[env_ids, :, :, 5] = 1
            else:
                raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")


# 定义 TimeToTargetCommand 类，继承自 ShadowingCommandBase，包含到达目标的时间。
class TimeToTargetCommand(ShadowingCommandBase):
    """到达目标时间的命令。"""

    cfg: TimeToTargetCommandCfg

    def __init__(self, cfg: TimeToTargetCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项。

        ## 注意
            该命令没有掩码。无效值应设置为 -1。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 调用基类初始化。
        super().__init__(cfg, env)
        # 数据维度为 1（时间）。
        data_dims = (1,)
        # 初始化命令张量。
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        # 如果需要当前状态，则生成状态缓冲区。
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
        # 更新命令。
        self._update_command()

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """为指定环境索引更新到达目标时间命令。"""
        # 计算到达目标帧的时间减去从上次更新经过的时间。
        self._command[env_ids, :, 0] = self._motion_reference.data.time_to_target_frame[
            env_ids
        ] - self._motion_reference.time_passed_from_update[env_ids].unsqueeze(-1)