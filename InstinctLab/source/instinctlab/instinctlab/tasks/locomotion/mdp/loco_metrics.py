"""Training diagnostics for the unified G1 locomotion teacher."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from instinctlab.monitors import MonitorTerm


class LocoTrackingMonitorTerm(MonitorTerm):
    """Report command coverage, tracking errors, falls, and action/joint statistics."""

    _BIN_NAMES = ("low", "mid", "high")

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._log: dict[str, torch.Tensor] = {}

    @staticmethod
    def _mean(value: torch.Tensor) -> torch.Tensor:
        return value.mean() if value.numel() else torch.zeros((), device=value.device)

    def update(self, *args, **kwargs):
        command = self._env.command_manager.get_command("base_velocity")
        robot = self._env.scene["robot"]
        velocity_yaw = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])
        error = velocity_yaw[:, :2] - command[:, :2]
        yaw_error = robot.data.root_ang_vel_w[:, 2] - command[:, 2]
        raw_action = self._env.action_manager.action
        joint_error = robot.data.joint_pos - robot.data.default_joint_pos

        log: dict[str, torch.Tensor] = {}
        for index, axis in enumerate(("vx", "vy", "wz")):
            values = command[:, index]
            log[f"command/{axis}_mean"] = values.mean()
            log[f"command/{axis}_std"] = values.std(unbiased=False)
            log[f"command/{axis}_min"] = values.min()
            log[f"command/{axis}_max"] = values.max()

        log["tracking/vx_rmse"] = torch.sqrt(torch.mean(error[:, 0].square()))
        log["tracking/vy_rmse"] = torch.sqrt(torch.mean(error[:, 1].square()))
        log["tracking/vxy_rmse"] = torch.sqrt(torch.mean(torch.sum(error.square(), dim=1)))
        log["tracking/wz_rmse"] = torch.sqrt(torch.mean(yaw_error.square()))
        log["fall_rate_step"] = self._env.reset_terminated.float().mean()
        log["action/mean"] = raw_action.mean()
        log["action/std"] = raw_action.std(unbiased=False)
        log["action/abs_max"] = raw_action.abs().max()

        for index, joint_name in enumerate(robot.joint_names):
            log[f"joint_error_rmse/{joint_name}"] = torch.sqrt(torch.mean(joint_error[:, index].square()))

        ranges = ((-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0))
        for axis_index, (axis, limits) in enumerate(zip(("vx", "vy", "wz"), ranges)):
            edges = torch.linspace(limits[0], limits[1], 4, device=command.device)
            for bin_index, bin_name in enumerate(self._BIN_NAMES):
                if bin_index == 2:
                    mask = (command[:, axis_index] >= edges[bin_index]) & (
                        command[:, axis_index] <= edges[bin_index + 1]
                    )
                else:
                    mask = (command[:, axis_index] >= edges[bin_index]) & (
                        command[:, axis_index] < edges[bin_index + 1]
                    )
                log[f"command_bins/{axis}_{bin_name}_fraction"] = mask.float().mean()
                log[f"command_bins/{axis}_{bin_name}_vxy_rmse"] = torch.sqrt(
                    self._mean(torch.sum(error[mask].square(), dim=1))
                )
                log[f"command_bins/{axis}_{bin_name}_wz_rmse"] = torch.sqrt(
                    self._mean(yaw_error[mask].square())
                )

        self._log = log

    def get_log(self, is_episode: bool = False) -> dict[str, torch.Tensor]:
        return self._log
