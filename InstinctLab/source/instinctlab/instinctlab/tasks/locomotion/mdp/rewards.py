# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.

Locomotion（行走）任务的奖励函数。每个函数返回形状为 [num_envs] 的张量，
即每个并行环境一个原始得分；权重由环境配置中的 RewTerm.weight 指定。
返回非负误差的 penalty 函数通常搭配负权重，返回相似度的函数通常搭配正权重。
函数存在不代表已启用，实际以具体任务的 rewards 配置为准。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stand_still_joint_penalty(
    env, command_name: str, asset_cfg: SceneEntityCfg,
    velocity: bool = False, linear_threshold: float = 0.1, angular_threshold: float = 0.1,
) -> torch.Tensor:
    """零速命令下的关节惩罚，配负权重使用；正常原地转向不启用。

    velocity=False 惩罚偏离默认姿态的平方和；True 惩罚关节速度平方和。
    """
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.vector_norm(command[:, :2], dim=1) < linear_threshold) & (
        command[:, 2].abs() < angular_threshold
    )
    asset = env.scene[asset_cfg.name]
    if velocity:
        error = asset.data.joint_vel[:, asset_cfg.joint_ids]
    else:
        error = (asset.data.joint_pos - asset.data.default_joint_pos)[:, asset_cfg.joint_ids]
    return error.square().sum(dim=1) * standing


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.

    单脚支撑的持续奖励：仅在恰好一只脚接触地面时给分。
    threshold 是得分上限（秒），并非允许腾空的最长时间；超过后仍持续给满分。
    只根据平移命令开关，原地转向不会单独激活该奖励。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 两脚当前连续腾空/接触的时长，形状为 [环境数, 脚数]，单位为秒。
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    # 落地脚取接触时长，腾空脚取腾空时长，再取两者中较短的时间。
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # 平移指令速度不超过 0.1 m/s 时不给分；这里不是实际运动速度。
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_contact_symmetry_penalty(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    max_time: float = 1.0,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize left-right asymmetry in completed biped foot phases.

    The selected contact sensor bodies must contain exactly two feet in the
    order left, right.  The term compares the most recently completed air and
    contact durations.  Initial zero values are ignored until both feet have
    completed the corresponding phase, and the penalty is disabled when the
    commanded xy speed is below ``command_threshold``.

    左右脚上一次已完成的腾空/接触时长差的平方和，搭配负权重使用。
    必须选中恰好两只脚，并启用传感器的 track_air_time。
    此项不直接惩罚正在持续的超长腾空，也不要求双脚交替接触。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError(
            "feet_air_contact_symmetry_penalty requires "
            "ContactSensorCfg(track_air_time=True)."
        )

    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]

    if last_air_time.shape[1] != 2:
        raise ValueError(
            "feet_air_contact_symmetry_penalty expects exactly two foot bodies "
            "in sensor_cfg.body_ids."
        )

    # 截断用于限制极端时长的影响，但两脚均超过 max_time 时会丢失差异。
    air_time = torch.clamp(last_air_time, min=0.0, max=max_time)
    contact_time = torch.clamp(last_contact_time, min=0.0, max=max_time)

    # 两脚都完成过相应阶段才比较，忽略初始化时的零值。
    air_valid = torch.all(last_air_time > 0.0, dim=1)
    contact_valid = torch.all(last_contact_time > 0.0, dim=1)

    penalty = torch.square(air_time[:, 0] - air_time[:, 1]) * air_valid
    penalty += torch.square(contact_time[:, 0] - contact_time[:, 1]) * contact_valid

    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > command_threshold
    return penalty * moving


def both_feet_airborne_penalty(
    env, sensor_cfg: SceneEntityCfg, contact_threshold: float = 1.0
) -> torch.Tensor:
    """双脚同时离地时返回 1，否则返回 0，配负权重用于抑制步行中的跳跃。

    使用接触力历史中的最大模长判断每只脚是否接触（阈值单位 N）。
    只要历史窗口内任一脚仍有接触，就不判为双脚腾空，减少短暂丢失接触
    的误罚；窗口长度取决于传感器 history_length 和更新周期。
    对所有速度命令生效，包括静止与原地转向；不适用于需要腾空的翻越任务。
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    if forces.shape[2] != 2:
        raise ValueError("both_feet_airborne_penalty requires exactly two foot bodies.")
    contacts = forces.norm(dim=-1).amax(dim=1) > contact_threshold
    return (~contacts.any(dim=1)).float()


def feet_gait(
    env,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward contact phases matching a periodic alternating gait.

    This follows Unitree RL Lab's G1 gait reward: ``offset`` contains one
    phase offset per selected foot, normally ``[0.0, 0.5]`` for a biped.

    周期接触奖励（沿用 Unitree 逻辑）：period 为完整周期的秒数，offset
    为每只脚的归一化相位偏移，顺序必须与 sensor_cfg.body_ids 一致。
    threshold 为支撑阶段占比，例如 0.55 表示一个周期中 55% 时间应接触。
    每只脚独立计分：两脚均符合目标返回 2，只有一脚符合仍返回 1。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    # 从 episode 已运行时间计算 [0, 1) 相位；此处计算不会自动添加相位观测。
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = [(global_phase + phase_offset) % 1.0 for phase_offset in offset]
    leg_phase = torch.cat(phases, dim=-1)
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for foot_idx in range(len(sensor_cfg.body_ids)):
        # 目标支撑/腾空状态与实际接触状态一致，就给该脚 1 分。
        is_stance = leg_phase[:, foot_idx] < threshold
        reward += (is_stance == is_contact[:, foot_idx]).float()
    if command_name is not None:
        # 使用整个速度指令向量的范数，包含转向；与上面的平移开关不同。
        command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= (command_norm > 0.1).float()
    return reward


def foot_clearance_reward(
    env,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """运动脚高度奖励，沿用 Unitree 的指数误差形式。

    target_height：脚部连杆原点的世界 z 坐标目标（米），不是脚底到地面的距离。
    std：指数分母中的误差尺度；这里直接除以 std，而非 std**2。
    tanh_mult：水平脚速权重的饱和速度由此参数控制。
    未按接触或步态相位筛选脚，也没有速度指令开关；两脚水平速度为零时
    误差加权后为零，返回 1，因此它本身不能保证抬脚或左右交替。
    """
    asset = env.scene[asset_cfg.name]
    foot_z_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # 水平运动越快，脚高误差的权重越大；tanh 将权重限制在 [0, 1)。
    foot_speed = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    error = foot_z_error * foot_speed
    return torch.exp(-torch.sum(error, dim=1) / std)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """水平速度跟踪奖励：将世界速度转到仅随机器人 yaw 旋转的坐标系。

    这样排除 roll/pitch 对水平速度比较的影响。误差为 xy 两方向平方和，
    以 exp(-误差/std**2) 给分；完全跟随时为 1，std 越小要求越严格。
    此项只评价速度，不区分走路、滑行或跳跃。
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """转向角速度跟踪：比较指令与基座世界坐标系下的 z 轴角速度。

    使用 exp(-误差平方/std**2)，完全跟随时得 1；不等同于跟踪目标朝向角。
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """静止指令下的关节姿态偏离量，通常配负权重作为惩罚。

    平移命令范数和转向命令绝对值均小于 0.1 才激活。
    返回所有关节相对默认角度的绝对误差之和，并非直接检测是否站稳。
    """
    asset = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    return (
        dof_error
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1)
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < 0.1)
    )
