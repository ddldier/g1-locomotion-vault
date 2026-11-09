# 启用延迟注解（PEP 563）：类型注解不会在运行时立即求值，
# 因此可以安全地引用仅在 TYPE_CHECKING 下导入的类。
from __future__ import annotations

# 导入 Gymnasium，用于处理动作空间、观测空间等标准接口。
import gymnasium as gym
# 导入 PyTorch，用于张量运算和设备管理。
import torch
# 导入类型检查工具：TYPE_CHECKING 用于避免运行时导入，Dict 用于类型标注。
from typing import TYPE_CHECKING, Dict

# 从 Isaac Lab 导入两种常用环境基类：
# - DirectRLEnv：直接实现 RL 的环境；
# - ManagerBasedRLEnv：基于管理器（观测、动作、奖励等）的环境。
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

# 仅在静态类型检查时导入配置类，避免运行时循环导入。
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

# 从 Instinct-RL 导入向量化环境基类 VecEnv。
from instinct_rl.env import VecEnv


# 该类把 Isaac Lab 环境包装成 Instinct-RL 所需的 VecEnv 接口。
class InstinctRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for Instinct-RL library
    Reference:
       https://github.com/project-instinct/instinct_rl/blob/master/instinct_rl/env/vec_env.py
    """

    # 初始化包装器。
    def __init__(self, env: ManagerBasedRLEnv):
        """Initializes the wrapper.
        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.
        Args:
            env: The environment to wrap around.
        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # 检查输入环境是否合法：其最底层未包装环境必须是 ManagerBasedRLEnv 或 DirectRLEnv。
        # 如果不是，则抛出 ValueError，并打印实际环境类型。
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # 保存被包装的环境对象。
        # 注意：这里保存的是传入的 env，而不是 env.unwrapped，以保留外层包装。
        self.env = env
        # 保存包装器需要暴露的信息。
        # 环境数量，来自底层未包装环境。
        self.num_envs = self.unwrapped.num_envs
        # 运行设备（CPU/GPU），来自底层未包装环境。
        self.device = self.unwrapped.device
        # 每个 episode 的最大长度，来自底层未包装环境。
        self.max_episode_length = self.unwrapped.max_episode_length
        # 如果底层环境有 action_manager，则动作维度从 action_manager 获取。
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        # 否则从单环境动作空间展平后计算动作维度。
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        # 如果底层环境有 observation_manager，则从观测管理器获取 policy 组的观测维度。
        if hasattr(self.unwrapped, "observation_manager"):
            # 获取 policy 组中每个观测项的维度元组。
            num_obs_dims = self.unwrapped.observation_manager.group_obs_term_dim["policy"]
            # 对每个观测项维度求乘积，得到该项展平后的元素数量。
            num_obs_dims = [torch.prod(torch.tensor(dim, device="cpu")).item() for dim in num_obs_dims]
            # 求和得到 policy 观测总维度。
            self.num_obs = int(sum(num_obs_dims))
        # 否则从单环境观测空间的 policy 键展平计算观测维度。
        else:
            # Not checked for DiectRlEnv
            # 注意：这里没有对 DirectRLEnv 做额外检查。
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        # 处理特权观测（critic 观测），用于非对称 actor-critic。
        # 如果底层环境有 observation_manager，并且其观测组维度中包含 "critic" 组。
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            # 获取 critic 组中每个观测项的维度元组。
            num_obs_dims = self.unwrapped.observation_manager.group_obs_term_dim["critic"]
            # 对每个观测项维度求乘积，得到该项展平后的元素数量。
            num_obs_dims = [torch.prod(torch.tensor(dim, device="cpu")).item() for dim in num_obs_dims]
            # 求和得到 critic 观测总维度。
            self.num_critic_obs = int(sum(num_obs_dims))
        # 否则，如果底层环境有 num_states，并且单环境观测空间中包含 "critic" 键。
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            # Not checked for DiectRlEnv
            # 从单环境观测空间的 critic 键展平计算维度。
            self.num_critic_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        # 如果以上条件都不满足，则没有 critic 观测。
        else:
            self.num_critic_obs = None
        # reset at the start since the Instinct-RL runner does not call reset
        # 在初始化时立即重置环境，因为 Instinct-RL 的 runner 不会主动调用 reset。
        self.env.reset()

    # 返回包装器名称和内部环境表示字符串。
    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    # 返回对象的字符串表示，直接复用 __str__。
    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    # 返回底层环境的配置对象。
    @property
    def cfg(self) -> ManagerBasedRLEnvCfg:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    # 返回底层环境的渲染模式。
    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    # 返回底层环境的观测空间。
    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    # 返回底层环境的动作空间。
    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    # 返回包装器类名。
    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    # 返回最底层未包装环境。
    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    # 获取当前观测。
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        # 如果底层环境有 observation_manager，则通过管理器计算观测。
        if hasattr(self.unwrapped, "observation_manager"):
            obs_pack = self.unwrapped.obs_buf
        # 否则调用底层环境的 _get_observations。
        else:
            obs_pack = self.unwrapped._get_observations()
        # 展平所有观测组中的观测项，保证每个组是拼接后的张量。
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        # 返回 policy 观测，以及包含所有观测组的字典。
        return obs_pack["policy"], {"observations": obs_pack}

    # 获取 episode 长度缓冲区。
    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    # 设置 episode 长度缓冲区。
    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    # 返回奖励项数量。
    @property
    def num_rewards(self) -> int:
        return self.unwrapped.num_rewards

    """
    Operations - MDP
    """

    # 设置随机种子。
    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    # 重置环境。
    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        # 调用内部环境 reset，返回观测包和 info（这里忽略 info）。
        obs_pack, _ = self.env.reset()
        # 展平所有观测组。
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        # return observations
        # 返回 policy 观测，以及包含所有观测组的字典。
        return obs_pack["policy"], {"observations": obs_pack}

    # 执行一步环境交互。
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        # 调用内部环境 step，得到观测包、奖励、终止标志、截断标志和额外信息。
        obs_pack, rew, terminated, truncated, extras = self.env.step(actions)
        # 展平所有观测组。
        obs_pack = self._flatten_all_obs_groups(obs_pack)
        # compute dones for compatibility with RSL-RL
        # 计算 done 标志：终止或截断任一为真，并转换为 long 类型，兼容 RSL-RL。
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict

        # 把所有观测组放入 extras 字典。
        extras["observations"] = obs_pack
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        # 如果不是有限视界任务，把截断标志作为 time_outs 放入 extras，用于无限视界任务。
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        # 取出 policy 观测。
        obs = obs_pack["policy"]
        # 如果奖励是字典，说明来自多奖励管理器。
        if isinstance(rew, dict):
            # returned by multi-reward manager
            # 将多个奖励项堆叠成 [batch, num_rewards] 形状。
            rew = self._stack_rewards(rew)
        else:
            # returned by regular reward manager
            # make sure rewards are always in shape of [batch, num_rewards]
            # 普通奖励管理器返回的是 [batch]，这里扩展为 [batch, 1]。
            rew = rew.unsqueeze(1)
        # 返回观测、奖励、done 和额外信息。
        return obs, rew, dones, extras

    # 关闭环境。
    def close(self):  # noqa: D102
        return self.env.close()

    """
    Operations -- Instinct-RL
    """

    # 获取指定观测组的观测分段信息。
    def get_obs_segments(self, group_name: str = "policy"):
        # 获取该观测组中活跃的观测项名称。
        obs_term_names = self.unwrapped.observation_manager.active_terms[group_name]
        # 获取该观测组中每个观测项的维度。
        obs_term_dims = self.unwrapped.observation_manager.group_obs_term_dim[group_name]
        # 调用内部方法构建名称到维度的映射。
        return self._get_obs_segments(obs_term_names, obs_term_dims)

    # 根据观测项名称和维度构建观测分段字典。
    def _get_obs_segments(self, obs_term_names, obs_term_dims):
        # assuming the computed obs_term_dim is in the same order as the obs_cfg
        # From Python 3.6+, dictionaries are ordered by insertion order
        # 假设计算得到的 obs_term_dim 顺序与 obs_cfg 中一致。
        # 从 Python 3.6+ 开始，字典按插入顺序保持有序。
        obs_segments = dict()
        # 按顺序配对观测项名称和维度。
        for term_name, term_dim in zip(obs_term_names, obs_term_dims):
            obs_segments[term_name] = term_dim

        return obs_segments

    # 返回所有观测组的格式信息。
    def get_obs_format(self) -> dict[str, dict[str, tuple]]:
        """Returns the observation information for all observation groups.
        Using this interface, so that, the algorithm / policy should not access env directly.
        But let the runner access env to get critical information.
        """
        # 初始化观测格式字典。
        obs_format = dict()
        # 遍历所有活跃观测组。
        for group_name in self.unwrapped.observation_manager.active_terms.keys():
            # 记录该组的观测分段信息。
            obs_format[group_name] = self.get_obs_segments(group_name)
        return obs_format

    """
    Internal Helpers
    """

    # 展平单个观测组。
    def _flatten_obs_group(self, obs_group: dict) -> torch.Tensor:
        """Considering observation_manager only concatenate observation terms of 1D tensors,
        this function flattens the observation terms of different shape and concatenate into
        a single tensor.
        """
        # 存储展平后的各观测项。
        obs = []
        # 遍历观测组中的每个观测项及其值。
        for obs_term, obs_value in obs_group.items():
            # 从第 1 维开始展平，保留 batch 维。
            obs.append(obs_value.flatten(start_dim=1))
        # 沿第 1 维拼接所有观测项。
        obs = torch.cat(obs, dim=1)
        return obs

    # 展平所有观测组。
    def _flatten_all_obs_groups(self, obs_pack: dict) -> dict:
        # 创建新的观测包字典。
        obs_pack_ = dict()
        # 遍历每个观测组。
        for obs_group_name, obs_group in obs_pack.items():
            # 如果该组是字典，则调用 _flatten_obs_group 展平；否则保持原值。
            obs_pack_[obs_group_name] = self._flatten_obs_group(obs_group) if isinstance(obs_group, dict) else obs_group
        return obs_pack_

    # 将多个奖励项堆叠为张量。
    def _stack_rewards(self, rewards_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # 存储各奖励项。
        rewards = []
        # 遍历奖励字典。
        for reward_term, reward_value in rewards_dict.items():
            rewards.append(reward_value)
        # 沿最后一维堆叠，得到 [batch, num_rewards]。
        rewards = torch.stack(rewards, dim=-1)
        return rewards
