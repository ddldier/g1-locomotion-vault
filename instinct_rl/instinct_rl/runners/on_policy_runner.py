# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# 导入 importlib，用于动态导入模块（后面会根据类名动态加载算法类）。
import importlib
# 导入 os，用于路径操作（保存/加载模型、创建目录等）。
import os
# 导入 statistics，用于计算均值、中位数等统计量。
import statistics
# 导入 time，用于计时（收集时间、学习时间、总时间等）。
import time
# 导入 deque，双端队列，用于维护固定长度的缓冲区（奖励、episode 长度等）。
from collections import deque

# 导入 PyTorch 主库。
import torch
# 导入 PyTorch 分布式训练模块，用于多进程训练（DDP、all_reduce 等）。
import torch.distributed as dist
# 导入 TensorBoardX 的 SummaryWriter，用于记录训练日志。
from tensorboardX import SummaryWriter

# 导入 instinct_rl 顶层包。
import instinct_rl
# 导入 instinct_rl.algorithms 模块，包含各种算法（如 PPO）。
import instinct_rl.algorithms as algorithms
# 导入 instinct_rl.modules 模块，包含网络模块、归一化器等构建函数。
import instinct_rl.modules as modules
# 从 instinct_rl.env 导入 VecEnv 类型，用于类型标注。
from instinct_rl.env import VecEnv
# 从 instinct_rl.utils 导入 ckpt_manipulator，用于检查点加载时的“操作器”（可能用于兼容旧模型）。
from instinct_rl.utils import ckpt_manipulator
# 从 instinct_rl.utils.utils 导入 get_subobs_size 和 store_code_state。
# get_subobs_size 用于计算观测子组展平后的总维度；
# store_code_state 用于记录代码仓库状态（git commit 等）。
from instinct_rl.utils.utils import get_subobs_size, store_code_state


# OnPolicyRunner 类是 on-policy 算法的训练运行器，负责与环境交互、收集数据、更新策略、日志和保存模型。
class OnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        """
        ## Args:
            - mp_stat_all_reduce: All reduce statistics in multi-processing if True. NOTE: make sure statistics across all processes have
                the same items, otherwise it will raise an error.
        """
        # 保存训练配置字典。
        self.cfg = train_cfg
        # 取出算法配置部分。
        self.alg_cfg = train_cfg["algorithm"]
        # 取出策略配置部分。
        self.policy_cfg = train_cfg["policy"]
        # 保存运行设备（CPU 或 GPU）。
        self.device = device
        # 保存向量化环境。
        self.env = env

        # 从环境获取观测格式信息（各观测组的每个观测项名称与维度）。
        obs_format = env.get_obs_format()

        # 根据策略配置构建 actor-critic 网络。
        # 注意：这里 pop("class_name") 会从 policy_cfg 中移除 class_name，避免后续重复传递。
        actor_critic = modules.build_actor_critic(
            self.policy_cfg.pop("class_name"),
            self.policy_cfg,
            obs_format,
            num_actions=env.num_actions,
            num_rewards=env.num_rewards,
        ).to(self.device)

        # 取出算法类名。
        alg_class_name = self.alg_cfg.pop("class_name")
        # 如果类名中包含冒号，则可能是 "module:class" 的形式，使用 importlib 动态导入；
        # 否则从 instinct_rl.algorithms 模块中按名称获取类。
        alg_class = importlib.import_module() if ":" in alg_class_name else getattr(algorithms, alg_class_name)
        # 实例化算法对象（例如 PPO），传入 actor-critic、设备和算法配置。
        self.alg: algorithms.PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        # 每个环境每次收集的步数。
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        # 每隔多少次迭代保存一次模型。
        self.save_interval = self.cfg["save_interval"]

        # handle normalizers if needed
        # 处理观测归一化器（如果配置中指定了 normalizers）。
        self.normalizers = {}
        # 遍历配置中的 normalizers 字典，键为观测组名，值为该组的归一化配置。
        for obs_group_name, config in self.cfg.get("normalizers", dict()).items():
            # 复制配置，避免修改原始配置。
            config: dict = config.copy()
            # 构建归一化器，输入形状为该观测组展平后的总维度，
            # 类名从配置中弹出，剩余参数作为归一化器关键字参数。
            normalizer = modules.build_normalizer(
                input_shape=get_subobs_size(obs_format[obs_group_name]),
                normalizer_class_name=config.pop("class_name"),
                normalizer_kwargs=config,
            )
            # 将归一化器移动到指定设备。
            normalizer.to(self.device)
            # 保存到字典中，键为观测组名。
            self.normalizers[obs_group_name] = normalizer

        # init storage and model
        # 初始化算法的存储缓冲区（用于 PPO 等 on-policy 算法）。
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_format=obs_format,
            num_actions=self.env.num_actions,
            num_rewards=self.env.num_rewards,
        )

        # Log
        # 日志目录。
        self.log_dir = log_dir
        # TensorBoard writer，初始为 None，在 learn 中创建。
        self.writer = None
        # 总时间步数。
        self.tot_timesteps = 0
        # 总训练时间。
        self.tot_time = 0
        # 当前学习迭代次数。
        self.current_learning_iteration = 0
        # 日志记录间隔（每多少次迭代记录一次）。
        self.log_interval = self.cfg.get("log_interval", 1)
        # 需要记录 git 状态的仓库文件列表，默认包含 instinct_rl 包自身。
        self.git_status_repos = [instinct_rl.__file__]  # store files whose repo status will be logged.

        # 初始化时先重置一次环境，返回的观测和额外信息被忽略。
        _, _ = self.env.reset()

    # learn 方法执行训练循环。
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # Must call distributed_data_parallel after load_run (in train.py). Otherwise, a DDP algo cannot
        # load from a checkpoint trained in non-DDP mode.
        # 如果已经初始化了分布式训练，则将算法包装为 DistributedDataParallel。
        if dist.is_initialized():
            self.alg.distributed_data_parallel()
            print(f"[INFO rank {dist.get_rank()}]: DistributedDataParallel enabled.")
        # initialize writer
        # 如果日志目录存在、writer 尚未创建，且当前进程不是分布式训练中的非 0 号进程，则创建 SummaryWriter。
        if self.log_dir is not None and self.writer is None and (not self.is_mp_rank_other_process()):
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        # 如果需要在随机 episode 长度处初始化，则将 episode_length_buf 随机化为 [0, max_episode_length) 之间的值。
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        # 获取初始观测和额外信息。
        obs, extras = self.env.get_observations()
        # 将观测移动到指定设备。
        obs = obs.to(self.device)
        # 从额外信息中获取 critic 观测（如果存在），否则为 None。
        critic_obs = extras["observations"].get("critic", None)
        # 如果 critic 观测存在，也移动到指定设备。
        critic_obs = critic_obs.to(self.device) if critic_obs is not None else None
        # 将模型切换到训练模式。
        self.train_mode()

        # The first action must use the same normalization as subsequent
        # rollout steps, including after resuming a saved normalizer.
        for group_name, normalizer in self.normalizers.items():
            if group_name == "policy":
                obs = normalizer(obs)
            elif group_name == "critic" and critic_obs is not None:
                critic_obs = normalizer(critic_obs)

        # 用于记录 episode 信息的列表。
        ep_infos = []
        # 用于记录 step 信息的列表。
        step_infos = []
        # 为每个奖励项创建固定长度的 deque，用于记录最近 2000 个奖励帧。
        rframebuffer = [deque(maxlen=2000) for _ in range(self.env.num_rewards)]
        # 为每个奖励项创建固定长度的 deque，用于记录最近 100 个 episode 的奖励总和。
        rewbuffer = [deque(maxlen=100) for _ in range(self.env.num_rewards)]
        # 记录最近 100 个 episode 的长度。
        lenbuffer = deque(maxlen=100)
        # 当前每个环境的累积奖励，形状 [num_envs, num_rewards]。
        cur_reward_sum = torch.zeros(self.env.num_envs, self.env.num_rewards, dtype=torch.float, device=self.device)
        # 当前每个环境的 episode 长度，形状 [num_envs]。
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 打印初始化完成信息。
        print(
            "[INFO{}]: Initialization done, start learning.".format(
                f" rank {dist.get_rank()}" if dist.is_initialized() else ""
            )
        )
        # 打印提示信息，关于可能出现的 NaN/Inf 警告。
        print(
            "NOTE: you may see a bunch of `NaN or Inf found in input tensor` once and appears in the log. Just ignore"
            " it if it does not affect the performance."
        )
        # 如果日志目录存在且当前进程是主进程，则记录代码状态（git 状态）。
        if self.log_dir is not None and (not self.is_mp_rank_other_process()):
            store_code_state(self.log_dir, self.git_status_repos)
        # 记录起始迭代次数。
        start_iter = self.current_learning_iteration
        # 总迭代次数 = 当前迭代 + 需要学习的迭代次数。
        tot_iter = self.current_learning_iteration + num_learning_iterations
        # 记录总开始时间。
        tot_start_time = time.time()
        # 记录本轮开始时间。
        start = time.time()
        # 训练主循环。
        while self.current_learning_iteration < tot_iter:
            # DAgger uses this iteration to sample Teacher execution masks.
            # Synchronize before collecting data, not only inside update().
            self.alg.current_learning_iteration = self.current_learning_iteration
            # Rollout
            # 使用推理模式进行 rollout（默认开启，可通过配置关闭）。
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                # 收集 num_steps_per_env 步的数据。
                for i in range(self.num_steps_per_env):
                    # 执行一步 rollout，返回新的观测、critic 观测、奖励、done 和信息。
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    # 如果奖励是一维的，扩展为 [batch, 1]。
                    if len(rewards.shape) == 1:
                        rewards = rewards.unsqueeze(-1)

                    # 如果需要记录日志。
                    if self.log_dir is not None:
                        # Book keeping
                        # 如果 info 中包含 "step"，则记录。
                        if "step" in infos:
                            step_infos.append(infos["step"])
                        # 如果 info 中包含 "log"，则记录。
                        if "log" in infos:
                            ep_infos.append(infos["log"])
                        # 累积奖励和 episode 长度。
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        # 找出已经结束的环境索引（done > 0）。
                        new_ids = (dones > 0).nonzero(as_tuple=False)[:, 0]
                        # 对每个奖励项，记录未结束环境当步奖励，以及结束环境的累计奖励。
                        for i in range(self.env.num_rewards):
                            rframebuffer[i].extend(rewards[dones < 1][:, i].cpu().numpy().tolist())
                            rewbuffer[i].extend(cur_reward_sum[new_ids][:, i].cpu().numpy().tolist())
                        # 记录结束环境的 episode 长度。
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        # 重置结束环境的累积奖励和 episode 长度。
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                # 记录收集时间。
                stop = time.time()
                collection_time = stop - start

                # Learning step
                # 记录学习开始时间。
                start = stop
                # 计算回报（使用 critic 观测，如果存在，否则使用 policy 观测）。
                self.alg.compute_returns(critic_obs if critic_obs is not None else obs)

            # 执行算法更新，返回损失和统计信息。
            losses, stats = self.alg.update(self.current_learning_iteration)
            # 记录学习结束时间。
            stop = time.time()
            # 学习时间。
            learn_time = stop - start
            # 如果日志目录存在，且达到日志记录间隔，则记录日志。
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            # 如果达到保存间隔，且不是起始迭代，则保存模型。
            if (
                self.current_learning_iteration % self.save_interval == 0
                and self.current_learning_iteration > start_iter
            ):
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            # 清空 episode 和 step 信息列表。
            ep_infos.clear()
            step_infos.clear()
            # 迭代次数加一。
            self.current_learning_iteration = self.current_learning_iteration + 1
            # 重置开始时间。
            start = time.time()

        # 训练结束后保存最终模型。
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # 执行一步 rollout：算法选择动作，环境执行一步，处理观测归一化，然后通知算法处理环境步。
    def rollout_step(self, obs, critic_obs):
        # 算法根据当前观测和 critic 观测选择动作。
        actions = self.alg.act(obs, critic_obs)
        # 环境执行动作，返回新的观测、奖励、done 和信息。
        obs, rewards, dones, infos = self.env.step(actions)
        # 从信息中获取 critic 观测。
        critic_obs = infos["observations"].get("critic", None)
        # 将所有张量移动到指定设备。
        obs, critic_obs, rewards, dones = (
            obs.to(self.device),
            critic_obs.to(self.device) if critic_obs is not None else None,
            rewards.to(self.device),
            dones.to(self.device),
        )
        # Dealing with obs normalizers. At this line, obs and critic_obs and all observations in info are still flattened
        # 处理观测归一化器。此时 obs、critic_obs 以及 info 中的所有观测都还是展平后的。
        for obs_group_name, normalizer in self.normalizers.items():
            # 如果归一化器对应 policy 组。
            if obs_group_name == "policy":
                # 对 policy 观测进行归一化，并更新 info 中的 policy 观测。
                obs = normalizer(obs)
                infos["observations"]["policy"] = obs
            # 如果归一化器对应 critic 组。
            elif obs_group_name == "critic":
                # Doesn't need to worry about `critic is None`. Otherwise, error shall occur when normalizers are being built
                # 不需要担心 critic 为 None，否则在构建归一化器时就会出错。
                critic_obs = normalizer(critic_obs)
                infos["observations"]["critic"] = critic_obs
            # 其他观测组。
            else:
                infos["observations"][obs_group_name] = normalizer(infos["observations"][obs_group_name])
        # 通知算法处理环境步，传入奖励、done、信息、观测和 critic 观测。
        self.alg.process_env_step(rewards, dones, infos, obs, critic_obs)
        # 返回处理后的观测、critic 观测、奖励、done 和信息。
        return obs, critic_obs, rewards, dones, infos

    """
    Logging
    """

    # 记录日志到 TensorBoard 和控制台。
    def log(self, locs, width=80, pad=35):
        # 更新总时间步数。
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        # 更新总训练时间。
        self.tot_time = time.time() - locs["tot_start_time"]
        # 本轮迭代时间 = 收集时间 + 学习时间。
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # 初始化 episode 信息字符串。
        ep_string = f""
        # 如果存在 episode 信息。
        if locs["ep_infos"]:
            # 遍历第一个 episode 信息的所有键。
            for key in locs["ep_infos"][0]:
                infotensor = []
                # 遍历所有 episode 信息。
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    # 处理标量和零维张量信息。
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor.append(ep_info[key].to(self.device))
                # 拼接所有张量。
                infotensor = torch.cat(infotensor)
                # 根据键名中的 _max、_min 或其他，选择聚合方式。
                if "_max" in key:
                    value = self.gather_stat_values(infotensor, "max")
                elif "_min" in key:
                    value = self.gather_stat_values(infotensor, "min")
                else:
                    value = self.gather_stat_values(infotensor, "mean")
                # 写入 TensorBoard。
                self.writer_mp_add_scalar(
                    (key if key.startswith("Episode") else "Episode/" + key),
                    value,
                    self.current_learning_iteration,
                )
                # 添加到日志字符串。
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        # 如果存在 step 信息。
        if locs["step_infos"]:
            # 遍历第一个 step 信息的所有键。
            for key in locs["step_infos"][0]:
                infotensor = []
                texts = []
                # 遍历所有 step 信息。
                for step_info in locs["step_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    # 如果值是字符串，收集到 texts 中并跳过。
                    if isinstance(step_info[key], str):
                        texts.append(step_info[key])
                        continue
                    # 否则处理标量和零维张量。
                    elif not isinstance(step_info[key], torch.Tensor):
                        step_info[key] = torch.Tensor([step_info[key]])
                    if len(step_info[key].shape) == 0:
                        step_info[key] = step_info[key].unsqueeze(0)
                    infotensor.append(step_info[key].to(self.device))
                # 拼接所有张量。
                infotensor = torch.cat(infotensor)
                # 根据键名选择聚合方式。
                if "_max" in key:
                    value = self.gather_stat_values(infotensor, "max")
                elif "_min" in key:
                    value = self.gather_stat_values(infotensor, "min")
                else:
                    value = self.gather_stat_values(infotensor, "mean")
                # 如果有文本信息且当前是主进程，则写入文本。
                if len(texts) > 0 and (not self.is_mp_rank_other_process()):
                    self.writer.add_text(
                        (key if key.startswith("Step") else "Step/" + key),
                        "\n".join(texts),
                        self.current_learning_iteration,
                    )
                # 否则写入标量。
                else:
                    self.writer_mp_add_scalar(
                        (key if key.startswith("Step") else "Step/" + key),
                        value,
                        self.current_learning_iteration,
                    )
                    ep_string += f"""{f'Mean step {key}:':>{pad}} {value:.4f}\n"""
        # 计算动作噪声标准差均值。
        mean_std = self.alg.actor_critic.action_std.mean()
        # 计算 FPS（每秒步数）。
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))
        # 如果是分布式训练，乘以世界大小。
        if dist.is_initialized():
            fps *= dist.get_world_size()

        # 记录损失。
        for k, v in locs["losses"].items():
            v = self.gather_stat_values(v, "mean")
            self.writer_mp_add_scalar("Loss/" + k, v.item(), self.current_learning_iteration)
        # 记录训练统计。
        for k, v in locs["stats"].items():
            v = self.gather_stat_values(v, "mean")
            self.writer_mp_add_scalar("Train/" + k, v.item(), self.current_learning_iteration)

        # 记录学习率、动作噪声标准差、FPS、收集时间、学习时间、GPU 内存等。
        self.writer_mp_add_scalar("Loss/learning_rate", self.alg.learning_rate, self.current_learning_iteration)
        self.writer_mp_add_scalar("Policy/mean_noise_std", mean_std.item(), self.current_learning_iteration)
        self.writer_mp_add_scalar("Perf/total_fps", fps, self.current_learning_iteration)
        self.writer_mp_add_scalar("Perf/collection_time", locs["collection_time"], self.current_learning_iteration)
        self.writer_mp_add_scalar("Perf/learning_time", locs["learn_time"], self.current_learning_iteration)
        self.writer_mp_add_scalar(
            "Perf/gpu_allocated", torch.cuda.memory_allocated(self.device) / 1024**3, self.current_learning_iteration
        )
        self.writer_mp_add_scalar(
            "Perf/gpu_total", torch.cuda.mem_get_info(self.device)[1] / 1024**3, self.current_learning_iteration
        )
        self.writer_mp_add_scalar(
            "Perf/gpu_global_free_mem",
            torch.cuda.mem_get_info(self.device)[0] / 1024**3,
            self.current_learning_iteration,
        )
        # 记录每个奖励项的平均每步奖励。
        for i in range(self.env.num_rewards):
            self.writer_mp_add_scalar(
                f"Train/mean_reward_each_timestep_{i}",
                statistics.mean(locs["rframebuffer"][i]),
                self.current_learning_iteration,
            )
        # 如果奖励缓冲区非空，记录更多统计信息。
        if len(locs["rewbuffer"][0]) > 0:
            for i in range(self.env.num_rewards):
                self.writer_mp_add_scalar(
                    f"Train/mean_reward_{i}", statistics.mean(locs["rewbuffer"][i]), self.current_learning_iteration
                )
                self.writer_mp_add_scalar(
                    f"Train/ratio_above_mean_reward_{i}",
                    statistics.mean(
                        [(1.0 if rew > statistics.mean(locs["rewbuffer"][i]) else 0) for rew in locs["rewbuffer"][i]]
                    ),
                    self.current_learning_iteration,
                )
                self.writer_mp_add_scalar(
                    f"Train/time/mean_reward_{i}", statistics.mean(locs["rewbuffer"][i]), self.tot_time
                )
            self.writer_mp_add_scalar(
                "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/median_episode_length", statistics.median(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/min_episode_length", min(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/max_episode_length", max(locs["lenbuffer"]), self.current_learning_iteration
            )
            self.writer_mp_add_scalar(
                "Train/time/mean_episode_length", statistics.mean(locs["lenbuffer"]), self.tot_time
            )

        # 构建控制台日志字符串。
        info_str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"][0]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{info_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean([statistics.mean(buf) for buf in locs['rewbuffer']]):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{info_str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            log_string += (
                f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        # 添加 episode 信息。
        log_string += ep_string
        # 添加总时间、迭代时间、总时间和 ETA。
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n"""
        )
        # 如果不是分布式训练中的非 0 号进程，则打印日志。
        if not self.is_mp_rank_other_process():
            print(log_string)

    # 添加需要记录 git 状态的仓库路径。
    def add_git_repo_to_log(self, repo_path):
        self.git_status_repos.append(repo_path)

    # 保存训练状态字典到文件。多进程时仅 rank 0 执行。
    def save(self, path, infos=None):
        """Save training state dict to file. Will not happen if in multi-process and not rank 0."""
        # 如果是分布式训练中的非 0 号进程，则直接返回。
        if self.is_mp_rank_other_process():
            return

        # 获取算法状态字典。
        run_state_dict = self.alg.state_dict()
        # 更新归一化器状态字典（如果没有归一化器则不会更新任何内容）。
        run_state_dict.update(  # Nothing will update if there is no normalizers
            {
                f"{group_name}_normalizer_state_dict": normalizer.state_dict()
                for group_name, normalizer in self.normalizers.items()
            }
        )
        # 更新迭代次数和额外信息。
        run_state_dict.update(
            {
                "iter": self.current_learning_iteration,
                "infos": infos,
            }
        )
        # 保存到文件。
        torch.save(run_state_dict, path)

    # 从文件加载训练状态字典。多进程时仅 rank 0 执行。
    def load(self, path):
        """Load training state dict from file. Will not happen if in multi-process and not rank 0."""
        # 如果是分布式训练中的非 0 号进程，则直接返回。
        if self.is_mp_rank_other_process():
            return

        # 加载检查点，仅加载权重。
        loaded_dict = torch.load(path, weights_only=True)
        # 如果配置中指定了 ckpt_manipulator，则使用该函数对加载的字典进行修改。
        if self.cfg.get("ckpt_manipulator", False):
            # suppose to be a string specifying which function to use
            print("\033[1;36m Warning: using a hacky way to load the model. \033[0m")
            loaded_dict = getattr(ckpt_manipulator, self.cfg["ckpt_manipulator"])(
                loaded_dict,
                self.alg.state_dict(),
                **self.cfg.get("ckpt_manipulator_kwargs", {}),
            )
            print("\033[1;36m Done: using a hacky way to load the model. \033[0m")

        # 将状态字典加载到算法中。
        self.alg.load_state_dict(loaded_dict)

        # 加载归一化器状态字典。
        for group_name, normalizer in self.normalizers.items():
            if not f"{group_name}_normalizer_state_dict" in loaded_dict:
                print(
                    f"\033[1;36m Warning, normalizer for {group_name} is not found, the state dict is not loaded"
                    " \033[0m"
                )
            else:
                normalizer.load_state_dict(loaded_dict[f"{group_name}_normalizer_state_dict"])

        # 恢复迭代次数。
        self.current_learning_iteration = loaded_dict["iter"]
        # 如果使用了 ckpt_manipulator，则尝试保存修改后的检查点。
        if self.cfg.get("ckpt_manipulator", False):
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            except Exception as e:
                print(f"\033[1;36m Save manipulated checkpoint failed with error: {str(e)} \n ignored... \033[0m")
        # 返回额外信息。
        return loaded_dict["infos"]

    # 获取用于推理的策略函数。
    def get_inference_policy(self, device=None):
        # 切换到评估模式。
        self.eval_mode()
        # 如果指定了设备，则将 actor-critic 移动到该设备。
        if device is not None:
            self.alg.actor_critic.to(device)

        # 如果存在 policy 归一化器，则先归一化再推理。
        if "policy" in self.normalizers:
            self.normalizers["policy"].to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.normalizers["policy"](x))  # noqa: E731
        else:
            policy = self.alg.actor_critic.act_inference

        return policy

    # 将模型导出为 ONNX 格式。
    def export_as_onnx(self, obs, export_model_dir):
        # 切换到评估模式。
        self.eval_mode()
        # 如果有 policy 归一化器，则先归一化观测，并导出归一化器。
        if "policy" in self.normalizers:
            obs = self.normalizers["policy"](obs)
            # also export obs normalizer
            self.normalizers["policy"].export(os.path.join(export_model_dir, "policy_normalizer.npz"))
        # 导出 actor-critic 为 ONNX。
        self.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    """
    Helper functions
    """

    # 检查当前进程是否应该执行 all_reduce 汇总训练统计和检查点（分布式训练中的 rank 0）。
    def is_mp_rank_zero_process(self):
        """Check if the current process should do all reduce to summarize the training statistics and checkpoint."""
        return dist.is_initialized() and dist.get_rank() == 0

    # 检查当前进程是否为分布式训练中的非 0 号进程，或者单进程。
    def is_mp_rank_other_process(self):
        """Check if current process is in torch distributed multi-processing and not rank 0, or single process."""
        return dist.is_initialized() and dist.get_rank() != 0

    # 跨进程收集统计值并汇总。
    def gather_stat_values(self, values: torch.Tensor, gather_op: str = "mean", remove_nan: bool = True):
        """Properly gather the value across all processes. summarize the input values directly if not in multi-processing.
        ## Args:
            - values: torch.Tensor, the value to summarize. Do not summarize this into a scalar before calling this function.
            - gather_op: dist.ReduceOp, the operation to contat the value.
            - remove_nan: bool, whether to remove NaN values before summarizing.
        ## Return:
            The summarized value across all processes.
        """
        # 如果需要，移除 NaN 值。
        if remove_nan:
            values = values[~values.isnan()]
        # 确保值在可用于 all_reduce 的设备上。
        values = values.to(self.device)  # make sure the values are on the all_reduc-able device
        # 根据操作类型进行汇总。
        if gather_op == "mean":
            # 计算元素数量。
            num_values = torch.tensor([torch.numel(values)]).to(self.device)
            # 求和。
            values = torch.sum(values)
            # 如果分布式已初始化，则对和与数量进行 all_reduce。
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.SUM)
                dist.all_reduce(num_values, dist.ReduceOp.SUM)
            # 计算均值。
            values = values / num_values.item()
        elif gather_op == "max":
            # 取最大值，如果为空则取负无穷。
            values = torch.max(values) if values.numel() > 0 else torch.tensor(float("-inf"), device=self.device)
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.MAX)
        elif gather_op == "min":
            # 取最小值，如果为空则取正无穷。
            values = torch.min(values) if values.numel() > 0 else torch.tensor(float("inf"), device=self.device)
            if dist.is_initialized():
                dist.all_reduce(values, dist.ReduceOp.MIN)
        else:
            raise ValueError(f"Unsupported gather_op: {gather_op}")
        return values

    # 向 TensorBoard 写入标量。多进程时仅 rank 0 执行。
    def writer_mp_add_scalar(self, key, value, step):
        """Add scalar to tensorboard writer. Will not happen if in multi-process and not rank 0."""
        if not self.is_mp_rank_other_process():
            self.writer.add_scalar(key, value, step)

    # 将所有相关模型切换到训练模式（例如 dropout）。
    def train_mode(self):
        """Change all related models into training mode (for dropout for example)"""
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        for normalizer in self.normalizers.values():
            normalizer.train()

    # 将所有相关模型切换到评估模式（例如 dropout）。
    def eval_mode(self):
        """Change all related models into evaluation mode (for dropout for example)"""
        self.alg.actor_critic.eval()
        for normalizer in self.normalizers.values():
            normalizer.eval()