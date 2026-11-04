# 导入 importlib，用于动态导入模块（支持通过字符串指定编码器类）。
import importlib
# 导入 os，用于路径拼接等操作。
import os
# 导入 OrderedDict，虽然此处未直接使用，但可能用于有序配置。
from collections import OrderedDict
# 导入 copy 和 deepcopy，分别用于浅拷贝和深拷贝。
from copy import copy, deepcopy
# 导入 Dict 类型，用于类型注解。
from typing import Dict

# 导入 numpy，通常用于数值计算，此处可能未直接使用。
import numpy as np
# 导入 PyTorch 主库。
import torch
# 导入 PyTorch 神经网络模块。
import torch.nn as nn

# 导入 instinct_rl.modules 并重命名为 instinct_modules，用于动态获取编码器类。
import instinct_rl.modules as instinct_modules
# 从 instinct_rl.utils.utils 导入 get_subobs_size，用于计算观测子段的总维度。
from instinct_rl.utils.utils import get_subobs_size


# EncoderActorCriticMixin 是一个混入类，为 ActorCritic 提供独立的观测编码器（encoder）功能。
# 它通过编码器将原始观测嵌入到潜在空间，然后再交给后续的 actor/critic 网络。
class EncoderActorCriticMixin:
    """A general implementation where a separate encoder is used to embed the obs/privileged_obs"""

    # 初始化混入类。
    def __init__(
        self,
        obs_format: Dict[str, Dict[str, tuple]],
        num_actions,
        encoder_configs: Dict[str, dict],
        critic_encoder_configs=None,  # None, "shared", or an ordereddict of configs (in the same order as encoder_configs)
        **kwargs,
    ):
        """NOTE: recurrent encoder is not implemented and tested yet.
        Args:
        """
        # 从 obs_format 中提取 policy 观测的分段信息（每个观测项的名称与维度）。
        self.__obs_segments = obs_format["policy"]
        # 提取 critic 观测的分段信息；如果不存在，则默认与 policy 相同。
        self.__critic_obs_segments = obs_format.get("critic", obs_format["policy"])
        # 计算 actor 观测的总维度。
        self.num_actor_obs = get_subobs_size(self.__obs_segments)
        # 计算 critic 观测的总维度。
        self.num_critic_obs = get_subobs_size(self.__critic_obs_segments)
        # 保存动作维度。
        self.num_actions = num_actions

        # 浅拷贝编码器配置，避免修改原始配置。
        self.encoder_configs = copy(encoder_configs)
        # 浅拷贝 critic 编码器配置。
        self.critic_encoder_configs = copy(critic_encoder_configs)
        # 从编码器配置中弹出类名，默认使用 "ParallelLayer"。
        encoder_class_name = encoder_configs.pop("class_name", "ParallelLayer")
        # 根据类名字符串动态获取编码器类。
        # 如果类名中包含冒号，格式为 "模块路径:类名"，则动态导入模块并获取类。
        # 否则从 instinct_modules 中按名称获取。
        EncoderClass = (
            getattr(importlib.import_module(encoder_class_name.split(":")[0]), encoder_class_name.split(":")[1])
            if ":" in encoder_class_name
            else getattr(instinct_modules, encoder_class_name)
        )
        # 实例化 actor 编码器，传入输入分段和块配置。
        encoders = EncoderClass(
            input_segments=self.__obs_segments,
            block_configs=self.encoder_configs,
        )
        # 根据 critic_encoder_configs 的值决定 critic 编码器：
        # - 如果为 "shared"，则与 actor 编码器共享同一个实例。
        # - 如果为 None，则不使用独立的 critic 编码器。
        # - 否则，根据配置创建独立的 critic 编码器。
        if self.critic_encoder_configs == "shared":
            critic_encoders = encoders
        elif self.critic_encoder_configs is None:
            critic_encoders = None
        else:
            # 从 critic 编码器配置中弹出类名，默认 "ParallelLayer"。
            critic_encoder_class_name = critic_encoder_configs.pop("class_name", "ParallelLayer")
            # 动态获取 critic 编码器类，逻辑同上。
            CriticEncoderClass = (
                getattr(
                    importlib.import_module(critic_encoder_class_name.split(":")[0]),
                    critic_encoder_class_name.split(":")[1],
                )
                if ":" in critic_encoder_class_name
                else getattr(instinct_modules, critic_encoder_class_name)
            )
            # 实例化 critic 编码器。
            critic_encoders = CriticEncoderClass(
                input_segments=self.__critic_obs_segments,
                block_configs=self.critic_encoder_configs,
            )

        # 深拷贝观测格式，用于构造嵌入后的观测格式。
        embedded_obs_format = deepcopy(obs_format)
        # 将 policy 观测格式替换为编码器的输出分段。
        embedded_obs_format["policy"] = encoders.output_segment
        # 如果有独立的 critic 编码器，也替换 critic 观测格式。
        if critic_encoders is not None:
            embedded_obs_format["critic"] = critic_encoders.output_segment

        # 调用父类（ActorCritic 或 ActorCriticRecurrent）的初始化方法，
        # 传入嵌入后的观测格式、动作维度和其余关键字参数。
        super().__init__(
            embedded_obs_format,
            num_actions,
            **kwargs,
        )
        # 保存 actor 编码器。
        self.encoders = encoders
        # 保存 critic 编码器（可能为 None）。
        self.critic_encoders = critic_encoders

        # 打印编码器信息，便于调试。
        print(f"Actor Encoder: {self.encoders}")
        print(f"Critic Encoder: {self.critic_encoders}")

        # 初始化用于缓存编码器潜在表示的字典（目前未使用）。
        self.encoder_latents_buf = dict()
        self.critic_encoder_latents_buf = dict()

    # 使用已展平的观测直接调用父类的 act 方法（跳过编码器）。
    def backbone_act(self, flatten_observations, **kwargs):
        super().act(flatten_observations, **kwargs)

    # 重写 act 方法：先通过 actor 编码器嵌入观测，再调用父类的 act。
    def act(self, observations, **kwargs):
        obs = self.encoders(observations)
        return super().act(obs, **kwargs)

    # 重写 act_inference 方法：先通过 actor 编码器嵌入观测，再调用父类的推理方法。
    def act_inference(self, observations):
        obs = self.encoders(observations)
        return super().act_inference(obs)

    # 使用已展平的观测直接调用父类的 evaluate 方法（跳过 critic 编码器）。
    def backbone_evaluate(self, flatten_observations, masks=None, hidden_states=None):
        """Evaluate the model with the backbone input, which is typically an MLP."""
        return super().evaluate(flatten_observations, masks=masks, hidden_states=hidden_states)

    # 重写 evaluate 方法：如果存在 critic 编码器，则先嵌入 critic 观测，再调用父类的 evaluate。
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        obs = self.critic_encoders(critic_observations) if self.critic_encoders is not None else critic_observations
        return super().evaluate(obs, masks=masks, hidden_states=hidden_states)

    # 实现 forward 方法，使模型可以作为 torch.nn.Module 使用，并兼容 torch.onnx.export。
    def forward(self, observations):
        """Fill the function of a torch.nn.Module and make it compatible with the torch.onnx.export"""
        return self.act_inference(observations)

    # 属性：返回 actor 观测的分段信息。
    @property
    def obs_segments(self):
        """NOTE: make sure to copy these methods in subclasses if the subclass has its own obs_segments"""
        return self.__obs_segments

    # 属性：返回 critic 观测的分段信息。
    @property
    def critic_obs_segments(self):
        return self.__critic_obs_segments

    # 将模型导出为 ONNX 文件。
    def export_as_onnx(self, observations, filedir, encoder_as_seperate_file=True):
        """Export the model as an ONNX file. Input should be batch-wise observations with batchsize 1."""
        # 切换到评估模式。
        self.eval()
        # 如果编码器作为单独文件导出。
        if encoder_as_seperate_file:
            # 导出 actor 编码器为 ONNX。
            self.encoders.export_as_onnx(observations, filedir, encoder_as_seperate_file)
            # 在无梯度模式下计算编码后的观测。
            with torch.no_grad():
                obs = self.encoders(observations)
            # 调用父类的导出方法，导出后续的 actor-critic 网络。
            super().export_as_onnx(obs, filedir)
        # 否则将整个模型（编码器 + actor-critic）作为一个 ONNX 文件导出。
        else:
            with torch.no_grad():
                # 使用 torch.onnx.export 导出整个模型。
                exported_program = torch.onnx.export(
                    self,
                    observations,
                    "/tmp/encoder_actor_critic.onnx",
                    # This file does not contain the model weight, we call the save later to save the onnx with model weight.
                    input_names=["input"],
                    output_names=["output"],
                    dynamo=True,
                    opset=17,
                )
                # 保存导出的 ONNX 模型到指定目录。
                exported_program.save(os.path.join(filedir, "encoder_actor_critic.onnx"))
                print(f"Exported encoder_actor_critic to {os.path.join(filedir, 'encoder_actor_critic.onnx')}")


# 从当前包导入 ActorCritic 类。
from .actor_critic import ActorCritic


# 定义 EncoderActorCritic 类，继承 EncoderActorCriticMixin 和 ActorCritic。
class EncoderActorCritic(EncoderActorCriticMixin, ActorCritic):
    pass


# 从当前包导入 ActorCriticRecurrent 类。
from .actor_critic_recurrent import ActorCriticRecurrent


# 定义 EncoderActorCriticRecurrent 类，继承 EncoderActorCriticMixin 和 ActorCriticRecurrent。
class EncoderActorCriticRecurrent(EncoderActorCriticMixin, ActorCriticRecurrent):
    pass