# 启用延迟注解，使得类型注解在运行时不会被求值，便于前向引用。
from __future__ import annotations

# 导入 Sequence 用于类型提示。
from collections.abc import Sequence
# 导入 Literal 用于字面量类型提示。
from typing import Literal

# 导入 Isaac Lab 的仿真工具。
import isaaclab.sim as sim_utils
# 导入可视化标记配置类。
from isaaclab.markers import VisualizationMarkersCfg
# 导入传感器基类配置。
from isaaclab.sensors import SensorBaseCfg
# 导入 schemas 模块，用于碰撞属性配置。
from isaaclab.sim import schemas
# 导入 configclass 装饰器。
from isaaclab.utils import configclass
# 导入 Isaac Nucleus 资产目录常量。
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 从当前包导入运动缓冲区、运动参考数据和状态类。
from .motion_buffer import MotionBuffer, MotionReferenceData, MotionReferenceState
# 从当前包导入运动参考管理器类。
from .motion_reference_manager import MotionReferenceManager


# 定义运动缓冲区配置类。
@configclass
class MotionBufferCfg:
    """运动缓冲区的配置。"""

    # 类类型，默认为 MotionBuffer。
    class_type: type = MotionBuffer

    # 是否将关节参考裁剪到机器人关节限制内。
    clip_joint_ref_to_robot_limits: bool = False
    """ 将关节参考裁剪到机器人关节限制内。 """


# 定义运动参考管理器配置类，继承自 SensorBaseCfg。
@configclass
class MotionReferenceManagerCfg(SensorBaseCfg):
    """运动参考管理器的配置。"""

    # 类类型，默认为 MotionReferenceManager。
    class_type: type = MotionReferenceManager

    # 数据类类型，默认为 MotionReferenceData。
    data_class_type: type = MotionReferenceData
    """ 运动参考数据的类类型。使用此配置可以覆盖默认的运动参考数据类。 """

    # 状态类类型，默认为 MotionReferenceState。
    state_class_type: type = MotionReferenceState
    """ 运动参考状态的类类型。使用此配置可以覆盖默认的运动参考状态类。 """

    # 场景中刚体名称列表（非机器人本体）。对象数量由此列表长度推断。
    scene_object_names: list[str] = []
    """ 场景配置中的刚体名称（str）列表（不是机器人的身体）。对象数量由此列表长度推断。
    注意：如果使用 RigidObjectCollection，这些应该是集合中对象的名称。
    """

    # 机器人模型（urdf）路径，用于构建机器人运动学链。
    robot_model_path: str | None = None
    """ 用于构建机器人运动学链的机器人模型（urdf）路径。
        如果为 None，则假设运动数据文件已经是机器人模型的形式。
    """

    # 感兴趣的链接名称序列。
    link_of_interests: Sequence[str] | None = None
    """ 如果设置，将计算 base_link 坐标系下的链接姿态。 """

    # 生成命令的帧数。
    num_frames: int = 1
    """ 生成命令的帧数。 """

    # 数据起始方式：从帧间隔开始或从当前时间开始。
    data_start_from: Literal["one_frame_interval", "current_time"] = "one_frame_interval"
    """ 数据从帧间隔开始还是从当前时间开始。
    - "one_frame_interval": 第 0 帧位于 `t+frame_interval_s`。
    - "current_time": 第 0 帧位于 `t`。
    由于之前的实验，默认设置为 "one_frame_interval"。
    """

    # 帧间隔（秒）。可以是单个浮点数或浮点数序列。
    frame_interval_s: float | Sequence[float] = 0.1
    """ 帧间隔（秒）。
        如果是列表，则在重置时从列表范围中随机选择帧间隔。
    """

    # 运动缓冲区时间索引的更新周期（秒）。
    update_period: float | Sequence[float] = 0.0
    """ 运动缓冲区时间索引的更新周期（秒）。

    如果是两个浮点数的列表/元组，则在重置时为每个环境采样初始化。
    """

    # 更新周期采样策略。
    update_period_sample_strategy: Literal["uniform", "uniform_frame_limits"] | None = "uniform"
    """ 更新周期采样策略。它将使用 update_period 的范围。

    - None: 不进行更新周期采样。（如果 update_period 是列表，则使用最小的 update_period）
    - "uniform": 从 update_period 范围中均匀采样。
    - "uniform_frame_limits": 忽略 `update_period`，在运动帧序列的最长时间范围内均匀采样。
    """

    # 运动缓冲区字典，用于构建所有类型的运动缓冲区。
    motion_buffers: dict[str, MotionBufferCfg] = {}
    """ 用于构建所有类型运动缓冲区的字典。
    """

    # 多进程分割方法。
    mp_split_method: Literal["None", "Even", "Segment"] = "None"
    """ 将整个运动数据分割给多个进程的方法。
    进程之间只通信 rank/world_size 和权重。
    ## 选项：
        - "None": 不分割运动数据。建议每个进程中的每个运动缓冲区被动地将运动数据加载到 GPU 内存中。
        - "Even": 在进程之间均匀分割运动数据。如果将所有的轨迹按顺序排列，每个进程的轨迹分配如下：
            （假设 4 个进程）
                1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, ...
        - "Segment": 将运动数据分割成段，并将每段分配给每个进程。
            （假设 4 个进程）
                1, 1, 1, 1, 1, ...., 2, 2, 2, 2, 2, ...., 3, 3, 3, 3, 3, ...., 4, 4, 4, 4, 4, ...
    """

    ### 数据增强 ###
    # 用于对称增强运动数据的关节索引。
    symmetric_augmentation_joint_mapping: Sequence[int] | None = None
    """ 用于对称增强运动数据的关节索引。 """
    # 用于对称增强运动数据的关节反向缓冲区。
    symmetric_augmentation_joint_reverse_buf: Sequence | None = None
    """ 用于对称增强运动数据的关节索引，
        如果为 None，则不进行对称增强。
    """
    # 链接映射，顺序与 `link_of_interests` 一致。
    symmetric_augmentation_link_mapping: Sequence[int] | None = None
    """ 链接映射的顺序与 `link_of_interests` 一致。 """

    ### 可视化 ###
    # 参考机器人的 prim 路径。
    reference_prim_path: str | None = None
    """ 参考机器人的 prim 路径。
        要激活机器人模型可视化，请生成另一个不与场景中任何其他对象发生碰撞的关节
        （但可以在查看器中可视化）。
        然后提供该参考关节的 prim 路径。
        如果为 None，则不可视化参考机器人。
    """

    # 可视化器配置。
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/MotionReference",
        markers={
            "root_frame_ref": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.15, 0.15, 0.15),
            ),
            "link_ref": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "relative_link_ref": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
            ),
        },
    )
    """ 链接参考和基座姿态参考的可视化配置。 """

    # 要可视化的标记类型列表。
    visualizing_marker_types: list[str] = []
    """ 要可视化的标记类型列表。
    ## 可用选项：
        - 'root' 用于根变换
        - 'links' 用于链接变换
        - 'relative_links' 用于相对链接变换
    """

    # 参考机器人的可视化偏移。
    visualizing_robot_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """ 为了减少真实机器人与参考机器人的重叠，
    我们可以偏移参考机器人。
    """

    # 获取和设置参考机器人状态的数据源。
    visualizing_robot_from: Literal["aiming_frame", "reference_frame"] = "aiming_frame"
    """ 获取和设置参考机器人状态的数据源。
    - 'aiming_frame': 获取瞄准帧索引并从 motion_reference 数据中选择
    - 'reference_frame': 直接从 motion_reference.reference_frame 获取参考状态
    """


# 定义无碰撞属性配置类，继承自 schemas.CollisionPropertiesCfg。
@configclass
class NoCollisionPropertiesCfg(schemas.CollisionPropertiesCfg):
    # 碰撞启用标志，默认为 False。
    collision_enabled: bool = False