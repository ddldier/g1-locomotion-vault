# ============================================================
# 文件：import copy.txt
# 说明：本文件是 Project-Instinct 中 G1 跑酷任务的配置文件。
# 以下仅添加注释，不修改任何原有代码。
# ============================================================

import copy
import os

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.assets.unitree_g1 import (
    G1_29DOF_LINKS,
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    beyondmimic_g1_29dof_actuators,
    beyondmimic_g1_29dof_delayed_actuators,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import get_link_prim_targets
from instinctlab.tasks.parkour.config.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg

# 当前文件所在目录，用于拼接相对路径资源
__file_dir__ = os.path.dirname(os.path.realpath(__file__))

# 复制 G1 基础配置，并针对跑酷任务做少量覆盖
G1_CFG = copy.deepcopy(G1_29DOF_TORSOBASE_POPSICLE_CFG)
# 合并固定关节，简化模型
G1_CFG.spawn.merge_fixed_joints = True
# 初始位置略微抬高
G1_CFG.init_state.pos = (0.0, 0.0, 0.9)

# 带鞋版本的 G1 配置：替换为带鞋的 URDF
G1_with_shoe_CFG = copy.deepcopy(G1_CFG)
G1_with_shoe_CFG.spawn.asset_path = os.path.abspath(
    f"{__file_dir__}/../../urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
)

# AMASS 动作参考配置：继承基础配置并覆盖跑酷相关参数
@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    # 数据集根目录
    path = r"/root/gpufree-data/datasets/project-instinct/hiking-in-the-wild/data&model/parkour_motion_reference"
    # 不做重定向
    retargetting_func = None
    # 过滤后的动作选择文件
    filtered_motion_selection_filepath = r"/root/gpufree-data/datasets/project-instinct/hiking-in-the-wild/data&model/parkour_motion_reference/parkour_motion_without_run.yaml"
    # 动作起始片段范围
    motion_start_from_middle_range = [0.0, 0.9]
    # 起始高度偏移
    motion_start_height_offset = 0.0
    # 不强制链接低于地面
    ensure_link_below_zero_ground = False
    # 缓冲区设备
    buffer_device = "output_device"
    # 双线性插值
    motion_interpolate_func = motion_interpolate_bilinear
    # 前向速度估计
    velocity_estimation_method = "frontward"


# 动作参考管理器配置：定义参考动作如何被加载、对齐和采样
motion_reference_cfg = MotionReferenceManagerCfg(
    # 机器人躯干链接
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    # 机器人模型路径
    robot_model_path=G1_CFG.spawn.asset_path,
    # 参考动作的 prim 路径
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    # 对称增强链接映射
    symmetric_augmentation_link_mapping=[0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12],
    # 对称增强关节映射
    symmetric_augmentation_joint_mapping=G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping,
    # 对称增强关节反向缓冲区
    symmetric_augmentation_joint_reverse_buf=G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf,
    # 帧间隔
    frame_interval_s=0.02,
    # 更新周期
    update_period=0.02,
    # 每次采样帧数
    num_frames=10,
    # 动作缓冲区名称
    motion_buffers={
        "run_walk": AmassMotionCfg(),
    },
    # 关注的关键链接，用于计算奖励/观测
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    # 运动片段切分方式
    mp_split_method="Even",
)


# 回放用地形配置：复制训练地形，但把墙的概率全部置零
ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


# G1 跑酷粗糙地形环境配置：训练用
@configclass
class G1ParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        # 场景：地形生成器
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        # 场景：机器人配置
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 机器人执行器：使用带延迟的 BeyondMimic 执行器
        self.scene.robot.actuators = beyondmimic_g1_29dof_delayed_actuators
        # 相机网格路径：加入 G1 各链接
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(G1_29DOF_LINKS))
        # 场景：动作参考管理器
        self.scene.motion_reference = motion_reference_cfg


# 鞋子配置混入类：用于切换到带鞋模型并调整相关高度参数
class ShoeConfigMixin:
    def apply_shoe_config(self):
        # 替换为带鞋机器人模型
        self.scene.robot = G1_with_shoe_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 调整腿部体积点生成器的 Z 范围
        self.scene.leg_volume_points.points_generator.z_min = -0.063
        self.scene.leg_volume_points.points_generator.z_max = -0.023
        # 调整脚部平面奖励的高度偏移
        self.rewards.rewards.feet_at_plane.params["height_offset"] = 0.058


# G1 跑酷粗糙地形环境配置：回放/评估用
@configclass
class G1ParkourRoughEnvCfg_PLAY(G1ParkourRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # 使用回放地形配置
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        # make a smaller scene for play
        # 回放场景使用更少的环境
        self.scene.num_envs = 10
        # 查看器配置：设置相机位置和注视点
        self.viewer = ViewerCfg(
            eye=[4.0, 0.75, 1.0],
            lookat=[0.0, 0.75, 0.0],
            origin_type="asset_root",
            asset_name="robot",
        )

        # 环境间距
        self.scene.env_spacing = 2.5
        # 回合长度（秒）
        self.episode_length_s = 10
        # 回放时不做根高度终止
        self.terminations.root_height = None
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # reduce the number of terrains to save memory
        # 在网格中随机生成机器人（而不是按地形级别）
        # 减少地形数量以节省内存
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 10

        # 开启腿部体积点调试可视化
        self.scene.leg_volume_points.debug_vis = True
        # 开启基础速度命令调试可视化
        self.commands.base_velocity.debug_vis = True
        # 回放时不需要物理材质事件
        self.events.physics_material = None
        # 重置机器人关节：位置和速度范围都设为 0，保证确定性
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


# 带鞋的 G1 跑酷环境配置：训练用
@configclass
class G1ParkourEnvCfg(G1ParkourRoughEnvCfg, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        # 应用鞋子配置
        self.apply_shoe_config()


# 带鞋的 G1 跑酷环境配置：回放/评估用
@configclass
class G1ParkourEnvCfg_PLAY(G1ParkourRoughEnvCfg_PLAY, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        # 应用鞋子配置
        self.apply_shoe_config()