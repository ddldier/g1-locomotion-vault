# ============================================================
# 基础库导入
# ============================================================
import math
import os
from dataclasses import MISSING

# ============================================================
# Isaac Lab 仿真、环境、管理器、传感器等导入
# ============================================================
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# ============================================================
# InstinctLab 自定义模块导入
# ============================================================
import instinctlab.envs.mdp as instinct_mdp
import instinctlab.tasks.parkour.mdp as mdp
import instinctlab.terrains as terrain_gen
from instinctlab.assets.unitree_g1 import beyondmimic_action_scale
from instinctlab.managers import MultiRewardCfg
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.sensors import Grid3dPointsGeneratorCfg, NoisyGroupedRayCasterCameraCfg, VolumePointsCfg
from instinctlab.terrains import GreedyconcatEdgeCylinderCfg, TerrainImporterCfg
from instinctlab.utils.noise import (
    CropAndResizeCfg,
    DepthArtifactNoiseCfg,
    DepthNormalizationCfg,
    GaussianBlurNoiseCfg,
    RandomGaussianNoiseCfg,
    RangeBasedGaussianNoiseCfg,
)

# 当前文件所在目录
__file_dir__ = os.path.dirname(os.path.realpath(__file__))

##
# Scene definition
##
# ============================================================
# 粗糙地形生成器配置
# ============================================================
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0,  # 随机种子，保证每次生成的地形一样，方便复现实验
    size=(8.0, 8.0),  # 每个地形块的大小：宽 8 米，长 8 米
    border_width=3,  # 地形块四周的边界宽度，单位米，用来隔开不同地形
    num_rows=10,  # 地形网格的行数
    num_cols=20,  # 地形网格的列数
    horizontal_scale=0.05,  # 水平方向分辨率：每个像素代表 0.05 米
    vertical_scale=0.005,  # 垂直方向分辨率：高度每单位代表 0.005 米
    slope_threshold=1.0,  # 坡度阈值，超过这个值可能被判定为太陡
    use_cache=False,  # 是否缓存生成的地形，False 表示不缓存，每次都重新生成
    curriculum=True,  # 是否使用课程学习：训练过程中地形难度逐渐增加
    sub_terrains={  # 定义多种子地形，每种占一定比例
        # 子地形：Perlin 粗糙平面
        "perlin_rough": terrain_gen.PerlinPlaneTerrainCfg(
            proportion=0.05,  # 该子地形占总地形的比例：5%
            noise_scale=[0.0, 0.1],  # 噪声幅度范围：0 到 0.1 米
            noise_frequency=20,  # 噪声频率：值越大，起伏越密集
            fractal_octaves=2,  # 分形八度数：叠加多少层噪声
            fractal_lacunarity=2.0,  # 分形间隙度：每层频率的倍数
            fractal_gain=0.25,  # 分形增益：每层振幅的衰减系数
            centering=True,  # 是否将地形高度居中（让平均高度为 0）
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙出现的概率，分别是前后左右，0.3 表示 30% 概率有墙
            wall_height=5.0,  # 墙的高度，单位米
            wall_thickness=0.05,  # 墙的厚度，单位米
            flat_patch_sampling={  # 平坦补丁采样配置：在子地形中找一些平坦区域
                "target": FlatPatchSamplingCfg(  # 采样目标名为 "target"
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15, 0.20],  # 补丁半径候选列表，单位米
                    max_height_diff=0.05  # 补丁内最大高度差，超过就不算平坦
                ),
            },
        ),
        # 子地形：Perlin 粗糙平面，用于站立/低速训练
        "perlin_rough_stand": terrain_gen.PerlinPlaneTerrainCfg(
            proportion=0.05,  # 占比 5%
            noise_scale=[0.0, 0.1],  # 噪声幅度 0 到 0.1
            noise_frequency=20,  # 噪声频率 20
            fractal_octaves=2,  # 分形八度 2
            fractal_lacunarity=2.0,  # 分形间隙度 2.0
            fractal_gain=0.25,  # 分形增益 0.25
            centering=True,  # 高度居中
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙概率 30%
            wall_height=5.0,  # 墙高 5 米
            wall_thickness=0.05,  # 墙厚 0.05 米
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15, 0.20],  # 半径候选
                    max_height_diff=0.05  # 最大高度差 0.05 米
                ),
            },
        ),
        # 子地形：方形间隙/沟壑
        "square_gaps": terrain_gen.PerlinSquareGapTerrainCfg(
            proportion=0.10,  # 占比 10%
            gap_distance_range=(0.1, 0.7),  # 间隙距离范围：0.1 到 0.7 米
            gap_depth=(0.4, 0.6),  # 间隙深度范围：0.4 到 0.6 米
            platform_width=2.5,  # 平台宽度：2.5 米
            border_width=1.0,  # 边界宽度：1.0 米
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙概率 30%
            wall_height=5.0,  # 墙高 5 米
            wall_thickness=0.05,  # 墙厚 0.05 米
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15, 0.20],  # 半径候选
                    max_height_diff=0.05,  # 最大高度差 0.05 米
                    x_range=(3.7, 3.7),  # 采样位置的 x 范围限制在 3.7 附近
                    y_range=(-0.0, 0.0),  # 采样位置的 y 范围限制在 0 附近
                ),
            },
        ),
        # 子地形：金字塔楼梯
        "pyramid_stairs": terrain_gen.PerlinPyramidStairsTerrainCfg(
            proportion=0.15,  # 占比 15%
            step_height_range=(0.05, 0.23),  # 台阶高度范围：0.05 到 0.23 米
            step_width=0.3,  # 台阶宽度：0.3 米
            platform_width=2.5,  # 平台宽度：2.5 米
            border_width=1.0,  # 边界宽度：1.0 米
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙概率 30%
            wall_height=5.0,  # 墙高 5 米
            wall_thickness=0.05,  # 墙厚 0.05 米
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(  # 在楼梯表面叠加 Perlin 噪声
                noise_scale=0.05,  # 噪声幅度 0.05 米
                noise_frequency=20,  # 噪声频率 20
                fractal_octaves=2,  # 分形八度 2
                fractal_lacunarity=2.0,  # 分形间隙度 2.0
                fractal_gain=0.25,  # 分形增益 0.25
                centering=True,  # 高度居中
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15, 0.20],  # 半径候选
                    max_height_diff=0.05,  # 最大高度差 0.05 米
                    x_range=(3.7, 3.7),  # x 范围限制
                    y_range=(-0.0, 0.0),  # y 范围限制
                ),
            },
        ),
        # 子地形：高金字塔楼梯
        "pyramid_stairs_high": terrain_gen.PerlinPyramidStairsTerrainCfg(
            proportion=0.10,  # 占比 10%
            step_height_range=(0.05, 0.45),  # 台阶更高：0.05 到 0.45 米
            step_width=1.5,  # 台阶更宽：1.5 米
            platform_width=4.0,  # 平台更大：4.0 米
            border_width=1.0,  # 边界宽度 1.0 米
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙概率 30%
            wall_height=5.0,  # 墙高 5 米
            wall_thickness=0.05,  # 墙厚 0.05 米
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(  # 叠加 Perlin 噪声
                noise_scale=0.05,  # 噪声幅度 0.05 米
                noise_frequency=20,  # 噪声频率 20
                fractal_octaves=2,  # 分形八度 2
                fractal_lacunarity=2.0,  # 分形间隙度 2.0
                fractal_gain=0.25,  # 分形增益 0.25
                centering=True,  # 高度居中
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15, 0.20],  # 半径候选
                    max_height_diff=0.05,  # 最大高度差 0.05 米
                    x_range=(3.7, 3.7),  # x 范围限制
                    y_range=(-0.0, 0.0),  # y 范围限制
                ),
            },
        ),
        # 子地形：倒金字塔楼梯
        "pyramid_stairs_inv": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.15,  # 占比 15%
            step_height_range=(0.05, 0.23),  # 台阶高度 0.05 到 0.23 米
            step_width=0.3,  # 台阶宽度 0.3 米
            platform_width=2.5,  # 平台宽度 2.5 米
            border_width=1.0,  # 边界宽度 1.0 米
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙概率 30%
            wall_height=5.0,  # 墙高 5 米
            wall_thickness=0.05,  # 墙厚 0.05 米
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(  # 叠加 Perlin 噪声
                noise_scale=0.05,  # 噪声幅度 0.05 米
                noise_frequency=20,  # 噪声频率 20
                fractal_octaves=2,  # 分形八度 2
                fractal_lacunarity=2.0,  # 分形间隙度 2.0
                fractal_gain=0.25,  # 分形增益 0.25
                centering=True,  # 高度居中
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15, 0.20],  # 半径候选
                    max_height_diff=0.05,  # 最大高度差 0.05 米
                    x_range=(3.7, 3.7),  # x 范围限制
                    y_range=(-0.0, 0.0),  # y 范围限制
                ),
            },
        ),
        # 子地形：高倒金字塔楼梯
        "pyramid_stairs_inv_high": terrain_gen.PerlinInvertedPyramidStairsTerrainCfg(
            proportion=0.10,  # 占比 10%
            step_height_range=(0.05, 0.45),  # 台阶更高：0.05 到 0.45 米
            step_width=1.5,  # 台阶更宽：1.5 米
            platform_width=4.0,  # 平台更大：4.0 米
            border_width=1.0,  # 边界宽度 1.0 米
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙概率 30%
            wall_height=5.0,  # 墙高 5 米
            wall_thickness=0.05,  # 墙厚 0.05 米
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(  # 叠加 Perlin 噪声
                noise_scale=0.05,  # 噪声幅度 0.05 米
                noise_frequency=20,  # 噪声频率 20
                fractal_octaves=2,  # 分形八度 2
                fractal_lacunarity=2.0,  # 分形间隙度 2.0
                fractal_gain=0.25,  # 分形增益 0.25
                centering=True,  # 高度居中
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15, 0.20],  # 半径候选
                    max_height_diff=0.05,  # 最大高度差 0.05 米
                    x_range=(3.7, 3.7),  # x 范围限制
                    y_range=(-0.0, 0.0),  # y 范围限制
                ),
            },
        ),
        # 子地形：离散障碍盒
        "boxes": terrain_gen.PerlinDiscreteObstaclesTerrainCfg(
            proportion=0.10,  # 占比 10%
            num_obstacles=20,  # 障碍物数量 20 个
            obstacle_height_mode="fixed",  # 障碍物高度模式：固定
            obstacle_width_range=(0.8, 1.5),  # 障碍物宽度范围：0.8 到 1.5 米
            obstacle_height_range=(0.05, 0.45),  # 障碍物高度范围：0.05 到 0.45 米
            platform_width=1.5,  # 平台宽度 1.5 米
            border_width=0.0,  # 边界宽度 0
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙概率 30%
            wall_height=5.0,  # 墙高 5 米
            wall_thickness=0.05,  # 墙厚 0.05 米
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(  # 叠加 Perlin 噪声
                noise_scale=0.05,  # 噪声幅度 0.05 米
                noise_frequency=20,  # 噪声频率 20
                fractal_octaves=2,  # 分形八度 2
                fractal_lacunarity=2.0,  # 分形间隙度 2.0
                fractal_gain=0.25,  # 分形增益 0.25
                centering=True,  # 高度居中
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15, 0.20],  # 半径候选
                    max_height_diff=0.05  # 最大高度差 0.05 米
                ),
            },
        ),
        # 子地形：随机多盒网格障碍
        "mesh_boxes": terrain_gen.PerlinMeshRandomMultiBoxTerrainCfg(
            proportion=0.10,  # 占比 10%
            box_height_mean=[0.1, 0.4],  # 盒子高度均值范围：0.1 到 0.4 米
            box_height_range=0.05,  # 盒子高度变化范围：±0.05 米
            box_length_mean=0.4,  # 盒子长度均值：0.4 米
            box_length_range=0.1,  # 盒子长度变化范围：±0.1 米
            box_width_mean=0.4,  # 盒子宽度均值：0.4 米
            box_width_range=0.1,  # 盒子宽度变化范围：±0.1 米
            platform_width=1.5,  # 平台宽度 1.5 米
            generation_ratio=0.3,  # 生成比例：30% 的区域生成盒子
            no_perlin_at_obstacle=True,  # 障碍物处不加 Perlin 噪声
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙概率 30%
            wall_height=5.0,  # 墙高 5 米
            wall_thickness=0.05,  # 墙厚 0.05 米
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15],  # 半径候选
                    max_height_diff=0.05  # 最大高度差 0.05 米
                ),
            },
        ),
        # 子地形：倒金字塔斜坡
        "hf_pyramid_slope_inv": terrain_gen.PerlinInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,  # 占比 10%
            slope_range=(0.0, 0.7),  # 坡度范围：0.0 到 0.7（弧度或比例）
            platform_width=1.5,  # 平台宽度 1.5 米
            border_width=1.0,  # 边界宽度 1.0 米
            wall_prob=[0.3, 0.3, 0.3, 0.3],  # 四面墙概率 30%
            wall_height=5.0,  # 墙高 5 米
            wall_thickness=0.05,  # 墙厚 0.05 米
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(  # 叠加 Perlin 噪声
                noise_scale=0.00,  # 噪声幅度 0，即不添加噪声
                noise_frequency=20,  # 噪声频率 20
                fractal_octaves=2,  # 分形八度 2
                fractal_lacunarity=2.0,  # 分形间隙度 2.0
                fractal_gain=0.25,  # 分形增益 0.25
                centering=True,  # 高度居中
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,  # 采样 50 个平坦补丁
                    patch_radius=[0.05, 0.10, 0.15, 0.20],  # 半径候选
                    max_height_diff=0.05  # 最大高度差 0.05 米
                ),
            },
        ),
    },
)


# ============================================================
# 交互场景配置
# ============================================================
# @configclass 是 Isaac Lab 的装饰器，把普通类变成配置类，方便统一管理参数。
@configclass
class SceneCfg(InteractiveSceneCfg):
    # 场景配置类，继承自 InteractiveSceneCfg。
    # 作用：定义仿真场景里有哪些东西——地形、机器人、传感器、灯光、运动参考等。

    # 地面地形
    # terrain 是地形配置，类型为 TerrainImporterCfg。
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",  # 地形在仿真世界中的路径，叫 /World/ground
        terrain_type="generator",  # 地形类型为“生成器”，即用程序自动生成，而不是导入现成模型
        terrain_generator=ROUGH_TERRAINS_CFG,  # 使用之前定义好的粗糙地形生成器配置
        max_init_terrain_level=5,  # 课程学习中，初始地形难度最高到第 5 级
        collision_group=-1,  # 碰撞组编号，-1 通常表示默认或特殊组
        physics_material=sim_utils.RigidBodyMaterialCfg(  # 物理材质，决定摩擦和弹性
            friction_combine_mode="multiply",  # 摩擦系数合并方式为“相乘”
            restitution_combine_mode="multiply",  # 恢复系数（弹性）合并方式为“相乘”
            static_friction=1.0,  # 静摩擦系数 1.0
            dynamic_friction=1.0,  # 动摩擦系数 1.0
        ),
        visual_material=sim_utils.MdlFileCfg(  # 视觉材质，决定地形看起来什么样
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",  # 材质文件路径，指向砖块纹理
            project_uvw=True,  # 启用 UVW 投影
            texture_scale=(0.25, 0.25),  # 纹理缩放
        ),
        debug_vis=False,  # 不显示调试可视化
        virtual_obstacles={  # 虚拟障碍物配置
            "edges": GreedyconcatEdgeCylinderCfg(  # 名字叫 edges，一种边缘圆柱配置
                cylinder_radius=0.05,  # 圆柱半径 0.05 米
                min_points=2,  # 最少 2 个点构成边缘
            ),
        },
    )

    # 机器人，由具体机器人配置文件填充
    # robot 是机器人配置，类型为 ArticulationCfg（关节式机器人）。
    # MISSING 是占位符，表示这里先空着，具体用哪个机器人由具体任务配置文件填充。
    robot: ArticulationCfg = MISSING

    # 传感器
    # 左脚高度扫描器
    # left_height_scanner 本质是一个射线投射器（RayCaster），测量左脚到地面的距离。
    left_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",  # 安装在机器人左脚的 left_ankle_roll_link 上
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),  # 相对安装点偏移到 z=20 米高处，然后向下发射射线
        ray_alignment="yaw",  # 射线只跟随偏航角旋转，保持垂直向下
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),  # 射线模式为网格，分辨率 0.12 米，大小 x=0.12、y=0
        debug_vis=False,  # 不显示调试
        mesh_prim_paths=["/World/ground"],  # 只对地面发射射线，检测地面高度
        update_period=0.02,  # 每 0.02 秒更新一次（50Hz）
    )

    # 右脚高度扫描器
    # 与左脚类似，只是安装在 right_ankle_roll_link 上。
    right_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",  # 安装在机器人右脚的 right_ankle_roll_link 上
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),  # 相对安装点偏移到 z=20 米高处，然后向下发射射线
        ray_alignment="yaw",  # 射线只跟随偏航角旋转，保持垂直向下
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),  # 射线模式为网格，分辨率 0.12 米，大小 x=0.12、y=0
        debug_vis=False,  # 不显示调试
        mesh_prim_paths=["/World/ground"],  # 只对地面发射射线，检测地面高度
        update_period=0.02,  # 每 0.02 秒更新一次（50Hz）
    )

    # 接触力传感器，记录历史并跟踪离地时间
    # contact_forces 监控机器人所有链接的接触力。
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # 监控机器人所有链接
        history_length=3,  # 保留最近 3 步的历史数据
        track_air_time=True,  # 跟踪离地时间，用来判断脚是否着地、腾空多久
    )

    # 腿部体积点传感器
    # leg_volume_points 在脚踝周围生成 3D 网格点，用于碰撞检测、奖励计算或观测。
    leg_volume_points = VolumePointsCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",  # 安装在所有踝关节 roll link 上
        points_generator=Grid3dPointsGeneratorCfg(  # 生成 3D 网格点
            x_min=-0.025,  # x 方向最小值 -0.025 米
            x_max=0.12,  # x 方向最大值 0.12 米
            x_num=10,  # x 方向分 10 个点
            y_min=-0.03,  # y 方向最小值 -0.03 米
            y_max=0.03,  # y 方向最大值 0.03 米
            y_num=5,  # y 方向分 5 个点
            z_min=-0.04,  # z 方向最小值 -0.04 米
            z_max=0.0,  # z 方向最大值 0 米
            z_num=2,  # z 方向分 2 个点
        ),
        debug_vis=False,  # 不显示调试
    )

    # 带噪声的分组射线相机
    # camera 模拟深度相机，输出带噪声的深度图，用于给策略提供视觉观测。
    camera = NoisyGroupedRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",  # 安装在机器人躯干上
        mesh_prim_paths=[
            "/World/ground",  # 相机能看到地面
            # NOTE: Don't forget to add the robot links in robot-specific configuration file.
            # 注意：机器人链接需要在具体机器人配置里添加。
        ],
        ray_alignment="yaw",  # 射线随偏航旋转
        pattern_cfg=PinholeCameraPatternCfg(  # 针孔相机模式
            focal_length=1.0,  # 焦距 1.0
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),  # fovx，水平视场角 89.51 度
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),  # fovy，垂直视场角 58.29 度
            width=64,  # 图像宽度 64
            height=36,  # 图像高度 36
        ),
        debug_vis=False,  # 不显示调试
        data_types=["distance_to_image_plane"],  # 输出到图像平面的距离（深度图）
        update_period=0.02,  # 每 0.02 秒更新一次
        depth_clipping_behavior="max",  # 深度裁剪行为为“取最大值”
        offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(  # 相机相对于躯干的位姿
            # G1 Robot head camera nominal pose
            # G1 机器人头部相机的标称位姿
            pos=(
                0.0487988662332928,  # 位置偏移 x
                0.01,  # 位置偏移 y
                0.4378029937970051,  # 位置偏移 z
            ),
            rot=(
                0.9135367613482678,  # 旋转四元数 w
                0.004363309284746571,  # 旋转四元数 x
                0.4067366430758002,  # 旋转四元数 y
                0.0,  # 旋转四元数 z
            ),
            convention="world",  # 旋转约定为世界坐标系
        ),
        min_distance=0.1,  # 最小距离 0.1 米
        # noise
        # 噪声处理流程
        noise_pipeline={
            "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),  # 裁剪并缩放，裁剪区域 (18, 0, 16, 16)
            "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),  # 高斯模糊，核大小 3，sigma=1
            "depth_normalization": DepthNormalizationCfg(  # 深度归一化
                depth_range=(0.0, 2.5),  # 深度范围 (0.0, 2.5)
                normalize=True,  # 启用归一化
                output_range=(0.0, 1.0),  # 输出范围 (0.0, 1.0)
            ),
        },
        data_histories={"distance_to_image_plane_noised": 37},  # 保存加噪后的深度图历史，长度 37 帧
    )

    # 灯光
    # sky_light 是天空光，给场景提供环境光照。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",  # 灯光路径
        spawn=sim_utils.DomeLightCfg(  # 生成一个穹顶灯
            intensity=750.0,  # 强度 750
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",  # 使用 HDR 天空纹理
        ),
    )

    # 运动参考管理器，由具体任务填充
    # motion_reference 负责加载和管理参考动作，用于模仿学习或奖励计算。
    # MISSING 是占位符，具体由任务配置文件填充。
    motion_reference: MotionReferenceManagerCfg = MISSING

# ============================================================
# 观测配置
# ============================================================
@configclass  # 将类标记为配置类
class ObservationsCfg:  # 观测配置类，定义 MDP 的观测
    """Observation specifications for the MDP."""  # 文档字符串：MDP 的观测规范

    # ========================================================
    # 策略观测组
    # ========================================================
    @configclass  # 策略观测组的配置类
    class PolicyCfg(ObsGroup):  # 继承 ObsGroup，表示一组观测
        """Observations for policy group."""  # 策略组的观测

        # observation terms (order preserved)
        # 观测项，顺序保持不变
        # 基础角速度观测
        base_ang_vel = ObsTerm(  # 观测项：基础角速度
            func=mdp.base_ang_vel,  # 使用的函数：计算基础角速度
            noise=Unoise(n_min=-0.2, n_max=0.2),  # 添加均匀噪声，范围 -0.2 到 0.2
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True,  # 将历史维度展平
            scale=0.25,  # 缩放因子 0.25
        )
        # 投影重力观测
        projected_gravity = ObsTerm(  # 观测项：投影重力
            func=mdp.projected_gravity,  # 计算投影重力
            noise=Unoise(n_min=-0.05, n_max=0.05),  # 均匀噪声，范围 -0.05 到 0.05
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True,  # 展平历史维度
        )
        # 速度指令观测
        velocity_commands = ObsTerm(  # 观测项：速度指令
            func=mdp.generated_commands,  # 生成命令的函数
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={"command_name": "base_velocity"},  # 参数：命令名为 base_velocity
            noise=None,  # 不加噪声
        )
        # 关节位置观测
        joint_pos = ObsTerm(  # 观测项：关节位置
            func=mdp.joint_pos_rel,  # 计算相对关节位置
            noise=Unoise(n_min=-0.01, n_max=0.01),  # 均匀噪声，范围 -0.01 到 0.01
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True  # 展平历史维度
        )
        # 关节速度观测
        joint_vel = ObsTerm(  # 观测项：关节速度
            func=mdp.joint_vel_rel,  # 计算相对关节速度
            noise=Unoise(n_min=-0.5, n_max=0.5),  # 均匀噪声，范围 -0.5 到 0.5
            scale=0.05,  # 缩放因子 0.05
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True,  # 展平历史维度
        )
        # 上一时刻动作观测
        actions = ObsTerm(  # 观测项：上一时刻动作
            func=mdp.last_action,  # 获取上一时刻动作
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True  # 展平历史维度
        )
        # 深度图像观测
        depth_image = ObsTerm(  # 观测项：深度图像
            func=mdp.delayed_visualizable_image,  # 延迟可视化图像函数
            params={  # 参数字典
                "data_type": "distance_to_image_plane_noised_history",  # 数据类型：加噪后的历史深度图
                "sensor_cfg": SceneEntityCfg("camera"),  # 传感器配置：相机
                "history_skip_frames": 5,  # 历史跳帧：每隔 5 帧取一帧
                "num_output_frames": 8,  # 输出帧数：8 帧
                "delayed_frame_ranges": (0, 1),  # 延迟帧范围：(0, 1)
                "debug_vis": False,  # 不显示调试可视化
            },
            noise=None,  # 不加噪声
        )

        def __post_init__(self):  # 初始化后处理函数
            # 是否启用观测噪声/扰动
            self.enable_corruption = True  # 启用噪声/扰动
            # 是否拼接各观测项
            self.concatenate_terms = False  # 不拼接各观测项

    # ========================================================
    # 评论家观测组
    # ========================================================
    @configclass  # 评论家观测组的配置类
    class CriticCfg(ObsGroup):  # 继承 ObsGroup
        """Observations for critic group."""  # 评论家组的观测

        # observation terms (order preserved)
        # 观测项，顺序保持不变
        # 基础线速度观测
        base_lin_vel = ObsTerm(  # 观测项：基础线速度
            func=mdp.base_lin_vel,  # 计算基础线速度
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True  # 展平历史维度
        )
        # 基础角速度观测
        base_ang_vel = ObsTerm(  # 观测项：基础角速度
            func=mdp.base_ang_vel,  # 计算基础角速度
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True,  # 展平历史维度
            scale=0.25,  # 缩放因子 0.25
        )
        # 投影重力观测
        projected_gravity = ObsTerm(  # 观测项：投影重力
            func=mdp.projected_gravity,  # 计算投影重力
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True  # 展平历史维度
        )
        # 速度指令观测
        velocity_commands = ObsTerm(  # 观测项：速度指令
            func=mdp.generated_commands,  # 生成命令
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={"command_name": "base_velocity"},  # 命令名为 base_velocity
            noise=None,  # 不加噪声
        )
        # 关节位置观测
        joint_pos = ObsTerm(  # 观测项：关节位置
            func=mdp.joint_pos_rel,  # 相对关节位置
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True  # 展平历史维度
        )
        # 关节速度观测
        joint_vel = ObsTerm(  # 观测项：关节速度
            func=mdp.joint_vel_rel,  # 相对关节速度
            scale=0.05,  # 缩放因子 0.05
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True  # 展平历史维度
        )
        # 上一时刻动作观测
        actions = ObsTerm(  # 观测项：上一时刻动作
            func=mdp.last_action,  # 获取上一时刻动作
            history_length=8,  # 保留最近 8 步历史
            flatten_history_dim=True  # 展平历史维度
        )
        # 深度图像观测
        depth_image = ObsTerm(  # 观测项：深度图像
            func=mdp.delayed_visualizable_image,  # 延迟可视化图像函数
            params={  # 参数字典
                "data_type": "distance_to_image_plane_noised_history",  # 加噪历史深度图
                "sensor_cfg": SceneEntityCfg("camera"),  # 相机传感器
                "history_skip_frames": 5,  # 跳帧 5
                "num_output_frames": 8,  # 输出 8 帧
                "delayed_frame_ranges": (0, 1),  # 延迟帧范围
                "debug_vis": False,  # 不显示调试
            },
            noise=None,  # 不加噪声
        )

        def __post_init__(self):  # 初始化后处理
            # 评论家观测不启用扰动
            self.enable_corruption = False  # 不启用噪声/扰动
            # 是否拼接各观测项
            self.concatenate_terms = False  # 不拼接各观测项

    # ========================================================
    # AMP 策略状态观测组
    # ========================================================
    @configclass  # AMP 策略状态观测组配置类
    class AmpPolicyStateObsCfg(ObsGroup):  # 继承 ObsGroup
        concatenate_terms = False  # 不拼接各观测项
        # 投影重力
        projected_gravity = ObsTerm(  # 观测项：投影重力
            func=mdp.projected_gravity,  # 计算投影重力
            params={  # 参数
                "asset_cfg": SceneEntityCfg("robot"),  # 资产配置：机器人
            },
            history_length=10,  # 保留最近 10 步历史
        )
        # 关节相对位置
        joint_pos_rel = ObsTerm(  # 观测项：关节相对位置
            func=mdp.joint_pos_rel,  # 计算相对关节位置
            history_length=10,  # 保留最近 10 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={  # 参数
                "asset_cfg": SceneEntityCfg(  # 资产配置
                    name="robot",  # 名字：robot
                    preserve_order=True,  # 保持顺序
                ),
            },
        )
        # 关节相对速度
        joint_vel = ObsTerm(  # 观测项：关节相对速度
            func=mdp.joint_vel_rel,  # 计算相对关节速度
            scale=0.05,  # 缩放因子 0.05
            history_length=10,  # 保留最近 10 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={  # 参数
                "asset_cfg": SceneEntityCfg(  # 资产配置
                    name="robot",  # 名字：robot
                    preserve_order=True,  # 保持顺序
                ),
            },
        )
        # 基础线速度
        base_lin_vel = ObsTerm(  # 观测项：基础线速度
            func=mdp.base_lin_vel,  # 计算基础线速度
            history_length=10,  # 保留最近 10 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={  # 参数
                "asset_cfg": SceneEntityCfg("robot"),  # 资产配置：机器人
            },
        )
        # 基础角速度
        base_ang_vel = ObsTerm(  # 观测项：基础角速度
            func=mdp.base_ang_vel,  # 计算基础角速度
            history_length=10,  # 保留最近 10 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={  # 参数
                "asset_cfg": SceneEntityCfg("robot"),  # 资产配置：机器人
            },
        )

    # ========================================================
    # AMP 参考状态观测组
    # ========================================================
    @configclass  # AMP 参考状态观测组配置类
    class AmpReferenceStateObsCfg(ObsGroup):  # 继承 ObsGroup
        concatenate_terms = False  # 不拼接各观测项
        # 参考投影重力
        projected_gravity = ObsTerm(  # 观测项：参考投影重力
            func=mdp.projected_gravity_reference_as_state,  # 参考投影重力作为状态
            params={  # 参数
                "asset_cfg": SceneEntityCfg(name="motion_reference"),  # 资产配置：运动参考
            },
            history_length=10,  # 保留最近 10 步历史
        )
        # 参考关节相对位置
        joint_pos_rel = ObsTerm(  # 观测项：参考关节相对位置
            func=mdp.joint_pos_rel_reference_as_state,  # 参考关节相对位置作为状态
            history_length=10,  # 保留最近 10 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={  # 参数
                "asset_cfg": SceneEntityCfg(name="motion_reference"),  # 资产配置：运动参考
            },
        )
        # 参考关节相对速度
        joint_vel = ObsTerm(  # 观测项：参考关节相对速度
            func=mdp.joint_vel_rel_reference_as_state,  # 参考关节相对速度作为状态
            scale=0.05,  # 缩放因子 0.05
            history_length=10,  # 保留最近 10 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={  # 参数
                "asset_cfg": SceneEntityCfg(name="motion_reference"),  # 资产配置：运动参考
            },
        )
        # 参考基础线速度
        base_lin_vel = ObsTerm(  # 观测项：参考基础线速度
            func=mdp.base_lin_vel_reference_as_state,  # 参考基础线速度作为状态
            history_length=10,  # 保留最近 10 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={  # 参数
                "asset_cfg": SceneEntityCfg(name="motion_reference"),  # 资产配置：运动参考
            },
        )
        # 参考基础角速度
        base_ang_vel = ObsTerm(  # 观测项：参考基础角速度
            func=mdp.base_ang_vel_reference_as_state,  # 参考基础角速度作为状态
            history_length=10,  # 保留最近 10 步历史
            flatten_history_dim=True,  # 展平历史维度
            params={  # 参数
                "asset_cfg": SceneEntityCfg(name="motion_reference"),  # 资产配置：运动参考
            },
        )

    # observation group
    # 策略观测组实例
    policy: PolicyCfg = PolicyCfg()  # 实例化策略观测组
    # critic group
    # 评论家观测组实例
    critic: CriticCfg = CriticCfg()  # 实例化评论家观测组
    # AMP training groups
    # AMP 策略状态观测组实例
    amp_policy: AmpPolicyStateObsCfg = AmpPolicyStateObsCfg()  # 实例化 AMP 策略状态观测组
    # AMP 参考状态观测组实例
    amp_reference: AmpReferenceStateObsCfg = AmpReferenceStateObsCfg()  # 实例化 AMP 参考状态观测组


# ============================================================
# 动作配置
# ============================================================
@configclass  # 将类标记为配置类
class ActionsCfg:  # 动作配置类，定义 MDP 的动作
    """Action specifications for the MDP."""  # 文档字符串：MDP 的动作规范

    # 关节位置动作
    joint_pos = mdp.JointPositionActionCfg(  # 关节位置动作配置
        asset_name="robot",  # 资产名：robot
        joint_names=[".*"],  # 关节名：所有关节（正则匹配）
        scale=beyondmimic_action_scale,  # 动作缩放因子，使用 beyondmimic_action_scale
        use_default_offset=True  # 使用默认偏移
    )


# ============================================================
# 指令配置
# ============================================================
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # 位姿速度指令
    base_velocity = mdp.PoseVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        rel_standing_envs=0.05,
        ranges=mdp.PoseVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)),
        random_velocity_terrain=["perlin_rough_stand"],
        velocity_ranges={
            "perlin_rough": {"lin_vel_x": (0.45, 1.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "perlin_rough_stand": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "square_gaps": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_high": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "pyramid_stairs_inv_high": {
                "lin_vel_x": (0.45, 0.8),
                "lin_vel_y": (0.0, 0.0),
                "ang_vel_z": (-1.0, 1.0),
            },
            "boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "mesh_boxes": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
            "hf_pyramid_slope_inv": {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)},
        },
        only_positive_lin_vel_x=True,
        lin_vel_threshold=0.0,
        ang_vel_threshold=0.0,
        target_dis_threshold=0.4,
    )


# ============================================================
# G1 奖励配置
# ============================================================
@configclass
class G1Rewards:
    """Reward terms for the MDP."""

    # Task rewards
    # 跟踪 x/y 线速度奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # 跟踪 z 轴角速度奖励
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    # 朝向误差惩罚
    heading_error = RewTerm(func=mdp.heading_error, weight=-1.0, params={"command_name": "base_velocity"})
    # 等待/停滞惩罚
    dont_wait = RewTerm(func=mdp.dont_wait, weight=-0.5, params={"command_name": "base_velocity"})
    # 存活奖励
    is_alive = RewTerm(func=mdp.is_alive, weight=3.0)
    # 站立不动惩罚
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.3, params={"command_name": "base_velocity", "offset": 4.0})

    # Regularization rewards
    # 体积点穿透惩罚
    volume_points_penetration = RewTerm(
        func=mdp.volume_points_penetration,
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg("leg_volume_points"),
        },
    )
    # 足部离地时间奖励
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "vel_threshold": 0.15,
        },
    )
    # 足部滑动惩罚
    feet_slide = RewTerm(
        func=mdp.contact_slide,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 1.0,
        },
    )
    # 髋关节偏置惩罚
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_square,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    # x/y 角速度 L2 惩罚
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 关节力矩 L2 惩罚
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    # 关节加速度 L2 惩罚
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # 关节速度 L2 惩罚
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # 动作变化率 L2 惩罚
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # 平坦朝向惩罚
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    # 骨盆朝向惩罚
    pelvis_orientation_l2 = RewTerm(
        func=mdp.link_orientation, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}
    )
    # 足部接触时朝向惩罚
    feet_flat_ori = RewTerm(
        func=mdp.feet_orientation_contact,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # 足部贴近平面惩罚
    feet_at_plane = RewTerm(
        func=mdp.feet_at_plane,
        weight=-0.1,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "left_height_scanner_cfg": SceneEntityCfg("left_height_scanner"),
            "right_height_scanner_cfg": SceneEntityCfg("right_height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "height_offset": 0.035,
        },
    )
    # 双脚 x/y 距离接近奖励
    feet_close_xy = RewTerm(
        func=mdp.feet_close_xy_gauss,
        weight=0.4,
        params={
            "threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "std": math.sqrt(0.05),
        },
    )
    # 电机功率平方惩罚
    energy = RewTerm(
        func=mdp.motors_power_square,
        weight=-5e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]),
            "normalize_by_stiffness": True,
        },
    )
    # 上半身冻结惩罚
    freeze_upper_body = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.004,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist.*", "waist_.*"]
            ),
        },
    )

    # Safety rewards
    # 关节位置限位惩罚
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # 关节速度限位惩罚
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # 力矩限位惩罚
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_by_ratio,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "limit_ratio": 0.8,
        },
    )
    # 非期望接触惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_ankle_roll_link).*"),
            "threshold": 1.0,
        },
    )


# ============================================================
# 多奖励配置
# ============================================================
@configclass
class RewardsCfg(MultiRewardCfg):
    rewards: G1Rewards = G1Rewards()


# ============================================================
# 终止条件配置
# ============================================================
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 超时终止
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 离开地形边界终止
    terrain_out_bound = DoneTerm(func=mdp.terrain_out_of_bounds, time_out=True, params={"distance_buffer": 2.0})
    # 躯干非法接触终止
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        },
    )
    # 姿态异常终止
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    # 根节点高度过低终止
    root_height = DoneTerm(func=mdp.root_height_below_env_origin_minimum, params={"minimum_height": 0.5})
    # 数据集耗尽终止
    dataset_exhausted = DoneTerm(
        func=instinct_mdp.dataset_exhausted,
        time_out=True,
        params={
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "print_reason": False,
            "reset_without_notice": True,
        },
    )


# ============================================================
# 事件配置
# ============================================================
@configclass
class EventCfg:
    """Configuration for events."""

    # 物理材质随机化
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.6),
            "restitution_range": (0.05, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    # reset
    # 重置根节点状态
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    # 注册虚拟障碍到传感器
    register_virtual_obstacles = EventTerm(
        func=instinct_mdp.register_virtual_obstacle_to_sensor,
        mode="startup",
        params={
            "sensor_cfgs": SceneEntityCfg("leg_volume_points"),
        },
    )

    # 重置机器人关节
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.15, 0.15),
            "velocity_range": (0.0, 0.0),
        },
    )


# ============================================================
# 课程学习配置
# ============================================================
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # 地形等级课程
    terrain_levels = CurrTerm(
        func=mdp.tracking_exp_vel, params={"lin_vel_threshold": (0.3, 0.6), "ang_vel_threshold": (0.0, 0.0)}
    )


# ============================================================
# 监控配置
# ============================================================
@configclass
class MonitorCfg:
    pass


##
# Environment configuration
##
# ============================================================
# Parkour 环境配置
# ============================================================
@configclass
class ParkourEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    # 场景设置
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    # 基础设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    # MDP 设置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    monitors: MonitorCfg = MonitorCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        # 通用设置
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        # 仿真设置
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**29
        # update sensor update periods
        # 更新传感器更新周期
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt