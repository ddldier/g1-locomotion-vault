"""Offline registry/config checks. No Isaac runtime or physics is exercised."""

import ast
import copy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "source/instinctlab/instinctlab"
G1 = PKG / "tasks/shadowing/perceptive/config/g1"


class Config:
    def __init__(self, **kwargs):
        for cls in reversed(type(self).__mro__):
            for key, value in vars(cls).items():
                if not key.startswith("_") and not callable(value):
                    setattr(self, key, copy.deepcopy(value))
        self.__dict__.update(kwargs)
        post_init = getattr(self, "__post_init__", None)
        if post_init:
            post_init()


class Critic(Config):
    task_id = Config(shape=(1,))


class BaseEnv(Config):
    def __post_init__(self):
        self.scene = Config(
            height_scanner=Config(
                prim_path="{ENV_REGEX_NS}/Robot/pelvis",
                pattern_cfg=Config(resolution=0.1, size=(3.0, 0.8)),
                mesh_prim_paths=["/World/ground", Config(
                    prim_expr="/World/envs/env_.*/VaultBox", track_mesh_transforms=True,
                )],
            ),
            loco_height_scanner=Config(mesh_prim_paths=["/World/ground"]),
            camera=Config(),
            motion_reference=Config(
                debug_vis=True,
                reference_prim_path="/World/envs/env_.*/RobotReference/pelvis",
                motion_buffers={"MimicKitSpeedVault": Config(
                    motion_bin_length_s=1.0,
                    env_starting_stub_sampling_strategy="concat_motion_bins",
                    motion_start_from_middle_range=[0.0, 0.0],
                )},
            ),
        )
        self.events = Config(randomize_ray_offsets=Config())
        self.sim = Config(dt=0.005)
        self.decimation = 4


class PlayEnv(BaseEnv):
    def __post_init__(self):
        super().__post_init__()
        self.play_setup_ran = True


class Runner(Config):
    algorithm = Config(class_name="DualTeacherDagger", using_ppo=False)
    normalizers = Config(policy=Config(class_name="EmpiricalNormalization"))


def registry():
    records = []
    gym = types.ModuleType("gymnasium")
    gym.register = lambda **kwargs: records.append(kwargs)
    with patch.dict(sys.modules, {"gymnasium": gym}):
        exec(compile((PKG / "tasks/__init__.py").read_bytes(), "tasks/__init__.py", "exec"), {})
    return records


def height_config():
    names = {
        "UnifiedHeightPolicyObsCfg", "UnifiedHeightObservationsCfg",
        "_configure_height_student", "G1UnifiedHeightDaggerEnvCfg",
        "G1UnifiedHeightDaggerEnvCfg_PLAY",
    }
    mdp = types.SimpleNamespace(**{
        key: (lambda *args, **kwargs: None)
        for key in ("height_scan", "projected_gravity", "base_ang_vel", "joint_pos_rel", "joint_vel", "last_action")
    })
    ns = dict(
        deepcopy=copy.deepcopy, configclass=lambda cls: cls,
        ObsGroupCfg=Config, ObsTermCfg=Config,
        SceneEntityCfg=lambda name: Config(name=name), UniformNoiseCfg=Config,
        mdp=mdp, patterns=types.SimpleNamespace(GridPatternCfg=Config),
        DualCriticObsCfg=Critic, G1DualTeacherDaggerEnvCfg=BaseEnv,
        G1DualTeacherDaggerEnvCfg_PLAY=PlayEnv,
        dual_student_velocity_command=lambda env: None,
        vault=types.SimpleNamespace(vault_cfg=types.SimpleNamespace(MOTION_NAME="MimicKitSpeedVault")),
    )
    tree = ast.parse((G1 / "dual_teacher_dagger_env_cfg.py").read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "height_config", "exec"), ns)
    ns.update(InstinctRlActorCriticCfg=Config, G1DualTeacherDaggerRunnerCfg=Runner)
    tree = ast.parse((G1 / "agents/dual_teacher_dagger_cfg.py").read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in {
        "UnifiedHeightPolicyCfg", "G1UnifiedHeightDaggerRunnerCfg",
    }]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "height_runner", "exec"), ns)
    return types.SimpleNamespace(**ns)


class ProjectTaskTests(unittest.TestCase):
    def test_only_ten_unique_tasks(self):
        records = registry()
        ids = {r["id"] for r in records}
        self.assertEqual(len(records), 10)
        self.assertEqual(len(ids), 10)
        self.assertEqual(sum("Play" in name for name in ids), 5)
        self.assertEqual(sum("Height" in name for name in ids), 2)

    def test_every_entry_point_resolves(self):
        for record in registry():
            for entry in (record["entry_point"], *record["kwargs"].values()):
                module, name = entry.split(":")
                path = ROOT / "source/instinctlab" / Path(*module.split("."))
                path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
                tree = ast.parse(path.read_text())
                declared = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
                declared.update(a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names)
                self.assertIn(name, declared, entry)

    def test_no_distributed_registration(self):
        sites = [p.relative_to(PKG / "tasks").as_posix()
                 for p in (PKG / "tasks").rglob("*.py") if "gym.register(" in p.read_text()]
        self.assertEqual(sites, ["__init__.py"])
        self.assertFalse((G1 / "unified_height_student_cfg.py").exists())
        self.assertNotIn("import_packages", (PKG / "tasks/__init__.py").read_text())

    def test_no_registration_from_subpackage_imports(self):
        # All task package initializers other than the central registry are empty
        # or docstrings, so importing configuration parents cannot add old tasks.
        for path in (PKG / "tasks").rglob("__init__.py"):
            if path.parent == PKG / "tasks":
                continue
            tree = ast.parse(path.read_text())
            self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                 and n.func.attr == "register" for n in ast.walk(tree)), str(path))

    def test_height_observation_and_sensors(self):
        cfg = height_config()
        policy = cfg.UnifiedHeightPolicyObsCfg()
        terms = [key for key, value in vars(policy).items() if isinstance(value, Config) and hasattr(value, "func")]
        self.assertEqual(terms, ["height_scan", "velocity_command", "projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action"])
        self.assertEqual(sum((31 * 17, 3, 3, 3, 29, 29, 29)), 623)
        self.assertIsNone(policy.history_length)
        self.assertEqual(policy.height_scan.history_length, 3)
        self.assertEqual(policy.velocity_command.history_length, 0)
        for name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action"):
            self.assertEqual(getattr(policy, name).history_length, 5)
        for cls in (cfg.G1UnifiedHeightDaggerEnvCfg, cfg.G1UnifiedHeightDaggerEnvCfg_PLAY):
            env = cls()
            original = env.scene.height_scanner
            scanner = env.scene.student_height_scanner
            self.assertEqual(original.pattern_cfg.size, (3.0, 0.8))
            self.assertEqual(scanner.pattern_cfg.size, (3.0, 1.6))
            self.assertIsNot(scanner.mesh_prim_paths, original.mesh_prim_paths)
            self.assertTrue(scanner.mesh_prim_paths[1].track_mesh_transforms)
            self.assertEqual(scanner.mesh_prim_paths[1].prim_expr, "/World/envs/env_.*/VaultBox")
            self.assertEqual(env.scene.loco_height_scanner.mesh_prim_paths, ["/World/ground"])
            self.assertEqual(scanner.update_period, env.sim.dt * env.decimation)
            self.assertIsNone(env.scene.camera)
            self.assertIsNone(env.events.randomize_ray_offsets)
            self.assertIsInstance(env.observations.critic, Critic)
            reference = env.scene.motion_reference
            self.assertFalse(reference.debug_vis)
            self.assertIsNone(reference.reference_prim_path)
            motion = reference.motion_buffers["MimicKitSpeedVault"]
            self.assertIsNone(motion.motion_bin_length_s)
            self.assertEqual(motion.env_starting_stub_sampling_strategy, "independent")
            self.assertEqual(motion.retargeted_frame_range, (102, 665))
            expected = [0.0, 0.9] if cls is cfg.G1UnifiedHeightDaggerEnvCfg else [0.0, 0.0]
            self.assertEqual(motion.motion_start_from_middle_range, expected)
        self.assertTrue(cfg.G1UnifiedHeightDaggerEnvCfg_PLAY().play_setup_ran)

    def test_mlp_fresh_run_and_raw_critic(self):
        runner = height_config().G1UnifiedHeightDaggerRunnerCfg()
        self.assertEqual(runner.policy.class_name, "ActorCritic")
        self.assertFalse(hasattr(runner.policy, "encoder_configs"))
        self.assertFalse(runner.resume)
        self.assertIsNone(runner.load_run)
        self.assertFalse(runner.algorithm.using_ppo)
        self.assertFalse(hasattr(runner.normalizers, "critic"))


if __name__ == "__main__":
    unittest.main()
