"""Preview the Parkour procedural terrain without spawning a robot."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview InstinctLab Parkour terrain in Isaac Sim.")
parser.add_argument(
    "--color_scheme",
    type=str,
    choices=["height", "random", "none"],
    default="height",
    help="Terrain coloring scheme.",
)
parser.add_argument(
    "--curriculum",
    action="store_true",
    help="Enable the terrain curriculum instead of showing the full difficulty range.",
)
parser.add_argument("--rows", type=int, default=None, help="Optional number of terrain rows for a quick preview.")
parser.add_argument("--cols", type=int, default=None, help="Optional number of terrain columns for a quick preview.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from instinctlab.tasks.parkour.config.parkour_env_cfg import ROUGH_TERRAINS_CFG
from instinctlab.terrains import TerrainImporter, TerrainImporterCfg


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))

    terrain_cfg = ROUGH_TERRAINS_CFG.replace(
        curriculum=args_cli.curriculum,
        color_scheme=args_cli.color_scheme,
        num_rows=args_cli.rows if args_cli.rows is not None else ROUGH_TERRAINS_CFG.num_rows,
        num_cols=args_cli.cols if args_cli.cols is not None else ROUGH_TERRAINS_CFG.num_cols,
    )
    importer_cfg = TerrainImporterCfg(
        num_envs=1,
        env_spacing=2.5,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_cfg,
        debug_vis=False,
    )

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)
    TerrainImporter(importer_cfg)

    terrain_width = terrain_cfg.size[0] * terrain_cfg.num_cols
    terrain_depth = terrain_cfg.size[1] * terrain_cfg.num_rows
    sim.set_camera_view(
        eye=[terrain_width * 0.72, terrain_depth * 0.95, max(terrain_width, terrain_depth) * 0.8],
        target=[0.0, 0.0, 0.0],
    )
    sim.reset()
    print("[INFO] Terrain preview ready. Close the Isaac Sim window to exit.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
