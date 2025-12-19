from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from engineai_lab.assets.pm01 import PM01_CFG


TARGET_COMMAND_NAME = "base_velocity"


def terrain_levels_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    target_velocity_names: str = TARGET_COMMAND_NAME,
):
    asset = env.scene["robot"]
    terrain = env.scene.terrain

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )

    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2

    try:
        command = env.command_manager.get_command(target_velocity_names)
        target_dist = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
        move_down = (distance < target_dist * 0.5) * ~move_up
    except KeyError:
        move_down = torch.zeros_like(move_up)

    return None


@configclass
class Pm01SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            curriculum=False,
            size=(10.0, 10.0),
            num_rows=10,
            num_cols=10,
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = PM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
