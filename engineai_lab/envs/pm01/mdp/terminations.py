from collections.abc import Sequence

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab.envs.mdp as mdp


def time_out_termination(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> bool:
    return env.episode_length_buf[env_ids] >= 50


def torso_ground_contact(env: ManagerBasedRLEnv, height_threshold: float = 0.5):
    robot = env.scene["robot"].data
    try:
        link_index = robot.body_names.index("link_base")
        base_height = robot.body_pos_w[:, link_index, 2]
    except ValueError:
        base_height = robot.root_pos_w[:, 2]
    return base_height < height_threshold


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(func=torso_ground_contact)
