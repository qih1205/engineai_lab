"""PM01 environment registration helpers."""

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

from .pm01_env_cfg import Pm01EnvCfg


def _make_pm01_env(**kwargs) -> ManagerBasedRLEnv:
    kwargs.pop("render_mode", None)
    env_cfg = Pm01EnvCfg()
    return ManagerBasedRLEnv(env_cfg)


gym.register(
    id="Template-PM01-Rough-v0",
    entry_point=_make_pm01_env,
    disable_env_checker=True,
)
