from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

from .mdp import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .mdp.curriculum import CurriculumCfg
from .mdp.scene import Pm01SceneCfg


@configclass
class Pm01EnvCfg(ManagerBasedRLEnvCfg):
    scene: Pm01SceneCfg = Pm01SceneCfg(num_envs=4, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = None
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 10
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.episode_length_s = 24.0
