from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import GaussianNoiseCfg
import isaaclab.envs.mdp as mdp


@configclass
class ObservationsCfg:
    """Observation group definitions for the robot policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=GaussianNoiseCfg(mean=0.0, std=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=GaussianNoiseCfg(mean=0.0, std=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
