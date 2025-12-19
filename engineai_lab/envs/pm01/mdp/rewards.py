import math

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp as mdp


@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.4,
        params={"std": math.sqrt(0.25), "command_name": "base_velocity"},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.1,
        params={"std": math.sqrt(0.25), "command_name": "base_velocity"},
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
