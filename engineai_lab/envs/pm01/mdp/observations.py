from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import GaussianNoiseCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat
import torch

# =============================================================================
# 开关：控制是否启用特权信息
# =============================================================================
ENABLE_PRIVILEGED_INFO = True

# =============================================================================
# 自定义 MDP 观测函数
# =============================================================================
# def root_body_mass(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """获取机器人的总质量 (Batch, 1)"""
#     asset = env.scene[asset_cfg.name]
#     return asset.root_physx_view.mass.unsqueeze(-1)

def base_euler_xyz(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """获取基座的欧拉角 (Roll, Pitch, Yaw)"""
    asset = env.scene[asset_cfg.name]
    # 获取四元数 (w, x, y, z)
    quat = asset.data.root_quat_w
    # 转换为欧拉角 (roll, pitch, yaw)
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    return torch.stack((roll, pitch, yaw), dim=-1)

# =============================================================================
# 观测配置
# =============================================================================
@configclass
class ObservationsCfg:
    """Observation group definitions for the robot policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor (策略) 的观测空间"""
        # 1. Base Linear Velocity
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=2.0,
            noise=GaussianNoiseCfg(mean=0.0, std=0.1),
        )
        # 2. Joint Positions
        dof_pos = ObsTerm(
            func=mdp.joint_pos,
            scale=1.0,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )
        # 3. Joint Velocities
        dof_vel = ObsTerm(
            func=mdp.joint_vel,
            scale=0.05,
            noise=GaussianNoiseCfg(mean=0.0, std=1.5),
        )
        # 4. Last Actions
        actions = ObsTerm(func=mdp.last_action, scale=1.0)
        
        # 5. Joint Position Error
        dof_pos_ref_diff = ObsTerm(
            func=mdp.joint_pos_rel, 
            scale=1.0,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )
        # 6. Base Angular Velocity
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=1.0,
            noise=GaussianNoiseCfg(mean=0.0, std=0.2),
        )
        
        # 7. Base Euler Angles [修改点: 使用自定义函数]
        base_euler_xyz = ObsTerm(
            func=base_euler_xyz,  # 这里调用上面定义的自定义函数
            scale=1.0,
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
        )
        
        # 8. Contact Mask
        # contact_mask = ObsTerm(
        #     func=mdp.contact_sensor,
        #     params={"sensor_cfg": mdp.ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll")},
        #     scale=1.0
        # )
        # 9. Height Measurements
        # height_measurements = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": mdp.RayCasterCfg(prim_path="{ENV_REGEX_NS}/Robot/height_scanner")},
        #     scale=5.0,
        #     clip=(-1.0, 1.0),
        # )
        # 10. Commands
        commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=1.0
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(PolicyCfg):
        if ENABLE_PRIVILEGED_INFO:
            # body_mass = ObsTerm(
            #     func=root_body_mass,
            #     params={"asset_cfg": SceneEntityCfg("robot")},
            #     scale=0.1
            # )
            # 其他特权信息...
            pass

        def __post_init__(self) -> None:
            super().__post_init__()
            self.enable_corruption = False 
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()