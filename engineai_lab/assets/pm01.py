import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from engineai_lab import ENGINEAI_LAB_EXT_DIR

# 定义 PM01 机器人的配置类
PM01_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(ENGINEAI_LAB_EXT_DIR, "assets/robots/biped/pm01/pm01.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9), # 对应 init_state.pos
        # 对应 init_state.default_joint_angles
        joint_pos={
            "j00_hip_pitch_l": -0.24,
            "j01_hip_roll_l": 0.0,
            "j02_hip_yaw_l": 0.0,
            "j03_knee_pitch_l": 0.48,
            "j04_ankle_pitch_l": -0.24,
            "j05_ankle_roll_l": 0.0,
            "j06_hip_pitch_r": -0.24,
            "j07_hip_roll_r": 0.0,
            "j08_hip_yaw_r": 0.0,
            "j09_knee_pitch_r": 0.48,
            "j10_ankle_pitch_r": -0.24,
            "j11_ankle_roll_r": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # 定义执行器 (Actuators)，对应 control.stiffness 和 control.damping
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            # 将 Isaac Gym 中的 P 控制映射为 ImplicitActuator
            stiffness={
                ".*_hip_pitch_.*": 70.0,
                ".*_hip_roll_.*": 50.0,
                ".*_hip_yaw_.*": 50.0,
                ".*_knee_pitch_.*": 70.0,
                ".*_ankle_pitch_.*": 20.0,
                ".*_ankle_roll_.*": 20.0,
            },
            damping={
                ".*_hip_pitch_.*": 7.0,
                ".*_hip_roll_.*": 5.0,
                ".*_hip_yaw_.*": 5.0,
                ".*_knee_pitch_.*": 7.0,
                ".*_ankle_pitch_.*": 0.2,
                ".*_ankle_roll_.*": 0.2,
            },
        ),
    },
    soft_joint_pos_limit_factor=0.9, # 对应 soft limits 的安全配置
)