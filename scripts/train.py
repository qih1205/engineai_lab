import argparse
import os
from datetime import datetime
from isaaclab.app import AppLauncher

# 1. 解析参数
parser = argparse.ArgumentParser(description="Train PM01 with RSL-RL.")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. 导入依赖
import gymnasium as gym
import isaaclab.envs
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# 导入环境和配置
import engineai_lab.envs.pm01
from engineai_lab.algo.ppo.runner_cfg import Pm01PPORunnerCfg

def main():
    # 设置日志目录
    log_root = os.path.join(os.path.dirname(__file__), "../logs/rsl_rl/pm01")
    experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, experiment_name)
    print(f"[INFO] Logging to: {log_dir}")

    # 创建环境
    env = gym.make("Template-PM01-Rough-v0", render_mode="rgb_array")

    # 实例化算法配置
    agent_cfg = Pm01PPORunnerCfg()
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed

    # 让 runner 使用环境所在设备
    agent_cfg.device = str(env.unwrapped.device)

    # 包装环境以符合 RSL-RL 接口
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建 Runner 并开始训练
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    env.close()

if __name__ == "__main__":
    main()