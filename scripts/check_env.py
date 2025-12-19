import argparse
from isaaclab.app import AppLauncher

# 1. 解析参数 (必须在导入 isaaclab 之前)
parser = argparse.ArgumentParser(description="Check the PM01 environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. 启动 Isaac Sim 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. 延迟导入 (必须在 app 启动后)
import torch
import gymnasium as gym
import isaaclab.envs
# 导入环境以触发注册
import engineai_lab.envs.pm01

def main():
    print("[INFO] Creating environment...")
    # 创建环境
    env = gym.make("Template-PM01-Rough-v0", render_mode=None)
    
    print(f"[INFO] Environment created. Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    # 重置环境
    print("[INFO] Resetting environment...")
    obs, _ = env.reset()
    print(f"[INFO] Reset successful. Obs shape: {obs['policy'].shape}")

    # 执行一步动作
    print("[INFO] Stepping environment...")
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    obs, rew, terminated, truncated, info = env.step(actions)
    
    print("[INFO] Step successful. Reward shape:", rew.shape)
    
    env.close()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()