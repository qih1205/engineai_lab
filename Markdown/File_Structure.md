engineai_lab/
├── assets/                     # 存放机器人 URDF 和 Meshes
│   └── pm01/
│       ├── urdf/
│       └── meshes/
├── engineai_lab/
│   ├── __init__.py
│   ├── algo/                   # 【新增】PPO 算法实现
│   │   ├── __init__.py
│   │   ├── ppo/
│   │   │   ├── __init__.py
│   │   │   ├── actor_critic.py # 策略与价值网络模型
│   │   │   ├── ppo.py          # PPO 核心更新逻辑
│   │   │   └── storage.py      # Rollout 数据存储
│   ├── envs/
│   │   ├── __init__.py
│   │   └── pm01/
│   │       ├── __init__.py
│   │       ├── pm01_env_cfg.py # 主环境配置文件 (Pm01EnvCfg)
│   │       └── mdp/            # MDP 组件的模块化定义
│   │           ├── __init__.py
│   │           ├── actions.py    # 动作空间定义
│   │           ├── observations.py # 观测空间定义
│   │           ├── rewards.py    # 奖励函数定义
│   │           ├── events.py     # 域随机化和重置事件定义
│   │           ├── terminations.py # 终止条件定义
│   │           ├── commands.py   # 命令生成器定义
│   │           └── scene.py      # 场景和机器人资产定义
│   └── utils/                  # [保留] 原有的工具库
├── scripts/
│   └── train.py                # 训练入口脚本 (待完善)
├── setup.py                    # Python 包安装脚本
└── pyproject.toml              # 代码格式化配置文件