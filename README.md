# MuJoCo UR5 + Robotiq85 PPO 抓取训练

这个文件夹里是一套可以直接运行的 MuJoCo 抓取训练工程：

- `assets/ur5_robotiq85_scene.xml`：UR5 + Robotiq85 风格夹爪 + 方块 + 桌面的 MJCF 场景
- `src/ur5_grasp_env/env.py`：Gymnasium 环境
- `train.py`：使用 Stable-Baselines3 的 PPO 训练脚本
- `evaluate.py`：加载训练好的模型并可视化测试
- `requirements.txt`：Python 依赖

说明：这里内置的是“简化几何版”UR5 和 Robotiq85，不依赖外部 mesh 文件，适合先把 PPO 训练流程跑通。如果后续你有学校/实验室提供的官方 UR5 XML/URDF 和 Robotiq85 模型，可以替换 `assets/ur5_robotiq85_scene.xml`，但建议保留这些名字：`shoulder_pan_joint`、`shoulder_lift_joint`、`elbow_joint`、`wrist_1_joint`、`wrist_2_joint`、`wrist_3_joint`、`left_finger_joint`、`right_finger_joint`、`pinch_site`、`cube`、`lift_target`。

## 1. 创建环境

推荐使用 Python 3.10 或 3.11。

```powershell
cd D:\CodeMujoco\mujoco_robotarm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果你的机器没有 `python` 命令，可以把上面的 `python` 换成 Anaconda 环境里的 Python。

## 2. 检查环境是否能加载

```powershell
python train.py --check-env --total-timesteps 1000 --n-envs 1
```

这一步会创建 MuJoCo 环境并短训练 1000 步。能看到 `Saved final model...` 就说明基本流程通了。

## 3. 开始 PPO 训练

CPU 训练可以先用：

```powershell
python train.py --total-timesteps 300000 --n-envs 4
```

如果电脑比较慢：

```powershell
python train.py --total-timesteps 100000 --n-envs 1
```

训练结果会保存在：

- `runs/ppo_ur5_grasp/ppo_ur5_grasp_final.zip`
- `runs/ppo_ur5_grasp/best_model/best_model.zip`
- `runs/ppo_ur5_grasp/checkpoints/`

## 4. 可视化测试

```powershell
python evaluate.py --model runs/ppo_ur5_grasp/best_model/best_model.zip --episodes 5 --deterministic
```

如果只想在命令行测试，不打开窗口：

```powershell
python evaluate.py --model runs/ppo_ur5_grasp/best_model/best_model.zip --episodes 20 --no-render --deterministic
```

## 5. 代码结构和训练逻辑

环境动作是 7 维：

- 前 6 维控制 UR5 六个关节的目标角度增量
- 第 7 维控制夹爪开合，`-1` 代表张开，`1` 代表闭合

环境观测包含：

- 机械臂关节角、关节速度
- 两个夹爪滑动关节的位置和速度
- 夹爪中心点 `pinch_site` 位置
- 方块位置和姿态
- 目标点位置
- 方块相对夹爪的位置

奖励函数主要由四部分组成：

- 夹爪靠近方块的距离奖励
- 靠近后闭合夹爪的奖励
- 方块被抬起后的高度奖励
- 方块靠近目标点的奖励

成功条件是方块被抬高超过桌面约 12 cm。

## 6. 替换成更真实的 UR5/Robotiq85 模型

如果你拿到了真实 UR5 XML 或 URDF：

1. 优先转换成 MuJoCo MJCF/XML。
2. 保留环境代码中使用的 joint、actuator、site、body 名字，或者同步修改 `src/ur5_grasp_env/env.py` 里的名称列表。
3. 确保夹爪末端有一个 site，名字为 `pinch_site`。
4. 确保被抓取物体 body 名字为 `cube`，自由关节名字为 `cube_freejoint`。
5. 确保 PPO 动作仍能映射到 6 个机械臂关节和 1 个夹爪开合指令。

真实模型接入后，建议先运行：

```powershell
python train.py --check-env --total-timesteps 1000 --n-envs 1
```

## 7. 常见问题

如果 `ModuleNotFoundError: No module named 'gymnasium'` 或 `No module named 'stable_baselines3'`，说明还没有安装依赖，请重新执行 `pip install -r requirements.txt`。

如果 MuJoCo 可视化窗口打不开，先用无窗口模式确认训练逻辑：

```powershell
$env:PYTHONPATH=".\src"
python evaluate.py --no-render
```

如果训练很久仍抓不起来，可以先增加训练步数到 `1000000`，或者降低随机化范围，把 `src/ur5_grasp_env/env.py` 中 `_sample_cube_xy` 的范围缩小，让策略先学会固定位置抓取。
