# MuJoCo UR5 + Robotiq85 PPO 抓取训练

## 训练与测试结果可视化

训练并自动画图：
```powershell
python train.py --total-timesteps 300000 --n-envs 4 --plot-after-train
```

打开 TensorBoard：
```powershell
tensorboard --logdir runs/ppo_ur5_grasp/tensorboard
```

评估并自动画图：
```powershell
python evaluate.py --model runs/ppo_ur5_grasp/best_model/best_model.zip --episodes 30 --no-render --deterministic --plot-after-eval
```

单独重新画训练图：
```powershell
python scripts/plot_training.py --log-dir runs/ppo_ur5_grasp
```

单独重新画测试图：
```powershell
python scripts/plot_evaluation.py --log-dir runs/ppo_ur5_grasp
```

对比多个实验：
```powershell
python scripts/compare_experiments.py --experiments runs/ppo_baseline runs/ppo_rrt_hybrid --labels "Pure PPO" "RRT + PPO" --output-dir runs/comparison_figures
```

一键实验：
```powershell
python scripts/run_experiment.py --name ppo_baseline --total-timesteps 300000 --n-envs 4 --eval-episodes 30
```

CSV 指标默认写入：
- `runs/ppo_ur5_grasp/metrics/train_step_metrics.csv`
- `runs/ppo_ur5_grasp/metrics/train_episode_metrics.csv`
- `runs/ppo_ur5_grasp/metrics/eval_step_metrics.csv`
- `runs/ppo_ur5_grasp/metrics/eval_episode_metrics.csv`

PNG 图像默认写入 `runs/ppo_ur5_grasp/figures/`：
- `train_episode_reward.png`：训练回合奖励变化
- `train_success_rate.png`：训练成功率变化
- `train_cube_lift_height.png`：方块最大抬升高度变化
- `train_ee_cube_distance.png`：夹爪和方块距离变化
- `train_reward_terms.png`：奖励函数各部分贡献
- `train_eval_mean_reward.png`：训练期间评估平均奖励变化
- `train_ppo_losses.png`：PPO policy/value/entropy/approx_kl/clip_fraction 等诊断曲线
- `eval_episode_reward.png`：每个测试回合奖励
- `eval_success_bar.png`：测试成功/失败统计
- `eval_lift_height_per_episode.png`：每个测试回合最大抬升高度
- `eval_final_distance_per_episode.png`：每个测试回合最终夹爪-方块距离
- `eval_step_distance_height.png`：测试过程中距离和抬升高度随 step 的变化

如果曲线不明显，优先把训练步数增加到 `1000000` 或更高；也可以降低 `--learning-rate`（例如 `1e-4`）、略增 `--ent-coef` 鼓励探索，或先缩小 `env.py` 中 `_sample_cube_xy` 的采样范围，让策略先学会更稳定的基础抓取。

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

## 性能修复说明

本版本修复了只按方块高度判断成功导致的假成功问题：成功现在要求方块被抬高、夹爪仍靠近方块，并且方块没有异常飞高。环境还新增了 `cube_out_of_workspace` 终止条件，方块飞出、掉落或远离夹爪时会提前结束 episode 并给予惩罚。

奖励函数改为阶段式奖励，包含 `reward_pregrasp`、`reward_xy_align`、`reward_height_align`、`reward_close_gripper`、`reward_lift`、`reward_target` 和 `reward_action_penalty`。训练脚本新增课程学习参数 `--curriculum-level`，默认从更小的方块随机范围开始训练；PPO 默认参数也调整为更稳的长 horizon、较小学习率和较小 clip range。

推荐先跑课程 0：

```powershell
python train.py --log-dir runs/ppo_fix_curriculum_0 --total-timesteps 500000 --n-envs 4 --curriculum-level 0 --plot-after-train
python evaluate.py --model runs/ppo_fix_curriculum_0/best_model/best_model.zip --episodes 40 --no-render --deterministic --plot-after-eval
```

稳定后扩大采样范围：

```powershell
python train.py --log-dir runs/ppo_fix_curriculum_1 --total-timesteps 800000 --n-envs 4 --curriculum-level 1 --plot-after-train
```

判断性能是否提升时，重点看 `figures/` 里的图：`train_cube_lift_height.png` 不应再出现 8m 级异常尖峰，`train_ee_cube_distance.png` 不应再出现 10m 到 20m 的异常距离；`eval_final_distance_per_episode.png` 多数回合应低于 0.10m 到 0.15m，`eval_lift_height_per_episode.png` 更稳定，`eval_success_bar.png` 中 Success 数量应增加。`train_reward_terms.png` 和 `train_grasp_phase_terms.png` 用来观察靠近、XY 对齐、高度对齐、闭合夹爪和抬升阶段是否逐步学起来。
