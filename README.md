# T-WAVES V1.1

T-WAVES V1.1 是面向工业作业的通用具身智能融合模型；技术架构名为 **T-WM-VFLA-S**。

- **T**：任务系统、机器人本体与部署工程。
- **WM**：Goal World Model，用于 Stage2 生成或编码 subgoal image / subgoal latent，辅助策略形成中间目标。
- **VFLA**：视觉、语言、力觉与动作联合策略。
- **S**：外挂安全层，在模型外进行规则校验、动作限幅、危险动作拒绝、碰撞约束与急停。

当前训练路线是“预训练基模 + 小规模微调”，不是从零大规模预训练。所有数据进入训练前都统一转换成 **LeRobot 格式**。

## 1. 数据统一规范

T-waves 训练时不直接混用原始坐标系。UMI、Ego、真机数据先进入一个中间 episode JSON，再由 `scripts/build_lerobot_dataset.py` 统一写成 LeRobot 数据集。

### 1.1 必需字段

LeRobot 数据集中每一帧包含：

```text
observation.images.main          主视角图像
observation.images.left_wrist    左腕部 / 操作视角图像
observation.images.right_wrist   右腕部 / 操作视角图像
observation.images.goal          Stage2 目标图像
observation.state                16维统一状态
action                           16维统一动作
action_mask                      动作有效维度
wrench                           6维 TCP 力 / 力矩
next_wrench                      下一时刻 TCP 力 / 力矩
force_mask                       力觉是否有效
action_valid                     动作监督是否有效
embodiment_id                    机器人 / 数据本体标识
source_id                        UMI / 机器人 / Ego 数据来源
action_space_id                  动作表示标识
subtask_id                       子任务标签
goal_source                      goal 来源标签
camera_mask                      [main, left_wrist, right_wrist]
task                             自然语言任务
```

### 1.2 图像视角约定

```text
camera_mask = [main, left_wrist, right_wrist]
```

常见设置：

- UMI 数据：通常没有主视角，使用操作相机或夹爪视角，`camera_mask = [0, 1, 1]` 或 `[0, 1, 0]`。
- Ego 数据：通常只有头戴主视角，`camera_mask = [1, 0, 0]`。
- 真机数据：建议采集主视角 + 两个腕部视角，`camera_mask = [1, 1, 1]`。

缺失的图像字段仍会被写入占位图像，但必须通过 `camera_mask` 告诉模型哪些视角真实可用。

### 1.3 动作空间约定

模型固定输出 16 维 action chunk。不同来源的数据通过 `action_mask` 和 `action_space_id` 对齐。

推荐主动作空间：

```text
tcp_local_delta:
[dx, dy, dz, drx, dry, drz, dgripper]
```

- 单臂：前 7 维有效。
- 双臂：前 14 维有效，即左手 7 维 + 右手 7 维。
- 额外维度：补 0，并用 `action_mask=0` 屏蔽。
- 无动作监督的普通 Ego 视频：`action_valid=0` 且 `action_mask=0`。

动作空间 ID 当前建议：

```text
0 = tcp_local_delta
1 = joint_delta / force-aware robot action
2 = ego_video_no_action
```

如果你要把 UMI、Ego、真机一起训练，最稳的做法是：Stage1/Stage2 尽量使用 `tcp_local_delta`；Stage3 再引入力觉和关节增量分支，并通过 `action_space_id` 区分。

## 2. 三类数据如何转换

### 2.1 UMI 数据

UMI 数据通常提供夹爪中心点或指尖附近坐标系下的轨迹。转换时使用“当前夹爪中心点坐标系下的下一步局部增量”：

```text
T_delta = inv(T_gripper_t) @ T_gripper_t+1
action = [delta_xyz, delta_rotvec, delta_gripper]
```

示例：

```bash
python scripts/prepare_twaves_embodiment_data.py \
  --umi-root /path/to/umi_lerobot_or_processed_data \
  --robot-root /path/to/robot_lerobot_or_processed_data \
  --ego-root /path/to/ego_raw_data \
  --robot-urdf /path/to/configured_robot.urdf \
  --output data/processed/twaves_stage12_mix \
  --umi-episode 0 \
  --robot-episode 0 \
  --ego-session session_000001 \
  --max-frames 256 \
  --tool-offset-m 0.155
```

说明：

- `--umi-root` 可以是已下载的 UMI/类 UMI 数据根目录。
- UMI 没有主视角时不要硬造主视角语义，交给 `camera_mask` 表示缺失。
- UMI 的动作默认对齐到夹爪中心点局部坐标系，不要再当作机器人 base 坐标系增量使用。

### 2.2 Ego 数据

Ego 数据分两种。

第一种是普通第一视角视频，没有可靠手部轨迹。这类数据只用于 Stage2 的目标图像、子任务和语义监督：

```bash
python scripts/prepare_ego_video_data.py \
  --input-root /path/to/ego_videos \
  --output-root data/processed/ego_video_only \
  --max-videos 20 \
  --max-frames-per-video 128 \
  --stride 6 \
  --image-size 256
```

这类数据会设置：

```text
action_valid = 0
action_mask = 0
action_space_id = 2
```

第二种是带 SLAM 与手部位姿重建的 Ego 数据。推荐使用“手部相对头部 / 相机轨迹 + SLAM 世界位姿”组合成可训练动作：

```text
T_world_hand = T_world_camera @ T_camera_hand
action = inv(T_world_hand_t) @ T_world_hand_t+1
```

在 `prepare_twaves_embodiment_data.py` 中，Ego raw 数据默认读取：

```text
ego_root/session_xxx/slam/pose.csv
ego_root/session_xxx/transform/left_hand/00_wrist.csv
ego_root/session_xxx/transform/right_hand/00_wrist.csv
ego_root/session_xxx/video/camera.mp4
```

如果手部关键点存在，脚本会用拇指与食指指尖距离估计 pinch / gripper。

### 2.3 真机数据

真机数据建议至少包含：

```text
主视角 RGB
左腕部 RGB
右腕部 RGB，可选但推荐
关节状态
抓夹开合
动作目标或下一时刻状态
6维力 / 力矩，可选但 Stage3 推荐
自然语言任务文本
```

如果原始动作是绝对关节目标，先转为小的关节增量：

```text
joint_delta = q_target - q_current
```

如果要和 UMI / Ego 的 TCP 动作一起训练，建议通过配置好的 URDF 做 FK，把当前关节和目标关节都转成 TCP pose，再计算 TCP 局部增量：

```text
T_tcp_current = FK(q_current) @ T_tool
T_tcp_target  = FK(q_target)  @ T_tool
action = inv(T_tcp_current) @ T_tcp_target
```

示例：

```bash
python scripts/prepare_twaves_embodiment_data.py \
  --umi-root /path/to/umi_data \
  --robot-root /path/to/robot_lerobot_data \
  --ego-root /path/to/ego_data \
  --robot-urdf /path/to/configured_robot.urdf \
  --output data/processed/twaves_stage12_mix \
  --robot-episode 0 \
  --max-frames 512 \
  --tool-offset-m 0.155
```

注意：

- README 不绑定具体机器人型号；只要求提供当前本体对应的 URDF。
- `--tool-offset-m` 是法兰到工具中心点 TCP 的 z 方向外移距离，按实际工具配置填写。
- 力 / 力矩建议统一到 TCP 坐标系；如果原始传感器在法兰，需要补偿工具偏置造成的力矩变化。

## 3. 构建 LeRobot 数据集

中间 episode JSON 生成后，统一写成 LeRobot：

```bash
python scripts/build_lerobot_dataset.py \
  --input-root data/processed/twaves_stage12_mix \
  --output-root data/lerobot/twaves_stage12_mix \
  --repo-id local/twaves_stage12_mix \
  --fps 20 \
  --image-size 224 \
  --overwrite
```

建议构建完成后检查：

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="local/twaves_stage12_mix",
    root="data/lerobot/twaves_stage12_mix",
)
print("frames:", len(ds), "fps:", ds.fps)
sample = ds[0]
for key in [
    "observation.state",
    "action",
    "action_mask",
    "camera_mask",
    "source_id",
    "action_space_id",
    "action_valid",
]:
    print(key, sample[key])
PY
```

重点看：

- `action` 是否是小增量，而不是绝对大数值。
- `observation.state` 单位是否统一，比如关节角使用弧度而不是角度。
- `action_mask` 是否正确屏蔽单臂 / Ego / 缺失动作维度。
- `camera_mask` 是否反映真实视角可用性。

## 4. 三阶段训练流程

训练依赖环境示例：

```bash
export ROOT=/home/tc/T-WM-VFLA-S
export PY=/home/tc/miniconda3/envs/t-wm-vfla-s/bin/python
export PYTHONPATH=$ROOT:$ROOT/src:$ROOT/external/foundation_runtime/src:$ROOT/external/foundation_runtime/third_party/pi_transformers/src
export CUDA_VISIBLE_DEVICES=0

BASE=checkpoints/foundation_policy/pretrained_model
TOKENIZER=checkpoints/foundation_tokenizer
DATA=data/lerobot/twaves_stage12_mix
REPO=local/twaves_stage12_mix
OUT=checkpoints/twaves_v1_1/my_run
mkdir -p $OUT
```

### 4.1 Stage1A：动作接口适配

目标：先不大幅破坏基模，只训练新本体和动作接口。

训练内容：

- State Projector
- Action Input / Output Projection
- Embodiment Embedding
- Action Space Embedding

命令：

```bash
$PY scripts/train_twaves_lerobot.py \
  --stage 1A \
  --dataset-root $DATA \
  --repo-id $REPO \
  --foundation-checkpoint $BASE \
  --tokenizer $TOKENIZER \
  --output $OUT/stage1A.pt \
  --steps 1000 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --fps 20 \
  --skip-inference
```

### 4.2 Stage1B：动作专家微调

目标：让 Action Expert 学会目标本体的动作分布。

训练内容：

- Action Expert
- VLM 顶部轻量适配层 / LoRA 风格上下文适配
- 保持视觉编码器冻结

建议数据比例起点：

```text
真机/仿真可执行数据：约 60%
UMI 物理验证数据：约 40%
Ego：不参与动作 loss，或仅少量用于语义 regularization
```

命令：

```bash
$PY scripts/train_twaves_lerobot.py \
  --stage 1B \
  --dataset-root $DATA \
  --repo-id $REPO \
  --foundation-checkpoint $BASE \
  --tokenizer $TOKENIZER \
  --init-delta $OUT/stage1A.pt \
  --output $OUT/stage1B.pt \
  --steps 2000 \
  --batch-size 1 \
  --learning-rate 5e-5 \
  --fps 20 \
  --skip-inference
```

### 4.3 Stage1C：小学习率联合收敛

目标：降低动作抖动，让状态、语言、视觉与动作头稳定对齐。

损失结构：

```text
L_stage1 =
  L_flow_action
  + lambda_s L_smooth
  + lambda_c L_contact_pseudo
```

命令：

```bash
$PY scripts/train_twaves_lerobot.py \
  --stage 1C \
  --dataset-root $DATA \
  --repo-id $REPO \
  --foundation-checkpoint $BASE \
  --tokenizer $TOKENIZER \
  --init-delta $OUT/stage1B.pt \
  --output $OUT/stage1C.pt \
  --steps 1000 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --fps 20 \
  --skip-inference
```

### 4.4 Stage2：Goal-WM + Goal-Conditioned VFLA

目标：引入世界模型生成或编码的 subgoal image / subgoal latent，让策略学会“看当前图像 + 任务语言 + 目标 latent 后行动”。

训练内容：

- Goal encoder / latent adapter
- Subtask generator head
- Goal-conditioned action expert
- 使用 Ego / UMI / 真机共同训练语义目标

Stage2 loss：

```text
L_stage2 =
  L_flow_action
  + lambda_gi L_goal_image
  + lambda_gl L_goal_latent
  + lambda_st L_subtask
  + lambda_s L_smooth
```

Cosmos Goal-WM 示例：

```bash
export TWM_COSMOS_PREDICT2_ROOT=$ROOT/external/cosmos-predict2.5

$PY scripts/train_twaves_lerobot.py \
  --stage 2 \
  --dataset-root $DATA \
  --repo-id $REPO \
  --foundation-checkpoint $BASE \
  --tokenizer $TOKENIZER \
  --init-delta $OUT/stage1C.pt \
  --output $OUT/stage2.pt \
  --steps 3000 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --fps 20 \
  --goal-wm-online \
  --goal-wm-checkpoint external/checkpoints/cosmos-predict2.5-2b-action-cond/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt \
  --goal-wm-reason-embedding external/checkpoints/cosmos-predict2.5-2b-action-cond/robot/action-cond/cr1_empty_string_text_embeddings.pt \
  --goal-wm-steps 1 \
  --skip-inference
```

如果显存或时间紧张，可以先关闭在线 WM，只使用数据中的 `observation.images.goal` 做 teacher goal；确认链路稳定后再打开 `--goal-wm-online`。

### 4.5 Stage3：力觉 MoE 融合

目标：在动作专家上融合力觉信息，使模型对插拔、开关、旋钮、阀门等接触任务更稳定。

训练内容：

- Force encoder
- Force Visual-Language MoE router
- next wrench prediction head
- contact prediction head
- action expert 小学习率继续收敛

Stage3 loss：

```text
L_stage3 =
  L_flow_action
  + lambda_f L_next_wrench
  + lambda_fc L_force_contact
  + lambda_moe L_moe_balance
  + lambda_gl L_goal_latent
  + lambda_st L_subtask
```

力觉数据转换示例：

```bash
python scripts/prepare_force_lerobot_data.py \
  --input-root /path/to/force_lerobot_dataset \
  --output-root data/processed/twaves_stage3_force \
  --image-size 256 \
  --max-episodes 100 \
  --max-joint-delta-rad 0.35 \
  --max-gripper-delta 1.2
```

构建 Stage3 LeRobot：

```bash
python scripts/build_lerobot_dataset.py \
  --input-root data/processed/twaves_stage3_force \
  --output-root data/lerobot/twaves_stage3_force \
  --repo-id local/twaves_stage3_force \
  --fps 20 \
  --image-size 224 \
  --overwrite
```

训练：

```bash
$PY scripts/train_twaves_lerobot.py \
  --stage 3 \
  --dataset-root data/lerobot/twaves_stage3_force \
  --repo-id local/twaves_stage3_force \
  --foundation-checkpoint $BASE \
  --tokenizer $TOKENIZER \
  --init-delta $OUT/stage2.pt \
  --output $OUT/stage3.pt \
  --steps 3000 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --fps 20 \
  --goal-wm-online \
  --goal-wm-checkpoint external/checkpoints/cosmos-predict2.5-2b-action-cond/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt \
  --goal-wm-reason-embedding external/checkpoints/cosmos-predict2.5-2b-action-cond/robot/action-cond/cr1_empty_string_text_embeddings.pt \
  --goal-wm-steps 1 \
  --skip-inference
```

## 5. 训练前检查清单

正式训练前建议检查：

- UMI 的 action 是否是相对夹爪中心点的局部增量。
- Ego 如果没有可靠动作，是否设置了 `action_valid=0`。
- 真机关节角是否统一为弧度；不要把角度制直接混进 `observation.state`。
- 真机绝对动作是否已转成 delta action。
- 力 / 力矩是否统一到 TCP 坐标系。
- 单臂数据是否只打开前 7 维 `action_mask`。
- 双臂数据是否只打开前 14 维 `action_mask`。
- Stage2 的 `observation.images.goal` 是否存在，或者 Goal-WM 是否能在线生成 latent。
- Stage3 数据中的 `force_mask=1` 是否只出现在真实有力觉的帧。

如果训练中出现异常大的 `flow_loss`，优先检查：

```text
1. action 是否混入绝对关节角 / 绝对 TCP pose
2. 角度单位是否混用 degree 和 radian
3. action_space_id 是否与部署/训练目标一致
4. action_mask 是否错误打开了无效维度
5. fps 是否与 LeRobot delta_timestamps 对齐
```

## 6. 推荐数据配比

小规模微调时可以从下面的配比开始：

```text
Stage1:
  UMI                 30% - 40%
  真机/仿真可执行数据 60% - 70%
  Ego                 0% - 10%，只做语义辅助

Stage2:
  UMI                 25% - 35%
  真机/仿真可执行数据 35% - 50%
  Ego                 20% - 35%

Stage3:
  有力觉真机数据       60% - 80%
  无力觉可执行数据     20% - 40%
  Ego                 可选，只参与 goal/subtask，不参与 force loss
```


