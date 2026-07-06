# T-WAES V1.1

T-WAES V1.1 是面向工业作业的通用具身智能融合模型；技术架构名为 **T-WM-VFLA-S**。

- **T**：天创任务、机器人本体与部署系统。
- **WM**：世界模型，用于生成 subtask、subgoal image 与 future latent。
- **VFLA**：视觉、语言、力觉与动作联合策略。
- **S**：外挂安全层，在模型外执行规则校验、动作修正、拒绝与急停。

当前路线是“基模 + 微调”，不要求从零大规模预训练。训练数据必须先转成 **LeRobot 3.x** 格式。

## 数据与动作空间

三类数据统一到当前 TCP 坐标系下的局部增量动作：

```text
[dx, dy, dz, drx, dry, drz, dgripper]
```

- 单臂使用前 7 维；
- 双臂使用前 14 维；
- 模型 action chunk 固定 16 维；
- 无效维度由 `action_mask` 屏蔽；
- 无真实动作的数据，例如普通 ego 视频，必须设置 `action_valid=0` 且 `action_mask=0`。

图像视角固定为：

```text
observation.images.main
observation.images.left_wrist
observation.images.right_wrist
observation.images.goal
camera_mask = [main, left_wrist, right_wrist]
```

当前约定：UMI/手持夹爪数据通常是 `[0,1,1]`，ego 视频是 `[1,0,0]`，完整机器人/仿真数据是 `[1,1,1]`。

## 新增数据转换脚本

### GenRobot RealOmin / UMI-like MCAP

```bash
python scripts/prepare_realomin_mcap_data.py \
  --input-root data/raw/modelscope_10kh_realomin_sample \
  --output-root data/processed/realomin_stage1_sample \
  --max-files 5 --max-frames-per-file 48 --stride 10 --image-size 256
```

### RoboTwin2.0 HDF5

RoboTwin2.0 的左右 `endpose` 会被转换成和 UMI 一致的双臂 TCP-local delta，而不是直接混用 joint action。

```bash
python scripts/prepare_robotwin2_hdf5_data.py \
  --input-root data/raw/robotwin2_sample \
  --output-root data/processed/robotwin2_stage1_sample \
  --max-episodes 20 --max-frames-per-episode 32 --stride 2 --image-size 256
```

### 开源 Ego 视频

普通 ego 视频没有机器人动作，只用于 Stage2 的 goal/subtask 语义监督。

```bash
python scripts/prepare_ego_video_data.py \
  --input-root data/hf/ego_pov_sample \
  --output-root data/processed/ego_open_stage2_sample \
  --max-videos 1 --max-frames-per-video 64 --stride 12 --image-size 256
```

### 合成 LeRobot 数据集

```bash
python scripts/build_lerobot_dataset.py \
  --input-root data/processed/stage12_realomin_robotwin2_ego_mix \
  --output-root data/lerobot/stage12_realomin_robotwin2_ego_mix \
  --repo-id local/stage12_realomin_robotwin2_ego_mix \
  --fps 20 --image-size 224 --overwrite
```

## 训练入口

```bash
export PY=/home/tc/miniconda3/envs/t-wm-vfla-s/bin/python
export ROOT=/home/tc/T-WM-VFLA-S
export PYTHONPATH=$ROOT:$ROOT/src:$ROOT/external/foundation_runtime/src:$ROOT/external/foundation_runtime/third_party/pi_transformers/src
export CUDA_VISIBLE_DEVICES=1

DATA=data/lerobot/stage12_realomin_robotwin2_ego_mix
REPO=local/stage12_realomin_robotwin2_ego_mix
BASE=checkpoints/foundation_policy/pretrained_model
TOKENIZER=checkpoints/foundation_tokenizer
OUT=checkpoints/twaes_v1_1
```

Stage1A：

```bash
$PY scripts/train_twaes_lerobot.py \
  --stage 1A --dataset-root $DATA --repo-id $REPO \
  --foundation-checkpoint $BASE --tokenizer $TOKENIZER \
  --output $OUT/stage1A.pt \
  --steps 1000 --batch-size 1 --learning-rate 1e-4
```

Stage2 + Cosmos Goal-WM latent：

```bash
$PY scripts/train_twaes_lerobot.py \
  --stage 2 --dataset-root $DATA --repo-id $REPO \
  --foundation-checkpoint $BASE --tokenizer $TOKENIZER \
  --init-delta $OUT/stage1A.pt \
  --output $OUT/stage2_cosmos_latent.pt \
  --steps 5000 --batch-size 1 --learning-rate 5e-5 \
  --goal-wm-online --goal-wm-latent-only \
  --goal-wm-checkpoint external/checkpoints/cosmos-predict2.5-2b-action-cond/robot/action-cond/38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt \
  --goal-wm-reason-embedding external/checkpoints/cosmos-predict2.5-2b-action-cond/robot/action-cond/cr1_empty_string_text_embeddings.pt \
  --goal-wm-steps 1
```


