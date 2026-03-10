#!/bin/bash
BASE=/home/bsrl/hongsenpang/RLbased/atom01_train/robolab
PYTHON=/home/bsrl/miniconda3/envs/thunder2/bin/python
CKPT=logs/rsl_rl/atom01_amp/best_checkpoint/model_24999.pt
cd $BASE

for GPU in 1 2 5 6 7; do
  LOG=${BASE}/amp_bsrl_gpu${GPU}.log
  echo "[AMP] Launching GPU $GPU -> $LOG"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup $PYTHON -u scripts/rsl_rl/train.py \
      --task Atom01-AMP-Getup-Phuma-V3 \
      --headless \
      --num_envs 2048 \
      --logger tensorboard \
      --device cuda:0 \
      --max_iterations 25000 \
      --checkpoint $CKPT \
      --seed $((40 + GPU)) \
    > $LOG 2>&1 &
  echo "  PID=$!"
done
echo 'All 5 AMP processes launched'
