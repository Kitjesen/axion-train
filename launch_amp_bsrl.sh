#!/bin/bash
# AMP Training Launch Script for BSRL
# Usage: bash launch_amp_bsrl.sh [GPU_ID] [CHECKPOINT]
# Example: bash launch_amp_bsrl.sh 1 logs/rsl_rl/atom01_amp/best_checkpoint/model_24999.pt

GPU=1
CKPT=""
BASE=/home/bsrl/hongsenpang/RLbased/atom01_train/robolab
PYTHON=/home/bsrl/miniconda3/envs/thunder2/bin/python
LOG=/home/bsrl/hongsenpang/RLbased/atom01_train/robolab/amp_bsrl_gpu${GPU}.log

CKPT_ARG=""
if [ -n "$CKPT" ]; then
    CKPT_ARG="--checkpoint $CKPT"
fi

echo "[AMP BSRL] GPU=$GPU CKPT=$CKPT"
cd $BASE
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PYTHON -u scripts/rsl_rl/train.py     --task Atom01-AMP-Getup-Phuma-V3     --headless     --num_envs 4096     --logger tensorboard     --device cuda:$GPU     --max_iterations 25000     $CKPT_ARG     > $LOG 2>&1 &
echo AMP_PID=$!
echo "Log: $LOG"
