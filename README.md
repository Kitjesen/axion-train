# Atom01 AMP Training — Robolab

Roboparty Atom01 humanoid robot RL training code based on **IsaacLab + rsl_rl + AMP (Adversarial Motion Priors)**.

## Task: Atom01-AMP-Getup-Phuma-V3

Combined task training three behaviors simultaneously:
- **Getup (52%)** — fall recovery from arbitrary poses
- **Walking (24%)** — forward locomotion and turning
- **PHUMA style (24%)** — motion style imitation from real human data (40,891 PKL clips)

## Results (2026-03-09)

| Metric | Value |
|--------|-------|
| Iterations | 25,000 |
| Style Reward | **0.881** |
| Mean Reward | ~300 |
| Episode Length | 2500 steps (full, no falls) |
| Training Time | ~17.5 hours on RTX 3090 |

## Structure

```
robolab/              # IsaacLab environment package
├── robolab/tasks/
│   └── manager_based/
│       ├── amp/      # AMP task definitions
│       │   ├── atom01_amp_getup_phuma_env_cfg.py
│       │   ├── amp_env.py
│       │   └── managers/
│       └── ...
├── scripts/rsl_rl/
│   ├── train.py
│   └── play_amp.py
└── data/robots/      # Robot URDF/MJCF (Atom01)

rsl_rl/               # rsl_rl with AMPRunner extension
scripts/              # Launch scripts
```

## Training

```bash
# Train from best checkpoint
CUDA_VISIBLE_DEVICES=0 python robolab/scripts/rsl_rl/train.py \
  --task=Atom01-AMP-Getup-Phuma-V3 --headless \
  --num_envs=2048 --device=cuda:0 \
  --checkpoint=<path>/model_24999.pt

# Play with 64 envs
python robolab/scripts/rsl_rl/play_amp.py \
  --task=Atom01-AMP-Getup-Phuma-V3-Play \
  --num_envs=64 --checkpoint=<path>/model_24999.pt \
  --video --video_length=600
```

## Config Key Points

- Push disturbance interval: 25-40s (V3, simulates real fall scenarios)
- Episode length: 50s
- Network: ActorCritic 512-256-128, PPOAMP algorithm
- Discriminator LR: 1e-4, adaptive schedule
- Motion library: `atom01_phuma_sample200/` + `atom01_getup/` + `atom01_walk/`
