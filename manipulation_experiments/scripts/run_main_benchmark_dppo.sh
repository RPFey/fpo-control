#!/usr/bin/env bash
#
# run_main_benchmark.sh
#
# Launches all 60 main benchmark runs:
#   5 tasks x 4 models (FPO++, DPPO Learned, Vanilla FPO, DPPO Fixed) x 3 seeds
#
# Usage:
#   bash scripts/run_main_benchmark.sh              # execute all runs
#   DRY_RUN=1 bash scripts/run_main_benchmark.sh    # print commands only
#   NUM_GPUS=4 bash scripts/run_main_benchmark.sh   # use 4 GPUs per run
#
# Each run is launched sequentially. To run in parallel, pipe DRY_RUN=1 output
# to a job scheduler or use & to background individual runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

NUM_GPUS="${NUM_GPUS:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Helper: run or print a command
# ---------------------------------------------------------------------------
run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$@"
    echo ""
  else
    echo "=== Running: $@ ==="
    "$@"
  fi
}

# ---------------------------------------------------------------------------
# Shared parameters for ALL 60 main benchmark runs
# ---------------------------------------------------------------------------
SHARED_ARGS=(
  --distributed True
  --base-policy-wandb-project flow-bc
  --load-ema True
  --wandb_project flow-bc-fpo-finetuning
  --wandb_enable True
  --gradient_accumulation_steps 1
  --num_minibatches 128
  --log_freq 1
  --save_freq 10
  --rollout_freq 1
  --eval_num_episodes 200
  --data_collection_steps 1600
  --do_chunk_level_ppo True
  --eval_ema False
  --exploration_noise_std None
  --freeze_vision_encoder True
  --gae_lambda 0.99
  --n_action_samples 8
  --wandb_entity leiboshu
  --num_envs 30
  --sampling_steps 10
  --spo_clip_coef 0.01
  --zero_sampling True
  --target_kl 0.1
)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
#              task_name       env_name             run_id    ckpt_step      total_ts  discount
TASKS=(
  "square        Square           trc7rbt0  step_110000  13000000   0.99"
  # "transport     Transport           trc7rbt0  step_110000  8000000   0.995"
)

###########################################################################
# 1.1  FPO++
###########################################################################
echo "====================================================================="
echo "  FPO++ (15 runs)"
echo "====================================================================="

# Per-task FPO++ parameters:
#   task_key  clip_coef  max_grad_norm  cfm_huber_delta  clamp_logratio  clamp_old_cfm n_action_steps
FPOP_PARAMS=(
  "square     0.02  25  1    5  4     4"
  # "transport  0.01  25  1    None  None     8"
)

for task_line in "${TASKS[@]}"; do
  read -r task_key env_name run_id ckpt_step total_ts disc <<< "$task_line"

  # Find FPO++ params for this task
  for fpop_line in "${FPOP_PARAMS[@]}"; do
    read -r fpop_key clip_coef mgn huber_delta clamp_lr clamp_old n_act_step <<< "$fpop_line"
    if [[ "$fpop_key" == "$task_key" ]]; then
      for SEED in 0 1 2; do
        run_cmd torchrun --nproc_per_node="$NUM_GPUS" finetune_online_rl.py \
          --base_policy_local_path /home/leiboshu/ActiveLearning/fpo-control/manipulation_experiments/runs/flow_bc_square_2026-05-13_22-07-50/checkpoints/best/policy \
          --wandb_entity leiboshu \
          --distributed True \
          --base-policy-wandb-project flow-bc \
          --load-ema True    \
          --experiment "finetune-fpo++-square-${SEED}" \
          --wandb_project flow-bc-fpo-finetuning \
          --image_observation_keys agentview_image \
          --total_timesteps 13000000 \
          --gradient_accumulation_steps 1 \
          --num_minibatches 128 \
          --log_freq 1 \
          --save_freq 10 \
          --rollout_freq 1 \
          --task Square \
          --eval_env Square \
          --eval_num_episodes 200 \
          --wandb_enable True \
          --cfm_loss_huber_delta 1.0 \
          --cfm_loss_use_huber True \
          --clamp_logratio 5.0 \
          --clamp_old_cfm_loss 4.0 \
          --clip_coef 0.02 \
          --data_collection_steps 1600 \
          --discount 0.99 \
          --do_chunk_level_ppo True \
          --eval_ema False \
          --exploration_noise_std None \
          --freeze_vision_encoder True \
          --gae_lambda 0.99 \
          --learning_rate_actor 0.00001 \
          --max_grad_norm 1.0 \
          --n_action_samples 16 \
          --n_action_steps 4 \
          --num_envs 50 \
          --sampling_steps 10 \
          --seed ${SEED} \
          --spo_clip_coef 0.05 \
          --trust_region_mode ppo \
          --zero_sampling True
      done
      break
    fi
  done
done

echo "====================================================================="
echo "  Main benchmark complete (60 runs total)"
echo "====================================================================="
