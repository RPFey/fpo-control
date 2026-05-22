#!/usr/bin/env bash
#
# run_pretrain_base_policies.sh
#
# Pretrains all 5 base policies (Flow Matching BC) used in FPO experiments.
# Each task produces a checkpoint that matches the base policy used in the
# main benchmark (W&B project: flow-bc).
#
# All base policies share these hyperparameters:
#   - network: MLP [1024,1024,1024] with ViT vision backbone
#   - flow_network_output_param=u, cfm_loss_mode=u (velocity prediction)
#   - cfm_loss_use_huber=False (L2 loss)
#   - grad_clip_norm=25
#   - horizon=16, n_action_steps=8, sampling_steps=10
#   - batch_size=512, learning_rate=1e-4, lr_backbone=1e-5
#   - ema_power=0.995, enable_geometric_augmentations=True
#   - seed=3
#
# Usage:
#   bash scripts/run_pretrain_base_policies.sh              # run all 5
#   DRY_RUN=1 bash scripts/run_pretrain_base_policies.sh    # print commands only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_DIR"

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
# Shared parameters across all 5 base policies
# ---------------------------------------------------------------------------
SHARED_ARGS=(
    --policy flowmatching
    --wandb_entity leiboshu
    --network_architecture mlp
    --mlp_dims "[768, 768, 768]"
    --vision_backbone vit
    --flow_network_output_param u
    --cfm_loss_mode u
    --cfm_loss_use_huber False
    --cfm_loss_huber_delta 1.0
    --grad_clip_norm 25
    --sampling_steps 10
    --batch_size 512
    --learning_rate 1e-4
    --lr_backbone 1e-5
    --weight_decay 1e-6
    --ema_power 0.999
    --enable_geometric_augmentations True
    --seed 3
    --num_workers 16
    --wandb_enable True
    --wandb_project flow-bc
    --eval_num_envs 25
    --eval_num_episodes 50
    --log_freq 5
    --save_freq 10000
    --rollout_freq 10000
)

###########################################################################
# Square
###########################################################################
# echo "====================================================================="
# echo "  Pretraining: Square"
# echo "====================================================================="

# run_cmd python pretrain_flow_bc.py \
#     "${SHARED_ARGS[@]}" \
#     --dataset ankile/robomimic-mh-square-image \
#     --image_observation_keys "agentview_image" \
#     --eval_env Square \
#     --steps 1000000 \
#     --horizon 4 \
#     --n_action_steps 4 \
#     --max_num_episodes 100 \
#     --experiment flow_bc_square

###########################################################################
# Transport (two-arm)
###########################################################################
echo "====================================================================="
echo "  Pretraining: Transport"
echo "====================================================================="

run_cmd python pretrain_flow_bc.py \
    "${SHARED_ARGS[@]}" \
    --dataset ankile/robomimic-mh-transport-image \
    --image_observation_keys "shouldercamera0_image shouldercamera1_image" \
    --eval_env Transport \
    --steps 1000000 \
    --horizon 8 \
    --n_action_steps 8 \
    --max_num_episodes 100 \
    --experiment flow_bc_transport_h8a8-v1.4.1

echo "====================================================================="
echo "  All base policy pretraining runs complete"
echo "====================================================================="
