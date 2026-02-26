#!/usr/bin/env bash
# =============================================================================
# Multi-Task Evaluation: Test Our Hospitality Model Across Multiple Environments
# =============================================================================
#
# Runs our hospitality-7ds model against different RoboCasa/GR00T-WBC environments
# that are similar to the training datasets:
#
#   Training Dataset          → RoboCasa/GR00T Eval Environment
#   ─────────────────────────────────────────────────────────────
#   G1_Fold_Towel             → Custom towel env (TODO)
#   G1_Clean_Table            → Surface cleaning tasks
#   G1_Wipe_Table             → Surface cleaning tasks
#   G1_Prepare_Fruit          → Pick-and-place fruit tasks
#   G1_Pour_Medicine          → Pouring tasks
#   G1_Organize_Tools         → Drawer/cabinet organization
#   G1_Pack_PingPong          → Pick-and-place tasks
#
# Phase 1: Validate pipeline with NVIDIA's pretrained PnP model
# Phase 2: Run our hospitality model on PnP (closest sim env available)
# Phase 3: Run our model on RoboCasa kitchen tasks (once available for G1)
#
# Prerequisites:
#   - Run setup_official_eval.sh first
#   - GROOT checkpoints available
#
# Usage:
#   bash scripts/eval/run_multi_task_eval.sh [--phase 1|2|3] [--episodes 5]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROOT_DIR="/workspace/Isaac-GR00T"
WBC_VENV="$GROOT_DIR/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"
EVAL_DIR="/tmp/groot_multi_eval_$(date +%Y%m%d_%H%M%S)"

# Our model checkpoints
NVIDIA_PNP_MODEL="nvidia/GR00T-N1.6-G1-PnPAppleToPlate"
OUR_HOSPITALITY_MODEL="/workspace/checkpoints/groot-g1-gripper-hospitality-7ds"
OUR_TOWEL_MODEL="/workspace/checkpoints/groot-g1-gripper-fold-towel-full"

# Defaults
PHASE="all"
N_EPISODES=5
N_ACTION_STEPS=20
MAX_EPISODE_STEPS=1440
SERVER_PORT=5555

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)    PHASE="$2"; shift 2 ;;
        --episodes) N_EPISODES="$2"; shift 2 ;;
        --port)     SERVER_PORT="$2"; shift 2 ;;
        *)          echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$EVAL_DIR"

echo "================================================================="
echo "  Multi-Task GR00T Evaluation"
echo "  Phase: $PHASE | Episodes per task: $N_EPISODES"
echo "  Output: $EVAL_DIR"
echo "================================================================="

cd "$GROOT_DIR"

# ─────────────────────────────────────────────────────────────────────
# Helper: Run one eval (server + client)
# ─────────────────────────────────────────────────────────────────────
run_eval() {
    local model_path="$1"
    local env_name="$2"
    local label="$3"
    local log_dir="$EVAL_DIR/$label"

    mkdir -p "$log_dir"

    echo ""
    echo "─────────────────────────────────────────────────────────────"
    echo "  [$label] Model: $(basename $model_path)"
    echo "  [$label] Env:   $env_name"
    echo "  [$label] Episodes: $N_EPISODES"
    echo "─────────────────────────────────────────────────────────────"

    # Start server
    echo "  Starting GROOT server..."
    python -u gr00t/eval/run_gr00t_server.py \
        --model-path "$model_path" \
        --embodiment-tag UNITREE_G1 \
        --use-sim-policy-wrapper \
        --port "$SERVER_PORT" \
        > "$log_dir/server.log" 2>&1 &
    local server_pid=$!

    # Wait for server
    local ready=0
    for i in $(seq 1 120); do
        if grep -q "Server started\|Serving\|Listening\|ready\|Running" "$log_dir/server.log" 2>/dev/null; then
            echo "  Server ready after ${i}s"
            ready=1
            break
        fi
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "  ERROR: Server died. Log:"
            tail -20 "$log_dir/server.log"
            return 1
        fi
        sleep 1
    done

    if [ $ready -eq 0 ]; then
        echo "  WARNING: Server didn't signal ready after 120s, proceeding anyway..."
    fi

    # Run client
    echo "  Running simulation client..."
    $WBC_VENV gr00t/eval/rollout_policy.py \
        --n_episodes "$N_EPISODES" \
        --max_episode_steps "$MAX_EPISODE_STEPS" \
        --env_name "$env_name" \
        --policy_client_host localhost \
        --policy_client_port "$SERVER_PORT" \
        --n_action_steps "$N_ACTION_STEPS" \
        --n_envs 1 \
        2>&1 | tee "$log_dir/client.log"

    local exit_code=$?

    # Extract results
    if grep -q "success rate" "$log_dir/client.log" 2>/dev/null; then
        local sr=$(grep "success rate" "$log_dir/client.log" | tail -1)
        echo "  RESULT: $sr"
        echo "$label | $model_path | $env_name | $sr" >> "$EVAL_DIR/results.txt"
    else
        echo "  RESULT: No success rate reported (exit=$exit_code)"
        echo "$label | $model_path | $env_name | exit=$exit_code" >> "$EVAL_DIR/results.txt"
    fi

    # Cleanup server
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true
    sleep 2  # Brief pause between evals

    return $exit_code
}

# ─────────────────────────────────────────────────────────────────────
# Phase 1: Validate pipeline with NVIDIA's pretrained model
# ─────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "1" ] || [ "$PHASE" = "all" ]; then
    echo ""
    echo "============================================================="
    echo "  PHASE 1: Validate pipeline (NVIDIA pretrained PnP model)"
    echo "============================================================="

    run_eval \
        "$NVIDIA_PNP_MODEL" \
        "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc" \
        "phase1_nvidia_pnp_apple" \
        || echo "  Phase 1 eval returned non-zero (may be normal)"
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 2: Test our hospitality model on PnP environment
# ─────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "2" ] || [ "$PHASE" = "all" ]; then
    echo ""
    echo "============================================================="
    echo "  PHASE 2: Our hospitality model on PnP environment"
    echo "============================================================="

    # Test with hospitality 7-dataset model
    if [ -d "$OUR_HOSPITALITY_MODEL" ]; then
        run_eval \
            "$OUR_HOSPITALITY_MODEL" \
            "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc" \
            "phase2_hospitality_pnp" \
            || echo "  Phase 2 hospitality eval returned non-zero"
    else
        echo "  SKIP: Hospitality model not found at $OUR_HOSPITALITY_MODEL"
    fi

    # Test with towel-only model
    if [ -d "$OUR_TOWEL_MODEL" ]; then
        run_eval \
            "$OUR_TOWEL_MODEL" \
            "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc" \
            "phase2_towel_pnp" \
            || echo "  Phase 2 towel eval returned non-zero"
    else
        echo "  SKIP: Towel model not found at $OUR_TOWEL_MODEL"
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# Phase 3: RoboCasa kitchen tasks (once G1 envs are available)
# ─────────────────────────────────────────────────────────────────────
if [ "$PHASE" = "3" ] || [ "$PHASE" = "all" ]; then
    echo ""
    echo "============================================================="
    echo "  PHASE 3: RoboCasa kitchen tasks"
    echo "============================================================="
    echo "  NOTE: Additional G1 locomanip envs need to be registered."
    echo "  Currently only LMPnPAppleToPlateDC is available."
    echo "  To add more tasks, create custom envs in gr00t_wbc."
    echo ""
    echo "  Available after custom env creation:"
    echo "    - Table clearing (maps to G1_Clean_Table training data)"
    echo "    - Surface wiping (maps to G1_Wipe_Table)"
    echo "    - Fruit preparation (maps to G1_Prepare_Fruit)"
    echo "    - Object organization (maps to G1_Organize_Tools)"
    echo "    - Pick-and-place (maps to G1_Pack_PingPong)"
    echo "    - Towel folding (custom deformable env needed)"
fi

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================="
echo "  Evaluation Summary"
echo "================================================================="
if [ -f "$EVAL_DIR/results.txt" ]; then
    echo ""
    cat "$EVAL_DIR/results.txt"
else
    echo "  No results (no evals completed)"
fi
echo ""
echo "  Full logs: $EVAL_DIR/"
echo "  Videos: /tmp/sim_eval_videos_*/"
echo "================================================================="
