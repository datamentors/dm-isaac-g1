#!/bin/bash
# ============================================
# DM-ISAAC-G1: Run All Phases
# ============================================
# Master script to run through all phases of the G1 project
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    source "$PROJECT_ROOT/.env" 2>/dev/null || true
fi

usage() {
    cat << EOF
DM-ISAAC-G1: Unitree G1 + Isaac Sim + GROOT

Usage: $0 [phase] [options]

Phases:
  check       - Check workstation capabilities and connectivity
  phase1      - Run G1 with GROOT inference (demonstration)
  phase2      - Train G1 with RL (g1_reach or ulc_vlm curriculum)
  phase3      - Fine-tune GROOT with RL demonstrations
  phase4      - Full navigation + inference + RL
  all         - Run all phases sequentially

Options:
  --help      - Show this help message

Examples:
  $0 check                    # Check workstation status
  $0 phase1                   # Run inference demo
  $0 phase2 g1_reach          # Train G1 reaching
  $0 phase2 ulc_vlm 7         # Train ULC stage 7

Phase Details:
  Phase 1: G1 robot in Isaac Sim with GROOT policy inference
  Phase 2: Reinforcement learning with curriculum (7+ stages)
  Phase 3: GROOT fine-tuning using RL-generated demonstrations
  Phase 4: Navigation + manipulation with combined policies

Workstation: ${WORKSTATION_HOST:-not configured}
GROOT Server: ${GROOT_SERVER_HOST:-not configured}:${GROOT_SERVER_PORT:-5555}
EOF
    exit 0
}

case "${1:-help}" in
    check)
        echo "Running workstation check..."
        "$SCRIPT_DIR/check_workstation.sh"
        ;;

    phase1)
        echo "Phase 1: G1 + GROOT Inference"
        "$SCRIPT_DIR/run_groot_inference.sh" "${@:2}"
        ;;

    phase2)
        echo "Phase 2: RL Training"
        "$SCRIPT_DIR/train_rl.sh" "${@:2}"
        ;;

    phase3)
        echo "Phase 3: GROOT Fine-tuning + RL"
        "$SCRIPT_DIR/finetune_groot.sh" "${@:2}"
        ;;

    phase4)
        echo "Phase 4: Navigation + Inference + RL"
        "$SCRIPT_DIR/run_navigation.sh" "${@:2}"
        ;;

    play)
        echo "Playing trained policy..."
        "$SCRIPT_DIR/play_policy.sh" "${@:2}"
        ;;

    all)
        echo "Running all phases..."
        echo ""
        echo "=== Phase 1: Inference Demo ==="
        "$SCRIPT_DIR/run_groot_inference.sh"
        echo ""
        echo "=== Phase 2: RL Training (g1_reach) ==="
        echo "Skipping training in 'all' mode - run manually with: $0 phase2 g1_reach"
        echo ""
        echo "=== Phase 3: Fine-tuning ==="
        echo "Skipping fine-tuning in 'all' mode - run manually with: $0 phase3"
        echo ""
        echo "=== Phase 4: Navigation ==="
        "$SCRIPT_DIR/run_navigation.sh"
        ;;

    help|--help|-h)
        usage
        ;;

    *)
        echo "Unknown command: $1"
        usage
        ;;
esac
