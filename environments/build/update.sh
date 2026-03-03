#!/usr/bin/env bash
# =============================================================================
# Pull the latest ECR image and update the running workstation container.
#
# Use this after build.sh has pushed a new image to ECR. This script pulls the
# latest image and restarts the container.
#
# Usage:
#   ./update.sh                  # Pull latest + restart container
#   ./update.sh --pull-only      # Pull only, don't restart
#   ./update.sh --restart-only   # Restart with current image (no pull)
#
# WARNING: This will stop and recreate the container. Any in-progress training
#          inside the container will be lost. Save checkpoints to S3/host first.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSTATION_DIR="$REPO_ROOT/environments/workstation"

source "$REPO_ROOT/.env"

PROFILE="${AWS_PROFILE:-elianomarques-dm}"
REGION="${AWS_ECR_REGION:-${AWS_REGION:-eu-west-1}}"
ECR_REGISTRY="${AWS_ECR_REGISTRY}"
ECR_REPO="${AWS_ECR_REPO}"
IMAGE="$ECR_REGISTRY/$ECR_REPO:latest"

DO_PULL=true
DO_RESTART=true

for arg in "$@"; do
    case "$arg" in
        --pull-only)    DO_RESTART=false ;;
        --restart-only) DO_PULL=false ;;
        -h|--help)      head -14 "$0" | tail -10; exit 0 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

log() { echo -e "\033[1;34m==>\033[0m \033[1m$*\033[0m"; }

if $DO_PULL; then
    log "Logging into ECR"
    aws ecr get-login-password --region "$REGION" --profile "$PROFILE" | \
        docker login --username AWS --password-stdin "$ECR_REGISTRY"

    log "Pulling $IMAGE"
    docker pull "$IMAGE"
    docker tag "$IMAGE" dm-workstation:latest
    echo "  Tagged as dm-workstation:latest"
fi

if $DO_RESTART; then
    log "Restarting container"
    cd "$WORKSTATION_DIR"

    echo "  Stopping..."
    docker compose -f docker-compose.unitree.yml stop groot

    echo "  Removing..."
    docker compose -f docker-compose.unitree.yml rm -f groot

    echo "  Starting..."
    docker compose -f docker-compose.unitree.yml up -d groot

    echo "  Container restarted."
    docker compose -f docker-compose.unitree.yml ps groot
fi

echo ""
echo "Done."
