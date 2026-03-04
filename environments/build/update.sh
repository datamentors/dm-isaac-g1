#!/usr/bin/env bash
# =============================================================================
# Pull the latest ECR image and update the running container.
#
# Use this after build.sh has pushed a new image to ECR. This script pulls the
# latest image and restarts the container.
#
# Usage:
#   ./update.sh                  # Pull latest workstation + restart
#   ./update.sh --spark          # Pull latest Spark + restart
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
SPARK_DIR="$REPO_ROOT/environments/spark"

source "$REPO_ROOT/.env"

PROFILE="${AWS_PROFILE:-elianomarques-dm}"
REGION="${AWS_ECR_REGION:-${AWS_REGION:-eu-west-1}}"
ECR_REGISTRY="${AWS_ECR_REGISTRY}"

PLATFORM="workstation"
DO_PULL=true
DO_RESTART=true

for arg in "$@"; do
    case "$arg" in
        --spark)        PLATFORM="spark" ;;
        --pull-only)    DO_RESTART=false ;;
        --restart-only) DO_PULL=false ;;
        -h|--help)      head -16 "$0" | tail -12; exit 0 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

log() { echo -e "\033[1;34m==>\033[0m \033[1m$*\033[0m"; }

# Platform-specific config
if [ "$PLATFORM" = "spark" ]; then
    ECR_REPO="${AWS_ECR_REPO_SPARK:-isaac-g1-spark}"
    IMAGE="$ECR_REGISTRY/$ECR_REPO:latest"
    LOCAL_TAG="dm-spark-workstation:latest"
    COMPOSE_DIR="$SPARK_DIR"
    COMPOSE_FILE="docker-compose.spark.yml"
    SERVICE="workstation"
else
    ECR_REPO="${AWS_ECR_REPO}"
    IMAGE="$ECR_REGISTRY/$ECR_REPO:latest"
    LOCAL_TAG="dm-workstation:latest"
    COMPOSE_DIR="$WORKSTATION_DIR"
    COMPOSE_FILE="docker-compose.unitree.yml"
    SERVICE="groot"
fi

echo "  Platform: $PLATFORM"
echo "  Image:    $IMAGE"

if $DO_PULL; then
    log "Logging into ECR"
    aws ecr get-login-password --region "$REGION" --profile "$PROFILE" | \
        docker login --username AWS --password-stdin "$ECR_REGISTRY"

    log "Pulling $IMAGE"
    docker pull "$IMAGE"
    docker tag "$IMAGE" "$LOCAL_TAG"
    echo "  Tagged as $LOCAL_TAG"
fi

if $DO_RESTART; then
    log "Restarting container ($SERVICE)"
    cd "$COMPOSE_DIR"

    echo "  Stopping..."
    docker compose -f "$COMPOSE_FILE" stop "$SERVICE"

    echo "  Removing..."
    docker compose -f "$COMPOSE_FILE" rm -f "$SERVICE"

    echo "  Starting..."
    docker compose -f "$COMPOSE_FILE" up -d "$SERVICE"

    echo "  Container restarted."
    docker compose -f "$COMPOSE_FILE" ps "$SERVICE"
fi

echo ""
echo "Done."
