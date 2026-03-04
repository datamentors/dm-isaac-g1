#!/usr/bin/env bash
# =============================================================================
# Build dm-workstation or dm-spark Docker images on a temporary EC2 instance
# and push to ECR.
#
# Launches a temp build instance matching the target architecture, uploads the
# build context, builds the image, pushes to ECR, and terminates.
#
# Prerequisites:
#   1. AWS SSO login:  aws sso login --profile elianomarques-dm
#   2. SSH key:        ~/.ssh/id_ed25519  (will be imported as EC2 key pair)
#
# Usage:
#   ./build.sh                    # Build workstation (x86_64, base + groot)
#   ./build.sh --base             # Build workstation base stage only
#   ./build.sh --groot            # Build workstation groot stage only
#   ./build.sh --spark            # Build Spark image (ARM64 Graviton)
#   ./build.sh --cleanup-only     # Terminate any leftover build instances
#   ./build.sh --no-terminate     # Keep instance alive after build
#
# Cost:
#   Workstation (c5.4xlarge x86_64): ~$1.50 for a 2-hour build
#   Spark (c7g.4xlarge ARM64):       ~$1.20 for a 1-hour build
# =============================================================================
set -euo pipefail

# ---------- Paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSTATION_DIR="$REPO_ROOT/environments/workstation"
SPARK_DIR="$REPO_ROOT/environments/spark"

source "$REPO_ROOT/.env"

# ---------- Configuration ----------
PROFILE="${AWS_PROFILE:-elianomarques-dm}"
REGION="${AWS_ECR_REGION:-${AWS_REGION:-eu-west-1}}"
ECR_REGISTRY="${AWS_ECR_REGISTRY}"
ECR_REPO="${AWS_ECR_REPO}"

# Spark uses a separate ECR repo (ARM64 images can't share tags with x86_64)
ECR_REPO_SPARK="${AWS_ECR_REPO_SPARK:-isaac-g1-spark}"

KEY_NAME="dm-docker-build"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_PUB="$HOME/.ssh/id_ed25519.pub"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o LogLevel=ERROR"

BUILD_TARGET="groot"  # default: base + groot (workstation)
BUILD_PLATFORM="workstation"  # workstation or spark
NO_TERMINATE=false
CLEANUP_ONLY=false

# ---------- Parse args ----------
for arg in "$@"; do
    case "$arg" in
        --base)          BUILD_TARGET="base" ;;
        --groot)         BUILD_TARGET="groot" ;;
        --spark)         BUILD_PLATFORM="spark"; BUILD_TARGET="spark" ;;
        --no-terminate)  NO_TERMINATE=true ;;
        --cleanup-only)  CLEANUP_ONLY=true ;;
        -h|--help)
            head -24 "$0" | tail -19
            exit 0 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Platform-specific configuration
if [ "$BUILD_PLATFORM" = "spark" ]; then
    INSTANCE_TYPE="c7g.4xlarge"    # 16 vCPU ARM64 Graviton3, 32 GB
    VOLUME_SIZE=100                # Spark image is smaller (~28 GB)
    TAG_NAME="dm-docker-build-spark"
    AMI_FILTER="ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"
    ACTIVE_ECR_REPO="$ECR_REPO_SPARK"
else
    INSTANCE_TYPE="c5.4xlarge"     # 16 vCPU x86_64, 32 GB
    VOLUME_SIZE=150                # Workstation image is larger (~42 GB)
    TAG_NAME="dm-docker-build"
    AMI_FILTER="ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"
    ACTIVE_ECR_REPO="$ECR_REPO"
fi

# ---------- Helpers ----------
aws_cmd() { aws --profile "$PROFILE" --region "$REGION" "$@"; }
log() { echo -e "\n\033[1;34m==>\033[0m \033[1m$*\033[0m"; }

remote() {
    ssh $SSH_OPTS -i "$SSH_KEY" ubuntu@"$PUBLIC_IP" "$@"
}

remote_script() {
    ssh $SSH_OPTS -i "$SSH_KEY" ubuntu@"$PUBLIC_IP" "bash -s" <<< "$1"
}

cleanup_instance() {
    log "Looking for existing build instances ($TAG_NAME)..."
    local ids
    ids=$(aws_cmd ec2 describe-instances \
        --filters "Name=tag:Name,Values=$TAG_NAME" \
                  "Name=instance-state-name,Values=running,pending,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null || true)
    if [ -n "$ids" ] && [ "$ids" != "None" ]; then
        echo "  Terminating: $ids"
        aws_cmd ec2 terminate-instances --instance-ids $ids --output text > /dev/null
        aws_cmd ec2 wait instance-terminated --instance-ids $ids 2>/dev/null || true
        echo "  Terminated."
    else
        echo "  No existing build instances."
    fi
}

cleanup_on_exit() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "  BUILD FAILED (exit code $exit_code)"
        echo "=========================================="
        echo ""
        echo "  Instance may still be running."
        echo "  To terminate:  $0 --cleanup-only"
        echo "  To debug:      ssh $SSH_OPTS -i $SSH_KEY ubuntu@\${PUBLIC_IP:-unknown}"
    fi
}
trap cleanup_on_exit EXIT

if $CLEANUP_ONLY; then
    # Clean up both workstation and spark build instances
    TAG_NAME="dm-docker-build" cleanup_instance
    TAG_NAME="dm-docker-build-spark" cleanup_instance
    exit 0
fi

# ---------- Preflight ----------
log "Preflight checks (platform: $BUILD_PLATFORM)"

if [ ! -f "$SSH_KEY" ]; then
    echo "ERROR: SSH key not found: $SSH_KEY"
    echo "Generate: ssh-keygen -t ed25519"
    exit 1
fi

if ! aws_cmd sts get-caller-identity > /dev/null 2>&1; then
    echo "ERROR: AWS session expired. Run: aws sso login --profile $PROFILE"
    exit 1
fi
echo "  AWS session valid."

# Ensure ECR repo exists (create if needed for spark)
if ! aws_cmd ecr describe-repositories --repository-names "$ACTIVE_ECR_REPO" > /dev/null 2>&1; then
    if [ "$BUILD_PLATFORM" = "spark" ]; then
        echo "  Creating ECR repo: $ACTIVE_ECR_REPO"
        aws_cmd ecr create-repository --repository-name "$ACTIVE_ECR_REPO" > /dev/null
    else
        echo "ERROR: ECR repo '$ACTIVE_ECR_REPO' not found in $REGION"
        exit 1
    fi
fi
echo "  ECR repo: $ECR_REGISTRY/$ACTIVE_ECR_REPO"

# ---------- Collect build context ----------
if [ "$BUILD_PLATFORM" = "spark" ]; then
    log "Collecting build context from environments/spark/"
    if [ ! -f "$SPARK_DIR/Dockerfile.spark" ]; then
        echo "ERROR: Missing: $SPARK_DIR/Dockerfile.spark"
        exit 1
    fi
    BUILD_FILES=("$SPARK_DIR/Dockerfile.spark")
else
    log "Collecting build context from environments/workstation/"
    for f in "$WORKSTATION_DIR/Dockerfile.unitree" "$WORKSTATION_DIR/requirements-groot.txt"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: Missing: $f"
            exit 1
        fi
    done
    BUILD_FILES=(
        "$WORKSTATION_DIR/Dockerfile.unitree"
        "$WORKSTATION_DIR/requirements-groot.txt"
    )
    if [ -d "$WORKSTATION_DIR/patches" ]; then
        PATCH_COUNT=$(find "$WORKSTATION_DIR/patches" -name '*.patch' | wc -l | tr -d ' ')
        echo "  Found $PATCH_COUNT patch file(s)"
        BUILD_FILES+=("$WORKSTATION_DIR/patches")
    fi
fi

echo "  Build files:"
for f in "${BUILD_FILES[@]}"; do
    echo "    $(basename "$f")"
done

# ---------- Clean up previous ----------
cleanup_instance

# ---------- SSH key pair ----------
log "Ensuring EC2 key pair"
if ! aws_cmd ec2 describe-key-pairs --key-names "$KEY_NAME" > /dev/null 2>&1; then
    echo "  Importing $SSH_PUB..."
    aws_cmd ec2 import-key-pair \
        --key-name "$KEY_NAME" \
        --public-key-material fileb://"$SSH_PUB" > /dev/null
fi
echo "  Key pair: $KEY_NAME"

# ---------- AMI ----------
log "Finding AMI ($AMI_FILTER)"
AMI_ID=$(aws_cmd ec2 describe-images \
    --owners 099720109477 \
    --filters \
        "Name=name,Values=$AMI_FILTER" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo "ERROR: AMI not found for filter: $AMI_FILTER"
    exit 1
fi
echo "  AMI: $AMI_ID"

# ---------- Security group ----------
log "Ensuring security group"
SG_NAME="dm-docker-build-sg"
SG_ID=$(aws_cmd ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    SG_ID=$(aws_cmd ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Docker image build - SSH only" \
        --query 'GroupId' --output text)
    aws_cmd ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 > /dev/null
fi
echo "  SG: $SG_ID"

# ---------- IAM instance profile ----------
log "Ensuring IAM role with ECR access"
ROLE_NAME="dm-docker-build-role"
IAM_PROFILE_NAME="dm-docker-build-profile"

if ! aws_cmd iam get-role --role-name "$ROLE_NAME" > /dev/null 2>&1; then
    aws_cmd iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' > /dev/null
    aws_cmd iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
fi

if ! aws_cmd iam get-instance-profile --instance-profile-name "$IAM_PROFILE_NAME" > /dev/null 2>&1; then
    aws_cmd iam create-instance-profile --instance-profile-name "$IAM_PROFILE_NAME" > /dev/null
    aws_cmd iam add-role-to-instance-profile \
        --instance-profile-name "$IAM_PROFILE_NAME" --role-name "$ROLE_NAME"
    echo "  Waiting 10s for IAM propagation..."
    sleep 10
fi
echo "  IAM: $IAM_PROFILE_NAME"

# ---------- Launch instance ----------
log "Launching $INSTANCE_TYPE ($BUILD_PLATFORM)"
INSTANCE_ID=$(aws_cmd ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile "Name=$IAM_PROFILE_NAME" \
    --block-device-mappings "[{
        \"DeviceName\": \"/dev/sda1\",
        \"Ebs\": {
            \"VolumeSize\": $VOLUME_SIZE,
            \"VolumeType\": \"gp3\",
            \"DeleteOnTermination\": true
        }
    }]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG_NAME}]" \
    --query 'Instances[0].InstanceId' --output text)

echo "  Instance: $INSTANCE_ID"
echo "  Waiting for running state..."
aws_cmd ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws_cmd ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "  IP: $PUBLIC_IP"

# ---------- Wait for SSH ----------
log "Waiting for SSH"
for i in $(seq 1 30); do
    if ssh $SSH_OPTS -i "$SSH_KEY" ubuntu@"$PUBLIC_IP" "echo ready" 2>/dev/null; then
        break
    fi
    echo "  Attempt $i/30..."
    sleep 10
done

# ---------- Install Docker ----------
log "Installing Docker + AWS CLI"
remote_script '
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io unzip > /dev/null 2>&1
sudo systemctl enable docker && sudo systemctl start docker
sudo usermod -aG docker ubuntu

# AWS CLI v2 (awscli apt package not available on Noble ARM64)
if ! command -v aws &> /dev/null; then
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscli.zip
    else
        curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip
    fi
    unzip -qq /tmp/awscli.zip -d /tmp
    sudo /tmp/aws/install
    rm -rf /tmp/awscli.zip /tmp/aws
fi
echo "Docker $(docker --version | cut -d, -f1 | cut -d\" \" -f3)"
echo "AWS CLI $(aws --version | cut -d/ -f2 | cut -d\" \" -f1)"
'

# ---------- Upload build context ----------
log "Uploading build context"
remote "mkdir -p /tmp/build-upload"

for f in "${BUILD_FILES[@]}"; do
    echo "  Uploading: $(basename "$f")"
    scp $SSH_OPTS -r -i "$SSH_KEY" "$f" ubuntu@"$PUBLIC_IP":/tmp/build-upload/
done

remote_script '
mkdir -p ~/build
cp -r /tmp/build-upload/* ~/build/
echo "Build context:"
find ~/build -type f | sort | sed "s|^|  |"
'

# ---------- ECR login ----------
log "ECR login on build instance"
remote_script "
aws ecr get-login-password --region $REGION | \
    sudo docker login --username AWS --password-stdin $ECR_REGISTRY
"

# ---------- Build ----------
if [ "$BUILD_PLATFORM" = "spark" ]; then
    # ── Spark ARM64 build ──
    SPARK_TAG="$ECR_REGISTRY/$ACTIVE_ECR_REPO:latest"
    DATE_TAG="$ECR_REGISTRY/$ACTIVE_ECR_REPO:$(date +%Y%m%d)"

    log "Building Spark image (ARM64)"
    echo "  Monitor: ssh $SSH_OPTS -i $SSH_KEY ubuntu@$PUBLIC_IP 'tail -f ~/build/build.log'"

    remote "
    set -euo pipefail
    cd ~/build

    echo '=== Spark build started at \$(date) ===' | tee build.log

    sudo docker build \
        --build-arg GITHUB_TOKEN='${GITHUB_TOKEN:-}' \
        -t '$SPARK_TAG' \
        -t '$DATE_TAG' \
        -f Dockerfile.spark \
        . 2>&1 | tee -a build.log

    echo '=== Build finished at \$(date) ===' | tee -a build.log

    echo '>>> Pushing latest...' | tee -a build.log
    sudo docker push '$SPARK_TAG' 2>&1 | tee -a build.log

    echo '>>> Pushing date tag...' | tee -a build.log
    sudo docker push '$DATE_TAG' 2>&1 | tee -a build.log

    echo '=== ALL DONE at \$(date) ===' | tee -a build.log
    "
else
    # ── Workstation x86_64 build ──
    BASE_TAG="$ECR_REGISTRY/$ACTIVE_ECR_REPO:base-latest"
    GROOT_TAG="$ECR_REGISTRY/$ACTIVE_ECR_REPO:latest"
    DATE_TAG="$ECR_REGISTRY/$ACTIVE_ECR_REPO:$(date +%Y%m%d)"

    log "Building workstation image (target: $BUILD_TARGET)"
    echo "  Monitor: ssh $SSH_OPTS -i $SSH_KEY ubuntu@$PUBLIC_IP 'tail -f ~/build/build.log'"

    remote "
    set -euo pipefail
    cd ~/build

    echo '=== Build started at \$(date) ===' | tee build.log

    echo '>>> Building base stage...' | tee -a build.log
    sudo docker build --target base \
        -t '$BASE_TAG' \
        -f Dockerfile.unitree \
        . 2>&1 | tee -a build.log
    echo '>>> Base complete at \$(date)' | tee -a build.log

    if [ '$BUILD_TARGET' = 'groot' ]; then
        echo '>>> Building groot stage...' | tee -a build.log
        sudo docker build --target groot \
            --build-arg GITHUB_TOKEN='${GITHUB_TOKEN:-}' \
            -t '$GROOT_TAG' \
            -t '$DATE_TAG' \
            -f Dockerfile.unitree \
            . 2>&1 | tee -a build.log
        echo '>>> Groot complete at \$(date)' | tee -a build.log
    fi

    echo '=== Build finished at \$(date) ===' | tee -a build.log

    echo '>>> Pushing base-latest...' | tee -a build.log
    sudo docker push '$BASE_TAG' 2>&1 | tee -a build.log

    if [ '$BUILD_TARGET' = 'groot' ]; then
        echo '>>> Pushing latest...' | tee -a build.log
        sudo docker push '$GROOT_TAG' 2>&1 | tee -a build.log
        echo '>>> Pushing date tag...' | tee -a build.log
        sudo docker push '$DATE_TAG' 2>&1 | tee -a build.log
    fi

    echo '=== ALL DONE at \$(date) ===' | tee -a build.log
    "
fi

# ---------- Verify ----------
log "ECR images"
aws_cmd ecr describe-images --repository-name "$ACTIVE_ECR_REPO" \
    --query 'sort_by(imageDetails, &imagePushedAt)[-3:].{tags:imageTags,pushed:imagePushedAt,sizeMB:to_string(div(imageSizeInBytes,`1048576`))}' \
    --output table

# ---------- Terminate ----------
if $NO_TERMINATE; then
    echo ""
    log "Instance kept alive: $INSTANCE_ID ($PUBLIC_IP)"
    echo "  SSH:       ssh $SSH_OPTS -i $SSH_KEY ubuntu@$PUBLIC_IP"
    echo "  Terminate: $0 --cleanup-only"
else
    log "Terminating $INSTANCE_ID"
    aws_cmd ec2 terminate-instances --instance-ids "$INSTANCE_ID" --output text > /dev/null
fi

# ---------- Summary ----------
echo ""
echo "=========================================="
echo "  BUILD & PUSH COMPLETE ($BUILD_PLATFORM)"
echo "=========================================="
echo ""
if [ "$BUILD_PLATFORM" = "spark" ]; then
    echo "  ECR images:"
    echo "    $ECR_REGISTRY/$ACTIVE_ECR_REPO:latest"
    echo "    $ECR_REGISTRY/$ACTIVE_ECR_REPO:$(date +%Y%m%d)"
    echo ""
    echo "  Pull on Spark:"
    echo "    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY"
    echo "    docker pull $ECR_REGISTRY/$ACTIVE_ECR_REPO:latest"
    echo "    docker tag $ECR_REGISTRY/$ACTIVE_ECR_REPO:latest dm-spark-workstation:latest"
else
    echo "  ECR images:"
    echo "    $ECR_REGISTRY/$ACTIVE_ECR_REPO:base-latest"
    if [ "$BUILD_TARGET" = "groot" ]; then
        echo "    $ECR_REGISTRY/$ACTIVE_ECR_REPO:latest"
        echo "    $ECR_REGISTRY/$ACTIVE_ECR_REPO:$(date +%Y%m%d)"
    fi
    echo ""
    echo "  Pull on workstation:"
    echo "    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY"
    echo "    docker pull $ECR_REGISTRY/$ACTIVE_ECR_REPO:latest"
    echo "    docker tag $ECR_REGISTRY/$ACTIVE_ECR_REPO:latest dm-workstation:latest"
fi
echo ""
echo "  Update running container:"
if [ "$BUILD_PLATFORM" = "spark" ]; then
    echo "    cd dm-isaac-g1/environments/spark"
    echo "    docker compose -f docker-compose.spark.yml stop workstation"
    echo "    docker compose -f docker-compose.spark.yml rm -f workstation"
    echo "    docker compose -f docker-compose.spark.yml up -d workstation"
else
    echo "    cd dm-isaac-g1/environments/workstation"
    echo "    docker compose -f docker-compose.unitree.yml stop groot"
    echo "    docker compose -f docker-compose.unitree.yml rm -f groot"
    echo "    docker compose -f docker-compose.unitree.yml up -d groot"
fi
echo ""
