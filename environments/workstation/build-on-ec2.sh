#!/usr/bin/env bash
# =============================================================================
# Build dm-workstation Docker images on a temporary EC2 instance and push to ECR.
#
# Prerequisites:
#   1. Run `aws sso login --profile elianomarques-dm` first (opens browser)
#   2. Ensure ~/.ssh/id_ed25519 exists (will be imported as EC2 key pair)
#
# Usage:
#   bash build-on-ec2.sh          # Full build (base + groot), push to ECR
#   bash build-on-ec2.sh --base   # Build base only
#   bash build-on-ec2.sh --cleanup-only  # Just terminate any leftover instance
#
# Cost: ~$1.50 for a 2-hour build on c5.4xlarge with 150 GB gp3
# =============================================================================
set -euo pipefail

# ---------- Configuration ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/.env"

PROFILE="${AWS_PROFILE:-elianomarques-dm}"
REGION="us-east-1"  # Same region as ECR for free transfer
ECR_REGISTRY="${AWS_ECR_REGISTRY}"
ECR_REPO="${AWS_ECR_REPO}"

INSTANCE_TYPE="c5.4xlarge"  # 16 vCPU, 32 GB RAM — no GPU needed for build
VOLUME_SIZE=150             # GB, gp3 — peak build uses ~120 GB
KEY_NAME="dm-docker-build"
TAG_NAME="dm-docker-build"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_PUB="$HOME/.ssh/id_ed25519.pub"

BUILD_TARGET="groot"  # default: build both base + groot
CLEANUP_ONLY=false

# ---------- Parse args ----------
for arg in "$@"; do
    case "$arg" in
        --base) BUILD_TARGET="base" ;;
        --cleanup-only) CLEANUP_ONLY=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ---------- Helpers ----------
aws_cmd() { aws --profile "$PROFILE" --region "$REGION" "$@"; }

log() { echo -e "\n\033[1;34m==>\033[0m \033[1m$*\033[0m"; }

cleanup_instance() {
    log "Looking for existing build instances to terminate..."
    local instance_ids
    instance_ids=$(aws_cmd ec2 describe-instances \
        --filters "Name=tag:Name,Values=$TAG_NAME" "Name=instance-state-name,Values=running,pending,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null || true)
    if [ -n "$instance_ids" ] && [ "$instance_ids" != "None" ]; then
        echo "  Terminating: $instance_ids"
        aws_cmd ec2 terminate-instances --instance-ids $instance_ids --output text > /dev/null
        aws_cmd ec2 wait instance-terminated --instance-ids $instance_ids 2>/dev/null || true
        echo "  Terminated."
    else
        echo "  No existing build instances found."
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
        echo "The EC2 instance may still be running."
        echo "To terminate it, run:"
        echo "  bash $0 --cleanup-only"
        echo ""
        echo "Or manually:"
        echo "  aws --profile $PROFILE --region $REGION ec2 terminate-instances --instance-ids \$INSTANCE_ID"
    fi
}
trap cleanup_on_exit EXIT

if $CLEANUP_ONLY; then
    cleanup_instance
    exit 0
fi

# ---------- Preflight checks ----------
log "Preflight checks"

if [ ! -f "$SSH_KEY" ]; then
    echo "ERROR: SSH key not found at $SSH_KEY"
    echo "Generate one: ssh-keygen -t ed25519"
    exit 1
fi

# Verify SSO session is valid
if ! aws_cmd sts get-caller-identity > /dev/null 2>&1; then
    echo "ERROR: AWS SSO session expired or invalid."
    echo "Run:  aws sso login --profile $PROFILE"
    exit 1
fi
echo "  SSO session valid."

# Verify ECR repo exists
if ! aws_cmd ecr describe-repositories --repository-names "$ECR_REPO" > /dev/null 2>&1; then
    echo "ERROR: ECR repo '$ECR_REPO' not found in $REGION"
    exit 1
fi
echo "  ECR repo exists: $ECR_REGISTRY/$ECR_REPO"

# Verify build files exist
for f in "$SCRIPT_DIR/Dockerfile.unitree" "$SCRIPT_DIR/requirements-groot.txt"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required build file not found: $f"
        exit 1
    fi
done
echo "  Build files present."

# ---------- Clean up any previous build instance ----------
cleanup_instance

# ---------- Import SSH key pair (idempotent) ----------
log "Ensuring EC2 key pair '$KEY_NAME' exists"
if ! aws_cmd ec2 describe-key-pairs --key-names "$KEY_NAME" > /dev/null 2>&1; then
    echo "  Importing $SSH_PUB as '$KEY_NAME'..."
    aws_cmd ec2 import-key-pair \
        --key-name "$KEY_NAME" \
        --public-key-material fileb://"$SSH_PUB" > /dev/null
    echo "  Imported."
else
    echo "  Key pair already exists."
fi

# ---------- Find Ubuntu 22.04 AMI ----------
log "Finding Ubuntu 22.04 AMI in $REGION"
AMI_ID=$(aws_cmd ec2 describe-images \
    --owners 099720109477 \
    --filters \
        "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo "ERROR: Could not find Ubuntu 22.04 AMI"
    exit 1
fi
echo "  AMI: $AMI_ID"

# ---------- Create security group (if not exists) ----------
log "Ensuring security group exists"
SG_NAME="dm-docker-build-sg"
SG_ID=$(aws_cmd ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    echo "  Creating security group '$SG_NAME'..."
    SG_ID=$(aws_cmd ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Temporary SG for Docker image builds — SSH only" \
        --query 'GroupId' --output text)
    aws_cmd ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 > /dev/null
    echo "  Created: $SG_ID"
else
    echo "  Exists: $SG_ID"
fi

# ---------- Create IAM instance profile (if not exists) ----------
log "Ensuring IAM instance profile with ECR access"
ROLE_NAME="dm-docker-build-role"
PROFILE_NAME="dm-docker-build-profile"

if ! aws_cmd iam get-role --role-name "$ROLE_NAME" > /dev/null 2>&1; then
    echo "  Creating IAM role '$ROLE_NAME'..."
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
    echo "  Role created and policy attached."
else
    echo "  Role exists."
fi

if ! aws_cmd iam get-instance-profile --instance-profile-name "$PROFILE_NAME" > /dev/null 2>&1; then
    echo "  Creating instance profile '$PROFILE_NAME'..."
    aws_cmd iam create-instance-profile --instance-profile-name "$PROFILE_NAME" > /dev/null
    aws_cmd iam add-role-to-instance-profile \
        --instance-profile-name "$PROFILE_NAME" \
        --role-name "$ROLE_NAME"
    echo "  Created. Waiting 10s for IAM propagation..."
    sleep 10
else
    echo "  Instance profile exists."
fi

# ---------- Launch EC2 instance ----------
log "Launching $INSTANCE_TYPE instance"
INSTANCE_ID=$(aws_cmd ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile "Name=$PROFILE_NAME" \
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

echo "  Instance ID: $INSTANCE_ID"
echo "  Waiting for instance to be running..."
aws_cmd ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws_cmd ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "  Public IP: $PUBLIC_IP"

# ---------- Wait for SSH to be ready ----------
log "Waiting for SSH to be ready on $PUBLIC_IP"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o LogLevel=ERROR"
for i in $(seq 1 30); do
    if ssh $SSH_OPTS -i "$SSH_KEY" ubuntu@"$PUBLIC_IP" "echo ready" 2>/dev/null; then
        break
    fi
    echo "  Attempt $i/30 — waiting..."
    sleep 10
done

# ---------- Remote helper ----------
remote() {
    ssh $SSH_OPTS -i "$SSH_KEY" ubuntu@"$PUBLIC_IP" "$@"
}

remote_script() {
    ssh $SSH_OPTS -i "$SSH_KEY" ubuntu@"$PUBLIC_IP" "bash -s" <<< "$1"
}

# ---------- Install Docker on the instance ----------
log "Installing Docker on EC2 instance"
remote_script '
set -euo pipefail
echo "Installing Docker..."
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io > /dev/null 2>&1
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
echo "Docker installed: $(docker --version)"
'

# ---------- Upload build files ----------
log "Uploading build files"
scp $SSH_OPTS -i "$SSH_KEY" \
    "$SCRIPT_DIR/Dockerfile.unitree" \
    "$SCRIPT_DIR/requirements-groot.txt" \
    ubuntu@"$PUBLIC_IP":/tmp/

remote_script '
mkdir -p ~/build
mv /tmp/Dockerfile.unitree ~/build/
mv /tmp/requirements-groot.txt ~/build/
ls -la ~/build/
'

# ---------- ECR login on the instance (via instance role) ----------
log "Logging into ECR on the instance"
remote_script "
set -euo pipefail
sudo apt-get install -y -qq awscli > /dev/null 2>&1
aws ecr get-login-password --region $REGION | \
    sudo docker login --username AWS --password-stdin $ECR_REGISTRY
echo 'ECR login successful.'
"

# ---------- Build Docker images ----------
log "Building Docker image (target: $BUILD_TARGET)"
echo "  This will take 75-115 minutes. You can monitor with:"
echo "  ssh $SSH_OPTS -i $SSH_KEY ubuntu@$PUBLIC_IP 'tail -f ~/build/build.log'"
echo ""

# Run build and push on the instance
BASE_TAG="$ECR_REGISTRY/$ECR_REPO:base-latest"
GROOT_TAG="$ECR_REGISTRY/$ECR_REPO:latest"

remote "
set -euo pipefail
cd ~/build

echo '=== Build started at \$(date) ===' | tee build.log

# Build base first (needed by groot)
echo '>>> Building base stage...' | tee -a build.log
sudo docker build --target base \
    -t '$BASE_TAG' \
    -f Dockerfile.unitree \
    . 2>&1 | tee -a build.log

echo '>>> Base build complete at \$(date)' | tee -a build.log

if [ '$BUILD_TARGET' = 'groot' ]; then
    echo '>>> Building groot stage...' | tee -a build.log
    sudo docker build --target groot \
        -t '$GROOT_TAG' \
        -f Dockerfile.unitree \
        . 2>&1 | tee -a build.log
    echo '>>> Groot build complete at \$(date)' | tee -a build.log
fi

echo '=== Build finished at \$(date) ===' | tee -a build.log

# Push to ECR
echo '>>> Pushing base-latest...' | tee -a build.log
sudo docker push '$BASE_TAG' 2>&1 | tee -a build.log
echo '>>> base-latest pushed.' | tee -a build.log

if [ '$BUILD_TARGET' = 'groot' ]; then
    echo '>>> Pushing latest...' | tee -a build.log
    sudo docker push '$GROOT_TAG' 2>&1 | tee -a build.log
    echo '>>> latest pushed.' | tee -a build.log
fi

echo '=== ALL DONE at \$(date) ===' | tee -a build.log
"

# ---------- Verify ----------
log "Verifying ECR images"
aws_cmd ecr describe-images --repository-name "$ECR_REPO" \
    --query 'imageDetails[*].{tag:imageTags[0],pushed:imagePushedAt,size:imageSizeInBytes}' \
    --output table

# ---------- Terminate instance ----------
log "Terminating EC2 instance $INSTANCE_ID"
aws_cmd ec2 terminate-instances --instance-ids "$INSTANCE_ID" --output text > /dev/null
echo "  Instance $INSTANCE_ID terminating."
echo ""

# ---------- Summary ----------
echo "=========================================="
echo "  BUILD & PUSH COMPLETE"
echo "=========================================="
echo ""
echo "  ECR images:"
echo "    $ECR_REGISTRY/$ECR_REPO:base-latest"
if [ "$BUILD_TARGET" = "groot" ]; then
    echo "    $ECR_REGISTRY/$ECR_REPO:latest"
fi
echo ""
echo "  To pull on the workstation:"
echo "    aws ecr get-login-password --region $REGION | \\"
echo "      docker login --username AWS --password-stdin $ECR_REGISTRY"
echo "    docker pull $ECR_REGISTRY/$ECR_REPO:latest"
echo "    docker tag $ECR_REGISTRY/$ECR_REPO:latest dm-workstation:latest"
echo ""
echo "  To replace the running container (AFTER training finishes):"
echo "    cd /home/datamentors/dm-isaac-g1/environments/workstation"
echo "    docker compose -f docker-compose.unitree.yml stop groot"
echo "    docker compose -f docker-compose.unitree.yml rm groot"
echo "    docker compose -f docker-compose.unitree.yml up -d groot"
echo ""
