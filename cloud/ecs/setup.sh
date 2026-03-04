#!/usr/bin/env bash
# =============================================================================
# ECS GPU Cluster Setup (one-time)
# =============================================================================
# Creates the ECS cluster, Auto Scaling Group with GPU instances, capacity
# provider, IAM roles, S3 bucket, and security group.
#
# Run this ONCE to set up the infrastructure. Then use run.sh to submit jobs.
#
# Usage:
#   ./setup.sh [--instance-type g5.2xlarge] [--max-instances 4]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
AWS_PROFILE="${AWS_PROFILE:-elianomarques-dm}"
AWS_REGION="${AWS_REGION:-eu-west-1}"
ECR_REGISTRY="${AWS_ECR_REGISTRY:-260464233120.dkr.ecr.eu-west-1.amazonaws.com}"
ECR_REPO="${AWS_ECR_REPO:-isaac-g1-sim-ft-rl}"

CLUSTER_NAME="dm-isaac-g1-gpu"
ASG_NAME="dm-isaac-g1-gpu-asg"
LAUNCH_TEMPLATE_NAME="dm-isaac-g1-gpu-lt"
CAPACITY_PROVIDER_NAME="dm-isaac-g1-gpu-cp"
S3_BUCKET="dm-isaac-g1-training-${AWS_REGION}"
SG_NAME="dm-isaac-g1-ecs-sg"
ROLE_NAME="dm-isaac-g1-ecs-instance"
TASK_ROLE_NAME="dm-isaac-g1-ecs-task"
KEY_NAME="dm-isaac-g1-training"
INSTANCE_TYPE="g5.2xlarge"
MAX_INSTANCES=4
DISK_SIZE_GB=200

# ECS-optimized GPU AMI — look it up dynamically
AMI_ID=""

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*" >&2; }

aws_cmd() { aws --profile "$AWS_PROFILE" --region "$AWS_REGION" "$@"; }

# ── Parse Arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --instance-type)  INSTANCE_TYPE="$2"; shift 2 ;;
        --max-instances)  MAX_INSTANCES="$2"; shift 2 ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Validate AWS Access ──────────────────────────────────────────────────────
log "Validating AWS credentials..."
ACCOUNT_ID=$(aws_cmd sts get-caller-identity --query Account --output text)
log "AWS Account: $ACCOUNT_ID, Region: $AWS_REGION"

# ── 1. Find ECS-Optimized GPU AMI ────────────────────────────────────────────
log "Finding ECS-optimized GPU AMI..."
AMI_ID=$(aws_cmd ssm get-parameters \
    --names /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id \
    --query "Parameters[0].Value" --output text 2>/dev/null)

if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    # Fallback: search for it
    AMI_ID=$(aws_cmd ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=amzn2-ami-ecs-gpu-hvm-*-x86_64-ebs" \
        --query "Images | sort_by(@, &CreationDate)[-1].ImageId" --output text)
fi
log "AMI: $AMI_ID"

# ── 2. S3 Bucket ─────────────────────────────────────────────────────────────
if ! aws_cmd s3api head-bucket --bucket "$S3_BUCKET" &>/dev/null; then
    log "Creating S3 bucket: $S3_BUCKET"
    aws_cmd s3api create-bucket \
        --bucket "$S3_BUCKET" \
        --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null
else
    log "S3 bucket exists: $S3_BUCKET"
fi

# ── 3. Key Pair ───────────────────────────────────────────────────────────────
KEY_FILE="$SCRIPT_DIR/${KEY_NAME}.pem"
if [[ ! -f "$KEY_FILE" ]]; then
    if aws_cmd ec2 describe-key-pairs --key-names "$KEY_NAME" &>/dev/null; then
        warn "Key pair exists in AWS but no local .pem. Deleting and recreating..."
        aws_cmd ec2 delete-key-pair --key-name "$KEY_NAME"
    fi
    log "Creating key pair: $KEY_NAME"
    aws_cmd ec2 create-key-pair --key-name "$KEY_NAME" --query "KeyMaterial" --output text > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
else
    log "Key pair exists: $KEY_FILE"
    # Ensure it exists in AWS too
    if ! aws_cmd ec2 describe-key-pairs --key-names "$KEY_NAME" &>/dev/null; then
        log "Re-importing key pair to AWS..."
        aws_cmd ec2 import-key-pair --key-name "$KEY_NAME" \
            --public-key-material "fileb://<(ssh-keygen -y -f $KEY_FILE)" 2>/dev/null || true
    fi
fi

# ── 4. Security Group ────────────────────────────────────────────────────────
VPC_ID=$(aws_cmd ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
SG_ID=$(aws_cmd ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null)

if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
    log "Creating security group: $SG_NAME"
    SG_ID=$(aws_cmd ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "DM Isaac G1 ECS GPU instances" \
        --vpc-id "$VPC_ID" \
        --query "GroupId" --output text)
    aws_cmd ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 >/dev/null
    # VNC — TurboVNC on port 5901 (already installed in Docker image)
    aws_cmd ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 5901 --cidr 0.0.0.0/0 >/dev/null
    # Allow all outbound (default)
else
    log "Security group exists: $SG_ID"
fi

# ── 5. IAM Role for EC2 Instances ────────────────────────────────────────────
if ! aws_cmd iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    log "Creating IAM instance role: $ROLE_NAME"
    aws_cmd iam create-role --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' >/dev/null

    # ECS agent needs these
    aws_cmd iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role
    # S3 access for training data
    aws_cmd iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    # ECR pull
    aws_cmd iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
    # CloudWatch logs
    aws_cmd iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
else
    log "IAM instance role exists: $ROLE_NAME"
fi

# Instance profile
if ! aws_cmd iam get-instance-profile --instance-profile-name "$ROLE_NAME" &>/dev/null; then
    log "Creating instance profile..."
    aws_cmd iam create-instance-profile --instance-profile-name "$ROLE_NAME" >/dev/null
    aws_cmd iam add-role-to-instance-profile --instance-profile-name "$ROLE_NAME" --role-name "$ROLE_NAME"
    log "Waiting for IAM propagation (15s)..."
    sleep 15
else
    log "Instance profile exists: $ROLE_NAME"
fi

# ── 6. IAM Role for ECS Tasks ────────────────────────────────────────────────
if ! aws_cmd iam get-role --role-name "$TASK_ROLE_NAME" &>/dev/null; then
    log "Creating ECS task role: $TASK_ROLE_NAME"
    aws_cmd iam create-role --role-name "$TASK_ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' >/dev/null

    aws_cmd iam attach-role-policy --role-name "$TASK_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    aws_cmd iam attach-role-policy --role-name "$TASK_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess

    # SSM permissions for ECS Exec (interactive shell access)
    aws_cmd iam put-role-policy --role-name "$TASK_ROLE_NAME" \
        --policy-name "ecs-exec-ssm" \
        --policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": [
                    "ssmmessages:CreateControlChannel",
                    "ssmmessages:CreateDataChannel",
                    "ssmmessages:OpenControlChannel",
                    "ssmmessages:OpenDataChannel"
                ],
                "Resource": "*"
            }]
        }'
    log "Added SSM permissions for ECS Exec"
else
    log "ECS task role exists: $TASK_ROLE_NAME"
    # Ensure SSM policy exists (idempotent)
    aws_cmd iam put-role-policy --role-name "$TASK_ROLE_NAME" \
        --policy-name "ecs-exec-ssm" \
        --policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": [
                    "ssmmessages:CreateControlChannel",
                    "ssmmessages:CreateDataChannel",
                    "ssmmessages:OpenControlChannel",
                    "ssmmessages:OpenDataChannel"
                ],
                "Resource": "*"
            }]
        }' 2>/dev/null || true
fi

# ECS task execution role (for pulling ECR images, writing logs)
EXEC_ROLE_NAME="dm-isaac-g1-ecs-exec"
if ! aws_cmd iam get-role --role-name "$EXEC_ROLE_NAME" &>/dev/null; then
    log "Creating ECS execution role: $EXEC_ROLE_NAME"
    aws_cmd iam create-role --role-name "$EXEC_ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' >/dev/null
    aws_cmd iam attach-role-policy --role-name "$EXEC_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
else
    log "ECS execution role exists: $EXEC_ROLE_NAME"
fi

# ── 7. ECS Cluster ───────────────────────────────────────────────────────────
if ! aws_cmd ecs describe-clusters --clusters "$CLUSTER_NAME" --query "clusters[?status=='ACTIVE'].clusterName" --output text | grep -q "$CLUSTER_NAME"; then
    log "Creating ECS cluster: $CLUSTER_NAME"
    aws_cmd ecs create-cluster --cluster-name "$CLUSTER_NAME" \
        --settings "name=containerInsights,value=enabled" \
        --configuration '{
            "executeCommandConfiguration": {
                "logging": "DEFAULT"
            }
        }' >/dev/null
else
    log "ECS cluster exists: $CLUSTER_NAME"
    # Ensure ECS Exec is enabled on existing cluster
    aws_cmd ecs update-cluster --cluster "$CLUSTER_NAME" \
        --configuration '{
            "executeCommandConfiguration": {
                "logging": "DEFAULT"
            }
        }' >/dev/null 2>&1 || true
fi

# ── 8. Launch Template ───────────────────────────────────────────────────────
# User data for ECS instances: register with cluster
USER_DATA=$(cat << USERDATA | base64
#!/bin/bash

# ECS configuration
echo ECS_CLUSTER=${CLUSTER_NAME} >> /etc/ecs/ecs.config
echo ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config
echo 'ECS_AVAILABLE_LOGGING_DRIVERS=["json-file","awslogs"]' >> /etc/ecs/ecs.config
echo ECS_ENABLE_EXECUTE_COMMAND=true >> /etc/ecs/ecs.config

# Load NVIDIA DRM kernel module for Vulkan/rendering support (creates /dev/dri/)
modprobe nvidia-drm modeset=1 2>/dev/null || true

# Install full NVIDIA Vulkan support on the host.
# The ECS GPU AMI only includes compute driver libs, not graphics/Vulkan.
# Isaac Sim and other rendering apps need the Vulkan producer library.
# We extract it from the NVIDIA .run installer matching the host driver version.
# The resulting libs are bind-mounted into containers via ECS task definition volumes.
DRIVER_VERSION=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
if [ -n "\$DRIVER_VERSION" ]; then
    if [ ! -f /usr/lib64/libnvidia-vulkan-producer.so.\${DRIVER_VERSION} ]; then
        echo "Installing NVIDIA Vulkan support for driver \${DRIVER_VERSION}..."
        DRIVER_URL="https://us.download.nvidia.com/tesla/\${DRIVER_VERSION}/NVIDIA-Linux-x86_64-\${DRIVER_VERSION}.run"
        curl -sL "\$DRIVER_URL" -o /tmp/nvidia-driver.run
        chmod +x /tmp/nvidia-driver.run
        /tmp/nvidia-driver.run --extract-only --target /tmp/nvidia-driver 2>/dev/null || true
        if [ -d /tmp/nvidia-driver ]; then
            cp -f /tmp/nvidia-driver/libnvidia-vulkan-producer.so.\${DRIVER_VERSION} /usr/lib64/ 2>/dev/null || true
            ln -sf libnvidia-vulkan-producer.so.\${DRIVER_VERSION} /usr/lib64/libnvidia-vulkan-producer.so 2>/dev/null || true
            # Also copy to dedicated dir for container bind-mounting (avoids exposing all of /usr/lib64)
            mkdir -p /opt/nvidia-vulkan
            cp -f /tmp/nvidia-driver/libnvidia-vulkan-producer.so.\${DRIVER_VERSION} /opt/nvidia-vulkan/ 2>/dev/null || true
            ln -sf libnvidia-vulkan-producer.so.\${DRIVER_VERSION} /opt/nvidia-vulkan/libnvidia-vulkan-producer.so 2>/dev/null || true
            mkdir -p /etc/vulkan/icd.d /usr/share/vulkan/icd.d
            cat > /usr/share/vulkan/icd.d/nvidia_icd.json << VICD
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libnvidia-vulkan-producer.so.\${DRIVER_VERSION}",
        "api_version" : "1.3"
    }
}
VICD
            cp /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/
            ldconfig
            echo "NVIDIA Vulkan support installed successfully"
        fi
        rm -rf /tmp/nvidia-driver /tmp/nvidia-driver.run
    fi
fi

# Install AWS CLI v2 for S3 operations inside tasks
if ! command -v aws &>/dev/null; then
    curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    /tmp/aws/install
    rm -rf /tmp/awscliv2.zip /tmp/aws
fi

# Note: VNC (TurboVNC), Chrome, XFCE4 are already in the ECR Docker image.
# No need to install on the host — they run inside the container.
USERDATA
)

LT_EXISTS=$(aws_cmd ec2 describe-launch-templates --launch-template-names "$LAUNCH_TEMPLATE_NAME" --query "LaunchTemplates[0].LaunchTemplateId" --output text 2>/dev/null || echo "None")

if [[ "$LT_EXISTS" == "None" || -z "$LT_EXISTS" ]]; then
    log "Creating launch template: $LAUNCH_TEMPLATE_NAME"
    aws_cmd ec2 create-launch-template \
        --launch-template-name "$LAUNCH_TEMPLATE_NAME" \
        --launch-template-data "{
            \"ImageId\": \"$AMI_ID\",
            \"InstanceType\": \"$INSTANCE_TYPE\",
            \"KeyName\": \"$KEY_NAME\",
            \"SecurityGroupIds\": [\"$SG_ID\"],
            \"IamInstanceProfile\": {\"Name\": \"$ROLE_NAME\"},
            \"BlockDeviceMappings\": [{
                \"DeviceName\": \"/dev/xvda\",
                \"Ebs\": {\"VolumeSize\": $DISK_SIZE_GB, \"VolumeType\": \"gp3\", \"Iops\": 6000, \"Throughput\": 400}
            }],
            \"UserData\": \"$USER_DATA\",
            \"TagSpecifications\": [{
                \"ResourceType\": \"instance\",
                \"Tags\": [{\"Key\": \"Name\", \"Value\": \"dm-isaac-g1-ecs-gpu\"}, {\"Key\": \"Project\", \"Value\": \"dm-isaac-g1\"}]
            }]
        }" >/dev/null
else
    log "Launch template exists: $LT_EXISTS"
fi

# ── 9. Auto Scaling Group ────────────────────────────────────────────────────
# Get all subnets in the VPC
SUBNET_IDS=$(aws_cmd ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "Subnets[*].SubnetId" --output text | tr '\t' ',')

ASG_EXISTS=$(aws_cmd autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG_NAME" \
    --query "AutoScalingGroups[0].AutoScalingGroupName" --output text 2>/dev/null || echo "None")

if [[ "$ASG_EXISTS" == "None" || -z "$ASG_EXISTS" ]]; then
    log "Creating Auto Scaling Group: $ASG_NAME (min=0, max=$MAX_INSTANCES)"
    aws_cmd autoscaling create-auto-scaling-group \
        --auto-scaling-group-name "$ASG_NAME" \
        --launch-template "LaunchTemplateName=$LAUNCH_TEMPLATE_NAME,Version=\$Latest" \
        --min-size 0 \
        --max-size "$MAX_INSTANCES" \
        --desired-capacity 0 \
        --vpc-zone-identifier "$SUBNET_IDS" \
        --new-instances-protected-from-scale-in \
        --tags "Key=Name,Value=dm-isaac-g1-ecs-gpu,PropagateAtLaunch=true" \
               "Key=Project,Value=dm-isaac-g1,PropagateAtLaunch=true"
else
    log "ASG exists: $ASG_NAME"
fi

# ── 10. Capacity Provider ────────────────────────────────────────────────────
ASG_ARN=$(aws_cmd autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG_NAME" \
    --query "AutoScalingGroups[0].AutoScalingGroupARN" --output text)

CP_EXISTS=$(aws_cmd ecs describe-capacity-providers --capacity-providers "$CAPACITY_PROVIDER_NAME" \
    --query "capacityProviders[0].status" --output text 2>/dev/null || echo "NONE")

if [[ "$CP_EXISTS" != "ACTIVE" ]]; then
    log "Creating capacity provider: $CAPACITY_PROVIDER_NAME"
    aws_cmd ecs create-capacity-provider \
        --name "$CAPACITY_PROVIDER_NAME" \
        --auto-scaling-group-provider "autoScalingGroupArn=$ASG_ARN,managedScaling={status=ENABLED,targetCapacity=100,minimumScalingStepSize=1,maximumScalingStepSize=$MAX_INSTANCES},managedTerminationProtection=ENABLED" >/dev/null

    # Associate capacity provider with cluster
    aws_cmd ecs put-cluster-capacity-providers \
        --cluster "$CLUSTER_NAME" \
        --capacity-providers "$CAPACITY_PROVIDER_NAME" \
        --default-capacity-provider-strategy "capacityProvider=$CAPACITY_PROVIDER_NAME,weight=1,base=0" >/dev/null
else
    log "Capacity provider exists: $CAPACITY_PROVIDER_NAME"
fi

# ── 11. CloudWatch Log Group ─────────────────────────────────────────────────
aws_cmd logs create-log-group --log-group-name "/ecs/dm-isaac-g1" 2>/dev/null || true
log "CloudWatch log group: /ecs/dm-isaac-g1"

# ── Done ──────────────────────────────────────────────────────────────────────
log ""
log "=== ECS GPU Cluster Setup Complete ==="
log ""
log "Cluster:           $CLUSTER_NAME"
log "Capacity Provider: $CAPACITY_PROVIDER_NAME"
log "ASG:               $ASG_NAME (min=0, max=$MAX_INSTANCES)"
log "Instance Type:     $INSTANCE_TYPE"
log "S3 Bucket:         $S3_BUCKET"
log "ECR Image:         ${ECR_REGISTRY}/${ECR_REPO}:latest"
log "SSH Key:           $KEY_FILE"
log ""
log "The ASG starts at 0 instances (zero cost). When you submit a task,"
log "ECS will automatically scale up a GPU instance, run the task, and"
log "scale back to 0 when done."
log ""
log "Interactive access: Team members can get a shell with:"
log "  ./run.sh shell --task mimic --motion cr7_06_tiktok_uefa"
log "  ./run.sh exec --task-arn <arn>"
log ""
log "Next step: ./run.sh submit --task mimic --motion cr7_06_tiktok_uefa"

# Save config for run.sh
cat > "$SCRIPT_DIR/.cluster-config" << EOF
CLUSTER_NAME=$CLUSTER_NAME
CAPACITY_PROVIDER_NAME=$CAPACITY_PROVIDER_NAME
S3_BUCKET=$S3_BUCKET
TASK_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/${TASK_ROLE_NAME}
EXEC_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/${EXEC_ROLE_NAME}
ACCOUNT_ID=$ACCOUNT_ID
EOF
