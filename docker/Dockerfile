# DM-ISAAC-G1 Docker Image
# For Isaac Sim/Lab development on Blackwell workstation

ARG ISAAC_SIM_VERSION=4.2.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Copy project files
COPY pyproject.toml .
COPY configs/ ./configs/
COPY src/ ./src/
COPY phases/ ./phases/

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    torch \
    torchvision \
    gymnasium \
    httpx \
    pyyaml \
    tensorboard \
    wandb \
    scipy \
    tqdm \
    pillow \
    opencv-python-headless \
    python-dotenv

# Set up Isaac Lab (if building with Lab support)
ARG INSTALL_ISAAC_LAB=false
RUN if [ "$INSTALL_ISAAC_LAB" = "true" ]; then \
    git clone https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab && \
    cd /opt/IsaacLab && \
    ./isaaclab.sh --install; \
    fi

# Expose ports
# Isaac Sim streaming
EXPOSE 47995-48012
EXPOSE 49000-49007
# Tensorboard
EXPOSE 6006
# GROOT client
EXPOSE 5555

# Default command
CMD ["/bin/bash"]
