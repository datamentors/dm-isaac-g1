# Spark Inference Environment Dockerfile
# Based on working dm-groot-inference/Dockerfile.arm64
# For DGX Spark (ARM64 Grace Hopper)

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
FROM ${BASE_IMAGE}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Download GR00T repository (using zip since git clone has issues in container)
RUN curl -L -o /tmp/groot.zip 'https://github.com/NVIDIA/Isaac-GR00T/archive/refs/heads/main.zip' && \
    unzip -o /tmp/groot.zip -d /workspace && \
    mv /workspace/Isaac-GR00T-main /workspace/gr00t && \
    rm /tmp/groot.zip

WORKDIR /workspace/gr00t

# Fix version constraints in pyproject.toml for NVIDIA container compatibility
RUN sed -i 's/requires-python = "==3.10.*"/requires-python = ">=3.10"/g' pyproject.toml && \
    sed -i 's/torch==2.7.1/torch>=2.5.0/g' pyproject.toml && \
    sed -i 's/torchvision==0.22.1/torchvision>=0.20.0/g' pyproject.toml && \
    sed -i 's/transformers==4.51.3/transformers>=4.40.0,<5.0.0/g' pyproject.toml && \
    sed -i 's/"torchcodec==0.4.0",/# "torchcodec==0.4.0",  # Not available on ARM64/g' pyproject.toml && \
    sed -i 's/"flash-attn==2.7.4.post1",/# "flash-attn==2.7.4.post1",  # Build issues on ARM64/g' pyproject.toml

# Fix NVIDIA container's constraint file
RUN sed -i 's/dm-tree==0.1.9/dm-tree==0.1.8/g' /etc/pip/constraint.txt 2>/dev/null || true

# Install GR00T without deps first, then install compatible packages
RUN pip install --no-deps -e . && \
    pip install --ignore-requires-python \
    albumentations av diffusers dm-tree lmdb msgpack msgpack-numpy \
    peft termcolor tyro datasets gymnasium omegaconf pyzmq deepspeed wandb \
    transformers==4.51.3

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && \
    python -c "import gr00t; print('GR00T imported successfully')"

# Go back to main workspace
WORKDIR /workspace

# Create cache directories
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch

# Set PYTHONPATH for dm-isaac-g1 (will be mounted)
ENV PYTHONPATH="/workspace/dm-isaac-g1/src:${PYTHONPATH}"

# Expose ports
# 5555: GROOT ZMQ server
# 8000: HTTP API
EXPOSE 5555
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

WORKDIR /workspace
CMD ["/bin/bash"]
