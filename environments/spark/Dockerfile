# Spark Inference Environment Dockerfile
# GROOT inference server for ARM64 (Jetson Orin)
# Base: NVIDIA L4T with PyTorch

ARG L4T_VERSION=r36.3.0
FROM nvcr.io/nvidia/l4t-pytorch:${L4T_VERSION}-pth2.1-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create workspace
WORKDIR /workspace

# Copy environment config
COPY pyproject.toml /workspace/env/

# Install dependencies with UV
WORKDIR /workspace/env
RUN uv pip install --system $(grep -E "^\s+\"" pyproject.toml | grep -v "^#" | tr -d '", ' | head -20) || \
    pip install numpy pyyaml python-dotenv tqdm transformers safetensors pillow huggingface-hub pyzmq msgpack httpx einops timm diffusers

# Install ARM64-specific packages
# Note: onnxruntime-gpu for Jetson requires NVIDIA's wheel
RUN pip install --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 \
    onnxruntime-gpu || \
    pip install onnxruntime

# Set PYTHONPATH for dm-isaac-g1 (will be mounted)
ENV PYTHONPATH="/workspace/dm-isaac-g1/src:${PYTHONPATH}"

# Expose GROOT inference port
EXPOSE 5555

WORKDIR /workspace
CMD ["/bin/bash"]
