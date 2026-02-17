# Workstation Environment Dockerfile
# Unified environment for GROOT training and Isaac Sim/Lab simulation
# Base: NVIDIA Isaac Sim with UV for dependency management

ARG ISAAC_SIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

# Run as root for package installation
USER root

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

# Fix apt permissions
RUN mkdir -p /var/lib/apt/lists/partial && chmod 755 /var/lib/apt/lists

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    htop \
    tmux \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Build CycloneDDS from source (required for unitree_sdk2py)
WORKDIR /opt
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x && \
    cd cyclonedds && \
    mkdir -p build install && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds/install && \
    cmake --build . --target install -j$(nproc)
ENV CYCLONEDDS_HOME=/opt/cyclonedds/install
ENV LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib:${LD_LIBRARY_PATH}"

# Install CycloneDDS Python bindings
RUN /isaac-sim/python.sh -m pip install cyclonedds

# Install unitree_sdk2py from source
# NOTE: pip install doesn't always copy all subdirectories (b2, g1, h1, comm)
# so we manually ensure they exist after installation
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python
WORKDIR /opt/unitree_sdk2_python
RUN /isaac-sim/python.sh -m pip install . && \
    SITE_PACKAGES=$(/isaac-sim/python.sh -c "import site; print(site.getsitepackages()[0])") && \
    for dir in b2 g1 h1 comm; do \
        if [ -d "unitree_sdk2py/$dir" ] && [ ! -d "$SITE_PACKAGES/unitree_sdk2py/$dir" ]; then \
            cp -r "unitree_sdk2py/$dir" "$SITE_PACKAGES/unitree_sdk2py/"; \
        fi; \
    done

# Create workspace directory
WORKDIR /workspace

# Copy environment config
COPY pyproject.toml /workspace/env/

# Install dependencies with UV
# Note: We install to the Isaac Sim Python environment
WORKDIR /workspace/env
RUN uv pip install --python /isaac-sim/kit/python/bin/python3 -r pyproject.toml

# Set PYTHONPATH for dm-isaac-g1 (will be mounted)
# Include IsaacLab source paths
ENV PYTHONPATH="/workspace/dm-isaac-g1/src:/workspace/Isaac-GR00T:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_assets:${PYTHONPATH}"

# PINK/PINOCCHIO PATHS - Set as variables, NOT added to PATH
# Isaac Sim 5.1.0 uses Python 3.11, so we use env_isaaclab (not .venv which is 3.10)
# IMPORTANT: We do NOT add these to PYTHONPATH at container startup because:
#   1. env_isaaclab has torch 2.7.0+cu126 which conflicts with Isaac Sim's torch
#   2. The CUDA libraries in cmeel.prefix conflict with Isaac Sim's CUDA libs
# Instead, these paths should be added AFTER Isaac Sim imports torch, before importing pink
ENV ISAACLAB_VENV_SITE="/workspace/IsaacLab/env_isaaclab/lib/python3.11/site-packages"
ENV ISAACLAB_CMEEL_SITE="${ISAACLAB_VENV_SITE}/cmeel.prefix/lib/python3.11/site-packages"
ENV ISAACLAB_CMEEL_LIB="${ISAACLAB_VENV_SITE}/cmeel.prefix/lib"

# NOTE: These paths are NOT added to PYTHONPATH or LD_LIBRARY_PATH here
# The inference script must add them at the right time (after torch imports)

# Expose ports
# Isaac Sim streaming
EXPOSE 47995-48012
# Isaac Sim services
EXPOSE 49000-49007
# TensorBoard
EXPOSE 6006
# GROOT ZMQ
EXPOSE 5555

WORKDIR /workspace
CMD ["/bin/bash"]
