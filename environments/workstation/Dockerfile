# Workstation Environment Dockerfile
# Unified environment for GROOT training and Isaac Sim/Lab simulation
# Base: NVIDIA Isaac Sim with UV for dependency management

ARG ISAAC_SIM_VERSION=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

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

# Install unitree_sdk2py from source
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python
WORKDIR /opt/unitree_sdk2_python
RUN /isaac-sim/python.sh -m pip install .

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

# CRITICAL: Include IsaacLab's .venv for pink/pinocchio IK libraries
# These are pre-installed in IsaacLab's virtual environment with proper cmeel configuration
ENV ISAACLAB_VENV_SITE="/workspace/IsaacLab/.venv/lib/python3.10/site-packages"
ENV ISAACLAB_CMEEL_SITE="${ISAACLAB_VENV_SITE}/cmeel.prefix/lib/python3.10/site-packages"
ENV PYTHONPATH="${ISAACLAB_VENV_SITE}:${ISAACLAB_CMEEL_SITE}:${PYTHONPATH}"

# Add cmeel.prefix libraries to LD_LIBRARY_PATH (required for hpp-fcl, pinocchio)
ENV LD_LIBRARY_PATH="${ISAACLAB_VENV_SITE}/cmeel.prefix/lib:${LD_LIBRARY_PATH}"

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
