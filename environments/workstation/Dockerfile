# Workstation Environment Dockerfile
# Unified environment for GROOT inference and Isaac Sim/Lab simulation
# Base: NVIDIA Isaac Sim 5.0.0 (stable, NumPy 1.x compatible)
#
# IMPORTANT: NumPy < 2.0 Requirement
# ==================================
# Isaac Sim 5.1.0 uses NumPy 2.x which has bugs in the synthetic data pipeline
# causing "Unable to write from unknown dtype" errors with cameras.
#
# We use Isaac Sim 5.0.0 which:
# - Uses NumPy 1.x (stable)
# - Has working camera/TiledCamera operations
# - Is recommended by Unitree for G1 simulation
#
# Reference: https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/doc/isaacsim5.0_install.md
# Reference: docs/NUMPY_COMPATIBILITY.md

ARG ISAAC_SIM_VERSION=5.0.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}

# Run as root for package installation
USER root

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV UV_LINK_MODE=copy

# Isaac Sim environment variables (required for proper initialization)
ENV ISAACSIM_PATH="/isaac-sim"
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
ENV CARB_APP_PATH="${ISAACSIM_PATH}/kit"
ENV EXP_PATH="${ISAACSIM_PATH}/apps"
ENV ISAAC_PATH="${ISAACSIM_PATH}"

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

# CRITICAL: Pin NumPy to < 2.0 for stability
# This ensures compatibility with:
# - Isaac Sim synthetic data pipeline
# - GR00T fine-tuning requirements (numpy>=1.23.5,<2.0.0)
# - pink/pinocchio IK libraries
RUN /isaac-sim/python.sh -m pip install "numpy>=1.26.0,<2.0.0"

# Install pink and pinocchio for IK
# IMPORTANT: Use pin-pink 3.1.0 (same as IsaacLab env_isaaclab)
# pin-pink 4.0.0+ requires NumPy 2.x which breaks Isaac Sim
RUN /isaac-sim/python.sh -m pip install "pin-pink==3.1.0" "pin==2.7.0"

# Install additional packages
RUN /isaac-sim/python.sh -m pip install pillow opencv-python-headless

# Install nlopt (required by IsaacLab dex_retargeting)
# Use nlopt < 2.9 which supports NumPy 1.x
RUN /isaac-sim/python.sh -m pip install "nlopt<2.9"

# FINAL: Re-pin NumPy to ensure it wasn't upgraded by dependencies
RUN /isaac-sim/python.sh -m pip install "numpy>=1.26.0,<2.0.0" --force-reinstall

# Set PYTHONPATH for dm-isaac-g1 (will be mounted)
# Include IsaacLab source paths
ENV PYTHONPATH="/workspace/dm-isaac-g1/src:/workspace/Isaac-GR00T:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_assets:/workspace/IsaacLab/source/isaaclab_contrib:/workspace/unitree_sim_isaaclab:${PYTHONPATH}"

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
