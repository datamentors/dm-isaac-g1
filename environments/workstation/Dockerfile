# Workstation Environment Dockerfile
# Unified environment for GROOT inference and Isaac Sim/Lab simulation
#
# APPROACH: Following IsaacLab Docker deployment guide
# https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html
#
# Base: NVIDIA Isaac Sim 5.0.0 → Install IsaacLab via ./isaaclab.sh --install
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

# Install system dependencies (includes IsaacLab requirements)
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
    unzip \
    xorg \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libglu1-mesa \
    libglib2.0-0 \
    ncurses-term \
    dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

# Install TurboVNC for remote visualization
RUN wget -q https://github.com/TurboVNC/turbovnc/releases/download/3.1.2/turbovnc_3.1.2_amd64.deb -O /tmp/turbovnc.deb && \
    dpkg -i /tmp/turbovnc.deb || apt-get install -f -y && \
    rm /tmp/turbovnc.deb
ENV PATH="/opt/TurboVNC/bin:${PATH}"

# Install AWS CLI v2 for ECR image push
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
    unzip -q /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ==========================================
# CycloneDDS (required for unitree_sdk2py)
# ==========================================
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

# ==========================================
# unitree_sdk2py
# ==========================================
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python
WORKDIR /opt/unitree_sdk2_python
RUN /isaac-sim/python.sh -m pip install . && \
    SITE_PACKAGES=$(/isaac-sim/python.sh -c "import site; print(site.getsitepackages()[0])") && \
    for dir in b2 g1 h1 comm; do \
        if [ -d "unitree_sdk2py/$dir" ] && [ ! -d "$SITE_PACKAGES/unitree_sdk2py/$dir" ]; then \
            cp -r "unitree_sdk2py/$dir" "$SITE_PACKAGES/unitree_sdk2py/"; \
        fi; \
    done

# ==========================================
# IsaacLab Installation (following official Docker approach)
# https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html
# ==========================================
WORKDIR /workspace

# Clone IsaacLab v2.2.0 (compatible with Isaac Sim 5.0.0 per Unitree docs)
RUN git clone https://github.com/isaac-sim/IsaacLab.git /workspace/IsaacLab && \
    cd /workspace/IsaacLab && \
    git checkout v2.2.0

# Create symlink to Isaac Sim (required by isaaclab.sh)
RUN ln -sf /isaac-sim /workspace/IsaacLab/_isaac_sim

# Install IsaacLab dependencies via official script
# This properly installs: warp, pink, pinocchio, and all other dependencies
WORKDIR /workspace/IsaacLab
RUN chmod +x isaaclab.sh && \
    ./isaaclab.sh --install && \
    # Ensure all isaaclab dependencies are installed (isaaclab.sh may miss some)
    /isaac-sim/python.sh -m pip install -e /workspace/IsaacLab/source/isaaclab

# ==========================================
# Additional dependencies for dm-isaac-g1
# ==========================================
WORKDIR /workspace

# Copy environment config
COPY pyproject.toml /workspace/env/

# Install dm-isaac-g1 dependencies with UV
WORKDIR /workspace/env
RUN uv pip install --python /isaac-sim/kit/python/bin/python3 -r pyproject.toml


# ==========================================
# CRITICAL: Pin NumPy to < 2.0 for stability
# ==========================================
# This ensures compatibility with:
# - Isaac Sim synthetic data pipeline
# - GR00T fine-tuning requirements (numpy>=1.23.5,<2.0.0)
# - pink/pinocchio IK libraries
RUN /isaac-sim/python.sh -m pip install "numpy>=1.26.0,<2.0.0" --force-reinstall

# ==========================================
# Environment Configuration
# ==========================================
# Set PYTHONPATH for all modules
# Note: warp is bundled in Isaac Sim's extscache, not installed via pip
ENV PYTHONPATH="/isaac-sim/extscache/omni.warp.core-1.7.1+lx64:/workspace/dm-isaac-g1/src:/workspace/Isaac-GR00T:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_tasks:/workspace/IsaacLab/source/isaaclab_rl:/workspace/IsaacLab/source/isaaclab_assets:/workspace/IsaacLab/source/isaaclab_contrib:/workspace/unitree_sim_isaaclab:${PYTHONPATH}"

# Expose ports
# Isaac Sim streaming
EXPOSE 47995-48012
# Isaac Sim services
EXPOSE 49000-49007
# TensorBoard
EXPOSE 6006
# GROOT ZMQ
EXPOSE 5555
# VNC
EXPOSE 5901

WORKDIR /workspace
CMD ["/bin/bash"]
