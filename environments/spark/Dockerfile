# =============================================================================
# DM Spark Workstation Dockerfile (ARM64 Grace Hopper / DGX Spark)
#
# Full workstation environment for DGX Spark — mirrors the x86_64 workstation
# with ARM64-specific versions. Includes Isaac Sim, IsaacLab, GR00T, MuJoCo,
# WBC, VNC desktop, and all training/inference tools.
#
# TWO-IMAGE STRATEGY (mirrors workstation):
#   Stage "builder" → compilation stage with nvcc (not pushed)
#   Stage "groot"   → dm-spark-workstation:latest (pushed to ECR)
#
# Build:
#   docker build --target groot -t dm-spark-workstation:latest .
#
# ARM64 Key Differences (vs x86_64 workstation):
#   - CUDA 13.0 (GB10 GPU requires CUDA ≥ 13)
#   - PyTorch 2.9.0+cu130 (aarch64 wheels)
#   - Isaac Sim 5.1.0 (multi-arch, ARM64 supported since 5.1.0)
#   - IsaacLab v2.3.0 (DGX Spark support added in 2.3.0)
#   - Miniforge instead of Miniconda (ARM64-native conda)
#   - TurboVNC arm64 .deb
#   - Chromium instead of Google Chrome (no official aarch64 .deb for Chrome)
#   - torchcodec built from source (no aarch64 pip wheels)
#   - Vulkan via nvidia-container-toolkit (not libnvidia-gl-550)
#
# ARM64 Limitations (Isaac Sim 5.1.0 on DGX Spark):
#   - No livestreaming
#   - No cuRobo/cuMotion
#   - No OBJ file import
#   - No SkillGen
#   - No Hub Workstation Cache
#
# Size: ~40 GB (estimated)
# =============================================================================

# ==============================
# Stage 1: Builder
# Compilation stage with nvcc for building CUDA extensions.
# ==============================
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Build dependencies (GCC 12 + GLU + Vulkan + cmake)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 g++-12 cmake build-essential unzip git git-lfs \
    libglu1-mesa-dev libvulkan1 vulkan-tools wget ca-certificates \
    libssl-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (ARM64-native conda, better than Miniconda for aarch64)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh \
        -O miniforge.sh && \
    bash miniforge.sh -b -p $CONDA_DIR && \
    rm miniforge.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Create Python 3.11 environment (required by Isaac Sim 5.1.0)
RUN conda create -n unitree_sim_env python=3.11 -y && \
    conda clean -afy

SHELL ["conda", "run", "-n", "unitree_sim_env", "/bin/bash", "-c"]

# ABI-compatible libgcc/libstdc++ for compiled wheels
RUN conda install -y -c conda-forge "libgcc-ng>=12" "libstdcxx-ng>=12" && \
    conda clean -afy

# PyTorch 2.9.0 cu130 — ARM64 aarch64 wheels from PyTorch index
# GB10 GPU requires CUDA 13.0; cu130 is the correct index.
# Per NVIDIA IsaacLab 2.3.0 aarch64 installation docs.
RUN pip install --upgrade pip uv && \
    pip install torch==2.9.0 torchvision==0.24.0 \
        --index-url https://download.pytorch.org/whl/cu130

# Isaac Sim 5.1.0 via pip (multi-arch, ARM64 supported since 5.1.0)
# numpy pinned <2.0: Isaac Sim synthetic data pipeline requires numpy 1.x
RUN pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com && \
    pip install "numpy>=1.26.0,<2.0.0" --force-reinstall

WORKDIR /home/code

# IsaacLab v2.3.0 (DGX Spark support added in this release)
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    git checkout v2.3.0 && \
    ./isaaclab.sh --install

# CycloneDDS — required by unitree_sdk2_python
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /cyclonedds && \
    cd /cyclonedds && mkdir build install && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install -j$(nproc)

ENV CYCLONEDDS_HOME=/cyclonedds/install

# unitree_sdk2_python — robot SDK
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python && \
    cd unitree_sdk2_python && pip install -e .

# unitree_sim_isaaclab — USD assets, G1 task definitions, pink/pinocchio IK
RUN git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git /home/code/unitree_sim_isaaclab && \
    cd /home/code/unitree_sim_isaaclab && pip install -r requirements.txt

# pink + pinocchio — IK libraries for G1 manipulation scenes
# PyPI has aarch64 wheels for pin (pinocchio) 3.9.0+
# Re-pin numpy <2.0 after pinocchio deps pull numpy 2.x.
RUN pip install pin-pink meshcat && \
    pip install "numpy>=1.26.0,<2.0.0" --force-reinstall

# flatdict — required by isaaclab.sim.simulation_context
RUN pip install flatdict

# MuJoCo 3.2.6 — required for closed-loop eval with deformable objects (flexcomp)
# MuJoCo Menagerie — standardized MJCF robot models (G1, Go1, etc.)
RUN pip install mujoco==3.2.6 && \
    git clone https://github.com/google-deepmind/mujoco_menagerie.git /home/code/mujoco_menagerie


# ==============================
# Stage 2: groot — full workstation image pushed to ECR
# Isaac Sim + IsaacLab + GR00T + MuJoCo + VNC desktop
# ==============================
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04 AS groot

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV OMNI_KIT_ALLOW_ROOT=1
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=Y

# ARM64-specific environment
# LD_PRELOAD: libgomp prevents OpenMP conflict between system and PyTorch's bundled version.
#             libopenblas fixes liblapack/gotoblas symbol error with torchcodec FFmpeg6.
ENV LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libopenblas.so.0"
ENV MUJOCO_GL=egl
ENV XDG_RUNTIME_DIR=/tmp/xdg
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Runtime system dependencies + VNC desktop + rendering libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa git git-lfs curl wget \
    zenity unzip vim htop tmux rsync \
    # X11 + Vulkan + GL rendering
    xorg libxext6 libxrender1 libxtst6 libxi6 \
    libglib2.0-0 libvulkan1 vulkan-tools \
    mesa-vulkan-drivers \
    libgl1 libegl1-mesa-dev libgles2-mesa-dev \
    libopenblas-dev \
    libsm6 libxrender-dev \
    # XFCE4 desktop (for VNC sessions)
    ncurses-term dbus-x11 \
    xfce4 xfce4-terminal xfce4-taskmanager \
    adwaita-icon-theme \
    # FFmpeg dev headers (for torchcodec source build verification)
    ffmpeg libavdevice-dev libavfilter-dev libavformat-dev \
    libavcodec-dev libavutil-dev libswresample-dev libswscale-dev \
    pkg-config \
    # Chromium (ARM64 — Google Chrome has no official aarch64 .deb)
    chromium-browser \
    && rm -rf /var/lib/apt/lists/*

# Desktop shortcuts for VNC session
RUN mkdir -p /root/Desktop && \
    printf '[Desktop Entry]\nType=Application\nName=Chromium\nExec=chromium-browser --no-sandbox --disable-gpu\nIcon=chromium-browser\nTerminal=false\n' > /root/Desktop/chrome.desktop && \
    printf '[Desktop Entry]\nType=Application\nName=Terminal\nExec=xfce4-terminal\nIcon=utilities-terminal\nTerminal=false\n' > /root/Desktop/terminal.desktop && \
    printf '[Desktop Entry]\nType=Application\nName=JupyterLab\nExec=chromium-browser --no-sandbox http://localhost:8888\nIcon=text-html\nTerminal=false\n' > /root/Desktop/jupyterlab.desktop && \
    printf '[Desktop Entry]\nType=Application\nName=VS Code\nExec=chromium-browser --no-sandbox http://localhost:8080\nIcon=text-editor\nTerminal=false\n' > /root/Desktop/vscode.desktop && \
    chmod +x /root/Desktop/*.desktop

# TurboVNC 3.1.2 — remote visualization (ARM64 .deb)
RUN wget -q https://github.com/TurboVNC/turbovnc/releases/download/3.1.2/turbovnc_3.1.2_arm64.deb \
        -O /tmp/turbovnc.deb && \
    dpkg -i /tmp/turbovnc.deb || apt-get install -f -y && \
    rm /tmp/turbovnc.deb
ENV PATH="/opt/TurboVNC/bin:${PATH}"

# VNC: set password and use XFCE4 as the desktop session
ENV TVNC_WM=xfce
RUN mkdir -p ~/.vnc && \
    printf "datament\ndatament\nn\n" | /opt/TurboVNC/bin/vncpasswd

# code-server — VS Code in the browser (port 8080)
RUN curl -fsSL https://code-server.dev/install.sh | sh && \
    mkdir -p /root/.config/code-server && \
    printf 'bind-addr: 0.0.0.0:8080\nauth: password\npassword: datamentors\ncert: false\n' \
        > /root/.config/code-server/config.yaml
EXPOSE 8080

# JupyterLab config — port 8888
# JupyterLab itself is installed below after conda env is copied.
RUN mkdir -p /root/.jupyter && \
    printf '{\n  "ServerApp": {\n    "ip": "0.0.0.0",\n    "port": 8888,\n    "token": "datamentors",\n    "open_browser": false,\n    "allow_root": true\n  }\n}\n' \
        > /root/.jupyter/jupyter_server_config.json
EXPOSE 8888

# Copy all artifacts from builder
COPY --from=builder /home/code/IsaacLab            /home/code/IsaacLab
COPY --from=builder /home/code/unitree_sdk2_python  /home/code/unitree_sdk2_python
COPY --from=builder /home/code/unitree_sim_isaaclab /home/code/unitree_sim_isaaclab
COPY --from=builder /home/code/mujoco_menagerie     /home/code/mujoco_menagerie
COPY --from=builder /cyclonedds                     /cyclonedds
COPY --from=builder /opt/conda                      /opt/conda

ENV CYCLONEDDS_HOME=/cyclonedds/install
ENV LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib:${LD_LIBRARY_PATH}"

# PROJECT_ROOT — required by unitree_sim_isaaclab scene configs to locate USD assets.
ENV PROJECT_ROOT=/workspace/unitree_sim_isaaclab

# PYTHONPATH: dm-isaac-g1, Isaac-GR00T, IsaacLab, unitree_sim_isaaclab, unitree_rl_lab
ENV PYTHONPATH="/workspace/dm-isaac-g1/src:/workspace/Isaac-GR00T:/workspace/unitree_rl_lab:/home/code/IsaacLab/source/isaaclab:/home/code/IsaacLab/source/isaaclab_tasks:/home/code/IsaacLab/source/isaaclab_rl:/home/code/IsaacLab/source/isaaclab_assets:/home/code/IsaacLab/source/isaaclab_contrib:/home/code/unitree_sim_isaaclab:${PYTHONPATH}"

RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate unitree_sim_env' >> ~/.bashrc && \
    echo 'export OMNI_KIT_ALLOW_ROOT=1' >> ~/.bashrc && \
    echo 'export ACCEPT_EULA=Y' >> ~/.bashrc && \
    echo 'export OMNI_KIT_ACCEPT_EULA=Y' >> ~/.bashrc && \
    echo 'export CYCLONEDDS_HOME=/cyclonedds/install' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/cyclonedds/install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc && \
    echo 'export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libopenblas.so.0"' >> ~/.bashrc

SHELL ["conda", "run", "-n", "unitree_sim_env", "/bin/bash", "-c"]

# Install uv — the ONLY approved Python package manager for this image.
RUN pip install uv

# GR00T extra deps (ARM64-adapted requirements)
COPY requirements-groot-spark.txt /tmp/requirements-groot-spark.txt
RUN uv pip install --system -r /tmp/requirements-groot-spark.txt

# flash-attn: try prebuilt wheel first, fall back to source build if unavailable.
RUN uv pip install --system flash-attn --no-build-isolation 2>/dev/null \
    || (apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-13-0 \
        && rm -rf /var/lib/apt/lists/* \
        && uv pip install --system flash-attn --no-build-isolation)

# conda-forge FFmpeg: provides libavutil.so shared libs required by torchcodec.
RUN conda install -n unitree_sim_env -c conda-forge ffmpeg -y && \
    conda clean -afy

# torchcodec: build from source for aarch64 (no PyPI wheels available)
# v0.5.0 is compatible with PyTorch 2.9.x
RUN cd /tmp && \
    git clone --depth 1 --branch v0.5.0 https://github.com/pytorch/torchcodec.git && \
    cd torchcodec && \
    ENABLE_CUDA=0 I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
    pip wheel . --no-build-isolation --no-binary=torchcodec -w /tmp/tc_wheel && \
    pip install /tmp/tc_wheel/torchcodec*.whl && \
    rm -rf /tmp/torchcodec /tmp/tc_wheel

# ==============================
# Clone workspace repos required at runtime
# On DGX Spark, dm-isaac-g1 and Isaac-GR00T are bind-mounted from the host.
# On ECS, there are NO bind mounts — these must be in the image.
# ==============================
WORKDIR /workspace

ARG GITHUB_TOKEN=""

# dm-isaac-g1 — our main G1 training/inference project (PRIVATE repo)
RUN if [ -n "$GITHUB_TOKEN" ]; then \
        git clone https://x-access-token:${GITHUB_TOKEN}@github.com/datamentors/dm-isaac-g1.git /workspace/dm-isaac-g1; \
    else \
        git clone https://github.com/datamentors/dm-isaac-g1.git /workspace/dm-isaac-g1; \
    fi && \
    cd /workspace/dm-isaac-g1 && uv pip install --system -e . && \
    git remote set-url origin https://github.com/datamentors/dm-isaac-g1.git

# unitree_rl_lab — Unitree's official RL training framework
RUN git clone https://github.com/unitreerobotics/unitree_rl_lab.git /workspace/unitree_rl_lab && \
    cd /workspace/unitree_rl_lab/source/unitree_rl_lab && uv pip install --system -e .

# Patch unitree_rl_lab with dm-isaac-g1 custom tasks and reward functions
RUN python -m dm_isaac_g1.rl.install_tasks \
    --unitree-rl-lab /workspace/unitree_rl_lab \
    --rewards-patch /workspace/dm-isaac-g1/environments/workstation/patches/rewards_custom.py \
    --dm-tasks-dir /workspace/dm-isaac-g1/src/dm_isaac_g1/rl/tasks

# Isaac-GR00T — NVIDIA's GR00T framework (inference + fine-tuning)
RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git /workspace/Isaac-GR00T

# GR00T-WholeBodyControl (Dex1 branch) — WBC eval pipeline
RUN git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git GR00T-WholeBodyControl-dex1

# video2robot — PromptHMR + GMR pipeline
RUN git clone https://github.com/AIM-Intelligence/video2robot.git video2robot

# Patch: robot_viser.py AV1 codec fallback
COPY patches/robot_viser_av1_fallback.patch /tmp/
RUN test -f video2robot/visualization/robot_viser.py \
    && cd video2robot && git apply /tmp/robot_viser_av1_fallback.patch \
    && echo "[ok] Patched robot_viser.py with AV1 fallback" \
    || echo "[skip] robot_viser.py not found — patch not needed"

# unitree_model — Unitree USD/URDF/MJCF models from HuggingFace
RUN git lfs install && \
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model unitree_model

# unitree_mujoco — G1 MuJoCo scene XML (for sim2sim pipeline)
RUN git clone https://github.com/unitreerobotics/unitree_mujoco.git /workspace/unitree_mujoco

# HumanoidTraining — educational units (exclude media/ directory to save ~451MB)
RUN git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/datamentors/HumanoidTraining.git /workspace/HumanoidTraining && \
    cd /workspace/HumanoidTraining && \
    git sparse-checkout set --no-cone '/*' '!/media'

# JupyterLab — install into conda env so notebooks can use Isaac Sim + GR00T packages
RUN uv pip install --system jupyterlab

# Create cache and data directories
RUN mkdir -p \
    /workspace/.cache/huggingface \
    /workspace/.cache/torch \
    /workspace/data \
    /workspace/eval_videos

# ---------- Entrypoint ----------
# Generates random credentials for code-server + JupyterLab on each start.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# ---------- Ports ----------
# 5555: GROOT ZMQ server
# 5901: TurboVNC
# 6006: TensorBoard
# 8000: HTTP API
# 8080: code-server (VS Code)
# 8888: JupyterLab
# 47995-48012: Isaac Sim PhysX
# 49000-49007: Isaac Sim comms
EXPOSE 5555
EXPOSE 5901
EXPOSE 6006
EXPOSE 8000
EXPOSE 8080
EXPOSE 8888
EXPOSE 47995-48012
EXPOSE 49000-49007

# ---------- Health check ----------
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD conda run -n unitree_sim_env python -c \
        "import torch; assert torch.cuda.is_available(); import mujoco; import gr00t" || exit 1

WORKDIR /workspace
CMD ["conda", "run", "--no-capture-output", "-n", "unitree_sim_env", "/bin/bash"]
