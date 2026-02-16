# DM-ISAAC-G1 Architecture & Flow Diagrams

This document provides visual diagrams of the system architecture, inference flows, and training pipelines.

## System Overview

```mermaid
graph TB
    subgraph "Local Machine"
        DEV[Developer Machine]
    end

    subgraph "Blackwell Workstation<br/>192.168.1.205"
        subgraph "Docker Containers"
            ISAAC[Isaac Sim 5.1.0<br/>Isaac Lab 2.3.2]
            MUJOCO[MuJoCo Environment]
        end
        VNC[VNC Server :5901]
        GPU1[RTX PRO 6000<br/>98GB VRAM]
    end

    subgraph "DGX Spark<br/>192.168.1.237"
        subgraph "GROOT Server"
            GROOT[GR00T N1.6<br/>Port 5555]
            MODEL[(Model Weights<br/>nvidia/GR00T-N1.6-3B)]
        end
        GPU2[GB10<br/>48GB VRAM]
    end

    DEV -->|SSH| ISAAC
    DEV -->|VNC| VNC
    ISAAC -->|ZMQ| GROOT
    MUJOCO -->|ZMQ| GROOT
    GROOT --> MODEL
    ISAAC --> GPU1
    MUJOCO --> GPU1
    GROOT --> GPU2
```

## Physics Engine Compatibility

```mermaid
graph LR
    subgraph "Training Environment"
        IG[Isaac Gym<br/>PhysX]
        MJ[MuJoCo]
    end

    subgraph "Policies"
        LOCO[Locomotion<br/>Policies]
        MANIP[Manipulation<br/>Policies]
    end

    subgraph "Inference Environment"
        ISIM[Isaac Sim<br/>PhysX]
        MJSIM[MuJoCo<br/>Simulator]
    end

    IG -->|Trained| LOCO
    MJ -->|Trained| MANIP

    LOCO -->|✅ Works| ISIM
    LOCO -->|❌ Mismatch| MJSIM
    MANIP -->|❌ Mismatch| ISIM
    MANIP -->|✅ Works| MJSIM

    style LOCO fill:#90EE90
    style MANIP fill:#90EE90
    style ISIM fill:#87CEEB
    style MJSIM fill:#87CEEB
```

## Track 1: Locomotion Inference (Isaac Sim)

```mermaid
sequenceDiagram
    participant User
    participant Script as policy_inference_locomotion.py
    participant Isaac as Isaac Sim
    participant Policy as Isaac Gym Policy<br/>(Local .pt file)
    participant Robot as G1 Robot

    User->>Script: python policy_inference_locomotion.py
    Script->>Isaac: Launch AppLauncher
    Isaac->>Robot: Spawn G1 robot

    Script->>Policy: Load checkpoint (.pt)
    Policy-->>Script: Model weights

    loop Inference Loop
        Robot->>Script: Observation<br/>(ang_vel, gravity, joints)
        Script->>Policy: Forward pass
        Policy-->>Script: Actions (29 DOF)
        Script->>Robot: Apply joint targets
        Robot->>Isaac: Physics step
    end

    Note over Script,Robot: No external server needed<br/>Policy runs locally
```

## Track 2: Manipulation Inference (MuJoCo)

```mermaid
sequenceDiagram
    participant User
    participant Script as policy_inference_mujoco.py
    participant MuJoCo as MuJoCo Sim
    participant Client as ZMQ Client
    participant Server as GROOT Server<br/>192.168.1.237:5555
    participant Model as GR00T N1.6

    User->>Script: python policy_inference_mujoco.py
    Script->>MuJoCo: Load G1 MJCF model
    Script->>Client: Connect to server

    loop Inference Loop
        MuJoCo->>Script: Get observation
        Script->>Script: Build flat observation<br/>(video.ego_view, state.*)
        Script->>Client: Send observation
        Client->>Server: ZMQ request
        Server->>Model: Forward pass
        Model-->>Server: Action trajectory<br/>(30 timesteps)
        Server-->>Client: ZMQ response
        Client-->>Script: Action dict
        Script->>MuJoCo: Apply action[0]<br/>(first timestep)
        MuJoCo->>MuJoCo: Physics step
    end
```

## Track 3: Fine-Tuning Pipeline

```mermaid
flowchart TB
    subgraph "Dataset Preparation"
        HF[(HuggingFace<br/>nvidia/GR00T-X-Embodiment)]
        DOWNLOAD[Download Dataset<br/>gr1_arms_only]
        DATASET[(Local Dataset<br/>~9k trajectories)]
    end

    subgraph "Training Configuration"
        CONFIG[unitree_g1_config.json]
        BASE[Base Model<br/>nvidia/GR00T-N1.6-3B]
    end

    subgraph "Training Loop<br/>(8-10 hours)"
        TRAIN[Fine-tune Script]
        CKPT1[Checkpoint 1000]
        CKPT2[Checkpoint 2000]
        CKPT3[Checkpoint 3000]
        CKPT4[Checkpoint 4000]
        FINAL[Final Checkpoint<br/>5000 steps]
    end

    subgraph "Deployment"
        DEPLOY[Deploy to GROOT Server]
        INFER[Run Inference]
    end

    HF --> DOWNLOAD
    DOWNLOAD --> DATASET
    DATASET --> TRAIN
    CONFIG --> TRAIN
    BASE --> TRAIN

    TRAIN --> CKPT1 --> CKPT2 --> CKPT3 --> CKPT4 --> FINAL
    FINAL --> DEPLOY --> INFER
```

## Observation Format (SimPolicyWrapper)

```mermaid
graph TB
    subgraph "Robot State"
        CAM[Head Camera<br/>256x256 RGB]
        LA[Left Arm<br/>7 joints]
        RA[Right Arm<br/>7 joints]
        LH[Left Hand<br/>6 joints]
        RH[Right Hand<br/>6 joints]
        W[Waist<br/>3 joints]
    end

    subgraph "Observation Dict"
        OBS["{<br/>  video.ego_view: [256,256,3]<br/>  state.left_arm: [7]<br/>  state.right_arm: [7]<br/>  state.left_hand: [6]<br/>  state.right_hand: [6]<br/>  state.waist: [3]<br/>}"]
    end

    subgraph "GROOT Server"
        POLICY[GR00T N1.6<br/>Policy Network]
    end

    subgraph "Action Output"
        ACT["action: [30, 29]<br/>(30 timesteps × 29 DOF)"]
    end

    CAM --> OBS
    LA --> OBS
    RA --> OBS
    LH --> OBS
    RH --> OBS
    W --> OBS

    OBS --> POLICY --> ACT
```

## Action Application Flow

```mermaid
flowchart TB
    subgraph "GROOT Output"
        TRAJ[Action Trajectory<br/>30 timesteps × 29 DOF]
    end

    subgraph "Action Processing"
        T0[timestep 0]
        T1[timestep 1]
        TN[timestep N]
        DOTS[...]
    end

    subgraph "Action Application"
        SCALE[Apply action_scale<br/>default: 0.1]
        DELTA[Compute Delta<br/>target = start + delta × scale]
        CLIP[Clip to Limits<br/>±3.14 rad]
    end

    subgraph "Robot Control"
        PD[PD Controller<br/>stiffness=3000<br/>damping=10]
        JOINTS[Joint Targets]
    end

    TRAJ --> T0 & T1 & TN
    T0 --> DOTS --> T1 --> TN

    T0 --> SCALE --> DELTA --> CLIP --> PD --> JOINTS

    Note1[Execute num_action_steps<br/>before new inference]
    TN --> Note1
```

## Infrastructure Setup

```mermaid
graph TB
    subgraph "Workstation 192.168.1.205"
        subgraph "Host"
            UBUNTU[Ubuntu + NVIDIA Driver]
            DOCKER[Docker Runtime]
        end

        subgraph "Containers"
            ISAAC_C[isaac-lab<br/>Isaac Sim 5.1.0]
            MUJOCO_C[mujoco-env<br/>MuJoCo + LeRobot]
        end

        subgraph "Services"
            VNC_S[TigerVNC :5901]
            SSH_S[SSH :22]
        end
    end

    subgraph "Spark 192.168.1.237"
        subgraph "Host ARM64"
            UBUNTU2[Ubuntu + GB10 Driver]
            DOCKER2[Docker Runtime]
        end

        subgraph "Container"
            GROOT_C[groot-server<br/>nvcr.io/nvidia/pytorch:25.04-py3]
        end

        subgraph "Services"
            ZMQ[ZMQ :5555]
            SSH_S2[SSH :22]
        end
    end

    DOCKER --> ISAAC_C & MUJOCO_C
    DOCKER2 --> GROOT_C
    ISAAC_C --> ZMQ
    MUJOCO_C --> ZMQ
```

## Overnight Fine-Tuning Workflow

```mermaid
gantt
    title Overnight Fine-Tuning Timeline
    dateFormat HH:mm
    axisFormat %H:%M

    section Preparation
    SSH & Setup           :prep, 22:00, 15m
    Download Dataset      :download, after prep, 45m
    Configure Training    :config, after download, 10m

    section Training
    Initialize Model      :init, 23:10, 10m
    Training Steps 0-1000 :t1, after init, 100m
    Checkpoint 1000       :milestone, after t1, 0m
    Training Steps 1001-2000 :t2, after t1, 100m
    Checkpoint 2000       :milestone, after t2, 0m
    Training Steps 2001-3000 :t3, after t2, 100m
    Checkpoint 3000       :milestone, after t3, 0m
    Training Steps 3001-4000 :t4, after t3, 100m
    Checkpoint 4000       :milestone, after t4, 0m
    Training Steps 4001-5000 :t5, after t4, 100m
    Final Checkpoint      :crit, milestone, after t5, 0m

    section Morning
    Verify Results        :verify, 07:30, 15m
    Deploy to Server      :deploy, after verify, 15m
    Test Inference        :test, after deploy, 30m
```

## File Structure

```mermaid
graph TB
    subgraph "dm-isaac-g1/"
        SCRIPTS[scripts/]
        DOCS[docs/]
        CONFIGS[configs/]
        PHASES[phases/]
    end

    subgraph "scripts/"
        S1[policy_inference_groot_g1.py<br/>Isaac Sim + GROOT]
        S2[policy_inference_locomotion.py<br/>Isaac Sim + Local Policy]
        S3[policy_inference_mujoco.py<br/>MuJoCo + GROOT]
        S4[finetune_groot_overnight.sh<br/>Training Script]
    end

    subgraph "docs/"
        D1[ARCHITECTURE.md<br/>This file]
        D2[PHYSICS_ENGINE_LEARNINGS.md<br/>Why sim-to-sim fails]
        D3[IMPLEMENTATION_PLAN.md<br/>Setup instructions]
        D4[FINETUNING_PLAN.md<br/>Training guide]
    end

    SCRIPTS --> S1 & S2 & S3 & S4
    DOCS --> D1 & D2 & D3 & D4
```

## Quick Reference: Which Script to Use

```mermaid
flowchart TD
    START[What do you want to do?]

    START --> Q1{Task Type?}

    Q1 -->|Walking/Running| LOCO[Locomotion]
    Q1 -->|Pick & Place| MANIP[Manipulation]
    Q1 -->|Training| TRAIN[Fine-tuning]

    LOCO --> LOCO_SIM{Simulator?}
    LOCO_SIM -->|Isaac Sim| L1[policy_inference_locomotion.py<br/>✅ Works]
    LOCO_SIM -->|MuJoCo| L2[Not recommended<br/>❌ Physics mismatch]

    MANIP --> MANIP_SIM{Simulator?}
    MANIP_SIM -->|MuJoCo| M1[policy_inference_mujoco.py<br/>✅ Works]
    MANIP_SIM -->|Isaac Sim| M2[policy_inference_groot_g1.py<br/>❌ Joint explosion]

    TRAIN --> T1[finetune_groot_overnight.sh<br/>8-10 hours on RTX 6000]

    style L1 fill:#90EE90
    style M1 fill:#90EE90
    style T1 fill:#90EE90
    style L2 fill:#FFB6C1
    style M2 fill:#FFB6C1
```

## Connection Details

| Service | Host | Port | Protocol | Purpose |
|---------|------|------|----------|---------|
| Workstation SSH | 192.168.1.205 | 22 | SSH | Remote access |
| VNC Server | 192.168.1.205 | 5901 | VNC | GUI visualization |
| GROOT Server | 192.168.1.237 | 5555 | ZMQ | Policy inference |
| Spark SSH | 192.168.1.237 | 22 | SSH | Server management |

## Environment Variables

See `.env` file for credentials and sensitive configuration. Key variables:

| Variable | Description |
|----------|-------------|
| `WORKSTATION_HOST` | Blackwell workstation IP (192.168.1.205) |
| `GROOT_SERVER_HOST` | DGX Spark server IP (192.168.1.237) |
| `GROOT_SERVER_PORT` | GROOT ZMQ port (5555) |
| `GROOT_MODEL_PATH` | Model path or HuggingFace ID |
| `HF_TOKEN` | HuggingFace access token |

## VNC Access for Visualization

The Blackwell workstation has VNC pre-installed for visualizing Isaac Sim and other GUI applications.

### VNC Connection Details

| Setting | Value |
|---------|-------|
| Host | 192.168.1.205 |
| Port | 5901 |
| Display | :1 |
| Password | (see .env) |

### Connecting with VNC Viewer

1. **macOS**: Use "Screen Sharing" app or any VNC client
   ```
   vnc://192.168.1.205:5901
   ```

2. **Windows/Linux**: Use TigerVNC, RealVNC, or Remmina
   ```
   192.168.1.205:5901
   ```

### Running Isaac Sim with Visualization

```bash
# SSH to workstation
ssh datamentors@192.168.1.205

# Set display for VNC
export DISPLAY=:1

# Run Isaac Sim with rendering (not headless)
cd /path/to/IsaacLab
./isaaclab.sh -p scripts/play.py \
    --task Isaac-Stack-RgyBlock-G129-Inspire-Joint \
    --num_envs 1

# Or using unitree_sim_isaaclab directly
cd /path/to/unitree_sim_isaaclab
python sim_main.py --device cuda --enable_cameras \
    --task Isaac-Stack-RgyBlock-G129-Inspire-Joint \
    --robot_type g129
```

### VNC Troubleshooting

```bash
# Check if VNC server is running
ps aux | grep -E 'vnc|Xvnc' | grep -v grep

# Check display
echo $DISPLAY
ls -la /tmp/.X11-unix/

# Start VNC if not running (usually already running)
vncserver :1 -geometry 1920x1080 -depth 24

# Kill and restart VNC
vncserver -kill :1
vncserver :1
```
