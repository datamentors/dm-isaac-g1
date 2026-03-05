# Ardia Robotics Platform - Strategic Report

**Date:** 2026-02-27
**Author:** Datamentors Engineering
**Status:** Strategic Planning Document
**Scope:** Repository architecture, version management, training pipelines, deployment strategy, and technology decisions for the Ardia humanoid robotics platform

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State Assessment](#2-current-state-assessment)
3. [The Core Problem](#3-the-core-problem)
4. [Isaac-GR00T Eval vs WBC: Architecture Deep Dive](#4-isaac-groot-eval-vs-wbc-architecture-deep-dive)
5. [VLA Ecosystem Landscape (Early 2026)](#5-vla-ecosystem-landscape-early-2026)
6. [Robotics Agent Skills: AI-Assisted Development Framework](#6-robotics-agent-skills-ai-assisted-development-framework)
7. [Closed-Loop Reasoning: CV/VLM Policies for Action-Time Evaluation](#7-closed-loop-reasoning-cvvlm-policies-for-action-time-evaluation)
8. [RL Post-Training for VLAs: The PI0 Path to Production Quality](#8-rl-post-training-for-vlas-the-pi0-path-to-production-quality)
9. [Autonomous Exploration & Semantic Navigation in Unknown Environments](#9-autonomous-exploration--semantic-navigation-in-unknown-environments)
10. [Recommended Repository Architecture](#10-recommended-repository-architecture)
11. [Strategic Decisions (Q1-Q6)](#11-strategic-decisions-q1-q6)
12. [Technology Stack Decisions](#12-technology-stack-decisions)
13. [Phased Roadmap](#13-phased-roadmap)
14. [Risk Assessment](#14-risk-assessment)
15. [Appendices](#15-appendices) (A-K: Dex1, DOF, Wrist, Files, GR00T, Datasets, Skills, References, **Sensor Stack**, **RL Recipe**, **VLA Ops**)

---

## 1. Executive Summary

Over the past several weeks, we have been building a V0 VLA pipeline for the Unitree G1 robot with Inspire (Dex1) hands. The work — documented extensively in `dm-isaac-g1/docs/` — has produced significant technical knowledge but has been hampered by fragile multi-repo dependencies, environment incompatibilities across targets (Mac, Workstation, Spark, Jetson), and difficulty isolating our customizations from rapidly evolving upstream NVIDIA/Unitree repositories.

This report proposes a clean break: a new **mono-repo architecture** (`ardia-robotics/`) that pins upstream repos as git submodules, standardizes on **LeRobot** for VLA training, **SONIC** for whole-body control deployment, and **ROS2** for high-level orchestration — all managed with **uv workspaces** for reproducible environments across every target.

**Key recommendations:**
- Start fresh with `ardia-robotics/` mono-repo; migrate dm-isaac-g1 knowledge
- Use LeRobot for all VLA training (GR00T N1.6 is natively integrated)
- Deploy via SONIC's C++ stack for real-time motor control
- Use ROS2 for high-level orchestration, DDS for low-level control
- Pin all upstream repos as git submodules in `vendor/`
- Build target-specific infrastructure with Docker + uv per environment
- Adopt `robotics-agent-skills` (Boston Dynamics engineer) as AI coding assistant foundation — extending with Ardia-specific skills for G1/Dex1, GROOT, and SONIC patterns
- Evaluate `forge` (same author) for data pipeline conversion between GR00T ↔ LeRobot ↔ RLDS formats
- **Implement closed-loop reasoning** at both planning AND action-execution time — CV/VLM monitors evaluate actions during execution, not just at planning stage (Section 7)
- **Adopt RL post-training** for VLAs following PI0's RECAP methodology — the path from demo-quality to production-quality reliability, targeting >2x throughput improvement (Section 8)
- **Maintain dual motor control paths** — SONIC (NVIDIA cutting-edge) alongside ros2_control (team-validated, 200Hz C++). The training team's strength is in traditional ROS2/RL methods; the strategy builds on this foundation (Appendices I-K)
- **Build autonomous exploration** for unknown environments — VLM-driven sign reading, frontier scoring, and scene graph memory enable the robot to navigate venues it has never seen before (Section 9)

---

## 2. Current State Assessment

### 2.1 Existing Repository Inventory

| Repository | Type | Status | Purpose |
|------------|------|--------|---------|
| **dm-isaac-g1** | Custom (Git) | Active | G1 training/inference suite, primary documentation hub |
| **GR00T-WholeBodyControl** | NVIDIA clone (Git) | Active | WBC pipeline: decoupled WBC + SONIC |
| **dm-groot-inference** | Custom (Git) | Active | GROOT server wrapper for Spark (192.168.1.237:5555) |
| **groot_temp** | NVIDIA clone (Git) | Active | Full Isaac-GR00T working copy |
| **RoboticsReposResearch/04_Training/Isaac-GR00T** | NVIDIA clone (Git) | Reference | Research reference version |
| **unitree_sim_isaaclab** | Unitree clone (Git) | Active | Isaac Lab simulation environments |
| **unitree_rl_lab** | Unitree clone (Git) | Active | RL training framework |
| **HumanoidTraining** | Custom | Active | Humanoid training utilities and notebooks |
| **RAIR** | Custom | Active | Robotics AI Research knowledge base (100+ repos tracked) |
| **RoboticsReposResearch** | Collection | Reference | Organized research clones (01-07 categories) |
| **robotics-agent-skills** | External (Git) | **NEW** | AI coding assistant skills for production robotics (Boston Dynamics engineer) |
| **ardia** | Custom (in linear-okr-planner) | Active | Web/backend platform with AI agent framework |

### 2.2 Infrastructure Map

| System | Address | GPU | Container | Role |
|--------|---------|-----|-----------|------|
| **Workstation** | 192.168.1.205 | RTX PRO 6000 (98GB VRAM) | `dm-workstation` (Python 3.13) | Training + Simulation |
| **Spark (DGX)** | 192.168.1.237 | GB10 | `groot-server` | GROOT inference (port 5555) |
| **Mac (local)** | — | — | — | Development, documentation |
| **Jetson** | TBD | Jetson Orin | TBD | Real robot deployment |

### 2.3 Models Deployed

| Model | Location | Embodiment | Training Data | Status |
|-------|----------|------------|---------------|--------|
| groot-g1-gripper-hospitality-7ds | Spark (5555) | UNITREE_G1 | 7 hospitality tasks, 1,400 eps, 1.4M frames | **Deployed** |
| groot-g1-gripper-fold-towel-full | HuggingFace | UNITREE_G1 | Fold towel, 200 eps | Uploaded |

### 2.4 Key Technical Knowledge Accumulated

Documented across 26 files in `dm-isaac-g1/docs/`:

- **Dex1 gripper value conversion**: Physical [-0.02, 0.024]m ↔ Training [5.4, 0.0] (inverted)
- **UNITREE_G1 DOF layout**: 31 DOF state (legs 12 + waist 3 + arms 14 + grippers 2), 23 DOF action
- **Wrist joint reordering**: GROOT (yaw,roll,pitch) ↔ WBC/MuJoCo (roll,pitch,yaw)
- **MuJoCo eval audit**: 3 critical issues identified (wrong robot model, initial pose mismatch, wrist ordering)
- **Fine-tuning pipeline**: 8 bugs fixed in Isaac-GR00T's LeRobot v3.0 compatibility; pinned to transformers 4.51.3
- **WBC architecture**: 2-process (collapsed) vs 3-process (full decoupled) patterns documented
- **38 Dex1 RoboCasa environments** auto-registered via `GR00T_LOCOMANIP_ENVS_ROBOTS`

---

## 3. The Core Problem

### 3.1 Dependency Hell

The current setup requires **5+ interdependent repositories** to be installed in the right order, with specific Python versions, specific library pins, and custom `.pth` files:

```
Isaac-GR00T (eval/rollout_policy.py)
    ↓ imports
GR00T-WholeBodyControl (gr00t_wbc)
    ↓ requires
unitree_sim_isaaclab (RoboCasa scenes)
    ↓ shares environments with
unitree_rl_lab (RL training)
    ↓ all need
Isaac Lab + Isaac Sim (GPU-specific builds)
```

Each repo has its own `pyproject.toml`, its own dependency versions, and its own assumptions about where other repos are installed. A change in one breaks others silently.

### 3.2 The Workstation Problem

Code modified in `RoboticsRepositoryResearch/04_Training/Isaac-GR00T/gr00t/eval/rollout_policy.py` on the Mac **does not work on the Workstation** because:

1. That path doesn't exist on the Workstation
2. The Workstation has `/workspace/Isaac-GR00T/` inside the container
3. `rollout_policy.py` requires `gr00t_wbc` imports which need `GR00T-WholeBodyControl` installed with correct `.pth` files
4. Python 3.13 on the Workstation container requires dataclass fixes not needed on Mac

### 3.3 The Upstream Sync Problem

When NVIDIA releases GR00T N1.7 or updates WBC:
- Pulling upstream changes conflicts with our Dex1 modifications
- `.pth` files and symlinks break
- Our patches are scattered across multiple files in multiple repos
- No clear way to identify "our changes" vs "upstream code"
- dm-isaac-g1 tried to isolate this but couldn't because the repos call each other directly

---

## 4. Isaac-GR00T Eval vs WBC: Architecture Deep Dive

### 4.1 How rollout_policy.py Actually Works

`rollout_policy.py` (Isaac-GR00T's eval script) is **not independent** — it directly imports and wraps GR00T-WholeBodyControl:

```python
# Lines 99-101 of rollout_policy.py
from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv
from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
```

**Architecture (collapsed 2-process):**
```
┌─────────────────────────────────────────────┐
│  Process 1: Simulation + WBC (merged)       │
│  rollout_policy.py                          │
│  ├── RoboCasa gym env (MuJoCo scene)        │
│  ├── WholeBodyControlWrapper (WBC inside)   │
│  │   ├── RL locomotion policy (ONNX)        │
│  │   ├── IK upper body solver               │
│  │   └── Dex1 gripper value conversion      │
│  └── PolicyClient → connects to GROOT       │
└─────────────────────┬───────────────────────┘
                      │ ZMQ (port 5555)
┌─────────────────────▼───────────────────────┐
│  Process 2: GROOT Server (Spark)            │
│  GR00T N1.6 neural network                  │
│  Embodiment: UNITREE_G1                     │
│  Input: 31 DOF state + ego camera           │
│  Output: 23 DOF action                      │
└─────────────────────────────────────────────┘
```

### 4.2 Validated 4-Process Deployment (HumanoidTraining Unit 29)

The training team has validated a **4-process closed-loop architecture** (documented in Appendix K.5):

```
GR00T Server (port 5556) ←→ Closed-Loop Bridge (10Hz) ←→ MuJoCo Sim (port 5557)
                                      ↕
                              WBC Controller (50Hz)
```

The Bridge runs at 10 Hz, fetching camera images from the sim, sending them to GR00T for inference, and passing the resulting action chunks to WBC. InterpolationPolicy smooths between discrete action chunks at the full 50 Hz control rate. This architecture is the foundation for the closed-loop reasoning extensions in Section 7.

### 4.3 Recommendation: Build from WBC, Not Isaac-GR00T Eval

Isaac-GR00T's `rollout_policy.py` is a thin orchestrator. The real work happens in GR00T-WholeBodyControl. Building directly from WBC gives you:

1. **Direct access** to the control layer without going through Isaac-GR00T's wrappers
2. **SONIC** — the latest unified whole-body controller that supersedes decoupled WBC
3. **Cleaner deployment** — SONIC's C++ stack is production-ready with 4 real-time DDS threads
4. **Less fragile** — one fewer repo in your dependency chain

### 4.4 SONIC vs Decoupled WBC

| Aspect | Decoupled WBC (Old) | SONIC (New) |
|--------|---------------------|-------------|
| Architecture | RL lower body + IK upper body | Unified end-to-end policy |
| Process model | 3-process (sim, WBC, GROOT) | Single C++ process, 4 threads |
| Training basis | Separate RL + IK | Large-scale human motion imitation |
| Capabilities | Walk + manipulate | Walk, run, crawl, jump, teleop, manipulate |
| Deployment | Python-based | C++, ONNX/TensorRT, real-time DDS |
| Latency | Higher (IPC overhead) | Lower (single process) |
| Status in repo | `decoupled_wbc/` | `gear_sonic/` + `gear_sonic_deploy/` |

**SONIC deployment threading model:**

| Thread | Rate | Responsibility |
|--------|------|---------------|
| Input | 100 Hz | Poll input interface, handle commands |
| Control | 50 Hz | Gather observations, run policy, compute motor targets |
| Planner | 10 Hz | Re-plan locomotion trajectory |
| Command Writer | 500 Hz | Publish motor commands via DDS |

---

## 5. VLA Ecosystem Landscape (Early 2026)

### 5.1 NVIDIA GR00T Platform

The GR00T platform has evolved into a comprehensive end-to-end stack:

```
Data Collection          Training              Deployment
┌──────────────┐   ┌──────────────┐    ┌──────────────────┐
│ GR00T-Teleop │──▶│ GR00T-Mimic  │    │ GR00T-Control    │
│ (Apple       │   │ (780K synth  │    │ (SONIC WBC)      │
│  Vision Pro) │   │  trajectories│    │                  │
└──────────────┘   │  from few    │    │ GR00T-Dexterity  │
                   │  demos)      │    │ (Grasping)       │
┌──────────────┐   └──────┬───────┘    │                  │
│ GR00T-Gen    │          │            │ GR00T-Mobility   │
│ (Diverse     │──────────▼            │ (Navigation +    │
│  sim envs)   │   ┌──────────────┐    │  COMPASS)        │
└──────────────┘   │ GR00T N1.6   │    └──────────────────┘
                   │ Foundation   │              ▲
                   │ Model (3B)   │──────────────┘
                   │ System 1+2   │
                   └──────────────┘
```

**GR00T N1.6** (CoRL 2025):
- Dual-system: System 2 (Cosmos Reason VLM for planning) + System 1 (2x larger diffusion transformer for actions)
- Loco-manipulation capable (walk + manipulate simultaneously)
- 3B parameters, available on HuggingFace (`nvidia/GR00T-N1.6-3B`)
- Natively stores data in LeRobotDataset v3.0 format

**GR00T-Mimic** (SkillMimicGen):
- Generates 780,000 synthetic trajectories from limited human demos
- Equivalent to 6,500 hours of demonstrations, generated in 11 hours
- 40% performance improvement when combined with real data

**GR00T-Gen**:
- Generative AI for simulation environments
- 100+ tasks, domain randomization, Cosmos Transfer for visual diversity
- Cross-embodiment support

**Supporting infrastructure:**
- **Newton Physics Engine** (open-source, Linux Foundation) — co-developed with Google DeepMind and Disney Research
- **Cosmos World Foundation Models** — Predict 2.5 (video generation), Transfer 2.5 (synthetic data), Reason (physical reasoning)
- **Jetson Thor** — onboard compute, adopted by Figure AI, Google DeepMind, Meta, Unitree

### 5.2 HuggingFace LeRobot

**LeRobot v0.4.3** (January 2026) has become the de facto standard for VLA training:

**Integrated VLA Models:**

| Model | Params | Source | Key Feature |
|-------|--------|--------|-------------|
| **GR00T N1.6** | 3B | NVIDIA | Humanoid foundation model, loco-manipulation |
| **PI0 / PI0-FAST** | — | Physical Intelligence | Flow matching VLA, 5x faster autoregressive variant |
| **PI0.5** | — | Physical Intelligence | Open-world generalization |
| **SmolVLA** | 450M | HuggingFace | Compact, trainable on single GPU, pre-trained on 10M frames |
| **X-VLA** | 0.9B | ICLR 2026 | Cross-embodiment, 290K episodes across 7 platforms |

**LeRobotDataset v3.0** (October 2025):
- File-based storage (multiple episodes per Parquet/MP4 file)
- Hub-native streaming (`StreamingLeRobotDataset`)
- Handles OXE-level datasets (400GB+) with chunked episodes
- 54.6% of new datasets (1,633/2,989 in Oct 2025) already use v3.0
- **NVIDIA's GR00T workflows now natively produce this format**

```
dataset/
  data/chunk-000/file-000.parquet        # Multiple episodes per file
  videos/camera/chunk-000/file-000.mp4   # Consolidated video chunks
  meta/episodes/chunk-000/file-000.parquet  # Structured metadata
```

**Why this matters for us:** One training pipeline, swap models freely. Same dataset format works with GR00T N1.6, PI0, SmolVLA, X-VLA, ACT, etc. Community flywheel: 3,000+ new datasets per month on HuggingFace.

### 5.3 Competing VLA Foundation Models

| Model | Org | Params | Open? | Key Innovation |
|-------|-----|--------|-------|----------------|
| **GR00T N1.6** | NVIDIA | 3B | Yes | Humanoid-specific, loco-manipulation, Cosmos Reason |
| **Gemini Robotics 1.5** | Google DeepMind | — | SDK only | "Thinks before acting", cloud+edge architecture |
| **PI0.6** | Physical Intelligence | — | Yes (openpi) | RL fine-tuning doubles robot throughput |
| **Helix 02** | Figure AI | — | No | System 0 for human-like balance at kHz, whole-body VLA |
| **SmolVLA** | HuggingFace | 450M | Yes | Compact, single-GPU trainable, 30% faster async inference |
| **X-VLA** | ICLR 2026 | 0.9B | Yes | Cross-embodiment via "Soft Prompts" |

### 5.4 SONIC and EgoScale

**SONIC** (NVIDIA, November 2025):
- Behavior foundation model for motor control
- Scales along 3 axes: network (1.2M-42M params), data (100M+ frames, 700hrs), compute (9K-32K GPU hours)
- Universal token space: VR teleoperation, human video, VLA outputs — all through same interface
- On G1 + GR00T N1.5: **95% success on mobile pick-and-place**
- Capabilities: running, jumping, crawling, cross-embodiment motion tracking

**EgoScale** (NVIDIA GEAR Lab, 2025):
- Trains VLAs from **20,000+ hours of human video** (20x larger than prior efforts)
- Log-linear scaling law (R² = 0.9983) between human data volume and validation loss
- 3-stage pipeline: pre-train on human video → mid-train with 54hrs aligned data → post-train on downstream tasks
- **One-shot task adaptation**: learn new tasks from a single teleoperated demo
- Cross-embodiment: trained for 22-DoF Sharpa hand, transferred to G1's 7-DoF tri-finger with 30% improvement

### 5.5 ICLR 2026 VLA Trends (164 submissions)

Key emerging patterns:
- **Discrete Diffusion VLAs**: Replacing autoregressive action generation for better precision
- **Reasoning VLAs**: Chain-of-thought embedded in policy learning (DiffusionVLA, Vlaser, UniVLA)
- **Efficient VLAs**: Real-time deployment of billion-parameter models
- **World Models**: VLA + video prediction for causal dynamics learning
- **Notable 2026 models**: NVIDIA DreamZero, Cosmos Policy, TwinBrainVLA, RDT2 (zero-shot cross-embodiment)

---

## 6. Robotics Agent Skills: AI-Assisted Development Framework

### 6.1 Overview

**Repository:** [github.com/arpitg1304/robotics-agent-skills](https://github.com/arpitg1304/robotics-agent-skills)
**Author:** Arpit Gupta — Senior Staff ML Platform Engineer at Boston Dynamics (WPI Robotics 2019)
**License:** Apache-2.0 | **Stars:** 57 | **Created:** February 26, 2026
**Status:** Cloned to `/Users/elianomarques/Documents/DataScienceProjects/Datamentors/robotics-agent-skills/`

This is **not** a robotics runtime library — it is a curated collection of **SKILL.md prompt files** designed to be injected into AI coding assistants (Claude Code, Cursor, Copilot) so they write production-quality robotics code. Each skill follows Anthropic's skill format with YAML frontmatter and structured sections containing patterns, anti-patterns, code examples, and checklists.

### 6.2 Available Skills

| Skill | File | Key Topics | Lines |
|-------|------|------------|-------|
| **robotics-software-principles** | `SKILL.md` | SOLID for robotics, fail-safe defaults, rate separation, composability, graceful degradation | ~900 |
| **ros1** | `SKILL.md` | catkin, rospy, roscpp, nodelets, tf, actionlib, launch XML, ROS1→ROS2 migration | ~800 |
| **ros2** | `SKILL.md` | rclpy, rclcpp, DDS, QoS, lifecycle nodes, components, Python launch, build system, production deployment | ~1200 |
| **robotics-design-patterns** | `SKILL.md` | 6-layer robot stack, behavior trees, FSMs, HAL, safety systems, sim-to-real, data recording | ~600 |
| **robot-perception** | `SKILL.md` | Camera calibration, depth processing, point clouds, ICP, multi-sensor fusion, object tracking | ~3500 |
| **robotics-testing** | `SKILL.md` | pytest + ROS2, launch_testing, mock hardware, golden files, sim testing, CI/CD workflows | ~600 |

**Planned skills (roadmap):**
- `robotics-data-pipelines/` — RLDS, LeRobot, Zarr, format conversion (gitignored, in progress)
- `robot-simulation/` — MuJoCo, Isaac Sim, Gazebo
- `robot-manipulation/` — Grasping, motion planning, force control
- `robot-navigation/` — Nav2, SLAM, path planning
- `robot-learning/` — RL, imitation learning, VLAs
- `robot-deployment/` — Docker, fleet management, OTA updates

### 6.3 Key Patterns Directly Relevant to Ardia

#### Behavior Trees as Decision Framework
The `robotics-design-patterns` skill recommends **Behavior Trees (BT) as the default** for robot decision-making — exactly aligned with our `ai_brain_bt_agentic_rnd.md` architecture document. It provides:
- Complete py_trees implementation with Blackboard state sharing
- Pick-and-place BT example with safety checks
- Decision matrix: BT vs FSM (FSM for simple sequential, BT for complex reactive)

#### Skill Composability (Principle 11)
Defines the exact composition pattern our skills layer needs:
```
MoveTo + Grasp + Release = Pick
Pick + Place = PickAndPlace
```
With full dependency injection and a `build_skill_library()` factory that wires primitives into composites.

#### Hardware Abstraction Layer (HAL)
Provides the sim-to-real pattern we need:
- `ArmInterface` (abstract) → `UR5Arm` (real) / `MuJoCoArm` (sim)
- `GripperInterface` → `RobotiqGripper` / `SimulatedGripper`
- Config-driven switching: same application code, different HAL
- This maps directly to our `controllers/` layer in ardia-robotics

#### Rate Separation (Principle 6)
Documents the exact rate hierarchy our SONIC + ROS2 architecture follows:
```
Safety monitor:    1000 Hz    HARD real-time
Joint controller:  500-1000 Hz HARD real-time  ← SONIC (500Hz cmd writer)
Trajectory exec:   100-200 Hz  Firm real-time
State estimation:  50-200 Hz   Firm real-time  ← SONIC (50Hz control)
Perception:        10-30 Hz    Soft real-time
Planning:          1-10 Hz     Best effort     ← SONIC (10Hz planner)
Task management:   0.1-1 Hz    Best effort     ← Agentic layer
```

#### Safety Systems (4-Level Hierarchy)
Matches our safety kernel design:
- Level 0: Hardware E-Stop (physical button)
- Level 1: Safety-rated controller (SIL2/SIL3)
- Level 2: Software watchdog (heartbeats, timeout)
- Level 3: Application safety (workspace limits, collision avoidance)

Includes complete `SafetyWatchdog` and `WorkspaceMonitor` implementations.

#### ROS2 Production Patterns
The most extensive skill covers:
- QoS compatibility matrix (the "#1 source of ROS2 bugs" — BEST_EFFORT pub + RELIABLE sub = silent failure)
- Lifecycle (managed) nodes with full state machine
- Components for zero-copy intra-process communication
- CycloneDDS configuration and tuning
- 10-item production deployment checklist
- Build system guide (ament_cmake vs ament_python)

### 6.4 Author's Broader Ecosystem

Arpit Gupta has built a coherent data-centric robotics ML pipeline across his repos:

| Repo | Stars | Purpose | Relevance to Ardia |
|------|-------|---------|-------------------|
| **forge** | 39 | Dataset format conversion: RLDS ↔ LeRobot ↔ Zarr ↔ HDF5 ↔ Rosbag ↔ GR00T | **HIGH** — directly useful for our data pipeline (LeRobotDataset v3 ↔ GR00T format conversion) |
| **ros-time-machine** | 15 | Event-triggered ROS2 recording with pre/post-event buffering (C++) | **MEDIUM** — useful for real-robot data collection |
| **tessera** | 9 | Episode embedding visualization for dataset curation | **MEDIUM** — useful for training data quality assessment |
| **robotics-agent-skills** | 57 | AI coding assistant skills for robotics | **HIGH** — immediate integration into our dev workflow |

### 6.5 Integration Plan for Ardia

**Immediate (Phase 1):**
1. Install skills into Claude Code's skill directory for all team members
2. Configure YAML frontmatter triggers so Claude Code auto-loads relevant skills when writing ROS2 code, designing safety systems, or building perception pipelines
3. Add our Ardia-specific skills on top:
   - `ardia-g1-dex1/` — Dex1 gripper conversion, DOF layout, wrist remapping
   - `ardia-groot-pipeline/` — GROOT N1.6 training/eval patterns
   - `ardia-sonic-deploy/` — SONIC deployment patterns for G1

**Near-term (Phase 2):**
4. Evaluate `forge` for our data conversion pipeline (LeRobotDataset v3 ↔ GR00T ↔ RLDS)
5. Adopt `ros-time-machine` for real-robot data collection on the G1

**When author ships planned skills:**
6. Adopt `robot-learning/` skill when published (covers VLA fine-tuning — directly relevant)
7. Adopt `robot-simulation/` skill for MuJoCo/Isaac patterns
8. Contribute our own skills back to the community (BT-governed VLA execution, humanoid-specific patterns)

### 6.6 Impact on Report Recommendations

This repo **reinforces** our existing strategic decisions:
- **ROS2 as orchestration layer** (Q5b) — the most extensive skill in the repo
- **Behavior Trees for agentic governance** — exactly matches our BT architecture doc
- **HAL pattern for sim-to-real** — validates our `controllers/` layer design
- **Skill composability** — the composition pattern is what our `skills/` directory should implement
- **Safety-first design** — the 4-level safety hierarchy should be adopted wholesale

It **adds** one new consideration:
- **AI-assisted development as a force multiplier** — by loading these skills into Claude Code, every team member writing robotics code gets production-grade patterns injected automatically. This is particularly valuable given the complexity of ROS2 QoS, DDS configuration, and safety systems where subtle bugs are hard to catch.
- **`forge` as a data pipeline tool** — should be evaluated as a replacement for or complement to our custom `data/converters/` in ardia-robotics. It already supports GR00T format natively.

---

## 7. Closed-Loop Reasoning: CV/VLM Policies for Action-Time Evaluation

### 7.1 The Problem: Open-Loop Execution

Current VLA deployment follows a **plan-then-execute** pattern:
1. VLM reasons about the scene and instruction (System 2, ~10 Hz)
2. VLA generates an action chunk (16-50 actions, covering 0.8-2.5 seconds)
3. Robot executes the chunk **without evaluating whether it's working**
4. After the chunk completes, a new observation triggers the next cycle

This is dangerously open-loop during action execution. If the robot bumps an object, the environment changes, or the action drifts — the robot won't notice until the next observation cycle. For navigation, this means the robot cannot course-correct mid-stride. For manipulation, it means dropped objects aren't detected until too late.

### 7.2 Closed-Loop Architecture: Reasoning at Both Planning AND Actioning

The goal is to have continuous evaluation at **two levels**:

```
┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 1: PLANNING-TIME REASONING (1-10 Hz)                    │
│  "What should I do next?"                                       │
│                                                                  │
│  VLM/Cosmos Reason:                                             │
│  ├── Scene understanding + task decomposition                   │
│  ├── Chain-of-thought semantic planning                         │
│  ├── Generate candidate action sequences                        │
│  └── Select best plan via value estimation                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Action commands
┌───────────────────────────▼─────────────────────────────────────┐
│  LEVEL 2: ACTION-TIME REASONING (10-50 Hz)                      │
│  "Is what I'm doing working? Should I adjust?"                  │
│                                                                  │
│  CV/Lightweight VLM Monitor:                                    │
│  ├── Track object state during manipulation                     │
│  ├── Detect grasp slippage, collisions, drift                   │
│  ├── Evaluate trajectory progress against expected              │
│  ├── Trigger replanning if deviation exceeds threshold          │
│  └── Provide feedback signal for RL training                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Motor commands
┌───────────────────────────▼─────────────────────────────────────┐
│  LEVEL 3: MOTOR CONTROL (50-500 Hz)                             │
│  SONIC / WBC executes joint-level commands                      │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Approaches to Closed-Loop Action-Time Reasoning

#### A. Dual-System VLAs (Best for Ardia)

**Hume (MIT, May 2025)** — Open-source, 3B params:
- **System 2 (4 Hz):** Generates N candidate action chunks, evaluates each with a **learned Q-value head**, selects best-of-N by highest value
- **System 1 (90 Hz):** Lightweight diffusion policy refines selected chunk using current observations
- **Failure recovery:** When System 1 would repeat a failing trajectory, System 2's value-guided selection corrects course
- **Relevance:** Open-source (MIT), tested on real robots (WidowX, Franka, AgiBot G-1), directly implementable

**GR00T N1.6 System 2 (Cosmos Reason):**
- Cosmos Reason 2 (2B edge / 8B cloud) provides chain-of-thought reasoning + trajectory coordinate prediction
- Runs at lower frequency (~10 Hz) than motor control
- Currently generates plans, not mid-action corrections — **gap we need to fill**

#### B. Visual Chain-of-Thought (CoT-VLA)

- Predicts **future image frames** as visual subgoals before generating action sequences
- Two-phase: (1) generate subgoal image with causal attention, (2) generate short action sequence to achieve subgoal
- **Closed-loop:** After executing actions, captures new observation, compares to predicted subgoal, adjusts
- 17% improvement over SOTA VLAs on real-world tasks
- **Relevance:** Can be integrated as a "visual progress checker" within our BT governance layer

#### C. Value-Guided Action Selection

**AutoHorizon (Feb 2026):**
- Dynamically estimates how many actions from a predicted chunk to actually execute, based on cross/self-attention analysis
- Uncertain situations → execute fewer actions → replan sooner
- **Relevance:** Simple, no additional model needed — just attention weight analysis on existing VLA

**Affordance Field Intervention:**
- Uses 3D Spatial Affordance Fields as plug-in to correct VLA behavior mid-execution
- Addresses "memory traps" where VLA repeats failing trajectories
- **Relevance:** Complementary to Hume's Q-value approach

#### D. World Model Planning (DreamDojo + DreamZero)

- **DreamDojo:** Generates predicted future video conditioned on candidate actions → evaluate before executing
- **DreamZero:** Joint video+action prediction enables implicit planning
- Both approach real-time (10.8 FPS, 150ms/chunk)
- **Relevance:** Applicable when planning-time reasoning needs to simulate consequences

### 7.4 Recommended Architecture for Ardia

> **Foundation:** The training team's validated 4-process deployment (Section 4.2, Appendix K.5) already implements a 10 Hz closed-loop bridge between GR00T and WBC. The architecture below extends this with CV monitoring and value-guided action selection.

```
┌─────────────────────────────────────────────────────────────┐
│  BT GOVERNANCE LAYER (0.1-1 Hz)                             │
│  Behavior Tree monitors task progress, triggers replanning  │
│  ├── TaskProgressMonitor (CV-based)                         │
│  ├── SafetyChecker (collision, workspace bounds)            │
│  └── ReplanTrigger (deviation threshold)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  VLA + REASONING LAYER (10 Hz)                              │
│  GR00T N1.6 System 2 (Cosmos Reason) for planning           │
│  + Hume-style Q-value head for action selection              │
│  ├── Generate N candidate action chunks                     │
│  ├── Score each with learned value function                 │
│  ├── Select highest-value chunk                             │
│  └── Monitor execution with lightweight CV                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  MOTOR LAYER (50-500 Hz)                                    │
│  SONIC executes actions with real-time DDS                  │
│  ├── Proprioceptive feedback loop (already closed)          │
│  └── Reports execution state back to reasoning layer        │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
1. **CV monitor at action-time:** Lightweight object tracker (FoundationPose, YOLO) runs at 10-30 Hz during manipulation to detect grasp failures, object drift, collisions
2. **AutoHorizon-style adaptive execution:** Don't execute full 50-action chunk blindly — use attention weights to determine confidence and reduce chunk length when uncertain
3. **Hume's Q-value head:** Train a value function alongside the VLA to score action quality. At inference, generate multiple candidates and select the best
4. **BT-governed replanning:** Behavior Tree monitors CV signals and triggers full replanning when deviation exceeds threshold

### 7.5 Navigation-Specific Closed-Loop

For the G1 navigating in hospitality environments:

```
Navigation Closed-Loop:
1. VLM plans path: "Walk to the kitchen table" → waypoints
2. While walking:
   a. SLAM provides continuous position updates (10-20 Hz)
   b. Depth camera detects obstacles (30 Hz)
   c. CV detects if target is still visible/accessible
   d. If obstacle appears → local path replan (Nav2)
   e. If target moved → global replan (VLM)
   f. If unexpected person → stop, ask for clearance (BT safety)
3. At destination: switch to manipulation mode
```

**ROS2 Nav2 is already deployed** by the training team via Docker (Appendix I.6) with Livox MID-360 LiDAR as the obstacle source, 10 Hz costmap updates, DWB local planner (max 0.3 m/s), and Fast-LIO for SLAM (Appendix I.3). Our addition is the **semantic layer** — using VLM reasoning to evaluate whether the navigation goal itself still makes sense (e.g., the cup you were going to fetch has already been picked up by someone else). The existing `g1_perception` package (Appendix I.5) provides a starting point for CV-based object tracking during navigation.

### 7.6 Implementation Priority

| Component | Priority | Effort | Impact |
|---|---|---|---|
| AutoHorizon adaptive chunk execution | **P0** | Low (attention analysis) | High (reduces blind execution) |
| CV object tracker during manipulation | **P0** | Medium (FoundationPose/YOLO) | High (grasp failure detection) |
| Nav2 + VLM semantic replan | **P1** | Medium (ROS2 integration) | High (robust navigation) |
| Hume Q-value action selection | **P1** | High (value function training) | Very High (action quality) |
| DreamDojo world model planning | **P2** | High (14B model deployment) | High (long-horizon planning) |

---

## 8. RL Post-Training for VLAs: The PI0 Path to Production Quality

### 8.1 The Quality Gap: Demo vs Production

The field's consensus is crystallizing: **imitation learning (SFT) alone produces demo-quality VLAs, not production-quality ones.** The gap is analogous to the LLM transition from pre-training to RLHF:

| Stage | LLM Analogy | VLA Reality |
|---|---|---|
| Pre-training | GPT-3 pre-training | VLA SFT on demonstrations |
| RLHF/DPO | GPT-4 alignment | **VLA RL post-training** |
| Deployment quality | "Helpful, harmless" | "Reliable, recoverable" |

PI0.6's RECAP demonstrates this concretely:
- SFT-only pi0.6: Demo-quality — works in controlled settings, fails on edge cases
- pi\*0.6 + RECAP: Production-quality — **>2x throughput, ~0.5x failure rate**, full-day continuous operation

### 8.2 The PI0 Family: Architecture and Evolution

**pi0 (October 2024):** Introduced flow-matching VLA with dedicated action expert:
- PaliGemma VLM (3B) + Action Expert (0.3B) = 3.3B total
- Flow matching generates 50-action chunks in 73ms
- Outperforms OpenVLA and Octo by >2x on real tasks

**pi0.5 (April 2025):** Open-world generalization:
- Chain-of-thought semantic reasoning before action generation
- Knowledge Insulation prevents catastrophic forgetting of VLM capabilities
- **94% success in entirely unseen homes** (vs 31% without multi-environment data)

**pi\*0.6 (November 2025):** RL self-improvement via RECAP:
- Learns from heterogeneous data: demonstrations + robot's own failures + human corrections
- Advantage conditioning: labels all data with quality signal, conditions on high-advantage at inference
- **>2x throughput** on hardest tasks (espresso, laundry, box assembly)

**Full distillation available:** `PapersDistilled/PI0/PI0.md`

### 8.3 RECAP: How It Works

Unlike GRPO (which discards low-reward samples), RECAP **keeps all data** and learns from both success and failure:

```
1. COLLECT HETEROGENEOUS DATA
   ├── Expert demonstrations (good)
   ├── Robot's own rollouts (mixed quality)
   └── Human corrections during autonomous execution (good, targeted)

2. LEARN VALUE FUNCTION V(s)
   ├── Predicts "situation quality" relative to other situations
   ├── For espresso: V(s) ≈ -(steps remaining to completion)
   └── V increases on progress, decreases on errors

3. COMPUTE ADVANTAGES
   ├── A(s,a) = V(s') - V(s)
   ├── Positive advantage = action improved situation
   └── Negative advantage = action worsened situation

4. TRAIN WITH ADVANTAGE CONDITIONING
   ├── All data labeled with advantage level
   ├── VLA conditioned on advantage during training
   └── At inference: condition on HIGH advantage → policy better than any data source
```

### 8.4 RL for VLAs: The Broader Landscape

RECAP is not the only approach. The field is rapidly developing RL methods for VLAs:

| Method | Source | Key Idea | Reward Type |
|---|---|---|---|
| **RECAP** | PI (Nov 2025) | Advantage conditioning, offline RL | Learned value function |
| **SimpleVLA-RL** | ICLR 2026 | GRPO with binary rewards + dynamic sampling | Binary (0/1 task success) |
| **TGRPO** | June 2025 | Trajectory-level GRPO + Claude-generated reward functions | Multi-stage (auto-generated) |
| **VLA-RL** | May 2025 | Process reward model from VLM fine-tuning | Learned process rewards |
| **VLA-RFT** | Oct 2025 | World model as simulator + dense trajectory rewards | Trajectory comparison |
| **FPO** | Oct 2025 | Flow-matching-specific policy optimization | Continuous reward |
| **SRPO** | Nov 2025 | Self-referential: model's own successes as reference | Self-reference |

**Key insight:** SimpleVLA-RL shows that even **binary 0/1 rewards** (task success/failure) with GRPO can achieve 120% improvement on real robots. You don't need complex reward engineering to get started.

### 8.5 Recommended RL Strategy for Ardia

#### Phase 1: Binary Reward GRPO (Simplest, Highest ROI)

Start with the simplest approach that works:

```
1. Train base VLA (GR00T N1.6 or SmolVLA) on demonstrations via LeRobot
2. Run autonomous rollouts in MuJoCo/RoboCasa simulation
3. Automatically score: did task succeed? (binary 0/1)
4. Apply GRPO (SimpleVLA-RL approach):
   - Sample N rollouts per task
   - Compute group-relative advantages
   - Update VLA with clipped surrogate objective
5. Repeat until convergence
```

**Why start here:**
- Binary reward requires NO reward engineering
- GRPO is simpler than RECAP (no value function training)
- Simulation rollouts are free (MuJoCo runs at 10,000+ fps)
- SimpleVLA-RL code is open-source

#### Phase 2: RECAP with Real Robot Data

Once we have a G1 physical robot and human operators:

```
1. Deploy Phase 1 VLA on real G1
2. Collect heterogeneous data:
   - Autonomous rollouts (robot tries tasks independently)
   - Human corrections (operator intervenes when robot struggles)
   - Expert demonstrations (for new tasks)
3. Train value function V(s) on all data
4. Apply RECAP advantage conditioning
5. Deploy improved policy, repeat
```

**Why RECAP for real-robot:**
- Corrections data is the most efficient form of teaching
- Value function enables learning from ALL data (including failures)
- Offline RL doesn't need a simulator — trains on collected data

#### Phase 3: Flow Policy Optimization (FPO)

For advanced VLA improvement with continuous rewards:
- FPO is specifically designed for flow-matching VLAs (like both GR00T and PI0)
- Addresses intractable importance sampling in flow-matching policies
- Integrates credit assignment across action chunks

### 8.6 Training Infrastructure Requirements

| Phase | Compute | Data | Timeline |
|---|---|---|---|
| Phase 1 (GRPO in sim) | Workstation RTX PRO 6000 | Existing LeRobot datasets + sim rollouts | Weeks 8-10 |
| Phase 2 (RECAP real) | Workstation + Spark | Real G1 rollouts + corrections | Weeks 14+ |
| Phase 3 (FPO) | Workstation | Mixed sim + real | Weeks 18+ |

### 8.7 Connection to Closed-Loop Reasoning (Section 7)

RL post-training and closed-loop reasoning are **complementary and synergistic**:

1. **Closed-loop CV monitoring (Section 7)** provides the **reward signal** for RL training:
   - Object tracker detects grasp success/failure → binary reward
   - Progress monitor measures task completion percentage → dense reward
   - These signals feed directly into GRPO or RECAP

2. **RL training (this section)** improves the **quality of action-time decisions**:
   - Better VLA → fewer failures to detect → fewer replanning triggers
   - Value function (from RECAP) can be reused as Q-value head for Hume-style action selection

3. **The virtuous cycle:**
```
Better CV monitoring → Better reward signals → Better RL training
        ↑                                              │
        └──────── Better VLA → Cleaner execution ──────┘
```

---

## 9. Autonomous Exploration & Semantic Navigation in Unknown Environments

### 9.1 The Gap: VLAs Execute Known Tasks, But Can't Explore

Current VLAs (GR00T N1.6, PI0, etc.) are trained on demonstrations of **specific tasks**: "pick up the apple," "fold the towel," "open the cabinet." They generate motor actions conditioned on a language instruction and visual observation.

But consider: **you drop the robot in an unknown airport and tell it "find gate A20."** No VLA can solve this because:

```
What a VLA can do:                     What the airport requires:
├── "Walk forward 0.3m/s"             ├── "Where am I?"
├── "Reach for the cup"               ├── "What signs do I see?"
├── "Turn left 30°"                   ├── "What do the signs say?"
└── (motor-level reactions)            ├── "A16-A20 are to the right"
                                       ├── "I should head right"
                                       ├── "I've been this way already"
                                       ├── "This corridor is a dead end"
                                       ├── "I need to backtrack and try another way"
                                       └── (reasoning, memory, strategy)
```

This is **not** a perception problem, a navigation problem, or a motor control problem in isolation. It requires all three plus **reasoning about unseen space** and **reading the built environment's own wayfinding cues**.

### 9.2 Clarifying the Terminology

These terms are often conflated but refer to distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  PERCEPTION ("What do I see?")                                   │
│  ├── CV: Object detection, segmentation, depth estimation        │
│  │   → "Red bag at 2.3m, 30° left"                              │
│  ├── SLAM/LiDAR: Spatial mapping, localization                   │
│  │   → "I am at (4.2, 1.8), facing north"                       │
│  ├── OCR/Sign Reading: Text in the environment                   │
│  │   → "Sign says: Gates A16-A30 →"                             │
│  └── Scene Understanding: Semantic interpretation                │
│      → "I'm in a hallway with overhead signs and people"         │
├─────────────────────────────────────────────────────────────────┤
│  NAVIGATION ("Where do I go?")                                   │
│  ├── Metric (Nav2): Follow costmap, avoid obstacles              │
│  │   → "Move to waypoint (12.5, 8.3) avoiding the wall"         │
│  ├── Topological: Room-to-room, landmark-based                   │
│  │   → "Go from lobby → corridor B → gate area"                 │
│  └── Semantic: Goal-driven, language-conditioned                 │
│      → "Find gate A20" (requires REASONING + EXPLORATION)        │
├─────────────────────────────────────────────────────────────────┤
│  MOTOR CONTROL ("How do I move my body?")                        │
│  ├── SONIC: 50Hz control, 500Hz cmd — BLIND, receives cmd_vel   │
│  ├── ros2_control: 200Hz C++ — BLIND, receives joint targets     │
│  └── Both are purely reactive; they do NOT perceive anything     │
└─────────────────────────────────────────────────────────────────┘
```

**Critical insight:** SONIC and ros2_control are completely blind. They receive velocity commands (`cmd_vel: vx=0.3, vy=0, wz=0.1`) and figure out how to move legs to achieve that. The "intelligence" that decides *where* to go must come from above.

### 9.3 State-of-the-Art: How Robots Explore Unknown Environments (2025-2026)

The field has converged on a **two-tier pattern**: VLMs/LLMs provide high-level exploration strategy and reasoning, while classical robotics stacks (SLAM, Nav2, path planners) handle low-level mapping and locomotion.

#### A. Frontier-Based Exploration + VLM Scoring

Classical frontier exploration (identify boundaries between mapped and unmapped space) enhanced with VLMs to decide **which** frontier is most promising.

| System | Source | Key Idea | Results |
|--------|--------|----------|---------|
| **VLFM** | Boston Dynamics AI, ICRA 2024 | Score each frontier with VLM cosine-similarity ("how likely does this frontier lead to the target?") | SOTA on HM3D/Gibson/MP3D; deployed on **real Spot robot** |
| **Think, Remember, Navigate** | NeurIPS 2025 | VLM as active strategist with chain-of-thought + action history (breaks exploration loops) | "Exceptionally direct" trajectories |
| **Berkeley Frontier+VLM** | UC Berkeley 2025 | Annotated top-down map + frontier images fed to VLM for spatial reasoning about building layout | Verified across 6 Matterport3D environments |

**Architecture:**
```
Depth Camera → Occupancy Grid → Identify Frontiers → VLM scores each frontier
                                                       → "This hallway likely leads to more rooms"
                                                       → Select highest-scoring frontier
                                                       → Nav2 plans path to frontier
                                                       → Robot walks there → repeat
```

#### B. Scene Graph + LLM Planning

Build structured semantic maps (graphs of rooms, objects, relationships) and use LLMs to reason about where targets likely are.

| System | Source | Key Idea | Results |
|--------|--------|----------|---------|
| **SayNav** | SRI International, ICAPS 2024 | Incrementally build 3D scene graph; LLM generates step-by-step plan ("toothbrush → likely in bathroom → find bathroom") | SOTA on MultiON; outperforms oracle by 8% |
| **OrionNav** | ICLR 2025 Workshop | Hierarchical scene graph with LLM-based room labeling ("these objects suggest this is a kitchen") | Real-time on **quadruped robot** |
| **UniGoal** | CVPR 2025 | Unifies object/image/text goals into a single goal graph; navigation = graph matching | SOTA zero-shot across 3 benchmarks with single model |

#### C. Sign Reading & Mapless Navigation (Most Relevant to Airport Scenario)

Navigate by reading the built environment's own wayfinding infrastructure — signs, labels, directional markers.

| System | Source | Key Idea | Results |
|--------|--------|----------|---------|
| **SignScene** | NUS, ICLR 2025 | Detect signs → VLM parses content → Construct map aligned to sign's frame → VLM grounds parsed instructions to robot paths | **88.6% success** across hospitals, malls, airports, campuses (real-world) |
| **Sign Language** | NUS, June 2025 | GroundingDINO detects signs → PaddleOCR fast text extraction → VLM parses location+direction associations | Demonstrated on **quadruped robot** |
| **ReasonNav** | CoRL 2025 | Full agentic VLM (GPT-4o): reads signs, remembers landmarks, reasons about building layout, can ask people for directions | Real **university buildings** + Isaac Sim hospital |

**ReasonNav is the closest system to solving the airport scenario.** It has a memory bank of landmarks (frontiers, doors, people, signs), and the VLM reasons step-by-step: *"Room 312 sign says rooms 300-320 are this way, so room 315 must be down this hall."*

#### D. World Model Navigation ("Imagine Before Moving")

The robot predicts what it will see at each frontier *before physically going there*.

| System | Source | Key Idea | Results |
|--------|--------|----------|---------|
| **WMNav** | IROS 2025 (Oral) | VLM as world model: predicts frontier views, compares predictions vs reality to correct for hallucinations | +13.5% SR on MP3D |
| **3DGSNav** | Feb 2026 | 3D Gaussian Splatting scene representation; VLM sees synthesized views from unvisited viewpoints | +25% SR, +52% SPL over prior SOTA |

#### E. Vision-Language-Action Models for Navigation

| System | Source | Key Idea | Results |
|--------|--------|----------|---------|
| **NaVILA** | NVIDIA+USC, RSS 2025 | VLA generates mid-level language actions ("move forward 75cm", "turn left 30°"); RL locomotion policy executes | **88% success**, tested on **Unitree Go2 + G1** |

**NaVILA is directly relevant** — it's tested on the G1 humanoid, uses the same two-tier pattern (VLM reasoning → motor execution), and comes from NVIDIA.

### 9.4 The Dominant Architecture Pattern

Every successful system follows the same structure:

```
┌─────────────────────────────────────────────────────────────────┐
│  STRATEGIC LAYER (0.5-2 Hz)                                      │
│  VLM/LLM Exploration Agent                                       │
│  ├── Reads signs (OCR → VLM parsing)                            │
│  ├── Maintains memory (scene graph / landmark bank / VLMaps)    │
│  ├── Reasons about unseen space ("A18 → A20 must be further")   │
│  ├── Selects exploration targets (frontier scoring / planning)   │
│  └── Detects dead ends, triggers backtracking                    │
├─────────────────────────────────────────────────────────────────┤
│  TACTICAL LAYER (5-10 Hz)                                        │
│  Nav2 + SLAM                                                     │
│  ├── Fast-LIO builds occupancy map from LiDAR                   │
│  ├── Nav2 plans path to VLM-selected waypoint                   │
│  ├── DWB avoids dynamic obstacles                                │
│  └── Reports progress / arrival / blockage back to strategic     │
├─────────────────────────────────────────────────────────────────┤
│  MOTOR LAYER (50-200 Hz)                                         │
│  SONIC or ros2_control                                           │
│  ├── Converts cmd_vel to joint commands                          │
│  ├── Maintains balance on uneven surfaces                        │
│  └── Completely blind — no perception here                       │
└─────────────────────────────────────────────────────────────────┘
```

### 9.5 How Real Systems Orchestrate VLA + WBC + Exploration

Before designing our architecture, let's look at how the leading systems actually stack these layers — because they solve the orchestration problem in fundamentally different ways.

#### Pattern A: Frequency-Separated Layers (NVIDIA GR00T + SONIC, Figure Helix)

This is the NVIDIA stack we are building on. Each layer runs at a different frequency, communicating through a defined interface:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: VLM REASONING (7-10 Hz)                               │
│  Cosmos Reason 2B / Custom VLM                                   │
│  ├── Decomposes "find gate A20" into step-by-step plan          │
│  ├── "Step 1: Read nearby signs. Step 2: Walk toward A gates."  │
│  └── Outputs: language instruction for current step              │
│                                                                   │
│  Interface: language instruction + optional waypoint             │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: VLA MOTOR POLICY (10-120 Hz)                           │
│  GR00T N1.6 (3B params)                                         │
│  ├── Takes: camera images + language instruction + proprio      │
│  ├── Outputs: TELEOPERATION-FORMAT COMMANDS                     │
│  │   ├── Head pose (where to look)                              │
│  │   ├── Left wrist pose + right wrist pose (manipulation)      │
│  │   ├── Base height + waist position                           │
│  │   └── Navigation velocity (vx, vy, ωz)                      │
│  └── VLM backbone at 10Hz, DiT action head at 120Hz            │
│                                                                   │
│  Interface: teleoperation-format poses + velocity commands       │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: WHOLE-BODY CONTROLLER (50-1000 Hz)                    │
│  SONIC (C++) or ros2_control (200Hz)                             │
│  ├── Upper body: tracks VLA wrist/head poses directly           │
│  ├── Lower body: kinematic planner converts nav velocity →      │
│  │   leg motions for walking/turning                            │
│  ├── Maintains balance on uneven surfaces                       │
│  └── Outputs: joint torques to hardware via DDS                 │
│                                                                   │
│  This layer is COMPLETELY BLIND — no cameras, no perception     │
└─────────────────────────────────────────────────────────────────┘
```

**Critical interface detail:** GR00T N1.6 outputs commands in the **same format a human teleoperator would send** — wrist poses, head direction, navigation velocity. SONIC doesn't know (or care) whether these come from a VLA, a VR headset, or a keyboard. This is what makes the layers composable.

**Mode switching:** There is no explicit "switch from walking to manipulation." GR00T N1.6 smoothly blends navigation velocity and manipulation poses in the same output. SONIC always processes both — if navigation velocity is zero and wrist poses are moving, the robot stands still and manipulates. If wrist poses are neutral and navigation velocity is nonzero, the robot walks. The transition is continuous, not a mode switch.

#### Pattern B: Orchestrator Calls VLA as a Tool (Google Gemini Robotics)

Google treats the VLA as a **callable tool** inside a larger agentic VLM:

```
┌─────────────────────────────────────────────────────────────────┐
│  VLM ORCHESTRATOR (Gemini Robotics-ER 1.5)                       │
│  ├── Receives: "find gate A20"                                   │
│  ├── Reasons: "I should look for signs first"                   │
│  ├── Calls TOOL: google_search("airport map terminal B")        │
│  ├── Calls TOOL: vla_action("walk to the nearest sign")         │
│  ├── Reads sign via camera, updates plan                        │
│  ├── Calls TOOL: vla_action("walk down corridor to the right")  │
│  └── Each tool call returns status; orchestrator decides next   │
│                                                                   │
│  The VLA is just one of many tools the orchestrator can call     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ natural language instruction
┌───────────────────────────▼─────────────────────────────────────┐
│  VLA ACTION MODEL (Gemini Robotics 1.5)                          │
│  ├── "Embodied Thinking": internal monologue before acting      │
│  ├── Decomposes instruction into few-second motion segments     │
│  └── Outputs joint-level actions                                │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** The interface between planner and VLA is **open-vocabulary natural language**. The orchestrator doesn't need to know joint angles or action spaces — it just says what it wants in words.

#### Pattern C: Unified End-to-End (Figure Helix 02, PI0.5)

No external orchestrator at all. Everything is inside one (or two) neural networks:

```
Figure Helix 02:
┌─────────────────────────┐
│  System 2 (7B VLM, 9Hz) │─── latent vector ──→┌─────────────────────────┐
│  "What should I do?"     │                     │  System 1 (200Hz)       │
└─────────────────────────┘                     │  ALL sensors → ALL joints│
                                                 │  Walking + manipulation │
                                                 │  = one unified policy   │
                                                 └────────┬────────────────┘
                                                          │ joint targets
                                                 ┌────────▼────────────────┐
                                                 │  System 0 (1000Hz)      │
                                                 │  Balance + contacts     │
                                                 └─────────────────────────┘

PI0.5:
┌───────────────────────────────────────────┐
│  Single 3.3B VLA                           │
│  Step 1: Predict subtask ("pick up plate") │
│  Step 2: Generate actions (50Hz)           │
│  No external planner, no WBC              │
│  (works on wheeled bases, NOT bipedal)     │
└───────────────────────────────────────────┘
```

**Figure's key innovation:** There is NO mode switch between walking and manipulation. System 1 connects every sensor to every joint through a single unified visuomotor transformer. The "mode" is implicit in the learned policy.

**PI0.5's limitation:** No WBC layer means it cannot handle bipedal balance. Works for wheeled mobile manipulators, not humanoids like G1.

#### Pattern D: Language-Mediated (NaVILA — tested on G1)

The VLA outputs **mid-level language commands** instead of joint angles:

```
┌───────────────────────────────────┐
│  VLA (fine-tuned VLM)              │
│  Takes: camera image + goal       │
│  Outputs: "move forward 75cm"     │
│           "turn left 30 degrees"   │
│  (NOT joint angles)               │
└─────────────┬─────────────────────┘
              │ natural language command
┌─────────────▼─────────────────────┐
│  Locomotion RL Policy              │
│  Trained to follow language cmds   │
│  + obstacle avoidance              │
│  Outputs: joint torques            │
└───────────────────────────────────┘
```

**Key advantage:** The same VLA works on **different robot bodies** (tested on Go2 quadruped AND G1 humanoid) because the interface is language, not joint-specific.

### 9.6 The Full Integrated Architecture for Ardia

Now here's how ALL of these pieces fit together for a G1 operating in an unknown hospitality environment. This is the complete orchestration — from "find gate A20" all the way down to joint torques:

```
┌═══════════════════════════════════════════════════════════════════════════┐
║                                                                           ║
║  MISSION: "Find gate A20 and deliver this coffee"                        ║
║                                                                           ║
║ ┌───────────────────────────────────────────────────────────────────────┐ ║
║ │  LAYER 4: AGENTIC ORCHESTRATOR (0.5-2 Hz)                            │ ║
║ │  BT-Governed LLM/VLM Agent (Claude / Cosmos Reason)                  │ ║
║ │                                                                       │ ║
║ │  ┌─────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐  │ ║
║ │  │ Sign Reader  │ │ Frontier     │ │ Scene Graph │ │ Task Planner │  │ ║
║ │  │ (OCR + VLM) │ │ Scorer       │ │ Memory      │ │ (LLM)        │  │ ║
║ │  └──────┬──────┘ └──────┬───────┘ └──────┬──────┘ └──────┬───────┘  │ ║
║ │         │                │                │               │           │ ║
║ │         └────────┬───────┘────────────────┘───────────────┘           │ ║
║ │                  ▼                                                    │ ║
║ │  Current Decision: "Sign says A16-A30 right. I haven't explored      │ ║
║ │  right. Frontier scorer confirms open corridor. Plan: walk right     │ ║
║ │  80m, look for gate numbers, holding coffee stable."                 │ ║
║ │                                                                       │ ║
║ │  Output: LANGUAGE INSTRUCTION to VLA                                 │ ║
║ │  → "Walk forward along the right corridor while carrying the cup.    │ ║
║ │     Look for gate A20 on the left side."                             │ ║
║ │                                                                       │ ║
║ │  Output: NAV2 WAYPOINT (optional, for metric navigation)             │ ║
║ │  → goal_pose: (x=12.5, y=8.3) in map frame                         │ ║
║ └───────────────────────────┬──────────────────────┬────────────────────┘ ║
║                             │                      │                      ║
║              language instruction          Nav2 waypoint                  ║
║                             │                      │                      ║
║ ┌───────────────────────────▼──────────────────────▼────────────────────┐ ║
║ │  LAYER 3: VLA POLICY (10-120 Hz)                                      │ ║
║ │  GR00T N1.6 / DreamZero / PI0                                        │ ║
║ │                                                                       │ ║
║ │  Inputs:                                                              │ ║
║ │  ├── RGB camera images (ego view)                                    │ ║
║ │  ├── Language instruction from Layer 4                               │ ║
║ │  ├── Proprioceptive state (31 DOF joint positions)                   │ ║
║ │  └── [Optional] Nav2 velocity hints                                  │ ║
║ │                                                                       │ ║
║ │  Outputs (teleoperation-format):                                     │ ║
║ │  ├── Head pose: look left (scanning for gate numbers)                │ ║
║ │  ├── Right wrist pose: holding cup stable                            │ ║
║ │  ├── Left wrist pose: neutral at side                                │ ║
║ │  ├── Base height: normal walking                                     │ ║
║ │  └── Navigation velocity: vx=0.3, vy=0, ωz=0.05 (slight right)     │ ║
║ │                                                                       │ ║
║ │  ┌──────────────────────────────────────────────────────────┐        │ ║
║ │  │ CV MONITOR (Section 7, 10-30 Hz)                         │        │ ║
║ │  │ ├── Is the cup still in hand? (grasp stability)          │        │ ║
║ │  │ ├── Is the robot making progress toward waypoint?        │        │ ║
║ │  │ ├── Any obstacles the VLA didn't notice?                 │        │ ║
║ │  │ └── INTERRUPT → replan if cup slipping or path blocked   │        │ ║
║ │  └──────────────────────────────────────────────────────────┘        │ ║
║ └───────────────────────────┬───────────────────────────────────────────┘ ║
║                             │                                             ║
║              teleoperation-format commands                                ║
║              (wrist poses + nav velocity + head pose)                     ║
║                             │                                             ║
║ ┌───────────────────────────▼───────────────────────────────────────────┐ ║
║ │  LAYER 2: TACTICAL NAVIGATION (5-10 Hz)                               │ ║
║ │  Nav2 + Fast-LIO SLAM                                                 │ ║
║ │                                                                       │ ║
║ │  ├── Fast-LIO: builds occupancy map from Livox MID-360              │ ║
║ │  ├── Nav2 DWB planner: local obstacle avoidance                      │ ║
║ │  ├── Costmap: 10Hz updates, 5m×5m rolling window                    │ ║
║ │  ├── Provides: position in map frame back to Layer 4                 │ ║
║ │  └── Can OVERRIDE VLA nav velocity if obstacle detected              │ ║
║ │                                                                       │ ║
║ │  Two operating modes:                                                 │ ║
║ │  a) VLA-driven: VLA generates nav velocity, Nav2 only overrides     │ ║
║ │     for safety (obstacle too close)                                   │ ║
║ │  b) Waypoint-driven: Nav2 generates nav velocity from waypoint,      │ ║
║ │     VLA focuses on upper body (carrying cup, looking around)         │ ║
║ └───────────────────────────┬───────────────────────────────────────────┘ ║
║                             │                                             ║
║              merged velocity commands + upper body poses                  ║
║                             │                                             ║
║ ┌───────────────────────────▼───────────────────────────────────────────┐ ║
║ │  LAYER 1: WHOLE-BODY CONTROLLER (50-1000 Hz)                          │ ║
║ │  SONIC (C++) or ros2_control (200Hz C++)                               │ ║
║ │                                                                       │ ║
║ │  ├── Upper body: IK solver tracks wrist/head poses from VLA          │ ║
║ │  ├── Lower body: locomotion policy converts nav velocity → leg gait  │ ║
║ │  ├── Balance controller: IMU feedback, maintains CoM stability       │ ║
║ │  ├── Safety: joint limits, torque limits, e-stop                     │ ║
║ │  └── Outputs: 29 joint torques via DDS at 500Hz                     │ ║
║ │                                                                       │ ║
║ │  THIS LAYER IS COMPLETELY BLIND — NO CAMERAS, NO PERCEPTION          │ ║
║ └───────────────────────────┬───────────────────────────────────────────┘ ║
║                             │                                             ║
║              joint torques via CycloneDDS                                 ║
║                             │                                             ║
║ ┌───────────────────────────▼───────────────────────────────────────────┐ ║
║ │  HARDWARE: Unitree G1 + Dex1 Hands                                    │ ║
║ │  29 motors, Livox MID-360 LiDAR, RealSense D435i, IMU               │ ║
║ └───────────────────────────────────────────────────────────────────────┘ ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 9.7 The Interface Between Layers — This Is Where the Magic Happens

The single most important design decision is **what crosses each boundary**:

| Boundary | What Flows Down | What Flows Up | Format |
|----------|----------------|---------------|--------|
| Layer 4 → 3 | Language instruction + optional waypoint | "I see gate A18" (VLA camera semantics) | Natural language + (x,y,θ) |
| Layer 3 → 2 | Teleoperation-format commands (poses + nav vel) | Obstacle alerts, position updates | Pose arrays + cmd_vel |
| Layer 2 → 1 | Merged velocity + upper body targets | Joint state feedback | cmd_vel + joint targets |
| Layer 1 → HW | 29 joint torques | 29 joint positions + IMU | DDS LowCmd / LowState |

**Why natural language between Layer 4 and Layer 3?** Following Google Gemini Robotics and NaVILA's pattern — the orchestrator doesn't need to know about joint angles or action spaces. It just describes what it wants. This makes the orchestrator **VLA-agnostic**: swap GR00T for DreamZero or PI0, and the orchestrator doesn't change.

**Why teleoperation-format between Layer 3 and Layer 1?** Following NVIDIA's pattern — the VLA outputs the same format a human teleoperator would. This means SONIC doesn't know (or care) whether commands come from a VLA, a VR headset, or a keyboard. This is what makes SONIC composable.

### 9.8 How Mode Switching Actually Works

Your airport scenario involves continuous mode switching — walking, looking around, reading signs, approaching a person, manipulating a door handle, carrying coffee. Here's how each layer handles this:

**Layer 4 (Orchestrator) — plans the mode:**
```
BT Root
├── [Sequence] Find Gate A20
│   ├── [Action] Explore: read signs, score frontiers
│   │   └── Instruction to VLA: "Walk forward, look for signs"
│   ├── [Condition] Sign found pointing right
│   │   └── Instruction to VLA: "Turn right, walk down corridor"
│   ├── [Condition] Gate A20 found
│   │   └── Instruction to VLA: "Approach gate A20"
│   └── [Sequence] Deliver Coffee
│       ├── [Action] Instruction: "Walk to the counter, offer the cup"
│       └── [Action] Instruction: "Place the cup on the counter"
```

**Layer 3 (VLA) — blends the execution seamlessly:**

The VLA doesn't "switch modes." In a single output, it simultaneously generates:
- Navigation velocity (walking)
- Head pose (looking at signs)
- Wrist poses (holding coffee stable)

When the orchestrator changes instruction from "walk and look for signs" to "place the cup on the counter," the VLA smoothly transitions:
- Navigation velocity → 0 (stop walking)
- Head pose → look at counter (fixate on target)
- Right wrist pose → extend toward counter (manipulation)

**No explicit mode switch.** The VLA's action space always includes both navigation and manipulation. The instruction determines the blend.

**Layer 1 (WBC) — doesn't know about modes at all:**

SONIC always does the same thing: track the upper-body poses from the VLA and convert navigation velocity into leg motion. If nav velocity is zero, the legs maintain a stable stance for manipulation. If wrist poses are neutral, the arms stay at rest during walking. The "mode" is implicit in the VLA's output.

### 9.9 Three Architecture Avenues for Ardia

Based on the research, here are three viable architectures in order of increasing ambition. Each integrates exploration, VLA task execution, and WBC motor control differently.

#### Avenue A: NVIDIA-Native Stack (Lowest Risk, Fastest to Deploy)

Use NVIDIA's own components wherever possible.

```
Cosmos Reason 2B (planning) → GR00T N1.6 (VLA) → SONIC (WBC)
       +
Our exploration agent (sign reader, frontier scorer, scene graph)
plugs into Cosmos Reason as additional context
```

| Component | Source | Role |
|-----------|--------|------|
| **Orchestrator** | Cosmos Reason 2B (NVIDIA) + our exploration modules | Plans, reads signs, selects waypoints |
| **VLA** | GR00T N1.6 (NVIDIA, 3B) | Generates teleoperation-format commands |
| **WBC** | SONIC (NVIDIA, C++) | 50Hz control, 500Hz cmd publishing |
| **SLAM** | Fast-LIO (team-validated) | Occupancy map + localization |
| **Nav2** | ROS2 Nav2 (team-validated) | Obstacle avoidance, metric navigation |

**Pros:** All NVIDIA components designed to work together. GR00T N1.6 was trained to output SONIC-compatible commands. COMPASS provides navigation fine-tuning data.
**Cons:** Cosmos Reason 2B may not be flexible enough for complex exploration reasoning. Vendor lock-in.
**Interface:** Language instruction (Cosmos → GR00T) + teleoperation commands (GR00T → SONIC).

#### Avenue B: Agentic Orchestrator + Swappable VLA (Most Flexible)

Follow Google's pattern — the VLA is a "tool" the orchestrator calls.

```
Claude/GPT-4o Orchestrator (agentic, tool-calling)
  ├── Tool: vla_action("walk toward the sign ahead")
  ├── Tool: read_sign(camera_image) → "Gates A16-A30 →"
  ├── Tool: nav2_goto(x=12.5, y=8.3)
  ├── Tool: vla_action("place the cup on the counter")
  └── Tool: ask_human("Excuse me, where is gate A20?")

VLA (GR00T N1.6 OR DreamZero OR PI0) → WBC (SONIC or ros2_control)
```

| Component | Source | Role |
|-----------|--------|------|
| **Orchestrator** | Claude API / GPT-4o (tool-calling agent) | Strategic planning, sign interpretation, human interaction |
| **VLA** | Swappable: GR00T N1.6, DreamZero, PI0 | Motor policy — receives language instructions |
| **WBC** | SONIC or ros2_control | Motor execution |
| **Sign Reader** | GroundingDINO + PaddleOCR + VLM | Perception tool called by orchestrator |
| **Scene Graph** | SayNav-style | Memory tool queried by orchestrator |

**Pros:** VLA is swappable without changing orchestrator. Orchestrator can use ANY tool (web search, databases, human interaction). Most flexible for unknown scenarios.
**Cons:** Requires reliable cloud connectivity for orchestrator LLM. Higher latency for planning decisions. More complex to build.
**Interface:** Natural language (Orchestrator → VLA). The VLA is a "function" the orchestrator calls.

#### Avenue C: World-Model-Driven (Most Advanced, Research-Grade)

Use DreamZero or Cosmos Policy as a world model that **imagines outcomes before acting**.

```
VLM Exploration Agent
  ├── "Should I go left or right?"
  ├── DreamZero IMAGINES going left → predicts: dead end
  ├── DreamZero IMAGINES going right → predicts: more gates visible
  └── Decision: go right (based on imagined outcomes)

DreamZero (14B, world model + action model) → WBC (SONIC)
```

| Component | Source | Role |
|-----------|--------|------|
| **Exploration Agent** | VLM + DreamZero imagination | Plans by simulating future outcomes |
| **World Model** | DreamZero (14B) or Cosmos Policy | Jointly predicts future video + actions |
| **WBC** | SONIC | Motor execution |
| **Frontier Evaluator** | 3DGSNav-style | Renders unseen viewpoints for VLM evaluation |

**Pros:** Robot can "think before acting" — predict consequences of exploration choices. Most robust for truly unknown environments.
**Cons:** DreamZero is 14B params (requires significant GPU). World model prediction adds latency. Most experimental — least real-world validation.
**Interface:** The world model IS the VLA — it generates both predictions and actions.

### 9.10 Recommended Avenue for Ardia: Start A, Evolve to B

**Phase 1-3 (Now → Week 12): Avenue A (NVIDIA-Native)**
- Deploy GR00T N1.6 + SONIC as the core motor stack
- Add our exploration modules (sign reader, frontier scorer) alongside Cosmos Reason
- Use the team's validated Fast-LIO + Nav2 stack for tactical navigation
- This gets a working system fastest with the least risk

**Phase 4+ (Week 13+): Evolve to Avenue B (Agentic Orchestrator)**
- Replace Cosmos Reason with a full agentic orchestrator (Claude API or equivalent)
- Make the VLA a callable tool — enables swapping GR00T for DreamZero or PI0 as they mature
- Add tool-calling capabilities: sign reading, scene graph queries, human interaction
- This gives maximum flexibility for complex scenarios like the airport

**Monitor Avenue C (Research Track):**
- Track DreamZero and Cosmos Policy progress
- When world models become real-time (they're approaching 7 Hz now), evaluate as a replacement for the VLA layer
- The "imagine before acting" paradigm is the ultimate solution for unknown environments but isn't production-ready yet

### 9.11 What We Already Have vs What We Need

**Already deployed (from HumanoidTraining, Appendix I):**
- Fast-LIO SLAM with Livox MID-360 (occupancy map, localization)
- Nav2 with DWB planner (path planning, obstacle avoidance)
- RealSense D435i (RGB-D for perception)
- g1_perception (basic CV: HSV detection, RGB-D 3D positioning, TF broadcasting)
- Sport mode cmd_vel driver (basic velocity control)
- GR00T N1.6 inference server (Spark, port 5555)
- 4-process closed-loop bridge at 10Hz (Appendix K.5)

**What we need to add:**

| Component | For Layer | Priority | Based On |
|-----------|-----------|----------|----------|
| **Sign Reader** | 4 (Orchestrator) | **P0** | SignScene + g1_perception |
| **Frontier Scorer** | 4 (Orchestrator) | **P0** | VLFM |
| **Exploration Memory** | 4 (Orchestrator) | **P1** | SayNav scene graph |
| **Language-to-VLA interface** | 4→3 boundary | **P1** | Gemini Robotics / NaVILA pattern |
| **Nav2-VLA velocity merger** | 2→1 boundary | **P1** | Custom: safety override logic |
| **Building Reasoner** | 4 (Orchestrator) | **P2** | ReasonNav chain-of-thought |
| **Agentic tool-calling orchestrator** | 4 (Orchestrator) | **P2** | Claude API tool use |
| **Human Interaction** | 4 (Orchestrator) | **P3** | ReasonNav |
| **World model imagination** | Research | **P3** | DreamZero / Cosmos Policy |

### 9.12 Connection to Other Sections

- **Section 4.2 (4-Process Deployment):** Our validated 10Hz bridge is Layer 3 → Layer 1 in the architecture above. Extend it to receive instructions from Layer 4.
- **Section 7 (Closed-Loop Reasoning):** The CV Monitor sits inside Layer 3, interrupting the VLA when execution diverges from expectations. AutoHorizon adapts within Layer 3.
- **Section 8 (RL Post-Training):** RL improves Layer 3 (better VLA actions). Exploration efficiency in Layer 4 can also be RL-trained — reward = "found target faster."
- **Appendix I (Sensor Stack):** Provides the complete hardware for Layers 1-2. Layer 4 only needs the camera feed that Layer 3 already captures.

### 9.13 Implementation Priority

| Phase | Component | Effort | Impact |
|-------|-----------|--------|--------|
| **Phase 2** | Sign Reader (GroundingDINO + OCR + VLM) | Medium | High — immediate wayfinding |
| **Phase 3** | VLFM frontier scoring | Medium | High — intelligent exploration |
| **Phase 3** | Exploration memory (visited rooms/signs) | Low | Medium — prevents loops |
| **Phase 3** | Nav2-VLA velocity merger (safety override) | Medium | High — safe autonomous walking |
| **Phase 4** | Agentic orchestrator with tool-calling | High | Very High — full autonomy |
| **Phase 4** | Scene graph + LLM planner | High | Very High — complex scenarios |
| **Phase 4** | Human interaction fallback | Medium | Medium — safety net |
| **Research** | DreamZero/Cosmos Policy world model integration | High | Transformative — imagine before acting |

### 9.14 Key References

**Exploration & Semantic Navigation:**

| System | Paper | Code | Real Robot? |
|--------|-------|------|-------------|
| **VLFM** | [arXiv:2312.03275](https://arxiv.org/abs/2312.03275) | [github](https://github.com/bdaiinstitute/vlfm) | Boston Dynamics Spot |
| **SayNav** | [arXiv:2309.04077](https://arxiv.org/abs/2309.04077) | [github](https://github.com/arajv/SayNav) | Sim (ProcTHOR) |
| **SignScene** | [ICLR 2025](https://arxiv.org/html/2602.12686v1) | — | Real: hospitals, malls, airports |
| **ReasonNav** | [arXiv:2509.21189](https://arxiv.org/abs/2509.21189) | [github](https://github.com/ReasonNav/ReasonNav) | Real university buildings |
| **NaVILA** | [arXiv:2412.04453](https://arxiv.org/abs/2412.04453) | [github](https://github.com/AnjieCheng/NaVILA) | **Unitree G1 + Go2** |
| **WMNav** | [arXiv:2503.02247](https://arxiv.org/abs/2503.02247) | [github](https://github.com/B0B8K1ng/WMNavigation) | Sim (Habitat) |
| **3DGSNav** | [arXiv:2602.12159](https://arxiv.org/abs/2602.12159) | — | Real robot |
| **UniGoal** | [arXiv:2503.10630](https://arxiv.org/abs/2503.10630) | [github](https://github.com/bagh2178/UniGoal) | Sim (Habitat) |

**Full System Architectures (VLA + WBC + Planning):**

| System | Source | Architecture Pattern | Key Interface |
|--------|--------|---------------------|---------------|
| **GR00T N1.6 + SONIC** | [NVIDIA Research](https://research.nvidia.com/labs/gear/gr00t-n1_6/) | Frequency-separated (Pattern A) | Teleoperation-format → SONIC |
| **Figure Helix 02** | [figure.ai/helix-02](https://www.figure.ai/news/helix-02) | Unified end-to-end (Pattern C) | S2 latent → S1 → S0 |
| **Gemini Robotics** | [ai.google.dev](https://ai.google.dev/gemini-api/docs/robotics-overview) | Orchestrator + tool-calling (Pattern B) | Natural language tool calls |
| **DreamZero** | [dreamzero0.github.io](https://dreamzero0.github.io/) | World model (Avenue C) | Joint video+action prediction |
| **Cosmos Policy** | [arXiv:2601.16163](https://arxiv.org/abs/2601.16163) | World model as policy | Latent frame injection |
| **WholeBodyVLA** | [arXiv:2512.11047](https://arxiv.org/abs/2512.11047) | Separate loco/manip LAMs | VQ-VAE latent actions |
| **LeVERB** | [arXiv:2506.13751](https://arxiv.org/abs/2506.13751) | VLA latent vocabulary → WBC | "Latent verbs" → RL controller |
| **SwitchVLA** | [arXiv:2506.03574](https://arxiv.org/abs/2506.03574) | Contact-phase task switching | Behavior modulation |

### 9.15 Long-Horizon Memory for Hospitality Tasks (MEM)

A critical gap in our current VLA pipeline is the **absence of memory**. GR00T N1.6 and PI0 both generate actions from single observations — they cannot remember what they did 30 seconds ago, let alone track a 15-minute kitchen cleanup.

**MEM (Multi-Scale Embodied Memory)**, from Physical Intelligence (Torne et al., 2026), solves this with a dual-memory architecture integrated into pi_0.6:

1. **Short-horizon video memory** (seconds): An efficient video encoder modifies the standard ViT attention pattern to process multiple past frames without latency explosion. Enables: occlusion recovery, re-grasp adaptation, dynamic tracking. Stays under 300ms with 18 frames (54 seconds).

2. **Long-horizon language memory** (minutes): A high-level policy maintains a compressed natural language summary of semantic events ("I placed a plate in the cabinet, wiped the counter, and moved to the sink"). Enables: recipe tracking, cleanup progress, multi-step planning across 15 minutes.

**Key results:**
- Solves 15-minute tasks (recipe setup, full kitchen cleanup, grilled cheese sandwich) that memoryless VLAs cannot
- Enables **in-context adaptation**: +41% on chopstick grasping, +62% on fridge opening after failures
- **No degradation** on standard manipulation tasks (unlike prior memory approaches)
- Architecture-agnostic: the video encoder adds no new parameters, applicable to any ViT-based VLA

**Integration with Ardia:** MEM directly addresses our hospitality use case. The language memory maps to our BT governance layer (tracking task progress), while the video memory maps to our closed-loop reasoning layer (Section 7). Full distillation: `PapersDistilled/MEM/MEM.md`.

### 9.16 Emerging Robotics Software Platforms: DimOS and Roboflow

Two additional systems inform our perception and navigation strategy:

#### DimOS (Dimensional OS) — Agent-Native Robotics SDK

[github.com/dimensionalOS/dimos](https://github.com/dimensionalOS/dimos) provides a Python-first alternative to ROS2 with native support for:
- **VLM-driven perception**: Integrates Florence, Moondream, Qwen, and OpenAI VLMs as first-class modules
- **Spatial memory**: Spatio-temporal RAG with object permanence tracking
- **Agent architecture**: LLM/VLM agents as native modules subscribing to perception and control streams via Model Context Protocol (MCP)
- **Navigation**: SLAM, dynamic obstacle avoidance, frontier exploration
- **Hardware**: Unitree G1 (beta), Go2 (stable), drones, arms

**Relevance to Ardia:**
- DimOS's perception module patterns (VLM → spatial memory → agent reasoning) validate our Section 7 closed-loop architecture
- The spatial memory system with object permanence is directly useful for hospitality (tracking where objects are)
- The agent-native design (agents as modules, not bolted on) aligns with our Avenue B agentic orchestrator
- **Not a replacement for our stack** — DimOS is alpha-stage and lacks SONIC-level motor control. But its perception and agent composition patterns are worth adopting

#### Roboflow — Robotics Vision AI Platform

[roboflow.com/industries/robotics](https://roboflow.com/industries/robotics) provides production-ready vision tooling:
- **Edge deployment**: Minimizes latency for real-time CV monitoring (our Section 7 action-time reasoning)
- **Object detection + tracking**: YOLO-based, directly applicable to our grasp failure detection
- **Pose estimation**: Relevant for manipulation monitoring
- **Customers**: Peer Robotics, Nexera Robotics, Standard Bots — production-validated

**Relevance to Ardia:**
- Roboflow could provide the **CV monitoring backbone** for our action-time reasoning layer (Section 7.2, Level 2)
- Edge deployment on Jetson Orin aligns with our deployment target
- Pre-trained models for object detection reduce our custom training burden
- Consider as the perception tooling layer feeding into our BT governance

---

## 10. Recommended Repository Architecture

### 10.1 Target Structure

```
datamentors/
├── ardia-robotics/                  # MONO-REPO (new, clean, our code)
│   ├── infra/                       # Infrastructure-as-code
│   │   ├── docker/
│   │   │   ├── Dockerfile.workstation   # RTX PRO 6000, CUDA, Python 3.13
│   │   │   ├── Dockerfile.spark         # GB10, ARM64, Grace Hopper
│   │   │   ├── Dockerfile.jetson        # Jetson Orin, real robot
│   │   │   └── Dockerfile.dev          # Local dev (Mac, CPU-only)
│   │   ├── deploy/
│   │   │   ├── spark/                  # GROOT server deployment scripts
│   │   │   ├── workstation/            # Training + sim scripts
│   │   │   └── jetson/                 # Real robot deployment
│   │   └── configs/
│   │       ├── uv.lock.workstation     # Pinned deps per target
│   │       ├── uv.lock.spark
│   │       └── uv.lock.jetson
│   │
│   ├── skills/                      # Robot skill library (BT-governed)
│   │   ├── manipulation/            # VLA-based manipulation skills
│   │   ├── locomotion/              # RL-based locomotion skills
│   │   ├── navigation/              # SLAM + planning
│   │   ├── perception/              # Scene understanding, object detection
│   │   └── registry.py              # Skill registry + composition engine
│   │
│   ├── training/                    # Training pipelines
│   │   ├── vla/                     # VLA fine-tuning (LeRobot-based)
│   │   │   ├── configs/             # Per-model training configs
│   │   │   ├── data_prep/           # Dataset conversion to LeRobotDataset v3
│   │   │   └── scripts/             # Training launch scripts
│   │   ├── rl/                      # RL training
│   │   │   ├── locomotion/          # IsaacLab extensions (PPO, SAC)
│   │   │   ├── vla_posttraining/    # VLA RL post-training (Section 8)
│   │   │   │   ├── grpo/            # SimpleVLA-RL binary GRPO
│   │   │   │   ├── recap/           # PI0-style advantage conditioning
│   │   │   │   ├── rewards/         # Reward functions (binary, CV-based, multi-stage)
│   │   │   │   └── value_fn/        # Value function training for RECAP
│   │   │   ├── configs/             # PPO, GRPO, RECAP configs per robot
│   │   │   └── scripts/
│   │   ├── mimic/                   # Synthetic data generation
│   │   └── teleop/                  # Teleoperation data collection
│   │
│   ├── eval/                        # Evaluation pipelines
│   │   ├── sim/
│   │   │   ├── mujoco/              # MuJoCo + RoboCasa evaluators
│   │   │   ├── isaac/               # Isaac Lab evaluators
│   │   │   └── scenes/              # Custom scene configs
│   │   ├── real/                    # Real-robot eval harness
│   │   └── metrics/                 # Success rate, trajectory quality
│   │
│   ├── data/                        # Dataset management
│   │   ├── converters/              # Format converters (all → LeRobotDataset v3)
│   │   ├── registry/                # Dataset catalog + metadata
│   │   └── validation/              # Data quality checks
│   │
│   ├── controllers/                 # Robot control layer (dual motor paths)
│   │   ├── ros2/                    # ROS2 generic interface (high-level)
│   │   ├── ros2_control/            # ros2_control 200Hz path (team-validated, Appendix I.4)
│   │   ├── unitree/                 # Unitree DDS bridge (low-level)
│   │   ├── sonic/                   # SONIC deployment wrapper
│   │   └── embodiments/             # Per-robot configs (G1, G1-Dex1, etc.)
│   │
│   ├── perception/                  # Sensor stack (Appendix I)
│   │   ├── lidar/                   # Livox MID-360 driver configs
│   │   ├── camera/                  # RealSense D435i configs + static TF
│   │   ├── slam/                    # Fast-LIO configs (g1_mid360.yaml)
│   │   └── cv/                      # g1_perception (HSV detection, RGB-D, TF)
│   │
│   ├── agents/                      # Agentic AI layer
│   │   ├── brain/                   # BT-governed decision making
│   │   ├── exploration/             # Autonomous exploration (Section 9)
│   │   │   ├── sign_reader/         # GroundingDINO + OCR + VLM sign parsing
│   │   │   ├── frontier_scorer/     # VLFM-style VLM frontier scoring
│   │   │   ├── scene_graph/         # Semantic scene graph (rooms, signs, landmarks)
│   │   │   └── exploration_bt/      # Exploration behavior tree
│   │   ├── world_model/             # Scene understanding + reasoning
│   │   ├── planner/                 # Task planning + decomposition
│   │   ├── reasoning/               # Closed-loop action-time reasoning (Section 7)
│   │   │   ├── cv_monitor/          # FoundationPose/YOLO object tracker
│   │   │   ├── value_head/          # Hume-style Q-value for action selection
│   │   │   └── adaptive_horizon/    # AutoHorizon attention-based chunk sizing
│   │   └── safety/                  # Safety kernel + constraints
│   │
│   ├── core/                        # Shared utilities
│   │   ├── robot_configs.py         # DOF layouts, gripper conversions (from dm-isaac-g1)
│   │   ├── remote.py                # SSH/SCP/Docker utilities
│   │   └── types.py                 # Shared type definitions
│   │
│   ├── docs/                        # Migrated from dm-isaac-g1/docs
│   │   ├── architecture/
│   │   ├── training/
│   │   ├── inference/
│   │   ├── evaluation/
│   │   ├── wbc/
│   │   └── issues/
│   │
│   └── pyproject.toml               # uv workspace root
│
├── vendor/                          # UPSTREAM REPOS (git submodules, pinned)
│   ├── Isaac-GR00T/                 # Pin: specific commit hash
│   ├── GR00T-WholeBodyControl/      # Pin: specific commit hash
│   ├── unitree_rl_lab/              # Pin: specific commit hash
│   ├── unitree_sim_isaaclab/        # Pin: specific commit hash
│   ├── lerobot/                     # Pin: specific commit hash
│   └── forge/                       # Pin: dataset format conversion (GR00T ↔ LeRobot ↔ RLDS)
│
├── skills/                          # AI CODING ASSISTANT SKILLS
│   ├── robotics-agent-skills/       # Upstream (git submodule) — Boston Dynamics patterns
│   │   ├── robotics-software-principles/
│   │   ├── ros2/
│   │   ├── robotics-design-patterns/
│   │   ├── robot-perception/
│   │   ├── robotics-testing/
│   │   └── ros1/
│   └── ardia-skills/                # OUR CUSTOM SKILLS (extend the above)
│       ├── ardia-g1-dex1/SKILL.md           # Dex1 gripper conversion, DOF layout
│       ├── ardia-groot-pipeline/SKILL.md    # GR00T N1.6 training/eval patterns
│       ├── ardia-sonic-deploy/SKILL.md      # SONIC deployment for G1
│       └── ardia-infra/SKILL.md             # Container, uv, target deployment
│
├── RAIR/                            # Knowledge base (keep as-is)
├── PapersDistilled/                 # Research reference (keep as-is)
├── HumanoidTraining/                # Training notebooks (keep, evolve)
└── dm-isaac-g1/                     # ARCHIVE (keep for reference, no new work)
```

### 10.2 Key Design Principles

1. **Our code lives in `ardia-robotics/`** — all custom logic, configs, scripts, and documentation
2. **Upstream code lives in `vendor/`** — git submodules, pinned to specific commits, NEVER modified
3. **Imports go one direction** — `ardia-robotics/` imports from `vendor/`, never the reverse
4. **Per-target environments** — Docker + uv lockfiles per deployment target
5. **LeRobotDataset v3 as canonical data format** — all data flows through this standard
6. **Skills as composable units** — every robot capability is a registered skill with contracts

### 10.3 Version Management Strategy

**Upstream sync workflow:**
```
1. NVIDIA releases GR00T N1.7
2. Update vendor/Isaac-GR00T submodule to new commit
3. Run ardia-robotics/ test suite
4. Fix any breakage in OUR code (not upstream)
5. Tag release: ardia-v0.2.0-groot-n1.7
```

**Commit tagging convention:**
- Upstream submodule pins: `vendor: bump Isaac-GR00T to abc123`
- Our code: `ardia: add G1-Dex1 towel folding eval`
- Breaking changes: `BREAKING: migrate training to LeRobot v0.5`

**Version format:** `ardia-v{MAJOR}.{MINOR}.{PATCH}-{upstream-tag}`
- Example: `ardia-v0.1.0-groot-n1.6`, `ardia-v0.2.0-groot-n1.7`

---

## 11. Strategic Decisions (Q1-Q6)

### Q1: Own repos vs importing from 3rd party repos?

**Decision: YES — use the `vendor/` submodule pattern.**

| Approach | Pros | Cons |
|----------|------|------|
| ~~Fork upstream repos~~ | Direct modification | Merge hell on every upstream update |
| ~~Copy code into our repo~~ | Full control | Lose upstream updates entirely |
| **Git submodules in `vendor/`** | **Clean separation, easy updates, clear ownership** | **Submodule UX requires team training** |
| ~~Pip install from upstream~~ | Simple | Version pins break, no local modifications |

**Implementation:**
```bash
# Initial setup
mkdir vendor && cd vendor
git submodule add https://github.com/NVIDIA/Isaac-GR00T.git
git submodule add https://github.com/NVlabs/GR00T-WholeBodyControl.git
git submodule add https://github.com/unitreerobotics/unitree_rl_lab.git
git submodule add https://github.com/unitreerobotics/unitree_sim_isaaclab.git
git submodule add https://github.com/huggingface/lerobot.git

# Pin to specific commits
cd Isaac-GR00T && git checkout <commit> && cd ..
git add . && git commit -m "vendor: pin all upstream repos"

# Update workflow
cd vendor/Isaac-GR00T && git fetch && git checkout <new-commit>
cd ../.. && git add vendor/Isaac-GR00T
git commit -m "vendor: bump Isaac-GR00T to <new-commit>"
```

**Infrastructure repo (`ardia-robotics/infra/`):**
- Docker containers per target with all deps pre-installed
- uv workspaces for Python environment management
- Target-specific lockfiles ensure reproducibility
- `deploy/` scripts handle container → target transfers

---

### Q2: Deploy policies in the same way as upstream repos?

**Decision: Use SONIC for VLA deployment, unitree_rl_lab patterns for RL deployment.**

**For VLA (manipulation + loco-manipulation):**
```
Training (LeRobot)  →  ONNX Export  →  SONIC C++ Stack  →  Robot
                                           │
                                    ┌──────┴──────┐
                                    │ 4 RT threads │
                                    │ Input: 100Hz │
                                    │ Control: 50Hz│
                                    │ Planner: 10Hz│
                                    │ Cmd: 500Hz   │
                                    └─────────────┘
```

SONIC handles:
- ONNX/TensorRT inference on Jetson
- Real-time DDS communication with robot
- Balance, locomotion, and safety monitoring
- Multiple input modes (keyboard, gamepad, ZMQ, ROS2, VR)

**For RL (locomotion):**
- Train in Isaac Lab using `unitree_rl_lab` extension pattern
- Export checkpoint to ONNX
- Deploy via SONIC's control policy interface or unitree_rl_lab's deploy scripts

**For high-level orchestration:**
- ROS2 actions/services for task-level commands
- BT-governed skill execution (from `ai_brain_bt_agentic_rnd.md` architecture)

---

### Q3: Own scenes/3D repo? Standardize on one sim?

**Decision: NO separate scenes repo. Standardize on MuJoCo + RoboCasa for eval, Isaac Lab for training.**

| Sim | Use Case | Reason |
|-----|----------|--------|
| **MuJoCo + RoboCasa** | Manipulation evaluation | Fast, NVIDIA uses it for WBC eval, 38 Dex1 envs already available |
| **Isaac Lab + Isaac Sim** | RL training | GPU-parallelized (thousands of simultaneous sims), domain randomization |
| **MuJoCo standalone** | Quick prototyping | Fastest iteration, no GPU required |

**Custom scenes go in `ardia-robotics/eval/sim/scenes/`** — either as:
- RoboCasa kitchen configs (for manipulation)
- Isaac Lab task extensions (for RL, following `IsaacLabExtensionTemplate`)

**Do NOT create a separate scenes repo.** RoboCasa already gives us 38 Dex1 environments. When we need custom scenes, they're configs within our mono-repo, not standalone assets.

**Scene catalog (already available):**
```
gr00tlocomanip_g1_dex1_sim/LMPnPAppleToPlateDC_G1Dex1_gear_wbc
gr00tlocomanip_g1_dex1_sim/LMPnPCupToCabinetDC_G1Dex1_gear_wbc
gr00tlocomanip_g1_dex1_sim/LMOpenSingleDoorDC_G1Dex1_gear_wbc
... (38 total across PnP, Open, Close, Turn tasks)
```

---

### Q4: Roadmap alignment with OKRs?

**Decision: 4-phase roadmap aligned to platform maturity.**

See [Section 13: Phased Roadmap](#13-phased-roadmap) for the detailed plan.

---

### Q5a: Leverage LeRobot for VLA training?

**Decision: Absolutely YES. This is the clearest strategic win.**

**Rationale:**

1. **GR00T N1.6 is natively integrated** into LeRobot — NVIDIA co-developed the integration
2. **LeRobotDataset v3.0** is the de facto data standard — NVIDIA's own workflows produce it
3. **Model interchangeability** — same dataset, same pipeline, swap between GR00T N1.6, PI0, SmolVLA, X-VLA
4. **Community data flywheel** — 3,000+ new datasets per month on HuggingFace
5. **Smaller models available** — SmolVLA (450M) trainable on single GPU for rapid iteration

**Training pipeline architecture:**
```
HuggingFace Hub ──────────────────────────────────────┐
(Unitree/NVIDIA datasets)                             │
                                                      ▼
Our Teleop Data ──▶ converters/ ──▶ LeRobotDataset v3 ──▶ LeRobot Training
                                        │                      │
GR00T-Mimic ──────────────────────────▶─┘                      ▼
(synthetic trajectories)                               ONNX Checkpoint
                                                           │
                                                    ┌──────┴──────┐
                                                    │  Evaluate   │
                                                    │  (MuJoCo)   │
                                                    └──────┬──────┘
                                                           │
                                                    ┌──────┴──────┐
                                                    │  Deploy     │
                                                    │  (SONIC)    │
                                                    └─────────────┘
```

---

### Q5b: Every robot SDK vs ROS2?

**Decision: ROS2 for high-level orchestration, DDS for low-level motor control.**

```
┌─────────────────────────────────────────────────────────┐
│                    AGENTIC LAYER                        │
│  BT-governed task planning, skill composition           │
│  Interface: ROS2 Actions + Services                     │
└───────────────────────┬─────────────────────────────────┘
                        │ ROS2
┌───────────────────────▼─────────────────────────────────┐
│                 ORCHESTRATION LAYER                      │
│  Nav2 (navigation), Fast-LIO (SLAM), RViz (viz)         │
│  Skill execution, state management                      │
│  Interface: ROS2 Topics + Actions                       │
└───────────────────────┬─────────────────────────────────┘
                        │ ROS2/DDS bridge
┌───────────────────────▼─────────────────────────────────┐
│              MOTOR CONTROL LAYER                        │
│  SONIC C++ stack (4 RT threads)                         │
│  Direct CycloneDDS to robot hardware                    │
│  50Hz control loop, 500Hz command publishing            │
│  Interface: DDS (rt/lowcmd, rt/lowstate)                │
└─────────────────────────────────────────────────────────┘
```

**Two validated motor control paths (keep BOTH):**

| Path | Rate | Latency | Status | Best For |
|------|------|---------|--------|----------|
| **SONIC** (C++) | 50Hz control, 500Hz cmd | <1ms | Production-ready (NVIDIA) | SONIC-integrated VLA deployment |
| **ros2_control** (C++) | 200Hz | 2-5ms | Team-validated (Appendix I.4) | RL policy deployment, FSM switching |

The training team has deep expertise in ros2_control and traditional ROS2/RL methods. SONIC represents the cutting-edge NVIDIA path. Both should be maintained as first-class deployment options — ros2_control for RL locomotion policies where the team has operational experience, and SONIC for VLA-driven whole-body control.

**Why ROS2 at the orchestration level:**
- Nav2 for navigation (free, well-tested)
- Fast-LIO for SLAM (validated with Livox MID-360, see Appendix I.3)
- RViz for visualization
- Robot-agnostic: if we add a second robot type, ROS2 interface stays the same
- Our agentic layer speaks ROS2, not robot-specific DDS topics

**Unitree G1 SDK access note:**
- G1 Basic ($21,600): No SDK access, cannot develop
- G1 EDU Standard ($43,500): Minimum for development — Python/C++ APIs, ROS2, Jetson Orin
- Both CycloneDDS (Unitree native) and ROS2 use DDS, so they interoperate natively

---

### Q6: Risk from new models (Sonic, EgoScale, new developments)?

**Decision: Medium risk, manageable with correct abstractions.**

**The field is converging on standards:**

| Layer | Convergence Point | Our Alignment |
|-------|-------------------|---------------|
| Data format | LeRobotDataset v3 | Aligned (Q5a decision) |
| Architecture | System 2 (VLM) + System 1 (diffusion/flow) | Aligned (GR00T N1.6) |
| Control | Whole-body controller (SONIC-type) | Aligned (Q2 decision) |
| Interface | ROS2 for orchestration | Aligned (Q5b decision) |

**What to watch, not build around yet:**

| Development | Impact | Action |
|-------------|--------|--------|
| **EgoScale** (human video pre-training) | Could eliminate need for teleoperation | Monitor; our LeRobot pipeline is compatible |
| **Gemini Robotics 1.5** (Google) | Cloud-edge split VLA | Monitor; different ecosystem, not open |
| **PI0.6** (Physical Intelligence) | RL fine-tuning for VLAs | Already in LeRobot; can switch when ready |
| **SmolVLA** (HuggingFace) | 450M param, single GPU | Test for rapid iteration scenarios |
| **DreamZero / Cosmos Policy** | World model + policy learning | Research track; ICLR 2026 trending |
| **Newton Physics Engine** | Better sim-to-real transfer | Will be integrated into Isaac Lab |
| **Discrete Diffusion VLAs** | Better action precision | Research track; compatible with LeRobot |

**Biggest risk is NOT new models** (they converge on similar interfaces) but **NVIDIA reorganizing their repos** (which they do frequently). The `vendor/` submodule strategy protects us here because we pin to specific commits and only update when ready.

**Risk mitigation strategy:**
1. Never modify upstream code directly
2. Keep our customizations thin and well-isolated
3. Use LeRobotDataset v3 as canonical format (model-agnostic)
4. Test new models as drop-in replacements before committing
5. Monitor NVIDIA release notes weekly (RAIR already does this)

---

## 12. Technology Stack Decisions

### 12.1 Consolidated Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Package management** | uv | Fastest Python package manager, workspace support, lockfiles per target |
| **Containers** | Docker | Reproducible environments across workstation/spark/jetson |
| **VLA Training** | LeRobot | Standard, GR00T N1.6 integrated, model-agnostic |
| **RL Training** | Isaac Lab + unitree_rl_lab | GPU-parallelized, official Unitree support |
| **Data format** | LeRobotDataset v3.0 | Industry standard, NVIDIA-compatible |
| **Eval (manipulation)** | MuJoCo + RoboCasa | Fast, 38 Dex1 envs, NVIDIA WBC compatible |
| **Eval (RL)** | Isaac Lab | GPU-parallelized, domain randomization |
| **WBC/Motor control (SONIC)** | SONIC (C++, ONNX/TensorRT) | Production-ready, 4 RT threads, single process |
| **WBC/Motor control (ros2_control)** | ros2_control + UnitreeSdk2 plugin (C++, 200Hz) | Team-validated, ONNX inference, joystick FSM switching (Appendix I.4) |
| **High-level control** | ROS2 (Humble) | Nav2, robot-agnostic |
| **SLAM** | Fast-LIO (LiDAR-inertial odometry) | Validated with Livox MID-360, not SLAM Toolbox (Appendix I.3) |
| **LiDAR** | Livox MID-360 (192.168.123.120) | 360°x59° FOV, 200K pts/s, 10Hz (Appendix I.1) |
| **Depth Camera** | Intel RealSense D435i | 640x480@30fps RGB+D, mounted 1.2m, 5cm forward (Appendix I.2) |
| **CV Perception** | g1_perception ROS2 package | HSV detection + RGB-D 3D positioning + TF broadcasting (Appendix I.5) |
| **Low-level control** | CycloneDDS (Unitree SDK2) | Deterministic latency, native to G1 |
| **Orchestration** | Behaviour Trees | Deterministic governance, auditable execution |
| **VLM reasoning** | Cosmos Reason (via GR00T N1.6) / Claude | Task planning, scene understanding |
| **Action-time CV** | FoundationPose / YOLO + AutoHorizon | Closed-loop manipulation monitoring (Section 7) |
| **RL post-training** | SimpleVLA-RL (GRPO) → RECAP (Phase 2) | VLA quality improvement (Section 8) |
| **Value function** | Hume-style Q-value head | Action candidate selection |
| **Sign detection** | GroundingDINO + PaddleOCR | Environmental text reading for exploration (Section 9) |
| **Frontier scoring** | VLFM (VLM + occupancy grid) | Intelligent exploration in unknown environments |
| **Scene graph** | SayNav-style semantic graph | Exploration memory (visited rooms, signs, landmarks) |
| **Version control** | Git + submodules | Clean upstream/downstream separation |
| **AI dev assistance** | robotics-agent-skills (SKILL.md) | Production-grade robotics patterns auto-injected into Claude Code |
| **Data format conversion** | forge (arpitg1304) | GR00T ↔ LeRobot ↔ RLDS ↔ Zarr ↔ HDF5 ↔ Rosbag |
| **CI/CD** | GitHub Actions | Automated testing per target |

### 12.2 Key Dependencies and Pinned Versions

| Dependency | Version | Notes |
|-----------|---------|-------|
| Python | 3.11 (training), 3.13 (workstation container) | 3.13 requires dataclass fixes |
| transformers | 4.51.3 | **Only working version** for GR00T fine-tuning |
| lerobot | v0.4.3+ | LeRobotDataset v3 support |
| Isaac Lab | 2.3+ | GR00T-Dexterity integration |
| MuJoCo | 3.2+ | Latest physics improvements |
| CUDA | 12.x | Required for Isaac Lab + TensorRT |
| ONNX Runtime | 1.18+ | SONIC deployment |
| TensorRT | 10.x | Jetson deployment optimization |
| CycloneDDS | 0.10.2 | Unitree G1 native |
| ROS2 | Humble | LTS, Unitree supported |

---

## 13. Phased Roadmap

### Phase 1: Foundation (Weeks 1-3)

**Goal:** Clean repo setup, migrate knowledge, get VLA eval working

| Task | Details |
|------|---------|
| Create `ardia-robotics/` repo | Mono-repo structure with uv workspace |
| Set up `vendor/` submodules | Pin Isaac-GR00T, WBC, unitree_rl_lab, unitree_sim_isaaclab, lerobot |
| Migrate dm-isaac-g1 docs | Organize into `docs/` subdirectories |
| Migrate core utilities | `robot_configs.py`, Dex1 conversions, remote.py |
| Create Docker targets | workstation, spark, jetson, dev Dockerfiles |
| Validate MuJoCo eval | Run PnP Apple task through new repo structure |
| Set up robotics-agent-skills | Install upstream skills + create Ardia-specific SKILL.md files for team |
| Evaluate `forge` for data pipeline | Test GR00T ↔ LeRobot format conversion with our hospitality datasets |
| Archive dm-isaac-g1 | Mark as read-only reference |

### Phase 2: Training Pipeline (Weeks 4-7)

**Goal:** LeRobot-based VLA training with clean data pipeline

| Task | Details |
|------|---------|
| LeRobot integration | Set up training configs for GR00T N1.6 fine-tuning |
| Data pipeline | Converters for Unitree hospitality datasets → LeRobotDataset v3 |
| Dataset registry | Catalog existing datasets (HuggingFace Unitree + NVIDIA) |
| Training validation | Reproduce hospitality-7ds model training through new pipeline |
| SmolVLA experiment | Test 450M model for rapid iteration (single GPU) |
| Evaluation framework | Automated eval across MuJoCo RoboCasa scenes |
| **Sign Reader prototype** | GroundingDINO + PaddleOCR + VLM sign parsing pipeline on RealSense D435i (Section 9) |

### Phase 3: Deployment + Closed-Loop Reasoning (Weeks 8-12)

**Goal:** SONIC deployment on real G1, ROS2 orchestration, closed-loop CV monitoring

| Task | Details |
|------|---------|
| SONIC integration | Build SONIC C++ stack for G1 with Dex1 |
| Jetson deployment | ONNX/TensorRT optimization for onboard inference |
| ROS2 bridge | High-level ROS2 → SONIC DDS bridge |
| Real robot eval | PnP task on physical G1 |
| Nav2 integration | Basic navigation + obstacle avoidance |
| Safety kernel | Joint velocity limits, restricted zones, e-stop |
| **CV action monitor** | Deploy FoundationPose/YOLO tracker during manipulation for grasp failure detection (Section 7) |
| **AutoHorizon adaptive chunks** | Implement attention-based dynamic chunk length — reduce blind execution on uncertain actions |
| **Nav2 + VLM semantic replan** | Add VLM-based semantic evaluation layer on top of Nav2 costmap navigation |
| **VLFM frontier scoring** | VLM-scored frontier exploration on top of Fast-LIO occupancy grid (Section 9) |
| **Exploration memory** | Scene graph tracking visited rooms, seen signs, dead ends — prevents exploration loops |

### Phase 3.5: RL Post-Training (Weeks 10-14)

**Goal:** Move from demo-quality to production-quality VLA via RL (Section 8)

| Task | Details |
|------|---------|
| **Binary GRPO in sim** | Run autonomous VLA rollouts in MuJoCo/RoboCasa, score success/failure, apply SimpleVLA-RL GRPO |
| **Reward pipeline** | Wire CV action monitor (Phase 3) outputs as reward signals for RL training |
| **Sim RL loop** | Automated: rollout → score → train → rollout cycle on workstation |
| **Measure improvement** | Track success rate and throughput on PnP, Open, Close tasks before/after RL |

### Phase 4: Agentic Layer + RECAP (Weeks 13+)

**Goal:** BT-governed skill composition, task planning, real-robot RL

| Task | Details |
|------|---------|
| Skill registry | Define manipulation, locomotion, navigation skills |
| BT framework | Implement behaviour tree governance from `ai_brain_bt_agentic_rnd.md` |
| World model | Scene understanding via VLM (Cosmos Reason / Claude) |
| Task planner | Decompose natural language instructions into skill sequences |
| Integration with Ardia | Connect robotics skills to Ardia platform agent framework |
| Multi-robot coordination | Extend ROS2 interface for fleet management |
| **Hume Q-value head** | Train value function for action selection — generate N candidates, select highest value (Section 7) |
| **RECAP on real G1** | Collect autonomous rollouts + human corrections on physical G1; apply RECAP advantage conditioning |
| **Continuous improvement loop** | Deploy → collect experience → RL train → redeploy — targeting >2x throughput (Section 8) |
| **Full exploration agent** | Scene graph + LLM planner for autonomous exploration in unknown venues (Section 9) |
| **Human interaction fallback** | If stuck >N minutes, approach person and ask for directions |

---

## 14. Risk Assessment

### 14.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NVIDIA reorganizes GR00T repos | High | Medium | Submodule pins; never modify upstream |
| Python 3.13 compatibility breaks | Medium | High | Dual Python version support in Docker |
| SONIC doesn't support Dex1 grippers | Medium | High | Keep decoupled WBC as fallback |
| LeRobot breaking changes (v0.5) | Medium | Medium | Pin version, migrate when stable |
| transformers version conflict | High | High | Pin to 4.51.3, isolate in training container |
| Sim-to-real gap for Dex1 | High | High | EgoScale pre-training, domain randomization |
| RL training instability for VLAs | Medium | Medium | Start with binary GRPO (simplest); RECAP as fallback |
| CV monitor latency too high for real-time | Low | Medium | Use YOLO (2ms) not FoundationPose (50ms) for speed-critical |
| Reward signal too sparse for RL | Medium | Medium | Multi-stage rewards via TGRPO; dense CV-based rewards |
| VLM exploration latency in real-time | Medium | Medium | Cache sign readings; frontier scoring runs at 1-2Hz not 10Hz |
| Exploration loops (revisiting same areas) | Medium | Medium | Scene graph memory + visited-frontier tracking |

### 14.2 Strategic Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GR00T becomes closed-source | Low | Critical | LeRobot pipeline works with PI0, SmolVLA as alternatives |
| Unitree discontinues G1 SDK | Low | High | ROS2 interface is robot-agnostic |
| Better VLA architecture emerges | High | Low | LeRobot abstracts model choice; swap easily |
| Team scaling challenges | Medium | Medium | Mono-repo + docs + automated environments |
| Competitor (Figure, Google) leapfrogs | Medium | Low | We build applications, not foundation models |

### 14.3 LMTowelFoldDC Environment Failure

This environment failed to load because:
1. **Deformable object simulation** (cloth physics) requires specific MuJoCo versions and RoboCasa configurations not available in all setups
2. The RoboCasa Dex1 auto-registration creates entries for **all 38 tasks** but not all have compatible physics for every robot configuration
3. Towel folding with Dex1 prismatic grippers is particularly challenging — the [-0.02, 0.024]m range cannot generate sufficient grasping force for soft objects
4. **Recommendation:** Focus on rigid-object tasks (PnP, Open, Close) for Dex1 eval; towel folding needs Dex3 or custom gripper physics

---

## 15. Appendices

### Appendix A: Dex1 Gripper Value Conversion Reference

```python
# Physical range: [-0.02, 0.024] meters (prismatic)
# Training range: [0.0, 5.4] (INVERTED)
# -0.02m (open) → 5.4 (training)
# 0.024m (closed) → 0.0 (training)

def physical_to_training(physical_value):
    """Convert physical gripper position to training space."""
    # Normalize to [0, 1] then scale and invert
    normalized = (physical_value - (-0.02)) / (0.024 - (-0.02))
    return 5.4 * (1.0 - normalized)

def training_to_physical(training_value):
    """Convert training space value to physical gripper position."""
    normalized = 1.0 - (training_value / 5.4)
    return normalized * (0.024 - (-0.02)) + (-0.02)
```

**Critical rule:** Any code touching Dex1 observations must convert physical → training. Any code touching Dex1 actions must convert training → physical.

### Appendix B: UNITREE_G1 DOF Layout

```
STATE (31 DOF):
  Legs:     12 DOF (6 per leg: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll)
  Waist:     3 DOF (yaw, roll, pitch)
  Arms:     14 DOF (7 per arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_yaw, wrist_roll, wrist_pitch)
  Grippers:  2 DOF (1 per hand, collapsed from 2 physical DOF for Dex1)

ACTION (23 DOF):
  Arms:     14 DOF (RELATIVE deltas)
  Grippers:  2 DOF (ABSOLUTE targets)
  Waist:     3 DOF (ABSOLUTE targets)
  Height:    1 DOF (ABSOLUTE target)
  Navigation: 3 DOF (ABSOLUTE: x_vel, y_vel, yaw_vel)
```

### Appendix C: Wrist Joint Reordering

```
GROOT model output:  [yaw, roll, pitch]
WBC/MuJoCo expects:  [roll, pitch, yaw]

Remapping: groot[0]→wbc[2], groot[1]→wbc[0], groot[2]→wbc[1]
```

### Appendix D: Key File Locations (Current → Proposed)

| Current Location | Proposed Location | Purpose |
|------------------|-------------------|---------|
| `dm-isaac-g1/src/dm_isaac_g1/core/robot_configs.py` | `ardia-robotics/core/robot_configs.py` | DOF layouts, gripper conversions |
| `dm-isaac-g1/docs/` (26 files) | `ardia-robotics/docs/` (organized) | All documentation |
| `GR00T-WholeBodyControl/decoupled_wbc/control/utils/n1_utils.py` | `vendor/GR00T-WholeBodyControl/` (submodule) | WBC wrapper, Dex1 conversion |
| `groot_temp/gr00t/eval/rollout_policy.py` | `vendor/Isaac-GR00T/` (submodule) | GROOT eval orchestrator |
| `dm-groot-inference/` | `ardia-robotics/infra/deploy/spark/` | GROOT server scripts |
| `RoboticsReposResearch/04_Training/Isaac-GR00T/` | `vendor/Isaac-GR00T/` (submodule) | Unified single upstream reference |

### Appendix E: GR00T Ecosystem Component Map

| Component | Purpose | Repository | Status |
|-----------|---------|-----------|--------|
| GR00T-Teleop | Data collection via Apple Vision Pro | Isaac Sim + CloudXR | Production |
| GR00T-Mimic | Synthetic trajectory generation | Isaac Lab + SkillMimicGen | Production |
| GR00T-Gen | Diverse sim environment generation | Isaac Lab + Cosmos Transfer | Production |
| GR00T-Dexterity | Dexterous grasping policies | Isaac Lab 2.3 | Production |
| GR00T-Mobility | Navigation + COMPASS | Isaac Lab | Production |
| GR00T-Control | SONIC whole-body controller | NVlabs/GR00T-WholeBodyControl | Production |
| GR00T N1.6 | Foundation VLA model (3B params) | nvidia/GR00T-N1.6-3B (HuggingFace) | Production |
| Cosmos Reason | VLM for physical reasoning | nvidia/Cosmos-Reason-2B | Production |
| Newton | GPU-accelerated physics engine | Linux Foundation (beta) | Beta |

### Appendix F: HuggingFace Datasets of Interest

**Unitree Official:**
- `unitreerobotics/g1_hospitality_*` (7 tasks, 1,400 episodes total)
- `unitreerobotics/g1_fold_towel` (200 episodes)

**NVIDIA Official:**
- `nvidia/GR00T-N1.6-G1-PnPAppleToPlate` (fine-tuned checkpoint)
- `nvidia/PhysicalAI-Robotics-GR00T-Teleop-*` (teleop demo data)

**Community (LeRobot):**
- 3,000+ new datasets/month being uploaded
- Cross-embodiment datasets: DROID, Robomind, Agibot, OXE

### Appendix G: Robotics Agent Skills — Integration Reference

**Installation for Claude Code:**
```bash
# Option 1: Place in Claude Code skills directory
cp -r robotics-agent-skills/skills/* /mnt/skills/user/

# Option 2: Reference via CLAUDE.md project instructions
# Add to .claude/projects/<project>/memory/MEMORY.md:
# "Load skills from /path/to/robotics-agent-skills/skills/ when writing ROS2 or robotics code"
```

**Custom Ardia Skills to Create:**

| Skill File | Triggers | Content |
|------------|----------|---------|
| `ardia-g1-dex1/SKILL.md` | "Dex1", "gripper conversion", "G1 DOF", "wrist remap" | Dex1 value conversion code, 31→23 DOF mapping, wrist ordering |
| `ardia-groot-pipeline/SKILL.md` | "GROOT training", "fine-tune VLA", "LeRobot GR00T" | Training configs, transformers pin, batch size, checkpoint management |
| `ardia-sonic-deploy/SKILL.md` | "SONIC deploy", "WBC G1", "motor control" | SONIC C++ build, ONNX export, DDS threading model |
| `ardia-infra/SKILL.md` | "deploy to spark", "workstation container", "Jetson deploy" | Docker targets, uv lockfiles, scp/docker cp patterns |

**Author's Related Repos to Evaluate:**

| Repo | URL | Priority |
|------|-----|----------|
| forge | github.com/arpitg1304/forge | **P0** — Data format conversion (GR00T ↔ LeRobot) |
| ros-time-machine | github.com/arpitg1304/ros-time-machine | P1 — Event-triggered ROS2 recording |
| tessera | github.com/arpitg1304/tessera | P2 — Episode embedding visualization |

### Appendix H: References and Sources

**NVIDIA GR00T:**
- [Isaac GR00T Developer Page](https://developer.nvidia.com/isaac/gr00t)
- [GR00T N1.6 on HuggingFace](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [GR00T WholeBodyControl GitHub](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [GR00T in LeRobot Blog](https://huggingface.co/blog/nvidia/nvidia-isaac-gr00t-in-lerobot)
- [SONIC Project Page](https://nvlabs.github.io/SONIC/)
- [SONIC Paper (arXiv:2511.07820)](https://arxiv.org/abs/2511.07820)
- [GR00T N1 Whitepaper](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T_1_Whitepaper.pdf)

**LeRobot:**
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot v0.4.0 Release](https://huggingface.co/blog/lerobot-release-v040)
- [LeRobotDataset v3.0 Documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [SmolVLA](https://huggingface.co/blog/smolvla)

**Competing VLAs:**
- [Physical Intelligence PI0.5](https://www.physicalintelligence.company/blog/pi05)
- [Google Gemini Robotics](https://deepmind.google/models/gemini-robotics/)
- [Figure AI Helix](https://www.figure.ai/helix)
- [VLA Survey](https://vla-survey.github.io/)
- [ICLR 2026 VLA Trends](https://mbreuss.github.io/blog_post_iclr_26_vla.html)

**Robotics Agent Skills (Arpit Gupta / Boston Dynamics):**
- [robotics-agent-skills GitHub](https://github.com/arpitg1304/robotics-agent-skills)
- [forge (dataset format conversion) GitHub](https://github.com/arpitg1304/forge)
- [ros-time-machine GitHub](https://github.com/arpitg1304/ros-time-machine)
- [tessera (episode embedding visualization) GitHub](https://github.com/arpitg1304/tessera)

**Unitree:**
- [unitree_rl_lab GitHub](https://github.com/unitreerobotics/unitree_rl_lab)
- [unitree_sim_isaaclab GitHub](https://github.com/unitreerobotics/unitree_sim_isaaclab)
- [Unitree G1 Developer Docs](https://support.unitree.com/home/en/G1_developer)
- [Unitree ROS2 GitHub](https://github.com/unitreerobotics/unitree_ros2)

**PI0 Family (Physical Intelligence):**
- [pi0 paper (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [pi0.5 paper (arXiv:2504.16054)](https://arxiv.org/abs/2504.16054)
- [pi*0.6 paper (arXiv:2511.14759)](https://arxiv.org/abs/2511.14759)
- [PI0 Blog](https://www.physicalintelligence.company/blog/pi0)
- [PI0.5 Blog](https://www.pi.website/blog/pi05)
- [PI*0.6 Blog](https://www.pi.website/blog/pistar06)
- [OpenPI (code)](https://github.com/Physical-Intelligence/openpi)
- [Knowledge Insulation paper](https://arxiv.org/html/2505.23705v1)

**RL for VLAs:**
- [SimpleVLA-RL (GRPO for VLAs)](https://arxiv.org/abs/2509.09674) — [Code](https://github.com/PRIME-RL/SimpleVLA-RL)
- [TGRPO (trajectory-level GRPO)](https://arxiv.org/html/2506.08440v1)
- [VLA-RL (process reward model)](https://arxiv.org/abs/2505.18719)
- [VLA-RFT (world model + GRPO)](https://openreview.net/forum?id=Jaut99EHeu)
- [FPO (flow policy optimization)](https://arxiv.org/abs/2510.09976)
- [SRPO (self-referential)](https://arxiv.org/abs/2511.15605)

**Closed-Loop Reasoning:**
- [Hume VLA (dual-system, Q-value)](https://arxiv.org/abs/2505.21432) — [Code](https://github.com/hume-vla/hume)
- [CoT-VLA (visual chain-of-thought)](https://cot-vla.github.io/)
- [AutoHorizon (adaptive execution horizon)](https://arxiv.org/abs/2602.21445)
- [World-VLA-Loop (co-evolving world model + VLA)](https://arxiv.org/abs/2602.06508)
- [NVIDIA Cosmos Reason 2](https://huggingface.co/blog/nvidia/nvidia-cosmos-reason-2-brings-advanced-reasoning)
- [Google RT-2](https://arxiv.org/abs/2307.15818)

**MEM (Multi-Scale Embodied Memory):**
- [MEM Research Page](https://pi.website/research/memory)
- [MEM Paper (PDF)](https://www.pi.website/download/Mem.pdf)
- Authors: Torne, Pertsch, Walke, Vedder, Nair, Ichter, Ren, Wang, Tang, Stachowicz, Dhabalia, Equi, Vuong, Springenberg, Levine, Finn, Driess (PI, Stanford, UC Berkeley, MIT)

**DimOS (Dimensional OS):**
- [DimOS GitHub](https://github.com/dimensionalOS/dimos)
- Python-first robotics SDK with agent-native architecture, VLM perception, spatial memory, and navigation
- Supports Unitree G1 (beta), Go2 (stable), drones, and arms
- Relevant for: perception stack patterns, spatial memory, VLM-driven navigation, agent composition

**Roboflow (Robotics Vision AI):**
- [Roboflow Robotics](https://roboflow.com/industries/robotics)
- Edge + cloud vision deployment for robotics (object detection, pose estimation, visual servoing)
- Relevant for: perception pipeline tooling, object tracking during manipulation, CV monitoring at action-time

**PapersDistilled:**
- [PI0 Family Distillation](../PapersDistilled/PI0/PI0.md)
- [MEM Distillation](../PapersDistilled/MEM/MEM.md)
- [Paper Connections & Relationships](../PapersDistilled/PaperConnections.md)

### Appendix I: Real-Robot Sensor & Control Stack

> Extracted from HumanoidTraining Units 10-18. This appendix documents the actual hardware stack our training team has deployed and validated on physical G1 robots. The team's strength is in these traditional ROS2/RL methods — the strategy should build on this foundation rather than replace it.

#### I.1 Livox MID-360 LiDAR

| Parameter | Value |
|-----------|-------|
| LiDAR IP | 192.168.123.120 |
| Host IP | 192.168.123.164 |
| FOV | 360° x 59° |
| Point Rate | 200,000 pts/s |
| Update Rate | 10 Hz |
| Range | 0.1 - 200m |
| Mounting | Roll = 180° (inverted) |
| ROS2 Topics | `/livox/lidar` (PointCloud2), `/livox/imu` |

**Port Configuration (MID360_config.json):**
```json
{
  "MID360": {
    "lidar_net_info": {
      "cmd_data_port": 56100,
      "push_msg_port": 56200,
      "point_data_port": 56300,
      "imu_data_port": 56400,
      "log_data_port": 56500
    },
    "host_net_info": {
      "cmd_data_ip": "192.168.123.164",
      "cmd_data_port": 56101,
      "push_msg_ip": "192.168.123.164",
      "push_msg_port": 56201,
      "point_data_ip": "192.168.123.164",
      "point_data_port": 56301,
      "imu_data_ip": "192.168.123.164",
      "imu_data_port": 56401
    }
  },
  "lidar_configs": [{
    "ip": "192.168.123.120",
    "extrinsic_parameter": { "roll": 180.0, "pitch": 0.0, "yaw": 0.0 }
  }]
}
```

**Launch:**
```bash
source ~/ws_livox_ros2/install/setup.bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

#### I.2 Intel RealSense D435i Camera

| Parameter | Value |
|-----------|-------|
| RGB Resolution | 640x480 @ 30 FPS |
| Depth Resolution | 640x480 @ 30 FPS |
| IMU Gyro Rate | 200 Hz |
| IMU Accel Rate | 250 Hz |
| Mount Height | 1.2m above base_link |
| Mount Offset | 5cm forward |
| Downward Tilt | ~10° (0.1745 rad) |
| Clip Distance | 4.0m |

**Static Transform (base_link → camera_link):**
```bash
ros2 run tf2_ros static_transform_publisher \
    0.05 0 1.2 0 0.1745 0 base_link camera_link
```

**Key Topics:**
- `/camera/color/image_raw` (30 Hz)
- `/camera/depth/image_rect_raw` (30 Hz)
- `/camera/depth/color/points` (PointCloud2, 30 Hz)
- `/camera/imu` (200 Hz)
- `/camera/color/camera_info` (calibration)

#### I.3 Fast-LIO SLAM (NOT SLAM Toolbox)

The actual SLAM system deployed on G1 is **Fast-LIO**, not SLAM Toolbox as previously stated in Section 11 (Q5b). Fast-LIO provides tightly-coupled LiDAR-inertial odometry optimized for the Livox MID-360.

**Key Configuration (g1_mid360.yaml):**
```yaml
common:
  lid_topic: "/livox/lidar"
  imu_topic: "/livox/imu"
preprocess:
  lidar_type: 4    # Generic pointcloud
  scan_rate: 10    # Hz
mapping:
  acc_cov: 0.1
  gyr_cov: 0.001          # Low — trust gyroscope
  b_gyr_cov: 0.0000001    # Extremely low — prevent drift
  fov_degree: 360.0
  det_range: 50.0
```

**Transform Chain:**
```
map → odom → camera_init → body → motion_link → base_link → imu_link → livox_frame
```

**Launch:**
```bash
ros2 launch fast_lio mapping.launch.py config_file:=g1_mid360.yaml rviz:=false
```

#### I.4 ros2_control Real-Time Interface

ros2_control provides a **parallel control path alongside SONIC**, running at 200Hz via direct C++ function calls. This is the team's validated production deployment method and should be maintained as a first-class option.

| Parameter | Value |
|-----------|-------|
| Control Loop | 200 Hz (5ms cycle) |
| Sensor-to-Actuator Latency | 2-5 ms |
| Hardware Plugin | `unitree/UnitreeSdk2` |
| Default Stiffness (kp) | 80 |
| Default Damping (kd) | 3 |
| Motor Count | 29 joints |

**Control Loop (C++ pseudocode):**
```cpp
// 200 Hz real-time loop
while(running) {
  hardware_interface->read();        // DDS receive (~1ms)
  walking_controller->update();      // ONNX policy inference (~2ms)
  hardware_interface->write();       // DDS transmit (~1ms)
}
```

**Joint Interfaces per Motor:**
- Command: stiffness, damping
- State: position (rad), velocity (rad/s), torque (Nm)

**IMU Sensor Interfaces (base_imu):**
- orientation (quaternion x,y,z,w)
- angular_velocity (x,y,z)
- linear_acceleration (x,y,z)

**Controller Activation (via Joystick):**
- L1 + A: Activate standby_controller (safe mode)
- R1 + A: Activate walking_controller (policy execution)
- B: Emergency stop (damping mode)

**Controller Lifecycle:** UNCONFIGURED → INACTIVE → ACTIVE

#### I.5 g1_perception ROS2 Package

Existing CV pipeline for object detection, already validated by the training team.

**HSV Red Bag Detection:**
```python
# Red color in HSV space (two ranges for red wrap-around)
mask1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
mask = mask1 | mask2
MIN_CONTOUR_AREA = 500  # pixels
```

**Pixel-to-3D Conversion (Pinhole Camera Model):**
```python
def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    Z = depth / 1000.0  # mm → m
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return (X, Y, Z)
```

**Depth Sampling:** 5x5 grid around centroid, filter zeros, compute min/max/avg.

**TF Broadcasting (with Y-axis inversion for optical → ROS frame):**
```python
t.header.frame_id = "camera_color_optical_frame"
t.child_frame_id = "reflex_bag"
t.transform.translation.y = -centroid_cam[1]  # INVERTED for ROS convention
```

**Published Topics:**
- `/bag_debug_image` (RGB with overlays)
- `/depth_debug_image` (colorized depth with markers)
- `/tf` (transform broadcasts)

**Processing Rate:** 1 Hz (configurable to 10 Hz for real-time)

#### I.6 Nav2 Docker Deployment

**Docker Configuration:**
```yaml
services:
  g1_nav2_humble:
    image: theconstructai/g1_nav2:latest
    privileged: true
    network_mode: host     # Required for DDS communication
    environment:
      - ROS_DOMAIN_ID=0    # Isolated from robot SDK (domain 1)
```

**Domain Isolation Strategy:**
- ROS_DOMAIN_ID=0: Nav2 stack
- ROS_DOMAIN_ID=1: Robot SDK, perception, sport mode
- Prevents DDS topic conflicts between navigation and motor control

**Local Costmap:**
```yaml
update_frequency: 10.0    # Hz
rolling_window: True
width: 5                  # 5m x 5m
height: 5
resolution: 0.10          # 10cm/cell
robot_radius: 0.25        # G1 radius
obstacle_max_range: 2.5
obstacle_min_range: 0.3
```

**DWB Local Planner Velocity Limits (G1):**
```yaml
min_vel_x: -0.3    # m/s
max_vel_x: 0.3
min_vel_y: -0.3
max_vel_y: 0.3
max_vel_theta: 2.0  # rad/s
```

**Goal Tolerances:**
- Position: 30cm (xy_goal_tolerance: 0.3)
- Heading: ~17° (yaw_goal_tolerance: 0.3)

#### I.7 Zenoh Bridge for Remote Visualization

Enables remote RViz access to G1 sensor data:
```bash
# On G1:
cd zenoh_g1/docker && docker-compose -f docker-compose-g1.yaml up -d
# On local PC:
cd zenoh_g1/docker && docker-compose -f docker-compose-local.yaml up -d
```

#### I.8 Complete System Startup Sequence (6 Terminals)

```bash
# T1: Sport mode cmd_vel driver
unset CYCLONEDDS_HOME
ros2 run g1_sport_mode_ros g1_sport_multiprocess.py --ros-args -p interface:=eth0

# T2: Livox LiDAR
ros2 launch livox_ros_driver2 pointcloud2_MID360_launch.py

# T3: Fast-LIO odometry
ros2 launch fast_lio odometry_only.launch.py use_sim_time:=true rviz:=false

# T4: Open3D global localization
ros2 launch open3d_global_localization global_localization_g1.launch.py

# T5: Nav2 (Docker)
cd ~/git-repo/g1_nav2/docker && docker compose up

# T6: Navigation goals
ros2 topic pub /goal_pose geometry_msgs/PoseStamped \
    '{header: {frame_id: "map"}, pose: {position: {x: 2.0, y: 1.0}, orientation: {w: 1.0}}}'
```

### Appendix J: RL Locomotion Training Recipe

> Extracted from HumanoidTraining Units 7-8. These are the validated PPO hyperparameters and reward functions used by the training team for G1 locomotion policies.

#### J.1 PPO Hyperparameters

```python
# Algorithm
clip_param = 0.2
entropy_coef = 0.01
num_learning_epochs = 5
num_mini_batches = 4
learning_rate = 1.0e-3
schedule = "adaptive"
gamma = 0.99
lam = 0.95
desired_kl = 0.01
max_grad_norm = 1.0
value_loss_coef = 1.0
use_clipped_value_loss = True
num_steps_per_env = 24
max_iterations = 50000
save_interval = 100
init_noise_std = 1.0

# Network
hidden_layers = [512, 256, 128]
activation = "ELU"

# Environment
num_envs = 4096
env_spacing = 2.5
```

#### J.2 Observation Space

**Actor (45 DOF):**
- Base angular velocity: 3 DOF (scale=0.2, noise ±0.2)
- Projected gravity: 3 DOF (noise ±0.05)
- Velocity commands (vx, vy, wz): 3 DOF
- Relative joint positions: 12 DOF (noise ±0.01)
- Relative joint velocities: 12 DOF (scale=0.05, noise ±1.5)
- Previous action: 12 DOF
- History length: 5 steps

**Critic (48 DOF):** Actor observations + base linear velocity (3 DOF, privileged)

#### J.3 Action Space (23 DOF)

- Legs: 12 DOF (hip pitch/roll/yaw, knee, ankle pitch/roll per leg)
- Waist: 1 DOF (yaw)
- Arms: 10 DOF (shoulder pitch/roll/yaw, elbow, wrist roll per arm)

#### J.4 Reward Function Components

**Positive rewards:**
```python
track_lin_vel_xy:   weight=2.0    std=sqrt(0.25)  # Track velocity commands
track_ang_vel_z:    weight=1.0    std=sqrt(0.25)  # Track yaw rate
alive:              weight=1.0                     # Survival bonus
```

**Penalties:**
```python
base_linear_velocity:  weight=-2.0      # Penalize Z velocity
base_angular_velocity: weight=-0.05     # Penalize roll/pitch rate
joint_vel:             weight=-0.001    # Smooth joint motion
joint_acc:             weight=-0.1e-7   # Smooth joint acceleration
action_rate:           weight=-0.05     # Smooth action changes
dof_pos_limits:        weight=-5.0      # Stay within joint limits
energy:                weight=-2e-5     # Energy efficiency
```

**Custom gait modifications (Wide Stance example):**
```python
base_height:              weight=-15   target=0.6m    # Lower stance (default 0.78m)
feet_too_near:            weight=-5.0  threshold=0.4  # Wide stance
zmp_deviation:            weight=-0.5                 # Balance (reduce from -10 to prevent divergence!)
torso_lean_back_penalty:  weight=-10.0                # Prevent backward leaning
gait:                     weight=0.5   period=0.8     # Gait frequency
```

#### J.5 Operational Lessons (Reward Tuning)

| Issue | Symptom | Solution |
|-------|---------|----------|
| ZMP weight too high (-10.0) | mean_reward → 0 early | Reduce to -0.5 |
| flat_orientation + ZMP conflict | Early convergence, plateau | Remove flat_orientation when using ZMP |
| Dual gait emergence | 50% lean forward, 50% backward | Add torso_lean_back_penalty=-10.0 |

**Training Performance:**
```
Throughput: 53,840 steps/s
Collection time: 1.505s
Learning time: 0.321s
Typical mean_reward: 21.51
Typical episode length: 1,002 steps
```

**Commands:**
```bash
# Train (headless, 2-3x faster)
python legged_gym/scripts/train.py --task=g1 --headless --num_envs=4096
# Resume
python legged_gym/scripts/train.py --task=g1 --resume --load_run="Jan01_12-00-00"
# Monitor
tensorboard --logdir logs/rsl_rl/unitree_g1_23dof_<TASK_NAME>
```

### Appendix K: VLA Fine-Tuning Operational Reference

> Extracted from HumanoidTraining Units 25-29. Critical operational knowledge for GR00T N1.6 fine-tuning that complements the LeRobot training pipeline (Section 11, Q5a).

#### K.1 Key Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `tune_diffusion_model` | `false` (single GPU) / `true` (multi-GPU) | false = adapter only (~14GB VRAM), true = full model (~20GB+ VRAM) |
| `batch_size` | 24 (single GPU) / 48 (multi-GPU) | Reduce if OOM |
| `action_horizon` | 30 | Number of future action steps predicted per chunk |
| `save_total_limit` | 2 | Keep only N most recent checkpoints |
| Loss target range | 0.029 - 0.055 | Monitor `update_s` metric in WANDB |
| Checkpoint size | ~6.96 GB (model.safetensors) | ~22GB total disk for full pipeline |
| Video backend | `pyav` | Most compatible; `decord` if available |

#### K.2 Single-GPU Training Command

```bash
python src/lerobot/scripts/lerobot_train.py \
  --output_dir="outputs/groot_pick_place_g1" \
  --save_checkpoint=true \
  --batch_size=24 \
  --steps=100 \
  --save_freq=100 \
  --eval_freq=100 \
  --log_freq=10 \
  --policy.type=groot \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id="theconstruct-ai/pick_place_g1" \
  --dataset.video_backend=pyav \
  --wandb.enable=true
```

#### K.3 Multi-GPU Training Command

```bash
export NUM_GPUS=8
accelerate launch \
  --multi_gpu --num_processes=$NUM_GPUS --mixed_precision=bf16 \
  $(which lerobot-train) \
  --batch_size=48 --steps=200000 \
  --policy.type=groot \
  --policy.tune_diffusion_model=true \
  --dataset.video_backend=pyav \
  --wandb.enable=true --wandb.disable_artifact=true
```

#### K.4 tune_diffusion_model Decision Guide

| Setting | VRAM | Speed | Quality | Use When |
|---------|------|-------|---------|----------|
| `false` | ~14 GB | Fast | Adapter-only, limited | Quick prototyping, small datasets, single GPU |
| `true` | ~20+ GB | Slower | Full model, better | Production models, large datasets, multi-GPU |

#### K.5 4-Process GROOT Deployment Architecture

The validated deployment from Unit 29 uses 4 concurrent processes:

```
┌─────────────────────────────────────────────────────────────┐
│  Process 1: GR00T Inference Server (port 5556)              │
│  - Loads GR00T N1.6 checkpoint on GPU                       │
│  - Receives observations, returns action chunks             │
│  - Command: run_gr00t_server.py --port 5556                 │
├─────────────────────────────────────────────────────────────┤
│  Process 2: MuJoCo Simulation (port 5557)                   │
│  - RoboCasa environment, sim_frequency=200 Hz               │
│  - Streams camera images on port 5557                       │
│  - Command: run_sim_loop.py --camera_port 5557              │
├─────────────────────────────────────────────────────────────┤
│  Process 3: WBC Controller                                  │
│  - Runs decoupled WBC at control_frequency=50 Hz            │
│  - Loads Balance + Walk ONNX policies                       │
│  - Command: run_g1_control_loop.py --control_frequency 50   │
├─────────────────────────────────────────────────────────────┤
│  Process 4: Closed-Loop Bridge (10 Hz)                      │
│  - Connects camera (5557) ↔ GROOT (5556) ↔ WBC             │
│  - InterpolationPolicy smooths between action chunks        │
│  - Language instruction: "Pick up the bottle."              │
│  - Command: run_groot_closed_loop_bridge.py --rate-hz 10    │
└─────────────────────────────────────────────────────────────┘
```

**Bridge rate (10 Hz)** is the effective closed-loop frequency — the rate at which new observations are sent to GR00T and new action chunks are received. InterpolationPolicy smoothly interpolates between chunks at the full 50 Hz control rate.

#### K.6 Training Monitoring Checklist

- [ ] WANDB dashboard showing `update_s` decreasing
- [ ] Loss in range 0.029-0.055
- [ ] No OOM errors (reduce batch_size if needed)
- [ ] Checkpoints saving at expected intervals
- [ ] `save_total_limit` set to prevent disk overflow
- [ ] GPU utilization >80% (check with `nvidia-smi`)

---

*This document should be reviewed and updated quarterly as the VLA ecosystem evolves. The vendor submodule strategy ensures we can adopt new upstream changes without disrupting our platform.*
