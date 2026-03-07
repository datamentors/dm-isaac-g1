# Sim2Sim C++ Deploy Pipeline

G1-29DOF MuJoCo deployment using the `unitree_mujoco` + `g1_ctrl` C++ pipeline
communicating via CycloneDDS.

## Architecture

```
Terminal 1: unitree_mujoco  ──(DDS over loopback)──  g1_ctrl  :Terminal 2
                                                        │
                                                   ONNX policy
                                                        │
Terminal 3: joystick_trigger.py ──(UDP:15001)──> JoystickInjector
```

- **unitree_mujoco**: C++ MuJoCo simulator publishing/subscribing robot state via DDS
- **g1_ctrl**: C++ controller that reads state, runs ONNX policy inference, sends commands
- **joystick_trigger.py**: Python script that injects FSM button presses via UDP

All three run inside the same container, communicating over the `lo` (loopback) interface.

## Prerequisites (Handled by Dockerfile)

| Component | Workstation (x86_64) | Spark (ARM64) |
|-----------|---------------------|---------------|
| C++ libs | libyaml-cpp-dev, libboost-all-dev, libeigen3-dev, libspdlog-dev, libfmt-dev | Same |
| unitree_sdk2 | Built from source → /usr/local | Same |
| ONNX Runtime C++ | v1.22.0 installed to /usr/local | v1.22.0 from thirdparty/ (aarch64) |
| CycloneDDS | Built from source (0.10.x) | Same |
| cyclonedds.xml | /etc/cyclonedds.xml (loopback) | Same |
| unitree_mujoco | Cloned + built | Same |
| g1_ctrl | Built from deploy/robots/g1_29dof/ | Same |

## Quick Start

### 1. Start MuJoCo Simulator

```bash
export CYCLONEDDS_URI=file:///etc/cyclonedds.xml
cd /workspace/unitree_mujoco/simulate/build
./unitree_mujoco
```

### 2. Start Robot Controller

```bash
export CYCLONEDDS_URI=file:///etc/cyclonedds.xml
cd /workspace/dm-isaac-g1/unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
```

### 3. Trigger a Policy

```bash
# Military March (base locomotion)
python /workspace/dm-isaac-g1/scripts/sim2sim/joystick_trigger.py --policy military_march

# Dance
python /workspace/dm-isaac-g1/scripts/sim2sim/joystick_trigger.py --policy dance_102

# Gangnam Style
python /workspace/dm-isaac-g1/scripts/sim2sim/joystick_trigger.py --policy gangnam_style

# 08Clip01Track1
python /workspace/dm-isaac-g1/scripts/sim2sim/joystick_trigger.py --policy 08clip01

# CR7 YouTube Run
python /workspace/dm-isaac-g1/scripts/sim2sim/joystick_trigger.py --policy cr7_youtube_run
```

### 4. MuJoCo Controls

- Press **8** to lower the robot to the ground
- Press **9** to release the elastic band (starts free-standing)

## unitree_mujoco Config

The simulator config is at `/workspace/unitree_mujoco/simulate/config.yaml`.
It must be configured for G1 29-DOF:

```yaml
robot: "g1"
robot_scene: "scene_29dof.xml"
domain_id: 0
interface: "lo"           # loopback for container
use_joystick: 0           # no physical joystick
enable_elastic_band: 1    # start with band ON
```

**Defaults from upstream** (must be changed): `robot: "go2"`, `robot_scene: "scene.xml"`,
`domain_id: 1`, `enable_elastic_band: 0`.

## FSM States & Transitions

Configured in `deploy/robots/g1_29dof/config/config.yaml`:

| State | ID | Type | Transition From | Button Combo |
|-------|----|------|----------------|--------------|
| Passive | 0 | — | (start) | — |
| FixStand | 1 | — | Passive | LT (2s hold) + UP |
| MilitaryMarch | 3 | RLBase | FixStand | RB + X |
| Velocity | 4 | RLBase | FixStand | (configured) |
| Mimic_Dance_102 | 101 | RLBase | MilitaryMarch | LT (2s) + DOWN |
| Mimic_Gangnam_Style | 102 | RLBase | MilitaryMarch | LT (2.5s) + LEFT |
| Mimic_08Clip01Track1 | 104 | RLBase | MilitaryMarch | RB + Y |
| Mimic_CR7YoutubeRun | 105 | RLBase | MilitaryMarch | RB + A |

## Available Policies

| Policy | Location | Source |
|--------|----------|--------|
| velocity/v0 | deploy/robots/g1_29dof/config/policy/velocity/v0/ | Unitree pretrained |
| military_march | deploy/robots/g1_29dof/config/policy/military_march/ | Our RL training |
| mimic/dance_102 | deploy/robots/g1_29dof/config/policy/mimic/dance_102/ | Unitree |
| mimic/gangnam_style | deploy/robots/g1_29dof/config/policy/mimic/gangnam_style/ | Unitree |
| mimic/08_clip01_track1 | deploy/robots/g1_29dof/config/policy/mimic/08_clip01_track1/ | Our training |
| mimic/cr7_youtube_run | deploy/robots/g1_29dof/config/policy/mimic/cr7_youtube_run/ | Our training |

Each policy directory contains: `policy.onnx` (or `exported/policy.onnx`), `deploy.yaml`,
and optionally a `.csv` motion capture reference file for mimic policies.

## CycloneDDS Configuration

Both `unitree_mujoco` and `g1_ctrl` must use the loopback interface for DDS
communication within a container. The config is at `/etc/cyclonedds.xml`:

```xml
<CycloneDDS xmlns="https://cdds.io/config">
    <Domain>
        <General>
            <Interfaces>
                <NetworkInterface name="lo" multicast="true" />
            </Interfaces>
            <AllowMulticast>true</AllowMulticast>
            <EnableMulticastLoopback>true</EnableMulticastLoopback>
        </General>
    </Domain>
</CycloneDDS>
```

Set via environment: `CYCLONEDDS_URI=file:///etc/cyclonedds.xml`

## Building g1_ctrl Manually

If you need to rebuild after code changes:

```bash
cd /workspace/dm-isaac-g1/unitree_rl_lab/deploy/robots/g1_29dof
rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc)
```

The CMakeLists.txt auto-detects architecture (x86_64 vs aarch64) and selects
the correct ONNX Runtime path.

## Code Changes Summary

### C++ Headers (deploy/include/)

- **JoystickInjector.h**: UDP server (127.0.0.1:15001) for programmatic button injection
- **FSM/FSMState.h**: Integrates JoystickInjector into FSM loop
- **isaaclab/algorithms/algorithms.h**: ONNX thread limiting (`SetIntraOpNumThreads(1)`)
- **isaaclab/envs/mdp/terminations.h**: NaN fix (`std::clamp` on `acos` input)

### CMakeLists.txt

- Multi-arch support (x86_64 + aarch64)
- `find_package(unitree_sdk2)` instead of hardcoded paths
- C++17 standard

## Known Issue: Robot Falls After Elastic Band Release

When pressing 9 (release elastic band), the robot falls regardless of policy.
This is unresolved and likely requires:
- A 29DOF velocity policy with better sim2sim transfer
- Upgrading MuJoCo to 3.4.0 (currently 3.2.6)
- Investigating 29DOF MuJoCo scene XML differences (joint limits, friction, gains)

The HumanoidTraining course used G1-23DOF with VelocityV2.6 on MuJoCo 3.4.0,
which worked. Our 29DOF setup on MuJoCo 3.2.6 does not transfer.

## Comparison: Python sim2sim vs C++ Deploy

| Aspect | Python sim2sim | C++ Deploy Pipeline |
|--------|---------------|-------------------|
| Location | `dm-isaac-g1/src/dm_isaac_g1/sim2sim/` | `unitree_rl_lab/deploy/` |
| MuJoCo | Python bindings (`mujoco`) | C++ (`unitree_mujoco`) |
| Communication | Direct (single process) | DDS over loopback (2 processes) |
| Inference | Python ONNX Runtime | C++ ONNX Runtime |
| Real robot ready | No | Yes (same DDS protocol) |
| Dependencies | Python only | C++ build toolchain |
| ECS command | `./run.sh sim2sim` | Manual (3 terminals) |
