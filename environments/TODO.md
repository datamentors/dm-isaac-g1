# Environments TODO

## Upgrade Workstation Isaac Sim 5.0.0 -> 5.1.0

The workstation Dockerfile uses `isaacsim==5.0.0`, while the Spark image and team's TeleOp
setup both use `5.1.0`. Update the workstation builder stage:

```dockerfile
# Line ~74 in environments/workstation/Dockerfile
RUN pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

Note: May require upgrading CUDA base image or torch version. Test IsaacLab compat after.

## Upgrade MuJoCo 3.2.6 -> 3.4.0

The C++ sim2sim pipeline (unitree_mujoco + g1_ctrl) has a known issue where the robot
falls after elastic band release. The HumanoidTraining course used MuJoCo 3.4.0 (we use
3.2.6). Upgrading may fix sim2sim transfer for 29DOF policies.

Affects: builder stage `pip install mujoco==3.2.6` + unitree_mujoco build.

## Configure unitree_mujoco at build time

Currently `unitree_mujoco/simulate/config.yaml` has upstream defaults (go2, scene.xml).
Should be patched during Docker build to use G1 29-DOF settings:
- `robot: "g1"`, `robot_scene: "scene_29dof.xml"`
- `domain_id: 0`, `interface: "lo"`, `use_joystick: 0`, `enable_elastic_band: 1`

---

## Completed

- ~~VNC password~~ -- runtime service password (entrypoint.sh)
- ~~noVNC~~ -- browser VNC on port 6080
- ~~Desktop icons~~ -- xfdesktop style=2 + gvfs trust
- ~~video2robot UI~~ -- HTTP Basic Auth on port 8000
- ~~Sim2Sim C++ build deps~~ -- libyaml-cpp-dev, libboost-all-dev, libeigen3-dev, libspdlog-dev, libfmt-dev
- ~~unitree_sdk2 C++ SDK~~ -- built from source, installed to /usr/local
- ~~ONNX Runtime C++ (x86_64)~~ -- installed to /usr/local for g1_ctrl
- ~~unitree_mujoco build~~ -- C++ simulator compiled in Docker image
- ~~g1_ctrl build~~ -- C++ controller compiled in Docker image
- ~~CycloneDDS config~~ -- cyclonedds.xml with loopback interface
- ~~CMakeLists.txt multi-arch~~ -- auto-detect x86_64 vs aarch64 ONNX paths
- ~~Python trigger scripts~~ -- scripts/sim2sim/joystick_trigger.py
