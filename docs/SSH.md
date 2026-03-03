# SSH Access Guide for Unitree G1

This guide explains how to connect to the Unitree G1 over SSH using the provided network details.

## Requirements

- Device and G1 on the same network
- SSH client (macOS/Linux Terminal or Windows PowerShell)

## Connection Details

- IP: `192.168.1.122`
- Username: `unitree`
- Password: *(ask team lead)*

## Steps

1. Open a terminal.
2. Run the SSH command:

```bash
ssh unitree@192.168.1.122
```

3. When prompted, enter the password (ask team lead for credentials).

4. If this is your first time connecting, confirm the host key by typing `yes`.

## Common Fixes

- **No route to host / connection timed out**: Verify the G1 is powered on and connected to the same network. Try `ping 192.168.1.122`.
- **Permission denied**: Recheck the username and password. Ensure caps lock is off.
- **Host key changed**: Remove the old key:

```bash
ssh-keygen -R 192.168.1.122
```

## Disconnect

To end the session:

```bash
exit
```

## G1 System Info

- Architecture: `aarch64` (ARM64)
- OS: Ubuntu 22.04.5 LTS
- GCC: 11.4.0
- Ethernet interface: `enP8p1s0` (use this for DDS communication, NOT `eth0`)

## 29DOF Deploy (unitree_rl_lab)

Deploying RL/Mimic policies from `https://github.com/unitreerobotics/unitree_rl_lab/`.

### ONNX Runtime Setup (RESOLVED 2026-03-04)

The G1 is aarch64 (Cortex-A78AE / Orin NX). Two issues needed to be fixed:

#### Issue 1: Headers (Build Error)
The x64 ONNX Runtime that ships with unitree_rl_lab won't compile on aarch64. You need the aarch64 package, but the **official release tarballs** from GitHub have a `-D_GLIBCXX_ASSERTIONS` flag that causes a runtime crash on the G1's ARM CPU (see Issue 2).

#### Issue 2: GLIBCXX Assertions (Runtime Crash)
All official ONNX Runtime aarch64 releases (v1.18.1, v1.20.1, v1.22.0, v1.23.2) are compiled on Red Hat with `-D_GLIBCXX_ASSERTIONS` enabled. This causes a `stl_vector` bounds assertion during ONNX session creation on the G1, because ONNX Runtime's internal cpuinfo code has an out-of-bounds access on this specific ARM platform (8 CPUs, only 4 online).

#### Solution: Use Locally-Built ONNX Runtime

The fix is to use a locally-compiled ONNX Runtime **without** assertions. This build already exists on the G1 at `~/onnxruntime/build/Linux/Release/`.

Setup the thirdparty directory with headers from the release + library from the local build:

```bash
cd ~/unitree_rl_lab/deploy/thirdparty
rm -rf onnxruntime-linux-aarch64-1.22.0
mkdir -p onnxruntime-linux-aarch64-1.22.0/{lib,include}

# Headers from the official release tarball (flat layout)
cp ~/Downloads/onnxruntime-linux-aarch64-1.22.0/include/*.h onnxruntime-linux-aarch64-1.22.0/include/

# Library from the local source build (no -D_GLIBCXX_ASSERTIONS)
cp ~/onnxruntime/build/Linux/Release/libonnxruntime.so* onnxruntime-linux-aarch64-1.22.0/lib/
```

#### Code Fix: Thread Affinity

Additionally, `deploy/include/isaaclab/algorithms/algorithms.h` needs two lines added after `SetGraphOptimizationLevel` to prevent thread pinning to offline CPUs:

```cpp
session_options.SetIntraOpNumThreads(1);
session_options.SetInterOpNumThreads(1);
```

This has already been applied on the G1.

### Build

```bash
cd ~/unitree_rl_lab/deploy/robots/g1_29dof
rm -rf build && mkdir build && cd build
cmake .. && make -j4
```

### Run

```bash
cd ~/unitree_rl_lab/deploy/robots/g1_29dof/build
export LD_LIBRARY_PATH=/usr/local/lib:/home/unitree/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib
./g1_ctrl --network enP8p1s0
```

**DDS requires wired Ethernet** — wireless interfaces will not work for robot control.

### Controller Usage

1. Robot must be in hoisting state first
2. Press **L2+R2** on remote to enter debug mode (damping state)
3. Press **L2+Up** to enter FixStand mode
4. Press **R1+X** to enter Velocity control mode
5. Use joystick to control movement
6. **L2+B** to return to Passive mode at any time

### Successful Run Output

```text
 --- Unitree Robotics ---
     G1-29dof Controller
[info] Waiting for connection to robot...
[info] Connected to robot.
[info] Initializing State_Passive ...
[info] Initializing State_FixStand ...
[info] Initializing State_Velocity ...
[info] Policy directory: .../config/policy/velocity/v0
[info] Initializing State_Mimic_Dance_102 ...
[info] Policy directory: .../config/policy/mimic/dance_102/
[info] Loaded motion file 'G1_Take_102.bvh_60hz' with duration 29.15s
[info] Initializing State_Mimic_Gangnam_Style ...
[info] Policy directory: .../config/policy/mimic/gangnam_style/
[info] Loaded motion file 'G1_gangnam_style_V01.bvh_60hz' with duration 32.37s
[info] FSM: Start Passive
```

### Previous Error (RESOLVED)

```text
/opt/rh/gcc-toolset-14/root/usr/include/c++/14/bits/stl_vector.h:1130: Assertion '__n < this->size()' failed.
Aborted (core dumped)
```

Root cause: Official ONNX Runtime aarch64 releases are built with `-D_GLIBCXX_ASSERTIONS` on Red Hat using gcc-toolset-14. An internal vector out-of-bounds in ONNX's cpuinfo code triggers the assertion on the G1's Cortex-A78AE CPU. Fixed by using a locally-compiled ONNX Runtime without assertions.
