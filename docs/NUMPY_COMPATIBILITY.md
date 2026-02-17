# NumPy Compatibility Guide

This document explains the NumPy version requirements across all Datamentors robotics environments.

## TL;DR

**Use NumPy < 2.0.0 everywhere.** NumPy 2.x causes crashes in Isaac Sim 5.1.0's synthetic data pipeline.

## Version Matrix

| Environment | Isaac Sim | Python | NumPy | Status |
|-------------|-----------|--------|-------|--------|
| **dm-groot-inference** (DGX Spark) | N/A | 3.10 | `>=1.26.0,<2.0.0` | ✅ Correct |
| **dm-isaac-g1** (Workstation) | 5.0.0 | 3.11 | `<2.0.0` | ✅ Correct |
| Isaac Sim 5.1.0 | 5.1.0 | 3.11 | 2.x (broken) | ❌ Avoid |
| Isaac Sim 6.0 (future) | 6.0 | 3.11 | `>2.0` | 🔮 Not ready |

## The Problem

Isaac Sim 5.1.0 ships with NumPy 2.x, which causes:
- `TypeError: Unable to write from unknown dtype, kind=f, size=0` in camera/synthetic data pipeline
- Crashes when using TiledCameraCfg or CameraCfg with annotators
- Incompatibility with GR00T fine-tuning requirements

This is documented in:
- [IsaacLab Issue #3312](https://github.com/isaac-sim/IsaacLab/issues/3312)
- [IsaacLab Issue #2235](https://github.com/isaac-sim/IsaacLab/issues/2235)

## Solution: Use Isaac Sim 5.0.0

Unitree officially recommends Isaac Sim 5.0.0 for their G1 simulation environments:
- [unitree_sim_isaaclab installation guide](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/doc/isaacsim5.0_install.md)

Isaac Sim 5.0.0:
- Uses NumPy 1.x (compatible with all dependencies)
- Has stable camera/TiledCamera operations
- Works with IsaacLab 2.2.0

## Environment Architecture

```
┌─────────────────────────────────┐
│  FINE-TUNING (DGX Spark)        │
│  Container: dm-groot-inference  │
├─────────────────────────────────┤
│  Python: 3.10                   │
│  NumPy: >=1.26.0,<2.0.0         │
│  Isaac Sim: NOT REQUIRED        │
│  CUDA: 12.4                     │
│  Purpose: GR00T model training  │
└─────────────────────────────────┘
              │
              │ ZMQ (port 5555)
              ▼
┌─────────────────────────────────┐
│  INFERENCE (Blackwell Workst.)  │
│  Container: dm-workstation      │
├─────────────────────────────────┤
│  Python: 3.11                   │
│  NumPy: <2.0.0 (pinned)         │
│  Isaac Sim: 5.0.0               │
│  Isaac Lab: 2.2.0               │
│  Purpose: Robot simulation      │
└─────────────────────────────────┘
```

## Pinning NumPy < 2.0

### In Dockerfile (after Isaac Sim install):
```dockerfile
# Pin NumPy to < 2.0 for stability
RUN /isaac-sim/python.sh -m pip install "numpy>=1.26.0,<2.0.0"
```

### In pyproject.toml:
```toml
dependencies = [
    "numpy>=1.26.0,<2.0.0",
]
```

### Quick fix in existing environment:
```bash
pip install "numpy>=1.26.0,<2.0.0"
```

## GR00T Requirements

From [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T):
- Python: 3.10 (not 3.11)
- NumPy: `>=1.23.5,<2.0.0`
- uv: v0.8.4+

Fine-tuning does NOT require Isaac Sim - it runs standalone.

## Isaac Sim Version Comparison

| Version | NumPy | Cameras | Recommendation |
|---------|-------|---------|----------------|
| 4.2.0 | 1.x | ✅ | Legacy, works |
| 5.0.0 | 1.x | ✅ | **Recommended** |
| 5.1.0 | 2.x | ❌ | Avoid |
| 6.0 | 2.x | 🔮 | Future (Newton) |

## References

- [Isaac-GR00T Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
- [Isaac Lab Development Update](https://github.com/isaac-sim/IsaacLab/discussions/4339)
- [Isaac Sim 5.0.0 Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim)
- [IsaacLab TiledCamera Fix PR #3808](https://github.com/isaac-sim/IsaacLab/pull/3808)

## Verification Commands

Check NumPy version in Isaac Sim:
```bash
/isaac-sim/python.sh -c "import numpy; print(numpy.__version__)"
```

Expected output for 5.0.0: `1.26.x`
Problematic output for 5.1.0: `2.x.x`
