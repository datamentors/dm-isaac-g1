#!/usr/bin/env python3
"""
Environment validation tests for dm-isaac-g1 containers.

Runs on both Workstation (x86_64) and Spark (ARM64) containers.
Tests marked with @x86_only are skipped on ARM64 (no Isaac Sim).

Usage:
  # Inside any container:
  python environments/tests/test_environment.py

  # Or via pytest:
  pytest environments/tests/test_environment.py -v

  # Run from host via docker exec:
  docker exec dm-spark-workstation python /workspace/dm-isaac-g1/environments/tests/test_environment.py
  docker exec dm-workstation conda run -n unitree_sim_env python /workspace/dm-isaac-g1/environments/tests/test_environment.py
"""

import os
import platform
import subprocess
import sys
import importlib
from dataclasses import dataclass
from typing import Optional

# ── Helpers ──────────────────────────────────────────────────────────────────

IS_ARM64 = platform.machine() in ("aarch64", "arm64")
IS_X86 = platform.machine() in ("x86_64", "AMD64")


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    skipped: bool = False


results: list[TestResult] = []


def test(name: str, skip_on_arm64: bool = False):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            if skip_on_arm64 and IS_ARM64:
                results.append(TestResult(name, True, "Skipped (ARM64)", skipped=True))
                return
            try:
                msg = func()
                results.append(TestResult(name, True, msg or "OK"))
            except Exception as e:
                results.append(TestResult(name, False, str(e)))
        return wrapper
    return decorator


# ── Core Tests ───────────────────────────────────────────────────────────────

@test("Python version")
def test_python():
    v = sys.version_info
    assert v.major == 3 and v.minor >= 10, f"Need Python 3.10+, got {v.major}.{v.minor}"
    return f"Python {v.major}.{v.minor}.{v.micro}"


@test("PyTorch + CUDA")
def test_torch():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    gpu = torch.cuda.get_device_name(0)
    # Quick tensor op on GPU
    t = torch.randn(100, 100, device="cuda")
    r = torch.mm(t, t.T)
    assert r.shape == (100, 100)
    return f"PyTorch {torch.__version__}, GPU: {gpu}"


@test("GPU memory allocation")
def test_gpu_memory():
    import torch
    # Allocate 1 GB on GPU
    t = torch.randn(256, 1024, 1024, device="cuda")
    mem_mb = torch.cuda.memory_allocated() / 1024**2
    del t
    torch.cuda.empty_cache()
    return f"Allocated ~{mem_mb:.0f} MB successfully"


@test("MuJoCo import")
def test_mujoco_import():
    import mujoco
    return f"MuJoCo {mujoco.__version__}"


@test("MuJoCo simulation step")
def test_mujoco_step():
    import mujoco
    import numpy as np
    xml = """
    <mujoco>
      <worldbody>
        <body>
          <joint type="hinge"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    return f"100 steps, time={data.time:.4f}s"


@test("MuJoCo EGL rendering")
def test_mujoco_egl():
    import mujoco
    import numpy as np
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <body>
          <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 64, 64)
    mujoco.mj_step(model, data)
    renderer.update_scene(data)
    img = renderer.render()
    assert img.shape == (64, 64, 3), f"Bad shape: {img.shape}"
    assert img.max() > 0, "Image is all black"
    renderer.close()
    return f"Rendered 64x64 frame, max pixel={img.max()}"


@test("MuJoCo Menagerie (G1 model)")
def test_menagerie():
    import mujoco
    # Check both possible locations
    paths = [
        "/workspace/mujoco_menagerie/unitree_g1/g1.xml",
        "/home/code/mujoco_menagerie/unitree_g1/g1.xml",
    ]
    found = None
    for p in paths:
        if os.path.exists(p):
            found = p
            break
    assert found, f"G1 model not found in: {paths}"
    model = mujoco.MjModel.from_xml_path(found)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    return f"G1 model loaded: {model.nq} qpos, {model.nv} qvel"


@test("GR00T import")
def test_groot():
    import gr00t
    return "GR00T imported OK"


@test("GR00T transformers version")
def test_transformers():
    import transformers
    v = transformers.__version__
    # Must be 4.51.x for GR00T compatibility
    assert v.startswith("4.51"), f"Need transformers 4.51.x, got {v}"
    return f"transformers {v}"


@test("DeepSpeed")
def test_deepspeed():
    import deepspeed
    return f"DeepSpeed {deepspeed.__version__}"


@test("wandb")
def test_wandb():
    import wandb
    return f"wandb {wandb.__version__}"


@test("pinocchio (IK)")
def test_pinocchio():
    import pinocchio
    return f"pinocchio {pinocchio.__version__}"


@test("pink (IK)")
def test_pink():
    import pink
    return f"pink {pink.__version__}"


@test("RSL-RL")
def test_rsl_rl():
    import rsl_rl
    return "rsl_rl imported OK"


@test("dm-isaac-g1 package")
def test_dm_isaac_g1():
    import dm_isaac_g1
    return "dm_isaac_g1 imported OK"


@test("ZMQ (GROOT server protocol)")
def test_zmq():
    import zmq
    return f"pyzmq {zmq.__version__}"


@test("scipy")
def test_scipy():
    import scipy
    return f"scipy {scipy.__version__}"


@test("numpy")
def test_numpy():
    import numpy as np
    v = np.__version__
    major = int(v.split(".")[0])
    return f"numpy {v}"


@test("pandas")
def test_pandas():
    import pandas
    return f"pandas {pandas.__version__}"


@test("tensorboard")
def test_tensorboard():
    import tensorboard
    return f"tensorboard {tensorboard.__version__}"


@test("huggingface-hub")
def test_hf_hub():
    import huggingface_hub
    return f"huggingface_hub {huggingface_hub.__version__}"


# ── Workspace Tests ──────────────────────────────────────────────────────────

@test("dm-isaac-g1 repo mounted")
def test_repo_mounted():
    assert os.path.exists("/workspace/dm-isaac-g1/pyproject.toml"), "dm-isaac-g1 not mounted at /workspace/dm-isaac-g1"
    return "OK"


@test("Isaac-GR00T available")
def test_groot_repo():
    # Check both baked-in (gr00t) and mounted (Isaac-GR00T)
    baked = os.path.exists("/workspace/gr00t/pyproject.toml")
    mounted = os.path.exists("/workspace/Isaac-GR00T/pyproject.toml")
    assert baked or mounted, "Neither /workspace/gr00t nor /workspace/Isaac-GR00T found"
    loc = "mounted" if mounted else "baked-in"
    return f"OK ({loc})"


@test("WBC repo present")
def test_wbc():
    assert os.path.exists("/workspace/GR00T-WholeBodyControl-dex1"), "WBC repo not found"
    return "OK"


@test("video2robot repo present")
def test_video2robot():
    assert os.path.exists("/workspace/video2robot"), "video2robot repo not found"
    return "OK"


@test("checkpoints dir accessible")
def test_checkpoints():
    assert os.path.isdir("/workspace/checkpoints"), "/workspace/checkpoints not found"
    contents = os.listdir("/workspace/checkpoints")
    return f"OK ({len(contents)} items)"


# ── x86_64 Only (Workstation/ECS) ───────────────────────────────────────────

@test("Isaac Sim import", skip_on_arm64=True)
def test_isaacsim():
    import isaacsim
    return "Isaac Sim imported OK"


@test("IsaacLab import", skip_on_arm64=True)
def test_isaaclab():
    import isaaclab
    return "IsaacLab imported OK"


@test("flash-attn")
def test_flash_attn():
    import flash_attn
    return f"flash-attn {flash_attn.__version__}"


@test("torchcodec", skip_on_arm64=True)
def test_torchcodec():
    import torchcodec
    return f"torchcodec {torchcodec.__version__}"


# ── GPU Stress Test ──────────────────────────────────────────────────────────

@test("GPU matmul benchmark")
def test_gpu_benchmark():
    import torch
    import time
    size = 4096
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)
    # Warmup
    torch.mm(a, b)
    torch.cuda.synchronize()
    # Timed
    start = time.time()
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    tflops = (2 * size**3 * 10) / elapsed / 1e12
    del a, b
    torch.cuda.empty_cache()
    return f"4096x4096 matmul: {elapsed:.2f}s (10 iters), ~{tflops:.1f} TFLOPS"


# ── Runner ───────────────────────────────────────────────────────────────────

def run_all():
    """Run all registered tests and print results."""
    print(f"{'='*60}")
    print(f"  Environment Validation — {platform.machine()}")
    print(f"  Python {sys.version.split()[0]}")
    print(f"{'='*60}")
    print()

    # Collect all test functions
    tests = [v for v in globals().values() if callable(v) and hasattr(v, "__wrapped__")]

    # Actually just run all the wrapper functions we created
    for name, obj in list(globals().items()):
        if callable(obj) and name.startswith("test_") and not name.startswith("test_environment"):
            obj()

    # Print results
    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = sum(1 for r in results if not r.passed)
    skipped = sum(1 for r in results if r.skipped)

    for r in results:
        if r.skipped:
            icon = "⊘"
        elif r.passed:
            icon = "✓"
        else:
            icon = "✗"
        print(f"  {icon} {r.name}: {r.message}")

    print()
    print(f"{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
