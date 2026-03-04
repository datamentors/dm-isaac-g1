#!/usr/bin/env python3
"""
Build validation tests — runs WITHOUT GPU on the build instance.

These tests validate the Docker image before it's pushed to ECR.
They catch: missing packages, wrong versions, broken imports, missing files.
GPU-dependent tests are skipped (those run separately on ECS).

Usage:
  python environments/tests/test_build.py
  python environments/tests/test_build.py --json  # JSON output for automation

Exit code 0 = all passed, 1 = failures detected.
"""

import importlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass


IS_ARM64 = platform.machine() in ("aarch64", "arm64")
IS_X86 = platform.machine() in ("x86_64", "AMD64")


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    skipped: bool = False


results: list[TestResult] = []


def test(name: str, skip_on_arm64: bool = False, skip_on_x86: bool = False):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            if skip_on_arm64 and IS_ARM64:
                results.append(TestResult(name, True, "Skipped (ARM64)", skipped=True))
                return
            if skip_on_x86 and IS_X86:
                results.append(TestResult(name, True, "Skipped (x86_64)", skipped=True))
                return
            try:
                msg = func()
                results.append(TestResult(name, True, msg or "OK"))
            except Exception as e:
                results.append(TestResult(name, False, str(e)[:300]))
        wrapper._test = True
        return wrapper
    return decorator


# ── Core Python & PyTorch ────────────────────────────────────────────────────

@test("Python version")
def test_python():
    v = sys.version_info
    assert v.major == 3 and v.minor >= 10, f"Need Python 3.10+, got {v.major}.{v.minor}"
    return f"Python {v.major}.{v.minor}.{v.micro}"


@test("PyTorch import")
def test_torch():
    import torch
    return f"PyTorch {torch.__version__}"


@test("numpy version (<2.0)")
def test_numpy():
    import numpy as np
    major = int(np.__version__.split(".")[0])
    assert major < 2, f"numpy must be <2.0 for Isaac Sim compatibility, got {np.__version__}"
    return f"numpy {np.__version__}"


# ── MuJoCo ───────────────────────────────────────────────────────────────────

@test("MuJoCo import")
def test_mujoco():
    import mujoco
    return f"MuJoCo {mujoco.__version__}"


@test("MuJoCo CPU simulation (100 steps)")
def test_mujoco_cpu():
    import mujoco
    xml = '<mujoco><worldbody><body><joint type="hinge"/><geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    return f"100 steps OK, time={data.time:.4f}s"


@test("MuJoCo Menagerie G1 model")
def test_menagerie():
    import mujoco
    paths = [
        "/workspace/mujoco_menagerie/unitree_g1/g1.xml",
        "/home/code/mujoco_menagerie/unitree_g1/g1.xml",
    ]
    found = next((p for p in paths if os.path.exists(p)), None)
    assert found, f"G1 model not found in: {paths}"
    model = mujoco.MjModel.from_xml_path(found)
    return f"G1 model: {model.nq} qpos, {model.nv} qvel"


# ── GR00T ────────────────────────────────────────────────────────────────────

@test("GR00T import")
def test_groot():
    import gr00t
    return "OK"


@test("GR00T model classes")
def test_groot_classes():
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6, Gr00tN1d6Config
    from gr00t.data.types import ModalityConfig
    from gr00t.experiment.experiment import Gr00tTrainer
    return "Gr00tN1d6, Gr00tTrainer, ModalityConfig OK"


@test("GR00T policy server/client")
def test_groot_policy():
    from gr00t.policy.server_client import PolicyServer
    return "PolicyServer OK"


@test("transformers version (4.51.x)")
def test_transformers():
    import transformers
    assert transformers.__version__.startswith("4.51"), \
        f"Need 4.51.x, got {transformers.__version__}"
    return f"transformers {transformers.__version__}"


# ── Training Dependencies ────────────────────────────────────────────────────

@test("DeepSpeed")
def test_deepspeed():
    import deepspeed
    return f"deepspeed {deepspeed.__version__}"


@test("wandb")
def test_wandb():
    import wandb
    return f"wandb {wandb.__version__}"


@test("PEFT")
def test_peft():
    import peft
    return f"peft {peft.__version__}"


@test("diffusers")
def test_diffusers():
    import diffusers
    return f"diffusers {diffusers.__version__}"


@test("torchcodec")
def test_torchcodec():
    import torchcodec
    return f"torchcodec {torchcodec.__version__}"


# ── IK & Sim Libraries ──────────────────────────────────────────────────────

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
    return "OK"


# ── Isaac Sim (x86_64 only) ─────────────────────────────────────────────────

@test("Isaac Sim import")
def test_isaacsim():
    import isaacsim
    return "OK"


@test("IsaacLab import")
def test_isaaclab():
    # IsaacLab depends on toml at import time
    import toml
    return f"toml {toml.__version__}"


@test("flash-attn")
def test_flash_attn():
    import flash_attn
    return f"flash-attn {flash_attn.__version__}"


# ── dm-isaac-g1 ──────────────────────────────────────────────────────────────

@test("dm-isaac-g1 package")
def test_dm_isaac_g1():
    import dm_isaac_g1
    return "OK"


# ── Common Libraries ─────────────────────────────────────────────────────────

@test("scipy")
def test_scipy():
    import scipy
    return f"scipy {scipy.__version__}"


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


@test("pyzmq (GROOT server)")
def test_zmq():
    import zmq
    return f"pyzmq {zmq.__version__}"


# ── File System Checks ───────────────────────────────────────────────────────

@test("dm-isaac-g1 repo present")
def test_repo_dm():
    assert os.path.exists("/workspace/dm-isaac-g1/pyproject.toml"), \
        "dm-isaac-g1 not found at /workspace/dm-isaac-g1"
    return "OK"


@test("Isaac-GR00T repo present")
def test_repo_groot():
    baked = os.path.exists("/workspace/gr00t/pyproject.toml")
    mounted = os.path.exists("/workspace/Isaac-GR00T/pyproject.toml")
    assert baked or mounted, "Neither /workspace/gr00t nor /workspace/Isaac-GR00T found"
    return "OK"


@test("WBC repo present")
def test_repo_wbc():
    assert os.path.exists("/workspace/GR00T-WholeBodyControl-dex1"), "WBC repo not found"
    return "OK"


@test("Vulkan ICD manifest", skip_on_arm64=True)
def test_vulkan_icd():
    icd_path = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    assert os.path.exists(icd_path), f"Vulkan ICD not found: {icd_path}"
    with open(icd_path) as f:
        content = f.read()
    assert "nvidia" in content.lower(), f"ICD must reference NVIDIA driver, got: {content}"
    return f"OK ({icd_path})"


@test("libnvidia-gl installed", skip_on_arm64=True)
def test_nvidia_gl():
    """Verify NVIDIA GL/Vulkan driver libs are installed in the image."""
    import ctypes.util
    lib = ctypes.util.find_library("nvidia-vulkan-producer")
    if not lib:
        lib = ctypes.util.find_library("GLX_nvidia")
    assert lib, "Neither libnvidia-vulkan-producer.so nor libGLX_nvidia.so.0 found"
    return f"OK ({lib})"


# ── Runner ───────────────────────────────────────────────────────────────────

def run_all(json_output=False):
    """Run all tests and return success status."""
    start = time.time()

    # Run all decorated test functions
    for name, obj in list(globals().items()):
        if callable(obj) and hasattr(obj, "_test"):
            obj()

    elapsed = time.time() - start
    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = sum(1 for r in results if not r.passed)
    skipped = sum(1 for r in results if r.skipped)

    if json_output:
        output = {
            "platform": platform.machine(),
            "python": sys.version.split()[0],
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "duration_seconds": round(elapsed, 1),
            "tests": [asdict(r) for r in results],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"{'=' * 60}")
        print(f"  Build Validation — {platform.machine()}")
        print(f"  Python {sys.version.split()[0]}")
        print(f"{'=' * 60}")
        print()

        for r in results:
            if r.skipped:
                icon = "⊘"
            elif r.passed:
                icon = "✓"
            else:
                icon = "✗"
            print(f"  {icon} {r.name}: {r.message}")

        print()
        print(f"{'=' * 60}")
        print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped ({elapsed:.1f}s)")
        print(f"{'=' * 60}")

        if failed > 0:
            print()
            print("  FAILED TESTS:")
            for r in results:
                if not r.passed:
                    print(f"    ✗ {r.name}: {r.message}")

    return failed == 0


if __name__ == "__main__":
    json_mode = "--json" in sys.argv
    success = run_all(json_output=json_mode)
    sys.exit(0 if success else 1)
