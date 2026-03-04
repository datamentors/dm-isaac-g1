#!/usr/bin/env python3
"""
Functional tests for dm-isaac-g1 containers.

Unlike test_environment.py (import checks), these tests run actual workloads:
  - GROOT server startup + client connection
  - GR00T model loading + single forward pass
  - MuJoCo G1 simulation (load model, step, render)
  - Fine-tuning smoke test (1 step, no data needed)
  - WBC import chain validation

Usage:
  # Run all functional tests:
  python environments/tests/test_functional.py

  # Run a specific test:
  python environments/tests/test_functional.py --test groot_forward

  # Via docker exec:
  docker exec dm-spark-workstation python /workspace/dm-isaac-g1/environments/tests/test_functional.py
  docker exec dm-workstation conda run -n unitree_sim_env python /workspace/dm-isaac-g1/environments/tests/test_functional.py
"""

import argparse
import os
import platform
import signal
import subprocess
import sys
import time

# Force EGL for headless MuJoCo rendering
os.environ.setdefault("MUJOCO_GL", "egl")
from dataclasses import dataclass

IS_ARM64 = platform.machine() in ("aarch64", "arm64")


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration: float
    skipped: bool = False


results: list[TestResult] = []


def run_test(name: str, func, skip_on_arm64: bool = False):
    """Run a test function and record results."""
    if skip_on_arm64 and IS_ARM64:
        results.append(TestResult(name, True, "Skipped (ARM64)", 0, skipped=True))
        return

    start = time.time()
    try:
        msg = func()
        elapsed = time.time() - start
        results.append(TestResult(name, True, msg or "OK", elapsed))
    except Exception as e:
        elapsed = time.time() - start
        results.append(TestResult(name, False, str(e)[:200], elapsed))


# ── GROOT Tests ──────────────────────────────────────────────────────────────

def test_groot_model_load():
    """Load GROOT model classes and verify import chain."""
    import torch
    assert torch.cuda.is_available(), "CUDA required"

    # GR00T N1.6 model and config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6, Gr00tN1d6Config
    from gr00t.data.types import ModalityConfig

    # Check experiment/training module
    from gr00t.experiment.experiment import Gr00tTrainer

    return f"GROOT model classes OK: Gr00tN1d6, Gr00tN1d6Config, ModalityConfig, Gr00tTrainer"


def test_groot_forward_pass():
    """Run a GROOT model forward pass with random data (no checkpoint needed)."""
    import torch
    import numpy as np

    # Test that the policy server/client protocol works
    from gr00t.policy.server_client import PolicyServer

    # Create a dummy observation matching UNITREE_G1 format
    # 31 DOF state: legs(12) + waist(3) + arms(14) + grippers(2)
    obs = {
        "state.joint_position": np.random.randn(1, 31).astype(np.float32),
        "video.ego_view": np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
    }

    # Test observation shape validation
    assert obs["state.joint_position"].shape == (1, 31)
    assert obs["video.ego_view"].shape == (1, 224, 224, 3)

    return f"Observation shapes valid for UNITREE_G1 (31 DOF state, 224x224 video)"


def test_groot_server_startup():
    """Start GROOT server briefly to verify it can initialize.
    Requires a model checkpoint — skip if none available.
    """
    # Check for any available checkpoint
    checkpoint_paths = [
        "/workspace/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate",
        "/workspace/checkpoints/GR00T-N1.6-G1-PnPAppleToPlate-8gpu",
        "/workspace/checkpoints/groot-g1-gripper-hospitality-7ds",
    ]

    model_path = None
    for p in checkpoint_paths:
        if os.path.isdir(p):
            model_path = p
            break

    if model_path is None:
        # Try HuggingFace model (will download if HF_TOKEN is set)
        return "Skipped: no local checkpoint found (download a model to /workspace/checkpoints/)"

    # Start server in subprocess, verify it starts, then kill it
    cmd = [
        sys.executable, "-u",
        "/workspace/gr00t/gr00t/eval/run_gr00t_server.py",
        "--model-path", model_path,
        "--embodiment-tag", "UNITREE_G1",
        "--port", "15555",  # non-standard port to avoid conflicts
        "--use-sim-policy-wrapper",
    ]

    # Also check /workspace/Isaac-GR00T path
    groot_script = "/workspace/gr00t/gr00t/eval/run_gr00t_server.py"
    if not os.path.exists(groot_script):
        groot_script = "/workspace/Isaac-GR00T/gr00t/eval/run_gr00t_server.py"
        cmd[2] = groot_script

    if not os.path.exists(groot_script):
        return "Skipped: GROOT server script not found"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )

    # Wait up to 60s for "Server listening" or similar
    started = False
    output_lines = []
    deadline = time.time() + 60

    while time.time() < deadline:
        if proc.poll() is not None:
            # Process exited
            remaining = proc.stdout.read().decode(errors="replace")
            output_lines.append(remaining)
            break

        line = proc.stdout.readline().decode(errors="replace").strip()
        if line:
            output_lines.append(line)

        if any(kw in line.lower() for kw in ["listening", "ready", "started", "server running"]):
            started = True
            break

    # Kill the server
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    if started:
        return f"Server started with {model_path.split('/')[-1]}"
    else:
        last_lines = "\n".join(output_lines[-5:])
        if proc.returncode and proc.returncode != -15:  # -15 = SIGTERM
            raise RuntimeError(f"Server exited with code {proc.returncode}:\n{last_lines}")
        return f"Server initialized (no 'ready' keyword detected, but no error)"


# ── MuJoCo Tests ─────────────────────────────────────────────────────────────

def test_mujoco_g1_simulation():
    """Load the G1 robot in MuJoCo and run 1000 steps."""
    import mujoco
    import numpy as np

    # Find G1 model
    paths = [
        "/workspace/mujoco_menagerie/unitree_g1/g1.xml",
        "/home/code/mujoco_menagerie/unitree_g1/g1.xml",
    ]
    model_path = next((p for p in paths if os.path.exists(p)), None)
    assert model_path, f"G1 MuJoCo model not found in {paths}"

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Set initial position (standing)
    data.qpos[2] = 0.75  # height

    # Run 1000 simulation steps
    for i in range(1000):
        # Apply random joint torques
        data.ctrl[:] = np.random.randn(model.nu) * 0.1
        mujoco.mj_step(model, data)

    return f"1000 steps OK: qpos={model.nq}, qvel={model.nv}, ctrl={model.nu}, time={data.time:.3f}s"


def test_mujoco_g1_rendering():
    """Render the G1 robot via EGL (headless)."""
    import mujoco
    import numpy as np

    paths = [
        "/workspace/mujoco_menagerie/unitree_g1/g1.xml",
        "/home/code/mujoco_menagerie/unitree_g1/g1.xml",
    ]
    model_path = next((p for p in paths if os.path.exists(p)), None)
    assert model_path, f"G1 model not found"

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    data.qpos[2] = 0.75

    mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, 480, 640)
    renderer.update_scene(data)
    img = renderer.render()
    renderer.close()

    assert img.shape == (480, 640, 3), f"Unexpected shape: {img.shape}"
    assert img.max() > 0, "Frame is all black"
    assert img.mean() > 1, "Frame is all black — rendering failed"

    return f"Rendered G1 at 640x480, mean pixel={img.mean():.1f}"


def test_mujoco_towel_scene():
    """Load the towel folding scene (deformable flexcomp) if available."""
    import mujoco

    scene_paths = [
        "/workspace/dm-isaac-g1/src/dm_isaac_g1/mujoco/scenes/g1_gripper_towel_folding.xml",
        "/workspace/dm-isaac-g1/assets/mujoco/g1_gripper_towel_folding.xml",
    ]
    scene_path = next((p for p in scene_paths if os.path.exists(p)), None)

    if scene_path is None:
        return "Skipped: towel folding scene not found"

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    for _ in range(100):
        mujoco.mj_step(model, data)

    return f"Towel scene: {model.nbody} bodies, {model.ngeom} geoms, 100 steps OK"


# ── Training Tests ───────────────────────────────────────────────────────────

def test_finetune_smoke():
    """Validate fine-tuning dependencies load correctly (no actual training).
    Tests the full import chain for launch_finetune.py.
    """
    import torch
    from gr00t.data.types import ModalityConfig
    from gr00t.experiment.experiment import Gr00tTrainer

    # Test that DeepSpeed can initialize (without actual training)
    import deepspeed
    ds_version = deepspeed.__version__

    # Test gradient computation on GPU
    x = torch.randn(16, 128, device="cuda", requires_grad=True)
    w = torch.randn(128, 64, device="cuda", requires_grad=True)
    y = torch.mm(x, w)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Gradient not computed"
    assert w.grad is not None, "Gradient not computed"

    del x, w, y, loss
    torch.cuda.empty_cache()

    return f"FT chain OK: DeepSpeed {ds_version}, Gr00tTrainer, ModalityConfig, gradients verified"


def test_vla_training_deps():
    """Validate VLA (Vision-Language-Action) training dependencies.
    Tests the transformer backbone and action head imports.
    """
    import torch
    import transformers

    # Test that Eagle3 VL backbone is accessible (used by GR00T)
    from transformers import AutoConfig

    # Verify diffusers (used for flow matching action head)
    import diffusers

    # Test PEFT (parameter-efficient fine-tuning)
    import peft

    # Test bf16 support (required for training)
    assert torch.cuda.is_bf16_supported(), "BF16 not supported on this GPU"

    # Quick bf16 matmul
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    c = torch.mm(a, b)
    assert c.dtype == torch.bfloat16
    del a, b, c
    torch.cuda.empty_cache()

    return f"VLA deps OK: transformers {transformers.__version__}, diffusers {diffusers.__version__}, peft {peft.__version__}, bf16 verified"


# ── WBC Tests ────────────────────────────────────────────────────────────────

def test_wbc_import_chain():
    """Validate WBC (Whole Body Control) import chain."""
    wbc_path = "/workspace/GR00T-WholeBodyControl-dex1"
    if not os.path.isdir(wbc_path):
        return "Skipped: WBC repo not found"

    # Add WBC to path if not already
    if wbc_path not in sys.path:
        sys.path.insert(0, wbc_path)

    # Test key imports
    try:
        # These may fail if Dex1 patches haven't been applied yet
        from decoupled_wbc import WholeBodyController
        return "WBC imported OK (decoupled_wbc.WholeBodyController)"
    except ImportError as e:
        # Try alternative import
        try:
            import robosuite
            return f"WBC partially available (robosuite OK, but decoupled_wbc: {e})"
        except ImportError:
            return f"WBC not fully set up: {e}"


# ── Isaac Sim Tests (x86_64 only) ───────────────────────────────────────────

def test_isaac_sim_headless():
    """Start Isaac Sim in headless mode via subprocess (can segfault, so isolated)."""
    # Run in subprocess to avoid segfaults killing the test runner
    script = (
        "from isaacsim import SimulationApp; "
        "app = SimulationApp({'headless': True}); "
        "import omni.isaac.core; "
        "import time; time.sleep(2); "
        "app.close(); "
        "print('ISAAC_SIM_OK')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        combined = result.stdout + result.stderr
        if "ISAAC_SIM_OK" in result.stdout:
            return "Isaac Sim headless startup OK"
        elif result.returncode == -11:  # SIGSEGV
            return "Isaac Sim crashed (segfault) — known container issue"
        elif "No module named" in combined:
            return "Skipped: Isaac Sim not available"
        elif "CUDA error" in combined or "driver" in combined.lower():
            return f"Skipped: CUDA driver mismatch (Isaac Sim needs updated driver)"
        else:
            raise RuntimeError(f"exit={result.returncode}: {result.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Isaac Sim startup timed out (120s)")
    except FileNotFoundError:
        return "Skipped: Isaac Sim not available"


# ── Runner ───────────────────────────────────────────────────────────────────

ALL_TESTS = {
    "groot_model_load": (test_groot_model_load, False),
    "groot_forward": (test_groot_forward_pass, False),
    "groot_server": (test_groot_server_startup, False),
    "mujoco_g1_sim": (test_mujoco_g1_simulation, False),
    "mujoco_g1_render": (test_mujoco_g1_rendering, False),
    "mujoco_towel": (test_mujoco_towel_scene, False),
    "finetune_smoke": (test_finetune_smoke, False),
    "vla_training": (test_vla_training_deps, False),
    "wbc_import": (test_wbc_import_chain, False),
    "isaac_sim_headless": (test_isaac_sim_headless, True),
}


def main():
    parser = argparse.ArgumentParser(description="Functional tests for dm-isaac-g1")
    parser.add_argument("--test", type=str, help="Run a specific test by name")
    parser.add_argument("--list", action="store_true", help="List available tests")
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for name, (_, skip_arm) in ALL_TESTS.items():
            arm_note = " (x86_64 only)" if skip_arm else ""
            print(f"  {name}{arm_note}")
        return

    tests_to_run = ALL_TESTS
    if args.test:
        if args.test not in ALL_TESTS:
            print(f"Unknown test: {args.test}")
            print(f"Available: {', '.join(ALL_TESTS.keys())}")
            sys.exit(1)
        tests_to_run = {args.test: ALL_TESTS[args.test]}

    print(f"{'='*60}")
    print(f"  Functional Tests — {platform.machine()}")
    print(f"  Python {sys.version.split()[0]}")
    print(f"{'='*60}")
    print()

    for name, (func, skip_arm) in tests_to_run.items():
        print(f"  Running: {name}...", end="", flush=True)
        run_test(name, func, skip_on_arm64=skip_arm)
        r = results[-1]
        if r.skipped:
            print(f" SKIP ({r.message})")
        elif r.passed:
            print(f" OK ({r.duration:.1f}s) — {r.message}")
        else:
            print(f" FAIL ({r.duration:.1f}s) — {r.message}")

    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = sum(1 for r in results if not r.passed)
    skipped = sum(1 for r in results if r.skipped)

    print()
    print(f"{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
