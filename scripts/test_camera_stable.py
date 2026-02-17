#!/usr/bin/env python3
"""
Stable camera test for Isaac Sim 5.1.0.

This script tests camera capture with all known workarounds for Isaac Sim 5.1.0:
1. Proper simulation app initialization with async rendering disabled
2. Correct extension loading order
3. World and scene setup before camera creation
4. Multiple render steps before capture

Based on:
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/known_issues.html
- https://github.com/unitreerobotics/unitree_sim_isaaclab

Usage:
    # Method 1: Use the stable wrapper script
    /usr/local/bin/isaac-python-stable scripts/test_camera_stable.py

    # Method 2: Direct with flags
    /isaac-sim/python.sh --/exts/isaacsim.core.throttling/enable_async=false scripts/test_camera_stable.py

    # Method 3: Set env vars first
    export ISAACSIM_DISABLE_ASYNC_RENDERING=1
    /isaac-sim/python.sh scripts/test_camera_stable.py
"""

import sys
import os
import argparse

# ============================================================================
# STEP 1: Parse args BEFORE importing anything from Isaac Sim
# ============================================================================
parser = argparse.ArgumentParser(description="Test camera in Isaac Sim 5.1.0")
parser.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
parser.add_argument("--output-dir", type=str, default="/tmp/groot_debug", help="Output directory")
parser.add_argument("--num-frames", type=int, default=60, help="Frames to render before capture")
args, unknown = parser.parse_known_args()

OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STEP 2: Launch SimulationApp with Isaac Sim 5.1.0 workarounds
# ============================================================================
print("=" * 60)
print("Isaac Sim 5.1.0 Stable Camera Test")
print("=" * 60)
print(f"\nStep 1: Launching Isaac Sim (headless={args.headless})...")

from isaacsim import SimulationApp

# Configuration with workarounds for Isaac Sim 5.1.0
# - anti_aliasing=0: Disable for synthetic data stability
# - Explicit renderer settings for camera capture
sim_config = {
    "headless": args.headless,
    "width": 1280,
    "height": 720,
    "anti_aliasing": 0,  # Disable AA for synthetic data
    "renderer": "RayTracedLighting",  # Explicit renderer
    "display_options": 3105,  # Standard display
}

# Add livestream only if not headless
if not args.headless:
    sim_config["livestream"] = 1

simulation_app = SimulationApp(sim_config)

print("  SimulationApp created successfully")

# ============================================================================
# STEP 3: Apply runtime workarounds via Carb settings
# ============================================================================
print("\nStep 2: Applying Isaac Sim 5.1.0 workarounds...")

import carb.settings

settings = carb.settings.get_settings()

# Disable async rendering for camera stability
# This prevents frame skipping with Replicator/synthetic data
settings.set("/exts/isaacsim.core.throttling/enable_async", False)
print("  - Disabled async rendering")

# Set DLSS to Quality mode for low resolution cameras
# Value 2 = Quality mode (recommended for < 600x600)
settings.set("/rtx/post/dlss/mode", 2)
print("  - Set DLSS to Quality mode")

# Increase subframes for material loading (synthetic data)
settings.set("/rtx/replicator/rt_subframes", 4)
print("  - Set rt_subframes=4 for material stability")

# ============================================================================
# STEP 4: Import Isaac Sim modules AFTER SimulationApp is running
# ============================================================================
print("\nStep 3: Importing Isaac Sim modules...")

import numpy as np
from PIL import Image

# Core Isaac Sim imports
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from pxr import Usd, UsdGeom, Gf

print("  - Core modules imported")

# ============================================================================
# STEP 5: Create World and add ground plane FIRST
# ============================================================================
print("\nStep 4: Creating world...")

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
print("  - World created with ground plane")

# ============================================================================
# STEP 6: Load robot (if available)
# ============================================================================
print("\nStep 5: Loading robot...")

G1_USD_PATHS = [
    "/workspace/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd",
    "/workspace/unitree_sim_isaaclab/assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd",
]

robot_loaded = False
for path in G1_USD_PATHS:
    if os.path.exists(path):
        print(f"  Loading: {path}")
        add_reference_to_stage(usd_path=path, prim_path="/World/Robot")
        robot_loaded = True
        break

if not robot_loaded:
    print("  WARNING: No G1 USD found, using simple test scene")
    # Add a simple object instead
    from omni.isaac.core.objects import DynamicCuboid
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/TestCube",
            name="test_cube",
            position=np.array([0.5, 0.0, 0.5]),
            size=0.2,
            color=np.array([0.2, 0.5, 0.8]),
        )
    )

# Add a red sphere as target object
from omni.isaac.core.objects import DynamicSphere
world.scene.add(
    DynamicSphere(
        prim_path="/World/Target",
        name="target",
        position=np.array([0.5, 0.0, 0.8]),
        radius=0.04,
        color=np.array([0.9, 0.1, 0.1]),
    )
)
print("  - Target sphere added")

# ============================================================================
# STEP 7: Create camera using simple USD approach (most stable)
# ============================================================================
print("\nStep 6: Creating camera via USD (stable method)...")

import math
from pxr import UsdGeom, Sdf

stage = omni.usd.get_context().get_stage()

# Camera parameters (Unitree head camera style)
CAMERA_HEIGHT = 1.2  # meters above ground
CAMERA_FORWARD = 0.15  # meters forward
CAMERA_PITCH_DOWN = 30  # degrees

# Create camera prim directly via USD (most reliable)
camera_path = "/World/StableCamera"
camera_prim = stage.DefinePrim(camera_path, "Camera")
camera_api = UsdGeom.Camera(camera_prim)

# Set camera properties
camera_api.CreateFocalLengthAttr(18.15)
camera_api.CreateHorizontalApertureAttr(20.955)
camera_api.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))

# Set camera transform
xform = UsdGeom.Xformable(camera_prim)
xform.ClearXformOpOrder()

# Translation
translate_op = xform.AddTranslateOp()
translate_op.Set(Gf.Vec3d(CAMERA_FORWARD, 0.0, CAMERA_HEIGHT))

# Rotation (pitch down to look at workspace)
# In USD, cameras look along -Z by default
# We rotate around X axis to pitch down
pitch_rad = math.radians(CAMERA_PITCH_DOWN)
rotate_op = xform.AddRotateXYZOp()
rotate_op.Set(Gf.Vec3f(-CAMERA_PITCH_DOWN, 0.0, 0.0))  # Negative = looking down

print(f"  - Camera created at {camera_path}")
print(f"  - Position: ({CAMERA_FORWARD}, 0.0, {CAMERA_HEIGHT})")
print(f"  - Pitch: -{CAMERA_PITCH_DOWN} degrees")

# ============================================================================
# STEP 8: Reset world and render multiple frames (critical for stability)
# ============================================================================
print(f"\nStep 7: Resetting world and rendering {args.num_frames} frames...")

world.reset()

# Render multiple frames to let everything settle
# This is critical for camera and physics stability
for i in range(args.num_frames):
    world.step(render=True)
    if (i + 1) % 20 == 0:
        print(f"  - Rendered {i + 1}/{args.num_frames} frames")

print("  - World stabilized")

# ============================================================================
# STEP 9: Capture frame using Replicator (Isaac Sim 5.1.0 method)
# ============================================================================
print("\nStep 8: Capturing camera frame...")

try:
    import omni.replicator.core as rep

    # Create render product for our camera
    render_product = rep.create.render_product(camera_path, (640, 480))

    # Create annotator for RGB capture
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])

    # Step and capture
    rep.orchestrator.step()

    # Get RGB data
    rgb_data = rgb_annotator.get_data()

    if rgb_data is not None and len(rgb_data) > 0:
        # Convert to image
        if isinstance(rgb_data, dict):
            rgb_array = rgb_data.get("data", rgb_data)
        else:
            rgb_array = rgb_data

        # Ensure correct shape and type
        rgb_array = np.array(rgb_array)
        if rgb_array.ndim == 3 and rgb_array.shape[2] == 4:
            rgb_array = rgb_array[:, :, :3]  # Remove alpha

        img = Image.fromarray(rgb_array.astype(np.uint8))
        output_path = os.path.join(OUTPUT_DIR, "camera_replicator.png")
        img.save(output_path)
        print(f"  - Saved: {output_path}")
        print(f"  - Shape: {rgb_array.shape}")
    else:
        print("  - WARNING: No RGB data from Replicator")

    # Cleanup
    rgb_annotator.detach([render_product])

except Exception as e:
    print(f"  - Replicator capture failed: {e}")
    print("  - Trying alternative viewport capture...")

    # Alternative: Capture from viewport
    try:
        import omni.kit.viewport.utility as viewport_util

        viewport = viewport_util.get_active_viewport()
        if viewport:
            # Set viewport camera
            viewport.set_active_camera(camera_path)

            # Wait for render
            for _ in range(10):
                world.step(render=True)

            # Capture
            import omni.kit.capture.viewport as capture
            output_path = os.path.join(OUTPUT_DIR, "camera_viewport.png")
            capture.capture_viewport_to_file(viewport, output_path)
            print(f"  - Saved viewport capture: {output_path}")
    except Exception as e2:
        print(f"  - Viewport capture also failed: {e2}")

# ============================================================================
# STEP 10: Print camera info
# ============================================================================
print("\n" + "=" * 60)
print("Camera Configuration:")
print("=" * 60)
print(f"  Path: {camera_path}")
print(f"  Position: ({CAMERA_FORWARD}, 0.0, {CAMERA_HEIGHT}) meters")
print(f"  Pitch: -{CAMERA_PITCH_DOWN} degrees (looking down)")
print(f"  Focal Length: 18.15 mm")
print(f"  Horizontal Aperture: 20.955 mm")
print(f"  Resolution: 640x480")
print(f"\nOutput directory: {OUTPUT_DIR}")

# ============================================================================
# STEP 11: Keep running or exit
# ============================================================================
if not args.headless:
    print("\n" + "=" * 60)
    print("Scene loaded. Connect via VNC to view:")
    print("  VNC address: 192.168.1.205:5901")
    print("=" * 60)
    print("\nPress Ctrl+C to exit...\n")

    try:
        while simulation_app.is_running():
            world.step(render=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
else:
    print("\nHeadless mode - shutting down...")

# Proper cleanup to avoid segfault
world.stop()
simulation_app.close()
print("Done.")
