#!/usr/bin/env python3
"""View G1 robot USD to inspect camera configuration.

This script loads the G1 robot USD directly (without ManagerBasedRLEnv)
to inspect the camera prim paths and configuration.

IMPORTANT: Run this with Isaac Sim's Python:
    /isaac-sim/python.sh scripts/view_g1_usd.py
"""

import sys
import os
import argparse

# ============================================================================
# STEP 1: Import Isaac Sim FIRST
# ============================================================================
print("Step 1: Launching Isaac Sim...")
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "width": 1920,
    "height": 1080,
    "anti_aliasing": 1,
    "livestream": 1,
})

# ============================================================================
# STEP 2: Import Omniverse modules after SimulationApp starts
# ============================================================================
print("Step 2: Importing Omniverse modules...")
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom, Sdf

# ============================================================================
# STEP 3: Find the G1 robot USD
# ============================================================================
print("Step 3: Finding G1 robot USD...")

# Common locations for G1 USD files
G1_USD_PATHS = [
    # Isaac Sim assets
    "/isaac-sim/exts/isaacsim.robot.assets/data/Robots/Unitree/G1",
    # IsaacLab assets
    "/workspace/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/G1",
    # Unitree sim isaaclab
    "/workspace/unitree_sim_isaaclab/assets/robots/g1",
]

# Find available USD files
available_usds = []
for base_path in G1_USD_PATHS:
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            for f in files:
                if f.endswith(".usd") or f.endswith(".usda") or f.endswith(".usdc"):
                    full_path = os.path.join(root, f)
                    available_usds.append(full_path)

print(f"\nFound {len(available_usds)} USD files:")
for i, usd in enumerate(available_usds[:20]):  # Show first 20
    print(f"  {i}: {usd}")

# ============================================================================
# STEP 4: Create world and load robot
# ============================================================================
print("\nStep 4: Creating world...")
world = World()
world.scene.add_default_ground_plane()

# Load first available USD (or specify one)
if available_usds:
    # Try to find an Inspire version
    inspire_usds = [u for u in available_usds if "inspire" in u.lower()]
    if inspire_usds:
        robot_usd = inspire_usds[0]
    else:
        robot_usd = available_usds[0]

    print(f"\nLoading robot USD: {robot_usd}")
    add_reference_to_stage(usd_path=robot_usd, prim_path="/World/Robot")
else:
    print("No G1 USD files found!")
    robot_usd = None

# ============================================================================
# STEP 5: Find all camera prims
# ============================================================================
print("\nStep 5: Searching for camera prims...")
stage = simulation_app.context.get_stage()

cameras = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera):
        cameras.append(prim)

print(f"\nFound {len(cameras)} cameras in scene:")
for cam in cameras:
    path = str(cam.GetPath())
    print(f"\n  Camera: {path}")

    # Get camera attributes
    cam_api = UsdGeom.Camera(cam)
    if cam_api:
        # Get local transform
        xform = UsdGeom.Xformable(cam)
        local_transform = xform.GetLocalTransformation()
        print(f"    Local Transform: {local_transform}")

        # Get focal length
        focal = cam_api.GetFocalLengthAttr().Get()
        print(f"    Focal Length: {focal}")

# Also search for prims that might be camera mounts (d435_link, etc)
print("\n\nSearching for camera-related prims (d435, camera, cam)...")
for prim in stage.Traverse():
    prim_name = prim.GetName().lower()
    if "d435" in prim_name or "camera" in prim_name or "cam" in prim_name:
        print(f"  Found: {prim.GetPath()}")

# ============================================================================
# STEP 6: Keep running to view in VNC
# ============================================================================
print("\n" + "="*60)
print("Scene loaded. Connect via VNC to view:")
print("  VNC address: 192.168.1.205:5901")
print("="*60)
print("\nPress Ctrl+C to exit.\n")

# Reset world
world.reset()

try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    print("\nShutting down...")

simulation_app.close()
