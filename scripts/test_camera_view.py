#!/usr/bin/env python3
"""
Simple test script to verify camera view in Isaac Sim.

This script loads a G1 robot and captures camera frames to verify
the camera position and orientation are correct for GROOT inference.

It uses the standard Isaac Sim camera API (not TiledCameraCfg) to avoid
compatibility issues with synthetic data pipelines.

Usage:
    export DISPLAY=:1
    export PYTHONPATH=/workspace/dm-isaac-g1/src:/workspace/IsaacLab/source/isaaclab:/workspace/IsaacLab/source/isaaclab_assets
    /isaac-sim/python.sh scripts/test_camera_view.py
"""

import sys
import os

print("Step 1: Launching Isaac Sim...")
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,  # Not headless - we want to see in VNC
    "width": 1280,
    "height": 720,
    "anti_aliasing": 1,
    "livestream": 1,  # Enable streaming for VNC
})

print("Step 2: Importing modules...")
import numpy as np
from PIL import Image

# Isaac Sim imports
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
from pxr import Usd, UsdGeom, Gf

# Camera imports
from omni.isaac.sensor import Camera

# Output directory
OUTPUT_DIR = "/tmp/groot_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Step 3: Creating world...")
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Find G1 robot USD
G1_USD_PATHS = [
    "/workspace/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd",
    "/workspace/unitree_sim_isaaclab/assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_29dof_inspire_hand.usd",
]

robot_usd = None
for path in G1_USD_PATHS:
    if path.startswith("http") or os.path.exists(path):
        robot_usd = path
        break

if robot_usd:
    print(f"Step 4: Loading robot USD: {robot_usd}")
    add_reference_to_stage(usd_path=robot_usd, prim_path="/World/Robot")
else:
    print("WARNING: No G1 USD found, creating simple test scene")

# Add a red sphere as a target object (like an apple)
from omni.isaac.core.objects import DynamicSphere
target = world.scene.add(
    DynamicSphere(
        prim_path="/World/Target",
        name="target",
        position=np.array([0.5, 0.0, 0.8]),  # 50cm in front, 80cm high
        radius=0.04,  # Apple-sized
        color=np.array([0.9, 0.1, 0.1]),  # Red
    )
)

print("Step 5: Creating camera...")

# Camera configuration based on Unitree d435_link settings
# Position relative to robot base
CAMERA_HEIGHT = 1.2  # Head height above ground
CAMERA_FORWARD = 0.15  # Forward from center
CAMERA_PITCH_DOWN = 30  # Degrees looking down at workspace

# Create camera at fixed world position looking at robot's workspace
camera = Camera(
    prim_path="/World/FrontCamera",
    name="front_camera",
    position=np.array([CAMERA_FORWARD, 0.0, CAMERA_HEIGHT]),
    frequency=30,
    resolution=(640, 480),
)

# Set camera orientation - looking forward and slightly down
# Rotation order: pitch down (around Y axis in camera frame)
import math
pitch_rad = math.radians(CAMERA_PITCH_DOWN)
# Quaternion for looking forward with pitch down
# In Isaac Sim, camera looks along -Z by default
# We need to rotate to look forward-down
quat = Gf.Quatf(
    math.cos(pitch_rad / 2),  # w
    0.0,  # x
    math.sin(pitch_rad / 2),  # y (pitch)
    0.0,  # z
)
camera.set_world_pose(
    position=np.array([CAMERA_FORWARD, 0.0, CAMERA_HEIGHT]),
    orientation=np.array([quat.GetReal(), *quat.GetImaginary()]),  # wxyz
)

world.scene.add(camera)

print("Step 6: Initializing camera...")
camera.initialize()

print("Step 7: Running simulation...")
world.reset()

# Take a few frames to let things settle
for i in range(30):
    world.step(render=True)

print("Step 8: Capturing camera frame...")
camera.add_motion_vectors_to_frame()

# Capture RGB image
rgb_data = camera.get_rgb()
if rgb_data is not None:
    # Save as PNG
    img = Image.fromarray(rgb_data)
    output_path = os.path.join(OUTPUT_DIR, "test_camera_view.png")
    img.save(output_path)
    print(f"Saved camera view to: {output_path}")
    print(f"Image shape: {rgb_data.shape}")
else:
    print("WARNING: No RGB data from camera")

# Get camera info
print("\nCamera Info:")
print(f"  Position: {camera.get_world_pose()[0]}")
print(f"  Orientation: {camera.get_world_pose()[1]}")
print(f"  Resolution: {camera.get_resolution()}")

print("\n" + "="*60)
print("Test complete!")
print(f"Camera frame saved to: {OUTPUT_DIR}/test_camera_view.png")
print("Connect to VNC at 192.168.1.205:5901 to see live view")
print("="*60)

# Keep running for VNC viewing
print("\nPress Ctrl+C to exit...")
try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    print("\nShutting down...")

simulation_app.close()
