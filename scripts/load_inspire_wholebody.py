#!/usr/bin/env python3
"""Load G1 Inspire Wholebody scene to view camera configuration.

This script loads the Unitree G1 Inspire wholebody scene which has cameras
defined (unlike the DEX3 scene) to verify camera setup.

IMPORTANT: Run this with Isaac Sim's Python:
    /isaac-sim/python.sh scripts/load_inspire_wholebody.py
"""

import sys
import os

# ============================================================================
# STEP 1: Import Isaac Sim FIRST (before any other imports)
# This ensures Isaac Sim's torch is used, not IsaacLab venv's
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
# STEP 2: Add pink/pinocchio paths AFTER Isaac Sim starts
# ============================================================================
print("Step 2: Setting up pink/pinocchio paths...")
ISAACLAB_VENV_SITE = "/workspace/IsaacLab/env_isaaclab/lib/python3.11/site-packages"
ISAACLAB_CMEEL_SITE = f"{ISAACLAB_VENV_SITE}/cmeel.prefix/lib/python3.11/site-packages"
ISAACLAB_CMEEL_LIB = f"{ISAACLAB_VENV_SITE}/cmeel.prefix/lib"

# Add to Python path for pink imports
if ISAACLAB_CMEEL_SITE not in sys.path:
    sys.path.insert(0, ISAACLAB_CMEEL_SITE)
if ISAACLAB_VENV_SITE not in sys.path:
    sys.path.insert(0, ISAACLAB_VENV_SITE)

# Add cmeel lib to LD_LIBRARY_PATH for shared libraries
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
if ISAACLAB_CMEEL_LIB not in current_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{ISAACLAB_CMEEL_LIB}:{current_ld}"

# ============================================================================
# STEP 3: Now import Isaac Lab and extensions
# ============================================================================
print("Step 3: Importing Isaac Lab modules...")
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict

# ============================================================================
# STEP 4: Import the Unitree task for G1 Inspire
# ============================================================================
print("Step 4: Importing Unitree G1 Inspire task...")

# Add unitree_sim_isaaclab to path
unitree_path = "/workspace/unitree_sim_isaaclab"
if unitree_path not in sys.path:
    sys.path.insert(0, unitree_path)

# Import the wholebody config directly
from tasks.locomotion.velocity.config.g1.g1_inspire_wholebody.rough_env_cfg import (
    G1Inspire29dofWholeBodyRoughEnvCfg,
)

print("Step 5: Creating environment configuration...")
env_cfg = G1Inspire29dofWholeBodyRoughEnvCfg()
env_cfg.scene.num_envs = 1

# Disable terrain for faster loading
env_cfg.scene.terrain.terrain_type = "plane"
env_cfg.scene.terrain.terrain_generator = None

print("Step 6: Creating environment...")
env = ManagerBasedRLEnv(cfg=env_cfg)

print("\n" + "="*60)
print("G1 Inspire Wholebody Scene Loaded Successfully!")
print("="*60)
print("\nScene configuration:")
print(f"  Robot: G1 29 DOF with Inspire hands")
print(f"  Number of environments: {env_cfg.scene.num_envs}")

# Print camera info if available
if hasattr(env_cfg.scene, "camera"):
    print(f"\nCamera configuration:")
    cam = env_cfg.scene.camera
    print(f"  Name: {getattr(cam, 'name', 'N/A')}")
    print(f"  Prim path: {getattr(cam, 'prim_path', 'N/A')}")
    print(f"  Offset: {getattr(cam, 'offset', 'N/A')}")
elif hasattr(env_cfg.scene, "tiled_camera"):
    print(f"\nTiled Camera configuration:")
    cam = env_cfg.scene.tiled_camera
    print(f"  Name: {getattr(cam, 'name', 'N/A')}")
    print(f"  Prim path: {getattr(cam, 'prim_path', 'N/A')}")
    print(f"  Offset: {getattr(cam, 'offset', 'N/A')}")
else:
    print("\nNo camera configuration found in scene config.")
    print("Checking for camera in robot definition...")

print("\nIsaac Sim is running. Connect via VNC to view the scene.")
print("VNC address: 192.168.1.205:5901")
print("Press Ctrl+C to exit.\n")

# Keep running
try:
    while simulation_app.is_running():
        env.step(env.action_space.sample() * 0)
except KeyboardInterrupt:
    print("\nShutting down...")

env.close()
simulation_app.close()
