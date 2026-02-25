#!/usr/bin/env bash
set -euo pipefail
# ============================================================================
# Setup MuJoCo Menagerie G1 model for custom towel scene
# Run inside dm-workstation container
# ============================================================================

SCENE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MENAGERIE_DIR="/workspace/mujoco_menagerie"

echo "Setting up MuJoCo G1 towel-folding scene..."

# 1. Clone MuJoCo Menagerie (has the best G1 MJCF)
if [ -d "$MENAGERIE_DIR/unitree_g1" ]; then
    echo "  MuJoCo Menagerie already present at $MENAGERIE_DIR"
else
    echo "  Cloning MuJoCo Menagerie..."
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git "$MENAGERIE_DIR"
    echo "  Done."
fi

# 2. Verify the G1 model loads
echo "  Verifying G1 model..."
python3 -c "
import mujoco
m = mujoco.MjModel.from_xml_path('$MENAGERIE_DIR/unitree_g1/scene.xml')
print(f'  G1 model loaded: {m.nq} qpos, {m.nv} dof, {m.nu} actuators')
print(f'  Bodies: {m.nbody}, Geoms: {m.ngeom}')
"

# 3. Create symlink in Menagerie dir (required for mesh resolution)
echo "  Creating symlink for scene in Menagerie dir..."
ln -sf "$SCENE_DIR/g1_towel_folding.xml" "$MENAGERIE_DIR/unitree_g1/g1_towel_folding.xml"

# 4. Verify towel scene loads (via programmatic loader which strips keyframes)
echo "  Verifying towel scene loads..."
python3 -c "
import sys
sys.path.insert(0, '$(dirname $SCENE_DIR)')
from run_mujoco_towel_eval import load_towel_scene
m = load_towel_scene('$SCENE_DIR/g1_towel_folding.xml')
print(f'  Towel scene: {m.nq} qpos, {m.nv} dof, {m.nu} actuators, {m.nflex} flex')
"

echo ""
echo "Setup complete."
echo ""
echo "Scene file: $SCENE_DIR/g1_towel_folding.xml"
echo "Symlink:    $MENAGERIE_DIR/unitree_g1/g1_towel_folding.xml"
echo ""
echo "To run eval:"
echo "  python3 /workspace/dm-isaac-g1/scripts/eval/run_mujoco_towel_eval.py \\"
echo "      --scene $SCENE_DIR/g1_towel_folding.xml \\"
echo "      --model-path /workspace/checkpoints/groot-g1-gripper-fold-towel-full"
