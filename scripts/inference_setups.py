"""
Inference Setup Configurations for GR00T Policy Inference

This module defines named experimental setups that can be selected when running
policy_inference_groot_g1.py via --setup <name>.

Each setup is an InferenceSetup dataclass that fully describes the inference
environment: cameras, scene, DOF layout, language command, and object placement.
Setups are robot-agnostic — the same config system supports G1 with Inspire hands,
G1 with Dex3 hands, or any future robot/hand combination.

Adding a new setup:
  1. Add an InferenceSetup instance to SETUPS dict below.
  2. Run inference with --setup <your_setup_name>.
  3. No other changes needed — the script reads all parameters from the setup.

=============================================================================
CAMERA CONFIGURATION
=============================================================================

Cameras are defined as a list of CameraSpec entries.  The first camera in the
list is used as the PRIMARY observation camera (mapped to "cam_left_high" by
default — the training-data key for the main camera).

Additional cameras are spawned in Isaac Sim and their images are passed to the
model as extra_camera_rgbs.

Special handling:
  - "DUPLICATE_PRIMARY" as prim_path: the primary camera image is duplicated
    under a different name (e.g. cam_right_high = same head camera view).
  - camera_parent=None → use d435_link (Unitree default head camera)
  - camera_parent="__world__" → world-fixed camera
  - camera_parent="<link_name>" → attach to named robot link

=============================================================================
DOF LAYOUT
=============================================================================

The dof_layout field maps DOF index ranges to body part groups, enabling the
script to correctly build state/action vectors for any DOF configuration:

  28 DOF (Dex3):   arms(14) + dex3_hands(14)
  53 DOF (Inspire): legs(12) + waist(3) + arms(14) + inspire_hands(24)

=============================================================================
SCENE COORDINATE SYSTEM — CRITICAL REFERENCE
=============================================================================

Robot spawn:   (-0.15, 0.0, 0.76) in world frame
Robot rotation: (0.7071, 0, 0, 0.7071) = 90° around X → robot faces +Y direction

Table (packing_table): world pos = (0.0, 0.55, -0.2)
  Table model height ≈ 1.07 m → table surface at world Z ≈ -0.2 + 1.07 = 0.87 m
  Table center is at Y=0.55 — directly in FRONT of robot (robot faces +Y)

Object reach zone (on table surface, reachable by arm):
  X: -0.35 to +0.05  (left-right relative to robot facing direction)
  Y:  0.35 to 0.55   (in front of robot — toward table center)
  Z:  0.84 to 0.87   (table surface + object height)

Camera (d435_link/front_cam):
  Parent: d435_link on robot head
  ROS quaternion: (0.5, -0.5, 0.5, -0.5) = forward-facing in robot frame
  Because robot is rotated 90° around X, "forward" in robot frame = +Y in world frame

=============================================================================
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraSpec:
    """Specification for a single camera in the inference setup.

    Attributes:
        name: Training-data camera name (e.g., "cam_left_high", "cam_left_wrist").
              This name is used as the key in the video observation dict sent to the model.
        prim_path: USD prim path template. Use {ENV_REGEX_NS} for env regex namespace.
                   Special value "DUPLICATE_PRIMARY" means this camera duplicates
                   the primary camera's image (no extra sensor spawned).
        camera_parent: Camera mount point:
                       None       → use d435_link (Unitree default head camera)
                       "__world__" → world-fixed camera
                       "<link>"   → attach to named robot link
        pos: (x, y, z) position offset from camera_parent.
        rot: (w, x, y, z) quaternion, ROS convention.
        focal_length: Camera focal length in mm.
        horizontal_aperture: Horizontal aperture in mm.
    """
    name: str
    prim_path: str = ""                          # empty → use d435_link default
    camera_parent: Optional[str] = None          # None → d435_link
    pos: tuple = (0.0, 0.0, 0.0)
    rot: tuple = (0.5, -0.5, 0.5, -0.5)         # default: Unitree d435 forward-facing
    focal_length: float = 7.6
    horizontal_aperture: float = 20.0

    @property
    def is_duplicate(self) -> bool:
        return self.prim_path == "DUPLICATE_PRIMARY"


@dataclass
class InferenceSetup:
    """Complete inference configuration: cameras, scene, DOF layout, objects.

    This dataclass is robot/hand/DOF-agnostic. The script reads all parameters
    from the setup and configures itself accordingly.
    """

    description: str

    # ----- Cameras -----
    # List of CameraSpec objects. First entry = primary camera ("cam_left_high").
    # Additional entries are extra cameras spawned in Isaac Sim.
    # If empty, a default d435_link camera is used (backward compat).
    cameras: list = field(default_factory=list)

    # Legacy single-camera fields (used when cameras list is empty)
    camera_parent: Optional[str] = None
    camera_pos: tuple = (0.0, 0.0, 0.0)
    camera_rot: tuple = (0.5, -0.5, 0.5, -0.5)
    focal_length: float = 7.6
    horizontal_aperture: float = 20.0

    # ----- Scene -----
    # Scene name from AVAILABLE_SCENES. If set, overrides --scene CLI arg.
    scene: Optional[str] = None

    # Language command. If set, overrides --language CLI arg.
    language: Optional[str] = None

    # Scene layout — initial world positions (x, y, z)
    object_pos: tuple = (-0.35, 0.40, 0.87)
    plate_pos: tuple = (0.15, 0.50, 0.865)
    plate_radius: float = 0.06

    # ----- DOF Layout -----
    # Maps body part name → (start_idx, end_idx) in the flat state/action vector.
    # If empty, the script auto-detects from statistics.json (backward compat).
    # Example for 28 DOF Dex3:
    #   {"left_arm": (0,7), "right_arm": (7,14), "left_hand": (14,21), "right_hand": (21,28)}
    dof_layout: dict = field(default_factory=dict)

    # ----- Joint Configuration -----
    # Maps body part name → ordered list of joint names for the robot.
    # Used to resolve joint indices from the robot articulation.
    # If empty, the script uses G1-specific defaults (backward compat).
    # Example:
    #   {"left_arm": ["left_shoulder_pitch_joint", ...], "left_hand": ["left_hand_*"]}
    joint_groups: dict = field(default_factory=dict)

    # Ordered list of joint name patterns for the action space.
    # These define which joints the action manager controls and in what order.
    # If empty, the script auto-detects from hand type (backward compat).
    # Supports regex patterns (e.g., "left_hand_.*").
    action_joint_patterns: list = field(default_factory=list)

    # ----- Robot Configuration -----
    # Hand type: "dex3", "inspire", "gripper", or None for auto-detect.
    # Controls which hand joint names to use and how to resolve them.
    hand_type: Optional[str] = None

    # Whether to fix the root link (True for manipulation, False for locomotion).
    fix_root_link: bool = True

    # Whether to disable lower body policy (True for tabletop manipulation).
    disable_lower_body: bool = True

    # Whether to spawn extra objects (apple + plate) on top of the scene.
    # Set False for scenes that already have their own objects (e.g. block stacking).
    spawn_objects: bool = True

    # Joint names whose action targets should be negated before sending to the sim.
    # This fixes sign convention mismatches between real robot and sim URDF.
    # For example, if the real Dex3 hand uses positive values for finger curl but
    # the sim URDF uses negative values for the same motion, the joint should be
    # listed here. The negation is applied AFTER the model predicts the action
    # and BEFORE sending to env.step().
    negate_action_joints: list = field(default_factory=list)

    def get_cameras(self) -> list:
        """Return camera list, building from legacy fields if needed."""
        if self.cameras:
            return self.cameras
        # Build single-camera list from legacy fields
        return [CameraSpec(
            name="cam_left_high",
            camera_parent=self.camera_parent,
            pos=self.camera_pos,
            rot=self.camera_rot,
            focal_length=self.focal_length,
            horizontal_aperture=self.horizontal_aperture,
        )]

    def get_primary_camera(self) -> CameraSpec:
        """Return the primary (first non-duplicate) camera."""
        for cam in self.get_cameras():
            if not cam.is_duplicate:
                return cam
        # Fallback: legacy fields
        return CameraSpec(
            name="cam_left_high",
            camera_parent=self.camera_parent,
            pos=self.camera_pos,
            rot=self.camera_rot,
            focal_length=self.focal_length,
            horizontal_aperture=self.horizontal_aperture,
        )

    def get_extra_cameras(self) -> list:
        """Return extra cameras (everything except the primary, excluding duplicates that reference primary)."""
        all_cams = self.get_cameras()
        if len(all_cams) <= 1:
            return []
        # Everything after the first camera, but only physical cameras (not duplicates)
        return [c for c in all_cams[1:] if not c.is_duplicate]

    def get_duplicate_cameras(self) -> list:
        """Return cameras that duplicate the primary camera image."""
        return [c for c in self.get_cameras() if c.is_duplicate]


# =============================================================================
# Unitree D435 head camera defaults (shared by many setups)
# =============================================================================
_D435_HEAD = CameraSpec(
    name="cam_left_high",
    camera_parent=None,  # d435_link
    pos=(0.0, 0.0, 0.0),
    rot=(0.5, -0.5, 0.5, -0.5),
)


# =============================================================================
# Named setups
# =============================================================================

SETUPS: dict[str, InferenceSetup] = {

    # =========================================================================
    # G1 + Inspire hands (53 DOF) — Pick & Place
    # =========================================================================

    # -------------------------------------------------------------------------
    "default": InferenceSetup(
        description=(
            "d435_link camera (Unitree default). "
            "Robot faces +Y, table at Y=0.55, surface Z=0.87. "
            "Apple at (-0.35, 0.40, 0.87) matches training cylinder position. "
            "Plate smaller (r=0.06m) and placed further from start pose to avoid hand collision."
        ),
        camera_parent=None,
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.35, 0.40, 0.87),
        plate_pos=(0.15, 0.50, 0.865),
        plate_radius=0.06,
    ),

    # -------------------------------------------------------------------------
    "option_a": InferenceSetup(
        description=(
            "d435_link camera. Apple at canonical table position (-0.35, 0.40, 0.87). "
            "Plate placed at (0.15, 0.50) — away from hands' initial pose, smaller radius. "
            "This is the primary test setup: correct camera, correct table positions."
        ),
        camera_parent=None,
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.35, 0.40, 0.87),
        plate_pos=(0.15, 0.50, 0.865),
        plate_radius=0.06,
    ),

    # -------------------------------------------------------------------------
    "option_a_apple_center": InferenceSetup(
        description=(
            "Apple at table center (X=-0.15, Y=0.45), plate further right (X=0.10, Y=0.52). "
            "Tests pick from center of reach zone. Smaller plate (r=0.06m)."
        ),
        camera_parent=None,
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.15, 0.45, 0.87),
        plate_pos=(0.10, 0.52, 0.865),
        plate_radius=0.06,
    ),

    # -------------------------------------------------------------------------
    "option_a_apple_left": InferenceSetup(
        description=(
            "Apple far left (X=-0.45, Y=0.35), plate center (X=0.05, Y=0.50). "
            "Tests reaching to the far-left side of the table."
        ),
        camera_parent=None,
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.45, 0.35, 0.87),
        plate_pos=(0.05, 0.50, 0.865),
        plate_radius=0.06,
    ),

    # -------------------------------------------------------------------------
    "option_a_closer": InferenceSetup(
        description=(
            "Apple near table edge (X=-0.30, Y=0.30), plate further (X=0.05, Y=0.45). "
            "Tests short-reach pickup from the near edge of the table."
        ),
        camera_parent=None,
        camera_pos=(0.0, 0.0, 0.0),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.30, 0.30, 0.87),
        plate_pos=(0.05, 0.45, 0.865),
        plate_radius=0.06,
    ),

    # -------------------------------------------------------------------------
    "training_matched": InferenceSetup(
        description=(
            "World-fixed camera at shoulder height (-0.15, 0.20, 1.05), 45 deg forward tilt. "
            "d435_link head is at Z=1.234; shoulder ~Z=1.05. Moved forward Y=0.20 to sit "
            "in front of torso like real cam_left_high. Same rotation (0.3827,-0.9239,0,0)."
        ),
        camera_parent="__world__",
        camera_pos=(-0.15, 0.20, 1.05),
        camera_rot=(0.3827, -0.9239, 0.0, 0.0),
        object_pos=(-0.35, 0.40, 0.87),
        plate_pos=(0.15, 0.50, 0.865),
        plate_radius=0.06,
    ),

    # -------------------------------------------------------------------------
    "option_b_worldcam": InferenceSetup(
        description=(
            "World-fixed camera — side-left view. "
            "Camera at (-0.80, 0.40, 1.00): to the robot's left at table height. "
            "Faces world +X (same rotation as d435_link option_a). "
            "Stable across all steps; shows both arms and apple from the side."
        ),
        camera_parent="__world__",
        camera_pos=(-0.80, 0.40, 1.00),
        camera_rot=(0.5, -0.5, 0.5, -0.5),
        object_pos=(-0.35, 0.40, 0.87),
        plate_pos=(0.15, 0.50, 0.865),
        plate_radius=0.06,
    ),

    # -------------------------------------------------------------------------
    "option_b_worldcam_v2": InferenceSetup(
        description=(
            "World-fixed camera — front view. "
            "Camera at (-0.15, 1.30, 1.10): in front of the table, looking back at robot (-Y). "
            "cam-Z=world -Y, cam-up=world +Z. Shows robot face-on, hands on table."
        ),
        camera_parent="__world__",
        camera_pos=(-0.15, 1.30, 1.10),
        camera_rot=(0.0, 0.0, 0.7071, -0.7071),
        object_pos=(-0.35, 0.40, 0.87),
        plate_pos=(0.15, 0.50, 0.865),
        plate_radius=0.06,
    ),

    # -------------------------------------------------------------------------
    "option_b_worldcam_top": InferenceSetup(
        description=(
            "World-fixed camera — overhead top-down view. "
            "Camera at (-0.10, 0.45, 1.80): above table center, looking straight down (-Z). "
            "cam-down=world +Y (table=bottom of image). Shows full table from above."
        ),
        camera_parent="__world__",
        camera_pos=(-0.10, 0.45, 1.80),
        camera_rot=(0.0, -1.0, 0.0, 0.0),
        object_pos=(-0.35, 0.40, 0.87),
        plate_pos=(0.15, 0.50, 0.865),
        plate_radius=0.06,
    ),

    # =========================================================================
    # G1 + Dex3 hands (28 DOF) — Block Stacking
    # =========================================================================

    # -------------------------------------------------------------------------
    # 4 cameras (cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist)
    # 28 DOF: left_arm(7) + right_arm(7) + left_dex3(7) + right_dex3(7)
    # Scene: stack_g1_dex3 (block stacking with Dex3 hands)
    # Model: groot-g1-dex3-28dof (checkpoint-10000)
    "dex3_stack": InferenceSetup(
        description=(
            "G1 Dex3 28-DOF block stacking with 4 cameras. "
            "Head camera (d435_link) = cam_left_high + cam_right_high (duplicate). "
            "Wrist cameras on hand_camera_base_link = cam_left_wrist + cam_right_wrist."
        ),
        cameras=[
            # Primary: head camera → cam_left_high
            CameraSpec(
                name="cam_left_high",
                camera_parent=None,  # d435_link
                pos=(0.0, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
            ),
            # Duplicate head camera → cam_right_high (same image, different name)
            CameraSpec(
                name="cam_right_high",
                prim_path="DUPLICATE_PRIMARY",
            ),
            # Left wrist camera → cam_left_wrist
            CameraSpec(
                name="cam_left_wrist",
                prim_path="{ENV_REGEX_NS}/Robot/left_hand_camera_base_link/left_wrist_cam",
                camera_parent="left_hand_camera_base_link",
                pos=(-0.04012, 0.07441, 0.15711),
                rot=(0.50809, 0.00539, 0.86024, 0.0424),
            ),
            # Right wrist camera → cam_right_wrist
            CameraSpec(
                name="cam_right_wrist",
                prim_path="{ENV_REGEX_NS}/Robot/right_hand_camera_base_link/right_wrist_cam",
                camera_parent="right_hand_camera_base_link",
                pos=(-0.04012, -0.07441, 0.15711),
                rot=(0.50809, -0.00539, 0.86024, -0.0424),
            ),
        ],
        scene="stack_g1_dex3",
        language="Stack the blocks",
        object_pos=(-0.35, 0.40, 0.87),
        plate_pos=(0.15, 0.50, 0.865),
        plate_radius=0.06,
        dof_layout={
            "left_arm": (0, 7),
            "right_arm": (7, 14),
            "left_hand": (14, 21),
            "right_hand": (21, 28),
        },
        joint_groups={
            "left_arm": [
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            ],
            "right_arm": [
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
            ],
            "left_hand": ["left_hand_.*"],
            "right_hand": ["right_hand_.*"],
        },
        action_joint_patterns=[
            # left arm (7 DOF)
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            # right arm (7 DOF)
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
            # DEX3 hands
            "left_hand_.*", "right_hand_.*",
        ],
        hand_type="dex3",
        spawn_objects=False,  # Scene has its own RGB blocks for stacking
        # Sign convention fix: these Dex3 hand joints have opposite sign in the sim
        # URDF compared to the real robot training data. The model predicts positive
        # values but the sim joint limits only allow negative (or vice versa).
        # Negating these action targets aligns sim behavior with real robot.
        negate_action_joints=[
            "left_hand_middle_0_joint",   # training mean +0.91, sim range [-1.49, -0.08]
            "left_hand_thumb_2_joint",    # training mean -0.55, sim range [+0.09, +1.66]
            "right_hand_index_0_joint",   # training mean -0.54, sim range [+0.08, +1.49]
            "right_hand_middle_0_joint",  # training mean -0.88, sim range [+0.08, +1.49]
            "right_hand_thumb_2_joint",   # training mean +0.43, sim range [-1.66, -0.09]
        ],
    ),

    # =========================================================================
    # G1 + Gripper (UNITREE_G1) — Hospitality tasks (Fold Towel, etc.)
    # =========================================================================

    # -------------------------------------------------------------------------
    # 1 camera (ego_view via d435_link), 31 DOF state, 23 DOF action
    # Mixed action format: arms RELATIVE, grippers/waist/nav ABSOLUTE
    # Scene: pickplace_g1_gripper (pick-and-place with gripper hands)
    # Model: groot-g1-gripper-hospitality-7ds (UNITREE_G1 embodiment)
    "gripper_unitree": InferenceSetup(
        description=(
            "G1 Gripper (UNITREE_G1 embodiment) with 1 ego-view camera. "
            "31 DOF state, 23 DOF mixed action (arms RELATIVE, grippers/waist/nav ABSOLUTE). "
            "For hospitality models: fold towel, clean table, wipe table, etc."
        ),
        cameras=[
            CameraSpec(
                name="ego_view",       # UNITREE_G1 training key (NOT cam_left_high)
                camera_parent=None,    # d435_link head camera
                pos=(0.0, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
            ),
        ],
        scene="pickplace_g1_gripper",
        language="fold the towel",
        spawn_objects=False,  # Use scene's native objects
        dof_layout={
            "left_leg":   (0, 6),
            "right_leg":  (6, 12),
            "waist":      (12, 15),
            "left_arm":   (15, 22),
            "right_arm":  (22, 29),
            "left_hand":  (29, 30),
            "right_hand": (30, 31),
        },
        joint_groups={
            "left_leg": [
                "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            ],
            "right_leg": [
                "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            ],
            "waist": [
                "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
            ],
            "left_arm": [
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            ],
            "right_arm": [
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ],
            "left_hand": ["left_gripper_joint"],
            "right_hand": ["right_gripper_joint"],
        },
        action_joint_patterns=[
            # left arm (7 DOF) — RELATIVE
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            # right arm (7 DOF) — RELATIVE
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            # grippers (2 DOF) — ABSOLUTE
            "left_gripper_joint", "right_gripper_joint",
            # waist (3 DOF) — ABSOLUTE
            "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
        ],
        hand_type="gripper",
    ),

    # -------------------------------------------------------------------------
    "gripper_redblock": InferenceSetup(
        description=(
            "G1 Gripper (UNITREE_G1) pick-and-place red block scene. "
            "Same config as gripper_unitree but with red block scene."
        ),
        cameras=[
            CameraSpec(
                name="ego_view",
                camera_parent=None,
                pos=(0.0, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
            ),
        ],
        scene="pickplace_redblock_g1_gripper",
        language="pick up the red block",
        spawn_objects=False,
        dof_layout={
            "left_leg":   (0, 6),
            "right_leg":  (6, 12),
            "waist":      (12, 15),
            "left_arm":   (15, 22),
            "right_arm":  (22, 29),
            "left_hand":  (29, 30),
            "right_hand": (30, 31),
        },
        joint_groups={
            "left_leg": [
                "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            ],
            "right_leg": [
                "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            ],
            "waist": [
                "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
            ],
            "left_arm": [
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint", "left_elbow_joint",
                "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            ],
            "right_arm": [
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ],
            "left_hand": ["left_gripper_joint"],
            "right_hand": ["right_gripper_joint"],
        },
        action_joint_patterns=[
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            "left_gripper_joint", "right_gripper_joint",
            "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
        ],
        hand_type="gripper",
    ),
}


def get_setup(name: str) -> InferenceSetup:
    """Return an InferenceSetup by name, raising ValueError if not found."""
    if name not in SETUPS:
        available = ", ".join(sorted(SETUPS.keys()))
        raise ValueError(f"Unknown setup '{name}'. Available: {available}")
    return SETUPS[name]


def list_setups() -> str:
    """Return a formatted string listing all available setups."""
    lines = ["Available inference setups:", ""]
    for name, setup in SETUPS.items():
        lines.append(f"  {name}")
        lines.append(f"    {setup.description}")
        cams = setup.get_cameras()
        if len(cams) > 1:
            cam_names = [c.name for c in cams]
            lines.append(f"    cameras: {cam_names}")
        else:
            primary = setup.get_primary_camera()
            lines.append(f"    camera_parent={primary.camera_parent}, pos={primary.pos}, rot={primary.rot}")
        if setup.scene:
            lines.append(f"    scene: {setup.scene}")
        if setup.dof_layout:
            total_dof = max(end for _, end in setup.dof_layout.values())
            lines.append(f"    dof: {total_dof} ({', '.join(f'{k}:{e-s}' for k, (s, e) in setup.dof_layout.items())})")
        lines.append(f"    object_pos={setup.object_pos}, plate_pos={setup.plate_pos}")
        lines.append("")
    return "\n".join(lines)
