"""
isaac_sim/base_scenes.py  —  BaseScene ABC and scene family base classes.

Import this module AFTER SimulationApp has been created in the calling script.
All Isaac Sim imports are at module level — safe because this file is always
imported after SimulationApp() is running.

Scene families
--------------
PickAndPlaceScene  Franka + 5 coloured cubes + single goal marker (tray or corner)
StackScene         Franka + 5 coloured cubes, no tray; one cube is fixed as the
                   stacking base, one is picked and placed on top
SortScene          Franka + 5 coloured cubes + 2 colour-coded trays; red cube →
                   red tray, blue cube → blue tray
"""

from abc import ABC, abstractmethod
import numpy as np

import omni.replicator.core        as rep
from omni.isaac.core               import World
from omni.isaac.core.objects       import DynamicCuboid, FixedCuboid
from omni.isaac.core.utils.types   import ArticulationAction
from omni.isaac.franka             import Franka
from omni.isaac.franka.controllers import RMPFlowController
from pxr                           import Gf, UsdGeom, UsdLux

OBS_H, OBS_W = 224, 224

# ── Colour palettes ───────────────────────────────────────────────────────── #

CUBE_COLORS = {
    "red":    np.array([0.80, 0.20, 0.10]),
    "blue":   np.array([0.10, 0.30, 0.90]),
    "yellow": np.array([0.90, 0.75, 0.10]),
    "orange": np.array([0.90, 0.45, 0.10]),
    "purple": np.array([0.55, 0.10, 0.80]),
}

# Green is reserved for the standard tray; red/blue variants are for SortScene.
TRAY_COLORS = {
    "green": np.array([0.10, 0.80, 0.10]),
    "red":   np.array([0.60, 0.10, 0.05]),
    "blue":  np.array([0.05, 0.10, 0.60]),
}

# Corner positions (x, y, z) in robot workspace coordinates.
# Front = smaller X (closer to robot base), Back = larger X.
# Left  = larger  Y, Right = smaller Y (from robot's forward-facing perspective).
CORNER_POSITIONS = {
    "front_left":  np.array([0.70,  0.35, 0.004]),
    "front_right": np.array([0.70, -0.35, 0.004]),
    "back_left":   np.array([0.30,  0.35, 0.004]),
    "back_right":  np.array([0.30, -0.35, 0.004]),
}


def _float_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        return arr.clip(0, 255).astype(np.uint8)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


# ============================================================================ #
#  Abstract base                                                                #
# ============================================================================ #

class BaseScene(ABC):
    @abstractmethod
    def reset(self, randomise: bool = True) -> None: ...
    @abstractmethod
    def get_obs(self) -> dict: ...
    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None: ...
    @abstractmethod
    def step(self) -> None: ...
    @abstractmethod
    def is_success(self) -> bool: ...
    @abstractmethod
    def scripted_action(self) -> np.ndarray: ...


# ============================================================================ #
#  Shared Franka infrastructure (_FrankaBase)                                  #
# ============================================================================ #

class _FrankaBase(BaseScene):
    """
    Internal base: shared Franka setup, annotator, RMPFlow state machine,
    and utility helpers.  Not intended for direct use.
    """

    LIFT_HEIGHT   = 0.22    # metres — carry height during transport phase
    CUBE_GROUND_Z = 0.025   # cube centre Z when resting on flat ground (half of 0.05 m)

    CAM_POS  = (2.39661, 0.46729, 1.05345)   # camera 1 — original fixed view
    CAM_ROT  = (65.86584, 0.0, 104.8045)
    CAM2_POS = (2.39661, -0.8, 1.05345)      # camera 2 — second fixed view (adjust to taste)
    CAM2_ROT = (65.19141, 0.0, 68.93048)

    CUBE_X_RANGE  = (0.30, 0.70)
    CUBE_Y_RANGE  = (-0.15, 0.30)
    MIN_CUBE_DIST = 0.12    # metres — minimum separation between cube centres

    # ── World + scene helpers ─────────────────────────────────────────── #

    def _init_world(self):
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt   = 1 / 120,
            rendering_dt = 4 / 120,
        )

    def _add_lights(self):
        stage = self.world.stage
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(300.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
        distant.CreateIntensityAttr(250.0)
        distant.CreateAngleAttr(0.53)
        distant.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

    def _add_franka(self):
        self.franka = self.world.scene.add(
            Franka(prim_path="/World/Franka", name="franka",
                   position=np.zeros(3))
        )

    def _cam_path_for(self, cam_id: int) -> str:
        paths = {
            1: "/World/ObsCamera1",
            2: "/World/ObsCamera2",
            3: "/World/Franka/panda_hand/WristCam",
        }
        if cam_id not in paths:
            raise ValueError(f"Unknown camera id {cam_id}; valid: 1, 2, 3")
        return paths[cam_id]

    def _add_cameras(self, camera_ids: list):
        stage = self.world.stage
        for cam_id in camera_ids:
            path = self._cam_path_for(cam_id)
            cam_prim = UsdGeom.Camera.Define(stage, path)
            xf = UsdGeom.Xformable(cam_prim)
            if cam_id == 1:
                xf.AddTranslateOp().Set(Gf.Vec3d(*self.CAM_POS))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(*self.CAM_ROT))
            elif cam_id == 2:
                xf.AddTranslateOp().Set(Gf.Vec3d(*self.CAM2_POS))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(*self.CAM2_ROT))
            elif cam_id == 3:
                # Eye-in-hand: pull 8 cm back from panda_hand origin (away from
                # fingertips) so the camera sits above the hand, then rotate 180°
                # around X to flip it to face the workspace below.
                xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 4.72257))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            print(f"[Camera {cam_id}] prim={path}")

    def _finish_init(self, camera_ids: list):
        """Call after world.reset() to bring up controllers and per-camera annotators."""
        print("[Isaac Sim] Warming up RTX Renderer and compiling shaders...")
        for _ in range(100):
            rep.orchestrator.step(rt_subframes=1, pause_timeline=True)

        self._camera_ids = list(camera_ids)
        self._annots = {}
        self._render_products = {}
        for cam_id in camera_ids:
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            rp    = rep.create.render_product(self._cam_path_for(cam_id), (OBS_W, OBS_H))
            annot.attach([rp])
            self._annots[cam_id]          = annot
            self._render_products[cam_id] = rp
        self._obs_checked = False

        self.controller = RMPFlowController(
            name="rmpflow", robot_articulation=self.franka
        )
        self.art_ctrl = self.franka.get_articulation_controller()
        self.controller.reset()

        self._phase      = 0
        self._grip_timer = 0
        self._grasp_pos  = None

    # ── Observation ───────────────────────────────────────────────────── #

    def get_obs(self) -> dict:
        """Return {cam_id: (H, W, 3) uint8} for every active camera."""
        result = {}
        for cam_id, annot in self._annots.items():
            raw = annot.get_data()
            if not self._obs_checked:
                self._obs_checked = True
                print(f"[Camera {cam_id}] annotator raw type: {type(raw)}")
            data = raw.get("data", None) if isinstance(raw, dict) else raw
            if data is None or (hasattr(data, "size") and data.size == 0):
                result[cam_id] = np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8)
                continue
            arr = np.asarray(data)
            if arr.ndim == 1:
                if arr.size == OBS_H * OBS_W * 4:
                    arr = arr.reshape(OBS_H, OBS_W, 4)
                elif arr.size == OBS_H * OBS_W * 3:
                    arr = arr.reshape(OBS_H, OBS_W, 3)
                    result[cam_id] = arr if arr.dtype == np.uint8 else _float_to_uint8(arr)
                    continue
                else:
                    result[cam_id] = np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8)
                    continue
            rgb = arr[:, :, :3] if arr.shape[-1] == 4 else arr
            if rgb.max() == 0:
                print(f"[Camera {cam_id}] Warning: black frame.")
            result[cam_id] = rgb if rgb.dtype == np.uint8 else _float_to_uint8(rgb)
        return result

    # ── Action / step ─────────────────────────────────────────────────── #

    def apply_action(self, action: np.ndarray):
        ee_pos, _ = self.franka.end_effector.get_world_pose()
        target    = ee_pos + action[:3].astype(np.float64)
        ctrl      = self.controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=None,
        )
        self.art_ctrl.apply_action(ctrl)
        self._set_gripper(open_gripper=float(action[6]) > 0.0)

    def step(self):
        rep.orchestrator.step(rt_subframes=4, pause_timeline=True)

    def _set_gripper(self, open_gripper: bool):
        finger = np.float32(0.04 if open_gripper else 0.0)
        self.art_ctrl.apply_action(ArticulationAction(
            joint_positions=np.array([finger, finger], dtype=np.float32),
            joint_indices=np.array([7, 8]),
        ))

    # ── Cube placement helper ─────────────────────────────────────────── #

    def _sample_non_overlapping(self, n: int) -> list:
        def _sample():
            return np.array([
                np.random.uniform(*self.CUBE_X_RANGE),
                np.random.uniform(*self.CUBE_Y_RANGE),
            ])
        while True:
            positions = [_sample() for _ in range(n)]
            if not any(
                np.linalg.norm(positions[i] - positions[j]) < self.MIN_CUBE_DIST
                for i in range(n) for j in range(i + 1, n)
            ):
                return positions

    # ── RMPFlow pick-and-place state machine ──────────────────────────── #

    def _rmpflow_pick_place_action(
        self,
        target_cube,
        goal_pos: np.ndarray,
    ) -> np.ndarray:
        """
        6-phase RMPFlow state machine used by all pick-and-place tasks.

        Phase 0  Approach    Move EE 10 cm above the target cube.
        Phase 1  Descend     Lower EE to grasp height.
        Phase 2  Grasp+Lift  Close gripper (30 steps), then lift to LIFT_HEIGHT.
        Phase 3  Transport   Carry cube to goal XY at LIFT_HEIGHT.
        Phase 4  Place       Descend at goal, open gripper.
        Phase 5  Retract     Hold gripper open; episode ends shortly after.

        Parameters
        ----------
        target_cube : DynamicCuboid  the cube to pick up
        goal_pos    : (3,) ndarray   world position of the placement surface
                      goal_pos[2] is the surface z-height (e.g. 0.004 for tray)
        """
        cube_pos, _ = target_cube.get_world_pose()
        ee_pos, _   = self.franka.end_effector.get_world_pose()

        if self._phase == 0:
            target       = cube_pos.copy()
            target[2]    = cube_pos[2] + 0.10
            gripper_open = True
            if np.linalg.norm(ee_pos[:2] - cube_pos[:2]) < 0.05:
                self._phase = 1

        elif self._phase == 1:
            target       = cube_pos.copy()
            target[2]    = cube_pos[2] + 0.01
            gripper_open = True
            if ee_pos[2] <= cube_pos[2] + 0.06:
                self._grasp_pos = ee_pos.copy()
                self._phase     = 2

        elif self._phase == 2:
            gripper_open     = False
            self._grip_timer += 1
            if self._grasp_pos is None:
                self._grasp_pos = ee_pos.copy()
            if self._grip_timer <= 30:
                target = self._grasp_pos.copy()
            else:
                target    = self._grasp_pos.copy()
                target[2] = self.LIFT_HEIGHT
                if ee_pos[2] >= self.LIFT_HEIGHT - 0.02:
                    self._phase      = 3
                    self._grip_timer = 0

        elif self._phase == 3:
            gripper_open = False
            target = np.array(
                [goal_pos[0], goal_pos[1], self.LIFT_HEIGHT], dtype=np.float64
            )
            if np.linalg.norm(ee_pos[:2] - goal_pos[:2]) < 0.05:
                self._phase = 4

        elif self._phase == 4:
            self._grip_timer += 1
            if self._grasp_pos is not None:
                ee_cube_offset = self._grasp_pos[2] - self.CUBE_GROUND_Z
                place_z = goal_pos[2] + 0.0025 + 0.025 + ee_cube_offset - 0.02
            else:
                place_z = goal_pos[2] + 0.07
            target       = np.array(
                [goal_pos[0], goal_pos[1], place_z], dtype=np.float64
            )
            at_height    = ee_pos[2] <= place_z + 0.05
            timed_out    = self._grip_timer > 50
            gripper_open = at_height or timed_out
            if gripper_open and self._grip_timer > 55:
                self._phase      = 5
                self._grip_timer = 0

        else:  # phase 5+
            target       = ee_pos.copy()
            target[2]    = self.LIFT_HEIGHT
            gripper_open = True

        ctrl = self.controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=None,
        )
        self.art_ctrl.apply_action(ctrl)
        self._set_gripper(gripper_open)

        delta       = np.clip((target - ee_pos), -1.0, 1.0).astype(np.float32)
        gripper_val = np.float32(1.0 if gripper_open else -1.0)
        return np.concatenate([delta, np.zeros(3, dtype=np.float32), [gripper_val]])

    @property
    def phase(self) -> int:
        return self._phase


# ============================================================================ #
#  PickAndPlaceScene                                                            #
# ============================================================================ #

class PickAndPlaceScene(_FrankaBase):
    """
    Franka + 5 coloured cubes + a single goal marker.

    Parameters
    ----------
    goal_type : "tray" | "corner"
    corner    : "front_left" | "front_right" | "back_left" | "back_right"
                Required when goal_type="corner".

    Subclasses must set self._target_cube in __init__ and implement
    scripted_action().  is_success() uses self._target_cube automatically.
    """

    TRAY_POS = np.array([0.50, -0.35, 0.004])

    def __init__(self, goal_type: str = "tray", corner: str | None = None,
                 camera_ids: list | tuple = (1,)):
        assert goal_type in ("tray", "corner"), f"Unknown goal_type: {goal_type!r}"
        if goal_type == "corner":
            assert corner in CORNER_POSITIONS, f"Unknown corner: {corner!r}"

        self.goal_type    = goal_type
        self.corner       = corner
        self._camera_ids  = list(camera_ids)
        self.goal_pos     = (
            CORNER_POSITIONS[corner].copy()
            if goal_type == "corner"
            else self.TRAY_POS.copy()
        )
        self._target_cube = None  # subclass sets this in its __init__

        self._init_world()
        self._build()
        self.world.reset()
        self._finish_init(self._camera_ids)

    def _build(self):
        self.world.scene.add_default_ground_plane()
        self._add_lights()
        self._add_franka()

        # ── 5 coloured cubes ─────────────────────────────────────────── #
        self.cubes: dict = {}
        for color, rgb in CUBE_COLORS.items():
            self.cubes[color] = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Cube_{color.capitalize()}",
                    name=f"cube_{color}",
                    position=np.array([0.5, 0.0, self.CUBE_GROUND_Z]),
                    scale=np.array([0.05, 0.05, 0.05]),
                    color=rgb,
                )
            )

        # ── Goal marker ───────────────────────────────────────────────── #
        if self.goal_type == "tray":
            self.world.scene.add(
                FixedCuboid(
                    prim_path="/World/Goal", name="goal",
                    position=self.goal_pos.copy(),
                    scale=np.array([0.20, 0.20, 0.005]),
                    color=TRAY_COLORS["green"],
                )
            )
        else:  # corner — small yellow marker at corner + border extending inward
            self.world.scene.add(
                FixedCuboid(
                    prim_path="/World/Goal", name="goal",
                    position=self.goal_pos.copy(),
                    scale=np.array([0.08, 0.08, 0.003]),
                    color=np.array([1.0, 1.0, 0.0]),
                )
            )
            # Border spans the full workspace defined by the four corner positions.
            _corners = np.array(list(CORNER_POSITIONS.values()))
            _min_x, _max_x = _corners[:, 0].min(), _corners[:, 0].max()
            _min_y, _max_y = _corners[:, 1].min(), _corners[:, 1].max()
            _cx   = (_min_x + _max_x) / 2
            _cy   = (_min_y + _max_y) / 2
            _cz   = self.goal_pos[2]
            _sx   = _max_x - _min_x   # width in X
            _sy   = _max_y - _min_y   # width in Y
            _w    = 0.015              # strip width (m)
            _h    = 0.003              # strip height (m)
            for _name, _pos, _sc in [
                ("goal_zone_px", [_max_x - _w/2, _cy,          _cz], [_w,        _sy, _h]),
                ("goal_zone_nx", [_min_x + _w/2, _cy,          _cz], [_w,        _sy, _h]),
                ("goal_zone_py", [_cx,            _max_y - _w/2, _cz], [_sx - 2*_w, _w, _h]),
                ("goal_zone_ny", [_cx,            _min_y + _w/2, _cz], [_sx - 2*_w, _w, _h]),
            ]:
                self.world.scene.add(
                    FixedCuboid(
                        prim_path=f"/World/{_name}", name=_name,
                        position=np.array(_pos),
                        scale=np.array(_sc),
                        color=np.array([1.0, 1.0, 0.0]),
                    )
                )

        self._add_cameras(self._camera_ids)

    def reset(self, randomise: bool = True):
        self.world.reset()
        self.controller.reset()
        self._phase      = 0
        self._grip_timer = 0
        self._grasp_pos  = None

        if randomise:
            positions = self._sample_non_overlapping(len(self.cubes))
            for cube, (x, y) in zip(self.cubes.values(), positions):
                cube.set_world_pose(
                    position=np.array([x, y, self.CUBE_GROUND_Z])
                )
                cube.set_linear_velocity(np.zeros(3))
                cube.set_angular_velocity(np.zeros(3))

        for _ in range(10):
            rep.orchestrator.step(rt_subframes=4, pause_timeline=True)
        self.get_obs()  # flush stale frame

    def is_success(self) -> bool:
        if self._target_cube is None:
            return False
        cube_pos, _ = self._target_cube.get_world_pose()
        xy_dist = np.linalg.norm(cube_pos[:2] - self.goal_pos[:2])
        return bool(xy_dist < 0.10 and cube_pos[2] < 0.035)

    def scripted_action(self) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} must implement scripted_action()"
        )


# ============================================================================ #
#  StackScene                                                                   #
# ============================================================================ #

class StackScene(_FrankaBase):
    """
    Franka + 5 coloured cubes, no tray.

    The base cube is a FixedCuboid placed at a fixed position so it cannot
    be knocked over during placing.  The target cube is a DynamicCuboid that
    the robot picks up and stacks on top of the base.

    Parameters
    ----------
    target_color : colour to pick up ("red", "blue", …)
    base_color   : colour of the fixed base cube to stack on
    """

    BASE_POS = np.array([0.50, -0.25, 0.025])

    def __init__(self, target_color: str, base_color: str,
                 camera_ids: list | tuple = (1,)):
        assert target_color in CUBE_COLORS, f"Unknown colour: {target_color!r}"
        assert base_color   in CUBE_COLORS, f"Unknown colour: {base_color!r}"
        assert target_color != base_color,  "target and base must differ"

        self.target_color = target_color
        self.base_color   = base_color
        self._camera_ids  = list(camera_ids)

        self._init_world()
        self._build()
        self.world.reset()
        self._finish_init(self._camera_ids)

    def _build(self):
        self.world.scene.add_default_ground_plane()
        self._add_lights()
        self._add_franka()

        # ── Dynamic cubes (all colours except the base) ───────────────── #
        self.cubes: dict = {}
        for color, rgb in CUBE_COLORS.items():
            if color == self.base_color:
                continue
            self.cubes[color] = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Cube_{color.capitalize()}",
                    name=f"cube_{color}",
                    position=np.array([0.5, 0.0, self.CUBE_GROUND_Z]),
                    scale=np.array([0.05, 0.05, 0.05]),
                    color=CUBE_COLORS[color],
                )
            )

        # ── Fixed base cube ───────────────────────────────────────────── #
        self._base_cube = self.world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Cube_{self.base_color.capitalize()}",
                name=f"cube_{self.base_color}",
                position=self.BASE_POS.copy(),
                scale=np.array([0.05, 0.05, 0.05]),
                color=CUBE_COLORS[self.base_color],
            )
        )

        self._add_cameras(self._camera_ids)

    @property
    def _stack_goal_pos(self) -> np.ndarray:
        """Top-surface centre of the base cube — the stacking target."""
        base_pos, _ = self._base_cube.get_world_pose()
        return np.array([base_pos[0], base_pos[1], base_pos[2] + 0.025])

    def reset(self, randomise: bool = True):
        self.world.reset()
        self.controller.reset()
        self._phase      = 0
        self._grip_timer = 0
        self._grasp_pos  = None

        if randomise:
            positions = self._sample_non_overlapping(len(self.cubes))
            for cube, (x, y) in zip(self.cubes.values(), positions):
                cube.set_world_pose(
                    position=np.array([x, y, self.CUBE_GROUND_Z])
                )
                cube.set_linear_velocity(np.zeros(3))
                cube.set_angular_velocity(np.zeros(3))

        for _ in range(10):
            rep.orchestrator.step(rt_subframes=4, pause_timeline=True)
        self.get_obs()

    def is_success(self) -> bool:
        target_pos, _ = self.cubes[self.target_color].get_world_pose()
        base_pos,   _ = self._base_cube.get_world_pose()
        xy_dist    = np.linalg.norm(target_pos[:2] - base_pos[:2])
        expected_z = base_pos[2] + 0.05  # stacked cube centre = base centre + cube height
        z_ok       = abs(target_pos[2] - expected_z) < 0.025
        return bool(xy_dist < 0.05 and z_ok)

    def scripted_action(self) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} must implement scripted_action()"
        )


# ============================================================================ #
#  SortScene                                                                    #
# ============================================================================ #

class SortScene(_FrankaBase):
    """
    Franka + 5 coloured cubes + 2 colour-coded trays.

    Red  cube → red  tray.
    Blue cube → blue tray.
    Yellow / orange / purple cubes are distractors.

    is_success() requires BOTH cubes placed on their correct trays.
    The scripted policy performs two sequential pick-and-place operations;
    self._sort_phase tracks which one is active (0 = red, 1 = blue).
    """

    RED_TRAY_POS  = np.array([0.50, -0.35, 0.004])
    BLUE_TRAY_POS = np.array([0.35, -0.35, 0.004])

    def __init__(self, camera_ids: list | tuple = (1,)):
        self._camera_ids = list(camera_ids)
        self._init_world()
        self._build()
        self.world.reset()
        self._finish_init(self._camera_ids)
        self._sort_phase = 0

    def _build(self):
        self.world.scene.add_default_ground_plane()
        self._add_lights()
        self._add_franka()

        # ── 5 coloured cubes ─────────────────────────────────────────── #
        self.cubes: dict = {}
        for color, rgb in CUBE_COLORS.items():
            self.cubes[color] = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Cube_{color.capitalize()}",
                    name=f"cube_{color}",
                    position=np.array([0.5, 0.0, self.CUBE_GROUND_Z]),
                    scale=np.array([0.05, 0.05, 0.05]),
                    color=rgb,
                )
            )

        # ── Colour-coded trays ────────────────────────────────────────── #
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/TrayRed", name="tray_red",
                position=self.RED_TRAY_POS.copy(),
                scale=np.array([0.18, 0.18, 0.005]),
                color=TRAY_COLORS["red"],
            )
        )
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/TrayBlue", name="tray_blue",
                position=self.BLUE_TRAY_POS.copy(),
                scale=np.array([0.18, 0.18, 0.005]),
                color=TRAY_COLORS["blue"],
            )
        )

        self._add_cameras(self._camera_ids)

    def reset(self, randomise: bool = True):
        self.world.reset()
        self.controller.reset()
        self._phase      = 0
        self._grip_timer = 0
        self._grasp_pos  = None
        self._sort_phase = 0

        if randomise:
            positions = self._sample_non_overlapping(len(self.cubes))
            for cube, (x, y) in zip(self.cubes.values(), positions):
                cube.set_world_pose(position=np.array([x, y, self.CUBE_GROUND_Z]))
                cube.set_linear_velocity(np.zeros(3))
                cube.set_angular_velocity(np.zeros(3))

        for _ in range(10):
            rep.orchestrator.step(rt_subframes=4, pause_timeline=True)
        self.get_obs()

    def is_success(self) -> bool:
        def _on_tray(cube, tray_pos: np.ndarray) -> bool:
            pos, _ = cube.get_world_pose()
            return bool(
                np.linalg.norm(pos[:2] - tray_pos[:2]) < 0.10
                and pos[2] < 0.035
            )
        return (
            _on_tray(self.cubes["red"],  self.RED_TRAY_POS) and
            _on_tray(self.cubes["blue"], self.BLUE_TRAY_POS)
        )

    def scripted_action(self) -> np.ndarray:
        raise NotImplementedError(
            f"{type(self).__name__} must implement scripted_action()"
        )
