"""
isaac_collector.py  —  Isaac Sim data collection script (Windows desktop).

Runs on: Windows desktop with RTX 3070 (8 GB)
Isaac Sim: Monitor (launch if required)

This script:
  1. Launches Isaac Sim
  2. Connects to the VLA inference server on the Linux server via ROS2
  3. Steps the Franka pick-and-place simulation
  4. At each step:
       a. Captures a 224×224 RGB observation from the camera
       b. Publishes it (JPEG-compressed, base64) to /vla/request
       c. Waits for the 7-DoF action string on /vla/response
       d. Applies the action to the robot
       e. Optionally injects interference
  5. Saves (obs_t, action_t, obs_{t+1}) triplets to HDF5

ROS2 environment (set on BOTH machines before launching)
---------------------------------------------------------
  Windows (cmd):
    set ROS_DOMAIN_ID=66
    set ROS_DISCOVERY_SERVER=10.32.33.41:11811

  Linux server:
    export ROS_DOMAIN_ID=66
    export ROS_DISCOVERY_SERVER=10.32.33.41:11811

Usage (run on Windows desktop — NOT on the server):
----------------------------------------------------
  # 1. First start the server on the Linux machine:
  #    python server/vla_server.py --model_dir outputs/openvla_finetuned

  # 2. Then run the collector on the Windows desktop:
  <isaac_sim_root>\\python.bat isaac_sim\\isaac_collector.py ^
      --output data\\stage12_clean.hdf5 ^
      --num_episodes 300 ^
      --episode_length 120

  # With interference (evaluation data):
  <isaac_sim_root>\\python.bat isaac_sim\\isaac_collector.py ^
      --output data\\eval_disturbed.hdf5 ^
      --num_episodes 100 ^
      --interference_action --interference_pose ^
      --interference_prob 0.10

Notes on headless mode
----------------------
Isaac Sim on Windows does not require a monitor when --headless is passed
to SimulationApp. You can run it over Remote Desktop or SSH with X-forwarding
disabled. The renderer still uses the GPU for physics and camera rendering.

If Isaac Sim shows a "no display" error, set:
    set DISPLAY=:0       (if WSL2 with X server)
or pass  "renderer": "RayCast"  instead of "RayTracedLighting" for a
purely physics-based renderer that skips visual rendering entirely.
"""

import argparse
import base64
import io
import json
import os
import queue
import random
import sys
import threading
import time

import numpy as np
import h5py

# ── Isaac Sim bootstrap ─────────────────────────────────────────────── #
# Parse --headless before SimulationApp is constructed.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--headless", action="store_true", default=True)
_pre_args, _ = _pre.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({
    # headless=False opens the Isaac Sim window on the desktop monitor.
    # This is required for the RTX renderer to fully initialize and populate
    # camera sensor buffers; headless=True leaves the render pipeline dormant
    # on a system that has a display attached.
    "headless":      False,
    "renderer":      "RayTracedLighting",
    "anti_aliasing": 0,
})

import omni.isaac.core.utils.prims    as prim_utils
import omni.replicator.core           as rep
from omni.isaac.core                  import World
from omni.isaac.core.objects          import DynamicCuboid, FixedCuboid, VisualCuboid
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.types      import ArticulationAction
from omni.isaac.franka                import Franka
from omni.isaac.franka.controllers    import RMPFlowController
from pxr                              import Gf, Sdf, Usd, UsdGeom, UsdShade

enable_extension("omni.isaac.franka")
enable_extension("omni.isaac.ros2_bridge")

OBS_H, OBS_W = 224, 224   # must match OpenVLA's expected input size


# ========================================================================== #
#  ROS2 Client  —  connects to vla_server.py on the Linux server             #
# ========================================================================== #

class VLAClient:
    """
    ROS2 client that publishes observations to /vla/request and
    subscribes to /vla/response to receive 7-DoF actions.

    Uses a sequence number to match each request to its response, and
    blocks in request_action() until the matching reply arrives.

    Required environment variables (set before launching Isaac Sim):
        ROS_DOMAIN_ID=66
        ROS_DISCOVERY_SERVER=10.32.33.41:11811

    Parameters
    ----------
    instruction  : str   Language instruction sent with every request
    unnorm_key   : str   Action un-normalisation key
    timeout_s    : float How long to wait for a reply before raising TimeoutError
    jpeg_quality : int   JPEG compression quality (lower = less bandwidth)
    """

    def __init__(
        self,
        instruction:  str   = "pick up the red cube and place it on the tray",
        unnorm_key:   str   = "franka_isaac",
        timeout_s:    float = 10.0,
        jpeg_quality: int   = 85,
    ):
        try:
            import rclpy
            from std_msgs.msg import String
        except ImportError:
            raise ImportError(
                "rclpy not found. Enable the ROS2 bridge extension and "
                "source your ROS2 workspace before launching Isaac Sim."
            )

        self.instruction  = instruction
        self.unnorm_key   = unnorm_key
        self.timeout_s    = timeout_s
        self.jpeg_quality = jpeg_quality

        self._seq      = 0
        self._pending  = {}   # seq → threading.Event
        self._results  = {}   # seq → reply dict
        self._lock     = threading.Lock()

        rclpy.init()
        self._node = rclpy.create_node("vla_client")
        String     = __import__("std_msgs.msg", fromlist=["String"]).String

        self._pub = self._node.create_publisher(String, "/vla/request", 10)
        self._node.create_subscription(
            String, "/vla/response", self._on_response, 10
        )

        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=executor.spin, daemon=True)
        self._spin_thread.start()

        print("[VLAClient] ROS2 node started.")
        print("[VLAClient] Publishing to  /vla/request")
        print("[VLAClient] Subscribed to  /vla/response")

    def _on_response(self, msg):
        try:
            data = json.loads(msg.data)
            seq  = data.get("seq", -1)
            with self._lock:
                if seq in self._pending:
                    self._results[seq] = data
                    self._pending[seq].set()
        except json.JSONDecodeError as e:
            print(f"[VLAClient] Malformed response JSON: {e}")

    def _compress(self, obs_rgb: np.ndarray) -> str:
        """Compress (H, W, 3) uint8 → base64-encoded JPEG string."""
        from PIL import Image as PILImage
        pil = PILImage.fromarray(obs_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def request_action(self, obs_rgb: np.ndarray) -> np.ndarray:
        """
        Publish observation, block until the server replies, return (7,) float32.

        Parameters
        ----------
        obs_rgb : (H, W, 3) uint8 numpy array

        Returns
        -------
        action : (7,) float32
        """
        from std_msgs.msg import String

        with self._lock:
            self._seq += 1
            seq = self._seq
            event = threading.Event()
            self._pending[seq] = event

        msg      = String()
        msg.data = json.dumps({
            "seq":         seq,
            "jpeg_b64":    self._compress(obs_rgb),
            "instruction": self.instruction,
            "unnorm_key":  self.unnorm_key,
        })
        self._pub.publish(msg)

        if not event.wait(timeout=self.timeout_s):
            with self._lock:
                self._pending.pop(seq, None)
            raise TimeoutError(
                f"No response from VLA server within {self.timeout_s}s. "
                "Is vla_server.py running and ROS_DISCOVERY_SERVER set correctly?"
            )

        with self._lock:
            reply = self._results.pop(seq)
            self._pending.pop(seq, None)

        if reply.get("status") != "ok":
            msg_txt = reply.get("message", "unknown error")
            print(f"[VLAClient] Server error: {msg_txt}. Using zero action.")
            return np.zeros(7, dtype=np.float32)

        return np.array(reply["action"], dtype=np.float32)

    def close(self):
        import rclpy
        rclpy.shutdown()


# ========================================================================== #
#  Interference Injector                                                      #
# ========================================================================== #

class InterferenceInjector:
    """
    Action perturbation + pose disturbance for evaluation data collection.

    Action perturbation
        Corrupt the action before execution.  The Predictor on the Edge
        Server receives the intended action but the robot executes a noisy
        version → z_t is unexpected → Δz spikes.

    Pose disturbance
        Teleport the cube mid-episode → sudden visual change → Δz spikes.
    """

    def __init__(
        self,
        action_prob:        float = 0.05,
        pose_prob:          float = 0.05,
        action_noise_scale: float = 0.30,
        pose_delta_max:     float = 0.15,
        seed:               int   = 0,
    ):
        self.action_prob        = action_prob
        self.pose_prob          = pose_prob
        self.action_noise_scale = action_noise_scale
        self.pose_delta_max     = pose_delta_max
        self.rng    = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.action_events = 0
        self.pose_events   = 0

    def reset(self):
        self.action_events = 0
        self.pose_events   = 0

    def maybe_perturb_action(self, action: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.action_prob:
            noise  = self.np_rng.randn(*action.shape).astype(np.float32)
            action = np.clip(action + noise * self.action_noise_scale, -1., 1.)
            self.action_events += 1
        return action

    def maybe_teleport_cube(self, cube_prim_path: str) -> bool:
        if self.rng.random() < self.pose_prob:
            prim = prim_utils.get_prim_at_path(cube_prim_path)
            if not prim.IsValid():
                return False
            dx = self.np_rng.uniform(-self.pose_delta_max, self.pose_delta_max)
            dy = self.np_rng.uniform(-self.pose_delta_max, self.pose_delta_max)
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    cur = op.Get()
                    op.Set(Gf.Vec3d(
                        float(cur[0]) + dx,
                        float(cur[1]) + dy,
                        float(cur[2]),
                    ))
                    break
            self.pose_events += 1
            return True
        return False


# ========================================================================== #
#  Franka Scene                                                               #
# ========================================================================== #

class FrankaScene:
    """
    Franka pick-and-place scene with a fixed workspace camera.

    Camera sits to the front-right of the robot and is angled to show
    both the arm and the cube/goal on the table surface.
    """

    # z = half-height so each object's bottom face sits flush on the ground plane
    CUBE_INIT  = np.array([0.50,  0.00, 0.025])   # half of 0.05 m cube
    GOAL_POS   = np.array([0.50, -0.35, 0.004])   # slightly above ground to avoid z-fighting

    # ── Random spawn region for the red cube ─────────────────────── #
    # Chosen to stay within the camera's field of view and within the
    # robot's reachable workspace.  Goal (y ≈ -0.35) is outside this
    # range so the cube never spawns on top of the marker.
    CUBE_X_RANGE = (0.30, 0.70)
    CUBE_Y_RANGE = (-0.15, 0.30)

    # ── Phase thresholds ──────────────────────────────────────────── #
    PREGRASP_HEIGHT = 0.18    # hover height above table before descending
    GRASP_HEIGHT    = 0.065   # EE height at which gripper closes on cube
    LIFT_HEIGHT     = 0.22    # height to lift cube before lateral move
    DIST_THRESHOLD  = 0.03    # metres — "close enough" to advance phase

    # Camera position and rotation read directly from the Isaac Sim viewport.
    CAM_POS = (1.94813, 1.0027, 0.88385)   # (x, y, z) in metres
    CAM_ROT = (66.05069, 0.0, 126.51205)   # Euler XYZ in degrees

    def __init__(self):
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt   = 1 / 120,
            rendering_dt = 4 / 120,
        )
        self._build()
        self.world.reset()

        # Annotator is created once; render product is recreated in reset()
        # because world.reset() invalidates existing render products.
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._render_product = None
        self._obs_checked = False

        self.controller = RMPFlowController(
            name="rmpflow", robot_articulation=self.franka
        )
        self.art_ctrl = self.franka.get_articulation_controller()
        self.controller.reset()
        self._phase      = 0     # current state-machine phase (0–4)
        self._grip_timer = 0     # counts steps in grasp/release phases
        self._grasp_pos  = None  # EE position saved when entering Phase 2

    def _build(self):
        self.world.scene.add_default_ground_plane()
        self._paint_ground(Gf.Vec3f(0.17, 0.42, 0.79))   # solid light blue

        from pxr import UsdLux
        stage = self.world.stage
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(800.0)
        dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        distant = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
        distant.CreateIntensityAttr(500.0)
        distant.CreateAngleAttr(0.53)
        distant.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))

        self.franka = self.world.scene.add(
            Franka(prim_path="/World/Franka", name="franka",
                   position=np.zeros(3))
        )
        # ── Target cube (red) — the one the robot picks up ─────────── #
        self.cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube", name="cube",
                position=self.CUBE_INIT.copy(),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.8, 0.2, 0.1]),
            )
        )
        
        # ── Extra distractor cubes (different colours) ───────────────── #
        self.cube_blue = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/CubeBlue", name="cube_blue",
                position=np.array([0.35, 0.25, 0.025]),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.1, 0.3, 0.9]),
            )
        )
        self.cube_yellow = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/CubeYellow", name="cube_yellow",
                position=np.array([0.60, 0.20, 0.025]),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.9, 0.75, 0.1]),
            )
        )

        # ── Goal marker — FixedCuboid so it sits on the ground and ───── #
        # has collision (the cube rests on it rather than passing through) #
        self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Goal", name="goal",
                position=self.GOAL_POS.copy(),
                scale=np.array([0.2, 0.2, 0.005]),
                color=np.array([0.1, 0.8, 0.1]),
            )
        )
        # Create a USD Camera prim with the exact transform the viewport reported.
        # Using UsdGeom.Camera + AddTranslateOp/AddRotateXYZOp mirrors what
        # Isaac Sim stores internally, so the render product captures the same
        # view the user verified in the viewport.
        self._cam_path = "/World/ObsCamera"
        cam_prim = UsdGeom.Camera.Define(stage, self._cam_path)
        xf = UsdGeom.Xformable(cam_prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(*self.CAM_POS))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*self.CAM_ROT))
        print(f"[Camera] prim={self._cam_path}  pos={self.CAM_POS}  rot={self.CAM_ROT}")

    def _paint_ground(self, rgb: Gf.Vec3f) -> None:
        """Bind a solid-colour UsdPreviewSurface material to the ground plane.

        Binds directly to every prim in the subtree (root + all descendants)
        so the default white-grid shader on child mesh prims is fully overridden.
        """
        stage = self.world.stage
        mat   = UsdShade.Material.Define(stage, "/World/GroundMat")
        sh    = UsdShade.Shader.Define(stage, "/World/GroundMat/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(rgb)
        sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.9)
        sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

        root = stage.GetPrimAtPath("/World/defaultGroundPlane")
        if root.IsValid():
            for prim in Usd.PrimRange(root):
                UsdShade.MaterialBindingAPI(prim).Bind(
                    mat, UsdShade.Tokens.strongerThanDescendants
                )

    def _attach_render_product(self):
        """(Re)create the replicator render product and attach the RGB annotator.

        world.reset() invalidates existing render products, so this is called
        after every world.reset().
        """
        if self._render_product is not None:
            try:
                self._rgb_annot.detach([self._render_product])
                self._render_product.destroy()
            except Exception:
                pass
            self._render_product = None

        self._render_product = rep.create.render_product(
            self._cam_path, (OBS_W, OBS_H)
        )
        self._rgb_annot.attach([self._render_product])
        self._obs_checked = False

    def reset(self, randomise: bool = True):
        self.world.reset()
        # Re-attach the render product every time — world.reset() invalidates it.
        self._attach_render_product()

        self.controller.reset()
        self._phase      = 0
        self._grip_timer = 0
        self._grasp_pos  = None
        if randomise:
            x = np.random.uniform(*self.CUBE_X_RANGE)
            y = np.random.uniform(*self.CUBE_Y_RANGE)
            self.cube.set_world_pose(
                position=np.array([x, y, self.CUBE_INIT[2]])
            )
            xb = np.random.uniform(*self.CUBE_X_RANGE)
            yb = np.random.uniform(*self.CUBE_Y_RANGE)
            self.cube_blue.set_world_pose(
                position=np.array([xb, yb, self.CUBE_INIT[2]])
            )
            xy = np.random.uniform(*self.CUBE_X_RANGE)
            yy = np.random.uniform(*self.CUBE_Y_RANGE)
            self.cube_yellow.set_world_pose(
                position=np.array([xy, yy, self.CUBE_INIT[2]])
            )

        # Warm-up: settle physics and prime the annotator buffer.
        # 30 render steps are enough for the RTX pipeline to fully stabilise;
        # the extra get_obs() call flushes any residual stale frame so that
        # the very first observation in the episode is guaranteed non-black.
        for _ in range(30):
            self.world.step(render=True)
        self.get_obs()   # discard — just flushes the annotator

    def get_obs(self) -> np.ndarray:
        """Return (OBS_H, OBS_W, 3) uint8 RGB frame via replicator annotator."""
        raw = self._rgb_annot.get_data()

        # Diagnostic on first call — shows exactly what the annotator returned
        if not self._obs_checked:
            self._obs_checked = True
            print(f"[Camera] annotator raw type : {type(raw)}")
            if isinstance(raw, dict):
                print(f"[Camera] annotator dict keys: {list(raw.keys())}")
                for k, v in raw.items():
                    arr = np.asarray(v) if hasattr(v, "__len__") else v
                    print(f"[Camera]   [{k}] type={type(v)} "
                          + (f"shape={arr.shape} dtype={arr.dtype} "
                             f"min={arr.min()} max={arr.max()}"
                             if hasattr(arr, "shape") and arr.size > 0 else str(v)))
            else:
                arr = np.asarray(raw)
                print(f"[Camera] annotator array   : shape={arr.shape} "
                      f"dtype={arr.dtype} min={arr.min()} max={arr.max()}")

        # In Isaac Sim 4.x replicator returns a dict {"data": ndarray, "info": ...};
        # older versions return the ndarray directly.
        if isinstance(raw, dict):
            data = raw.get("data", None)
        else:
            data = raw

        if data is None or (hasattr(data, "size") and data.size == 0):
            return np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8)

        arr = np.asarray(data)
        if arr.ndim == 1:
            # Flat buffer — reshape to (H, W, 4)
            if arr.size == OBS_H * OBS_W * 4:
                arr = arr.reshape(OBS_H, OBS_W, 4)
            elif arr.size == OBS_H * OBS_W * 3:
                arr = arr.reshape(OBS_H, OBS_W, 3)
                return arr.astype(np.uint8) if arr.dtype == np.uint8 \
                    else (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                print(f"[Camera] unexpected flat size {arr.size}; returning zeros")
                return np.zeros((OBS_H, OBS_W, 3), dtype=np.uint8)

        if arr.shape[-1] == 4:
            rgb = arr[:, :, :3]
        else:
            rgb = arr  # already RGB

        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        return rgb

    def apply_action(self, action: np.ndarray):
        """
        Apply a 7-DoF end-effector delta action.
        action[:3] = position delta (metres); action[6] = gripper (+1 open, -1 closed).
        """
        ee_pos, _ = self.franka.end_effector.get_world_pose()
        target    = ee_pos + action[:3].astype(np.float64)
        ctrl      = self.controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=None,
        )
        self.art_ctrl.apply_action(ctrl)
        self._set_gripper(open_gripper=float(action[6]) > 0.0)

    def step(self):
        # render=True drives the RTX viewport renderer each physics step.
        # In non-headless mode this fills the GPU frame buffer and the replicator
        # annotator (rgb) can then read the result via get_data().
        self.world.step(render=True)

    def _set_gripper(self, open_gripper: bool) -> None:
        """Set finger joints via a dedicated ArticulationAction with joint_indices.

        Using joint_indices=[7, 8] means this call ONLY touches the two finger
        joints and cannot overwrite arm-joint targets set by the preceding
        RMPFlow apply_action call.
        """
        finger = np.float32(0.04 if open_gripper else 0.0)
        self.art_ctrl.apply_action(ArticulationAction(
            joint_positions=np.array([finger, finger], dtype=np.float32),
            joint_indices=np.array([7, 8]),
        ))

    def _compute_rmpflow_action(self) -> np.ndarray:
        """
        Four-phase pick-and-place state machine using RMPFlow.

        Phase 0  Approach   : move EE above cube (10 cm above cube centre)
        Phase 1  Descend    : lower EE to grasp height (4 cm above cube centre,
                              same offset that the old single-phase controller used)
        Phase 2  Grasp+Lift : close gripper (30 steps), then lift to LIFT_HEIGHT
        Phase 3  Transport  : carry cube to goal XY, descend, open gripper
        Phase 4  Done       : hold (episode ends shortly)
        """
        cube_pos, _ = self.cube.get_world_pose()
        ee_pos, _   = self.franka.end_effector.get_world_pose()

        # ── Phase 0: align XY above cube, keep gripper open ──────────── #
        if self._phase == 0:
            target = cube_pos.copy()
            target[2] = cube_pos[2] + 0.10   # hover 10 cm above cube centre
            gripper_open = True

            # Transition on XY alignment only — Z convergence is not required
            # here; Phase 1 will handle the descent.
            xy_err = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
            if xy_err < 0.05:
                self._phase = 1

        # ── Phase 1: descend straight down to just above cube ────────── #
        elif self._phase == 1:
            target = cube_pos.copy()
            target[2] = cube_pos[2] + 0.01   # aim for 1 cm above cube top
            gripper_open = True

            # Advance when EE is within ~2 cm of the cube top (z = cube_z + 0.025)
            if ee_pos[2] <= cube_pos[2] + 0.06:
                self._grasp_pos = ee_pos.copy()
                self._phase = 2

        # ── Phase 2: close gripper, wait, then lift ───────────────── #
        elif self._phase == 2:
            gripper_open = False
            self._grip_timer += 1

            if self._grasp_pos is None:
                self._grasp_pos = ee_pos.copy()

            if self._grip_timer <= 30:
                target = self._grasp_pos.copy()          # hold in place
            else:
                target = self._grasp_pos.copy()
                target[2] = self.LIFT_HEIGHT             # rise with cube
                if ee_pos[2] >= self.LIFT_HEIGHT - 0.02:
                    self._phase      = 3
                    self._grip_timer = 0

        # ── Phase 3: move to goal XY at lift height (no descent yet) ─ #
        elif self._phase == 3:
            gripper_open = False
            target = np.array([self.GOAL_POS[0], self.GOAL_POS[1],
                               self.LIFT_HEIGHT], dtype=np.float64)
            xy_dist = np.linalg.norm(ee_pos[:2] - self.GOAL_POS[:2])
            if xy_dist < 0.05:
                self._phase = 4           # XY aligned — start descent

        # ── Phase 4: descend and release ──────────────────────────── #
        elif self._phase == 4:
            # Count every step so the timeout is always available.
            self._grip_timer += 1

            if self._grasp_pos is not None:
                ee_cube_offset = self._grasp_pos[2] - self.CUBE_INIT[2]
                place_z = self.GOAL_POS[2] + 0.0025 + 0.025 + ee_cube_offset - 0.02
            else:
                place_z = self.GOAL_POS[2] + 0.07
            target = np.array([self.GOAL_POS[0], self.GOAL_POS[1],
                               place_z], dtype=np.float64)

            # Open gripper when arm is low enough OR after 50 steps timeout.
            # The +0.05 tolerance handles cases where contact with the
            # FixedCuboid marker stops the arm just above the strict threshold.
            at_height  = ee_pos[2] <= place_z + 0.05
            timed_out  = self._grip_timer > 50
            gripper_open = at_height or timed_out

            if gripper_open and self._grip_timer > 55:
                self._phase      = 5
                self._grip_timer = 0

        # ── Phase 5: open gripper and hold ────────────────────────── #
        else:
            target = ee_pos.copy()
            target[2] = self.LIFT_HEIGHT
            gripper_open = True

        # ── Arm: RMPFlow (unchanged ctrl, no in-place mutation) ───── #
        ctrl = self.controller.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=None,
        )
        self.art_ctrl.apply_action(ctrl)      # arm joints

        # ── Gripper: separate call, joint_indices keeps arm safe ──── #
        self._set_gripper(gripper_open)

        delta       = np.clip((target - ee_pos), -1.0, 1.0).astype(np.float32)
        gripper_val = np.float32(1.0 if gripper_open else -1.0)
        return np.concatenate([delta, np.zeros(3, dtype=np.float32), [gripper_val]])

    @property
    def phase(self) -> int:
        """Current state-machine phase (0–5). Useful for logging."""
        return self._phase


# ========================================================================== #
#  HDF5 Writer                                                               #
# ========================================================================== #

class HDF5Writer:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.f  = h5py.File(path, "w")
        self.ep = 0
        print(f"[HDF5Writer] Writing to: {path}")

    def write(self, observations: list, actions: list,
              metadata: dict | None = None):
        grp = self.f.create_group(f"episode_{self.ep}")
        grp.create_dataset(
            "observations", data=np.stack(observations), compression="lzf"
        )
        grp.create_dataset(
            "actions", data=np.stack(actions), compression="lzf"
        )
        if metadata:
            for k, v in metadata.items():
                grp.attrs[k] = v
        self.ep += 1
        if self.ep % 20 == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()
        print(f"[HDF5Writer] Closed — {self.ep} episodes saved.")


# ========================================================================== #
#  Main collection loop                                                       #
# ========================================================================== #

def collect(args):
    if args.seed is not None:                                                                                                                                         
        np.random.seed(args.seed)                                                                                                                                     
        random.seed(args.seed)

    # ── VLA client (skipped in scripted mode) ─────────────────────────── #
    if args.scripted:
        client = None
        print("[Collector] Scripted mode — RMPFlow policy, no VLA server needed.")
    else:
        client = VLAClient(
            instruction  = args.instruction,
            unnorm_key   = args.unnorm_key,
            timeout_s    = args.timeout_s,
            jpeg_quality = args.jpeg_quality,
        )

    # ── Build scene ───────────────────────────────────────────────────── #
    scene    = FrankaScene()
    writer   = HDF5Writer(args.output)
    injector = InterferenceInjector(
        action_prob        = args.interference_prob if args.interference_action else 0.0,
        pose_prob          = args.interference_prob if args.interference_pose   else 0.0,
        action_noise_scale = args.action_noise_scale,
        pose_delta_max     = args.pose_delta_max,
        seed               = args.seed,
    )

    print(f"\n[Collector] {args.num_episodes} episodes × {args.episode_length} steps")
    if args.scripted:
        print("[Collector] Mode: scripted RMPFlow (no VLA server)")
    else:
        print("[Collector] VLA server: ROS2 /vla/request → /vla/response")
    print(f"[Collector] Interference: "
          f"action={args.interference_action}  "
          f"pose={args.interference_pose}  "
          f"p={args.interference_prob}")
    print(f"[Collector] Output: {args.output}\n")

    # Success threshold: cube must be within this XY distance of GOAL_POS
    # and below this height (resting on tray, not hovering).
    SUCCESS_XY_THRESH = 0.10   # metres
    SUCCESS_Z_MAX     = 0.07   # metres (tray top ≈ 0.004 + cube half-height 0.025)

    def is_success() -> bool:
        cube_pos, _ = scene.cube.get_world_pose()
        xy_dist = np.linalg.norm(cube_pos[:2] - scene.GOAL_POS[:2])
        return bool(xy_dist < SUCCESS_XY_THRESH and cube_pos[2] < SUCCESS_Z_MAX)

    t0             = time.time()
    total_latency  = 0.0
    total_requests = 0
    n_success      = 0

    for ep in range(args.num_episodes):
        scene.reset(randomise=True)
        injector.reset()
        obs_buf  = []
        act_buf  = []
        success  = False

        for step in range(args.episode_length):
            # Capture observation BEFORE action
            obs_t = scene.get_obs()                         # (H, W, 3) uint8

            if args.scripted:
                # _compute_rmpflow_action computes the target, applies it via
                # art_ctrl internally, and returns the (7,) action for HDF5.
                # Do NOT call scene.apply_action() again — that would double-apply.
                action = scene._compute_rmpflow_action()
            else:
                t_req  = time.perf_counter()
                action = client.request_action(obs_t)       # (7,) float32
                total_latency  += (time.perf_counter() - t_req) * 1000
                total_requests += 1
                if args.interference_action:
                    action = injector.maybe_perturb_action(action)
                scene.apply_action(action)

            if args.interference_pose:
                injector.maybe_teleport_cube("/World/Cube")

            obs_buf.append(obs_t)
            act_buf.append(action)

            scene.step()

            # Check success after physics settles each step
            if is_success():
                success = True
                print(f"  ep {ep+1:4d}  SUCCESS at step {step+1}")
                break

        if success:
            n_success += 1

        writer.write(obs_buf, act_buf, metadata={
            "disturbed":       int(args.interference_action or args.interference_pose),
            "n_action_events": injector.action_events,
            "n_pose_events":   injector.pose_events,
            "success":         int(success),
            "steps":           len(obs_buf),
        })

        elapsed     = time.time() - t0
        eta         = elapsed / (ep + 1) * (args.num_episodes - ep - 1)
        success_str = f"success={n_success}/{ep+1} ({100*n_success/(ep+1):.1f}%)"
        if args.scripted:
            print(
                f"  ep {ep+1:4d}/{args.num_episodes}  "
                f"phase={scene.phase}  "
                f"{success_str}  Δp={injector.pose_events}  ETA {eta:.0f}s"
            )
        else:
            avg_lat = total_latency / max(total_requests, 1)
            print(
                f"  ep {ep+1:4d}/{args.num_episodes}  "
                f"{success_str}  "
                f"Δa={injector.action_events}  Δp={injector.pose_events}  "
                f"avg_vla={avg_lat:.0f}ms  ETA {eta:.0f}s"
            )

    final_rate = 100.0 * n_success / max(args.num_episodes, 1)
    print(f"\n[Collector] Success rate: {n_success}/{args.num_episodes} ({final_rate:.1f}%)")
    writer.close()
    if client is not None:
        client.close()
    simulation_app.close()
    print(f"[Collector] Done. Total time: {time.time()-t0:.0f}s")


# ========================================================================== #
#  CLI                                                                        #
# ========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="Isaac Sim collector — sends obs to Linux server via ROS2, "
                    "receives actions on /vla/response"
    )
    # Network
    p.add_argument("--timeout_s",    type=float, default=10.0,
                   help="Seconds to wait for a VLA response before raising TimeoutError")
    p.add_argument("--jpeg_quality", type=int,   default=85,
                   help="JPEG compression quality (lower = less bandwidth)")

    # OpenVLA
    p.add_argument("--instruction", type=str,
                   default="pick up the red cube and place it on the tray")
    p.add_argument("--unnorm_key",  type=str, default="franka_isaac")

    # Collection
    p.add_argument("--output",           type=str,
                   default="data/trajectories.hdf5")
    p.add_argument("--num_episodes",     type=int,   default=300)
    p.add_argument("--episode_length",   type=int,   default=120)

    # Interference (all OFF by default → clean Stage-2 training data)
    p.add_argument("--interference_action", action="store_true")
    p.add_argument("--interference_pose",   action="store_true")
    p.add_argument("--interference_prob",   type=float, default=0.10)
    p.add_argument("--action_noise_scale",  type=float, default=0.30)
    p.add_argument("--pose_delta_max",      type=float, default=0.15)

    # Scripted mode
    p.add_argument("--scripted", action="store_true",
                   help="Use built-in RMPFlow policy instead of a VLA server "
                        "(no --server_host required)")

    # Misc
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


if __name__ == "__main__":
    collect(parse_args())