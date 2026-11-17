"""
isaac_sim/collector_runner.py  —  Task-agnostic data collection infrastructure.

Contains:
  VLAClient          ROS2 client that sends obs → Linux server, receives actions
  InterferenceInjector  Action perturbation + pose disturbance for eval data
  HDF5Writer         Saves (obs_t, action_t) trajectories to HDF5
  parse_collector_args   Shared CLI argument parser for per-task entry points
  collect            Main collection loop; accepts any BaseScene instance

Per-task entry points (tasks/<name>/isaac_collector.py) create SimulationApp,
instantiate their scene subclass, then call collect(scene, args).
"""

import argparse
import base64
import io
import json
import os
import queue
import random
import re
import threading
import time

import numpy as np

from isaac_sim.base_scenes import BaseScene


# ============================================================================ #
#  VLA Client                                                                   #
# ============================================================================ #

class VLAClient:
    """
    ROS2 client: publishes observations to /vla/request, receives actions on
    /vla/response.  Uses sequence numbers to match each request to its reply.

    Parameters
    ----------
    instruction  : task language instruction sent with every request
    unnorm_key   : action un-normalisation key expected by the server
    timeout_s    : seconds to wait before raising TimeoutError
    jpeg_quality : JPEG compression quality (lower → less bandwidth)
    topic_prefix : namespace for the topics, e.g. "/B" → /B/vla/request and
                   /B/vla/response. Empty (default) keeps /vla/request and
                   /vla/response. Use it to run several inference stacks at once.
    """

    def __init__(
        self,
        instruction:  str   = "",
        unnorm_key:   str   = "franka_isaac",
        timeout_s:    float = 10.0,
        jpeg_quality: int   = 85,
        topic_prefix: str   = "",
    ):
        try:
            import rclpy
        except ImportError:
            raise ImportError(
                "rclpy not found.  Enable the ROS2 bridge extension and "
                "source your ROS2 workspace before launching Isaac Sim."
            )

        self.instruction  = instruction
        self.unnorm_key   = unnorm_key
        self.timeout_s    = timeout_s
        self.jpeg_quality = jpeg_quality

        self._seq     = 0
        self._episode = 0
        self._pending = {}
        self._results = {}
        self._lock    = threading.Lock()

        prefix          = topic_prefix.strip().strip("/")
        node_suffix     = re.sub(r"[^A-Za-z0-9_]", "_", prefix)   # ROS node names: alphanumerics and _ only
        prefix          = f"/{prefix}" if prefix else ""
        self.req_topic  = f"{prefix}/vla/request"
        self.resp_topic = f"{prefix}/vla/response"

        rclpy.init()
        self._node = rclpy.create_node(f"vla_client_{node_suffix}" if node_suffix else "vla_client")
        String     = __import__("std_msgs.msg", fromlist=["String"]).String

        self._pub = self._node.create_publisher(String, self.req_topic, 10)
        self._node.create_subscription(
            String, self.resp_topic, self._on_response, 10
        )

        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=executor.spin, daemon=True)
        self._spin_thread.start()

        print("[VLAClient] ROS2 node started.")
        print(f"[VLAClient] Publishing to  {self.req_topic}")
        print(f"[VLAClient] Subscribed to  {self.resp_topic}")

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
        from PIL import Image as PILImage
        pil = PILImage.fromarray(obs_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _pose_json(pose):
        """pose: list of 12-float rows [ee_x, ee_y, ee_z, joint_1..joint_9] -> plain rounded lists."""
        return [[round(float(v), 5) for v in row] for row in pose]

    def send_between(self, obs_rgb, step: int, pose=None):
        """
        Publish one in-between observation as its own small message, fire-and-forget.

        {"type": "between", "episode": E, "step": k, "images_b64": {cam_id (str): JPEG},
         "pose": [[12 floats]]}
        k = actions executed so far in the episode. "pose" is the robot pose measured at the
        same instant as the frame (one row). The bridge forwards it to the server, which
        stores the frame and publishes no reply, so this neither waits nor pauses the simulation.
        """
        from std_msgs.msg import String

        views = obs_rgb if isinstance(obs_rgb, dict) else {1: obs_rgb}
        payload = {
            "type":       "between",
            "episode":    self._episode,
            "step":       int(step),
            "images_b64": {str(k): self._compress(v) for k, v in sorted(views.items())},
        }
        if pose is not None:
            payload["pose"] = self._pose_json(pose)
        msg      = String()
        msg.data = json.dumps(payload)
        self._pub.publish(msg)

    def new_episode(self):
        """
        Mark the start of a new episode.  Stateful servers (semcom_server.py)
        key their per-episode history off this, so it MUST be called after every
        scene.reset() — otherwise the world model conditions on frames from the
        previous episode.  Stateless servers ignore the field.
        """
        self._episode += 1

    def request_action(self, obs_rgb, disturbed: bool = False, pose=None) -> np.ndarray:
        """
        Publish obs, block until server replies, return the action array, or None
        if the server replied with an error (caller should hold, not act).

        obs_rgb   : one (H, W, 3) uint8 image, or {cam_id: image} for several
                    cameras. One image -> "jpeg_b64" (legacy format). Several
                    -> "images_b64" {cam_id (str): JPEG}, with no "jpeg_b64".

        disturbed : True if an interference event (action perturbation or cube
                    teleport) fired since the previous request — i.e. this
                    observation reflects an injected disturbance rather than
                    clean rollout. Servers that don't care can ignore the field.

        pose      : list of rows [ee_x, ee_y, ee_z, joint_1..joint_9] (12 floats, same
                    quantities/units/order as robot_ee_pos + robot_joint_pos in the
                    datasets). One row per control step executed since the previous
                    request, row j = pose at the instant of the frame after step j; the last
                    row belongs to the image in this request. The first request of an
                    episode has a single row (pose when its image was taken).
        """
        from std_msgs.msg import String

        with self._lock:
            self._seq += 1
            seq        = self._seq
            event      = threading.Event()
            self._pending[seq] = event

        views = obs_rgb if isinstance(obs_rgb, dict) else {1: obs_rgb}
        images_b64 = {str(k): self._compress(v) for k, v in sorted(views.items())}

        payload = {
            "seq":         seq,
            "episode":     self._episode,
            "disturbed":   disturbed,
            "instruction": self.instruction,
            "unnorm_key":  self.unnorm_key,
        }
        if len(views) == 1:
            # Single view: legacy field only, identical to the old wire format.
            payload["jpeg_b64"] = next(iter(images_b64.values()))
        else:
            # Multi view: images_b64 only. Do NOT also send jpeg_b64 — it would
            # duplicate a camera and push the sample past ~64 KB, which forces
            # RTPS fragmentation that Fast DDS drops across subnets.
            payload["images_b64"] = images_b64
        if pose is not None:
            payload["pose"] = self._pose_json(pose)

        msg      = String()
        msg.data = json.dumps(payload)
        n_subs = self._pub.get_subscription_count()
        if len(msg.data) > 60_000:
            print(f"[VLAClient] WARNING: payload {len(msg.data)/1024:.0f} KiB is near the "
                  f"64 KiB UDP datagram limit; lower --jpeg_quality if the server stops receiving.")
        if seq == 1 or n_subs == 0:
            print(f"[VLAClient] seq={seq} publish {self.req_topic}  cams={sorted(views)}  "
                  f"payload={len(msg.data) / 1024:.0f} KiB  matched_subscribers={n_subs}")
        self._pub.publish(msg)

        if not event.wait(timeout=self.timeout_s):
            with self._lock:
                self._pending.pop(seq, None)
            raise TimeoutError(
                f"No response from VLA server within {self.timeout_s}s "
                f"(seq={seq}, matched_subscribers={n_subs}; 0 means nothing on this "
                f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '').strip() or '<unset>'} "
                f"is listening on {self.req_topic})."
            )

        with self._lock:
            reply = self._results.pop(seq)
            self._pending.pop(seq, None)

        if reply.get("status") != "ok":
            print(f"[VLAClient] Server error: {reply.get('message')}. "
                  f"No action received — holding position.")
            return None

        return np.array(reply["action"], dtype=np.float32)

    def close(self):
        import rclpy
        rclpy.shutdown()


# ============================================================================ #
#  Interference Injector                                                        #
# ============================================================================ #

class InterferenceInjector:
    """
    Action perturbation + pose disturbance for evaluation data collection.

    Action perturbation  corrupt the action before execution.
    Pose disturbance     teleport a cube mid-episode for sudden visual change.
    """

    def __init__(
        self,
        action_prob:        float = 0.05,
        pose_prob:          float = 0.05,
        action_noise_scale: float = 0.30,
        pose_delta_max:     float = 0.15,
        seed:               int   = 0,
        min_gap:            int   = 0,
    ):
        # min_gap: minimum number of steps between two events of the same kind
        # (action noise / cube teleport). 0 = no limit (independent roll every step).
        self.min_gap            = max(int(min_gap), 0)
        self._action_wait       = 0
        self._pose_wait         = 0
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
        self._action_wait  = 0
        self._pose_wait    = 0

    def maybe_perturb_action(self, action: np.ndarray) -> np.ndarray:
        if self._action_wait > 0:          # inside the min_gap window: no roll
            self._action_wait -= 1
            return action
        if self.rng.random() < self.action_prob:
            noise  = self.np_rng.randn(*action.shape).astype(np.float32)
            action = np.clip(action + noise * self.action_noise_scale, -1., 1.)
            self.action_events += 1
            self._action_wait = max(self.min_gap - 1, 0)
        return action

    def maybe_teleport_cubes(
        self,
        movable_prim_paths: list,
        all_prim_paths: list | None = None,
        x_range: tuple | None = None,
        y_range: tuple | None = None,
        min_dist: float = 0.12,
    ) -> bool:
        """
        Single probability roll shared by all cubes; if it fires, every cube in
        movable_prim_paths teleports together (each gets its own random offset).
        x_range/y_range, if given, clamp each post-teleport position so cubes
        stay within the robot's reachable workspace (e.g. scene.CUBE_X_RANGE).
        min_dist enforces the same non-overlap spacing used at episode reset
        (scene.MIN_CUBE_DIST) against every cube in all_prim_paths, including
        ones not being moved (e.g. one currently grasped by the gripper).
        """
        if self._pose_wait > 0:            # inside the min_gap window: no roll
            self._pose_wait -= 1
            return False
        if self.rng.random() >= self.pose_prob:
            return False
        self._pose_wait = max(self.min_gap - 1, 0)
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import Gf, UsdGeom

        def _get_prim_and_pos(prim_path):
            prim = prim_utils.get_prim_at_path(prim_path)
            if not prim.IsValid():
                return None, None, None
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    return prim, op, op.Get()
            return prim, None, None

        movable_set = set(movable_prim_paths)
        placed_xy   = []  # (x, y) of every cube already fixed for this event
        for prim_path in (all_prim_paths or []):
            if prim_path in movable_set:
                continue
            _, _, cur = _get_prim_and_pos(prim_path)
            if cur is not None:
                placed_xy.append((float(cur[0]), float(cur[1])))

        for cube_prim_path in movable_prim_paths:
            prim, op, cur = _get_prim_and_pos(cube_prim_path)
            if prim is None or op is None:
                continue

            new_x, new_y = float(cur[0]), float(cur[1])
            for _attempt in range(20):
                dx = self.np_rng.uniform(-self.pose_delta_max, self.pose_delta_max)
                dy = self.np_rng.uniform(-self.pose_delta_max, self.pose_delta_max)
                cand_x = float(cur[0]) + dx
                cand_y = float(cur[1]) + dy
                if x_range is not None:
                    cand_x = float(np.clip(cand_x, *x_range))
                if y_range is not None:
                    cand_y = float(np.clip(cand_y, *y_range))
                if all(
                    np.hypot(cand_x - px, cand_y - py) >= min_dist
                    for px, py in placed_xy
                ):
                    new_x, new_y = cand_x, cand_y
                    break
                # else: too close to another cube — resample (falls back to
                # leaving the cube where it is if no valid spot is found)

            op.Set(Gf.Vec3d(new_x, new_y, float(cur[2])))
            placed_xy.append((new_x, new_y))

        self.pose_events += 1
        return True


# ============================================================================ #
#  HDF5 Writer                                                                  #
# ============================================================================ #

class HDF5Writer:
    def __init__(self, path: str, append: bool = False):
        import h5py
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        mode = "a" if (append and os.path.exists(path)) else "w"
        try:
            self.f = h5py.File(path, mode)
        except OSError:
            print(f"[HDF5Writer] Corrupt file detected, overwriting: {path}")
            self.f = h5py.File(path, "w")
            mode = "w"
        self.ep = len([k for k in self.f.keys() if k.startswith("episode_")])
        print(f"[HDF5Writer] {'Appending to' if mode=='a' else 'Writing to'}: {path} (starting at ep {self.ep})")

    def write(self, observations: list, actions: list,
              metadata: dict | None = None, disturbed: list | None = None,
              extras: dict | None = None):
        """
        extras    : optional {name: list of per-step arrays}, each stored as its own
                    dataset of length T (e.g. robot_ee_pos, robot_joint_pos).

        disturbed : optional per-step bool list/array, same length as
                    observations/actions. disturbed[i] marks whether an
                    interference event (actuator noise or cube teleport) fired
                    while step i was executing — i.e. a frame-aligned version
                    of the episode-level n_action_events/n_pose_events counts.
        """
        grp = self.f.create_group(f"episode_{self.ep}")
        # observations is a list of dicts {cam_id: (H,W,3)}
        for cam_id in observations[0].keys():
            grp.create_dataset(
                f"observations_cam{cam_id}",
                data=np.stack([obs[cam_id] for obs in observations]),
                compression="lzf",
            )
        grp.create_dataset(
            "actions", data=np.stack(actions), compression="lzf"
        )
        if disturbed is not None:
            grp.create_dataset(
                "disturbed", data=np.asarray(disturbed, dtype=bool), compression="lzf"
            )
        for name, series in (extras or {}).items():
            grp.create_dataset(name, data=np.stack(series), compression="lzf")
        if metadata:
            for k, v in metadata.items():
                grp.attrs[k] = v
        self.ep += 1
        self.f.flush()   # every episode, so a crash loses at most the current one

    def close(self):
        self.f.flush()
        self.f.close()
        print(f"[HDF5Writer] Closed — {self.ep} episodes saved.")


# ============================================================================ #
#  CLI helpers                                                                  #
# ============================================================================ #

def parse_collector_args(
    default_output:      str,
    default_instruction: str,
    default_unnorm_key:  str = "franka_isaac",
) -> argparse.Namespace:
    """
    Shared argument parser for all per-task isaac_collector.py entry points.

    Parameters
    ----------
    default_output      : default HDF5 output path (task-specific)
    default_instruction : default language instruction (task-specific)
    default_unnorm_key  : action un-normalisation key
    """
    p = argparse.ArgumentParser(
        description="Isaac Sim collector — sends obs to Linux server via ROS2, "
                    "receives actions on /vla/response"
    )
    p.add_argument("--timeout_s",    type=float, default=10.0)
    p.add_argument("--jpeg_quality", type=int,   default=85)
    p.add_argument("--topic_prefix", type=str,   default="",
                   help="Topic namespace, e.g. /B -> /B/vla/request, /B/vla/response")
    p.add_argument("--instruction",  type=str,   default=default_instruction)
    p.add_argument("--unnorm_key",   type=str,   default=default_unnorm_key)
    p.add_argument("--output",       type=str,   default=default_output)
    p.add_argument("--num_episodes", type=int,   default=300)
    p.add_argument("--episode_length", type=int, default=120)
    p.add_argument("--interference_action", action="store_true")
    p.add_argument("--interference_pose",   action="store_true")
    p.add_argument("--interference_prob",   type=float, default=0.10)
    p.add_argument("--action_noise_scale",  type=float, default=0.30)
    p.add_argument("--pose_delta_max",      type=float, default=0.15)
    p.add_argument("--pose_flush_subframes", type=int, default=32,
                   help="Render subframes for the step right after a cube teleport, so the old "
                        "cube position doesn't linger as a faded ghost (normal steps use 4)")
    p.add_argument("--interference_min_gap", type=int,  default=5,
                   help="Minimum steps between two interference events of the same kind "
                        "(default 5; 0 = no limit)")
    p.add_argument("--scripted", action="store_true",
                   help="Use the scene's scripted_action() instead of the VLA server")
    p.add_argument("--record_failures", action="store_true",
                   help="Also save failed episodes to the HDF5 (tagged success=0) "
                        "instead of discarding them, for debugging failure cases")
    p.add_argument("--headless", action="store_true",
                   help="Run without the Isaac Sim GUI window")
    p.add_argument("--camera", nargs="+", type=int, default=[1],
                   help="Camera IDs to collect (1=fixed, 2=second fixed, 3=wrist). "
                        "E.g. --camera 2 3")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


# ============================================================================ #
#  Main collection loop                                                         #
# ============================================================================ #

def collect(scene: BaseScene, args: argparse.Namespace) -> None:
    """
    Run the data collection loop for any BaseScene instance.

    Parameters
    ----------
    scene : already-constructed BaseScene subclass instance
    args  : parsed namespace from parse_collector_args()
    """
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.scripted:
        client = None
        print("[Collector] Scripted mode — scene.scripted_action(), no VLA server.")
    else:
        client = VLAClient(
            instruction  = args.instruction,
            unnorm_key   = args.unnorm_key,
            timeout_s    = args.timeout_s,
            jpeg_quality = args.jpeg_quality,
            topic_prefix = args.topic_prefix,
        )

    writer   = HDF5Writer(args.output)
    injector = InterferenceInjector(
        action_prob        = args.interference_prob if args.interference_action else 0.0,
        pose_prob          = args.interference_prob if args.interference_pose   else 0.0,
        action_noise_scale = args.action_noise_scale,
        pose_delta_max     = args.pose_delta_max,
        seed               = args.seed or 0,
        min_gap            = args.interference_min_gap,
    )

    print(f"\n[Collector] {args.num_episodes} episodes × {args.episode_length} steps")
    print(f"[Collector] Scripted: {args.scripted}")
    print(f"[Collector] Interference: action={args.interference_action}  "
          f"pose={args.interference_pose}  p={args.interference_prob}")
    print(f"[Collector] Output: {args.output}\n")

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

        for _ in range(5):                        # flush post-reset render before first obs
            scene.step()
        obs_t       = scene.get_obs()             # dict {cam_id: (H,W,3)}
        primary_cam = min(obs_t.keys())           # lowest cam id used for VLA

        for step in range(args.episode_length):
            pose_event = False            # set when a cube teleport fires this step
            if args.scripted:
                action = scene.scripted_action()
            else:
                t_req  = time.perf_counter()
                action = client.request_action(obs_t[primary_cam])
                total_latency  += (time.perf_counter() - t_req) * 1000
                total_requests += 1
                if action is not None:
                    if args.interference_action:
                        action = injector.maybe_perturb_action(action)
                    scene.apply_action(action)

            if args.interference_pose and hasattr(scene, "cubes"):
                # Single shared roll: if it fires, all cubes teleport together,
                # except the one currently grasped (phase >= 2) — teleporting a
                # cube physics is already holding via the gripper destabilises
                # the articulation.
                held_cube = (
                    getattr(scene, "grasp_target", None)
                    if getattr(scene, "phase", 0) >= 2 else None
                )
                x_range   = getattr(scene, "CUBE_X_RANGE", None)
                y_range   = getattr(scene, "CUBE_Y_RANGE", None)
                min_dist  = getattr(scene, "MIN_CUBE_DIST", 0.12)
                all_paths = [cube.prim_path for cube in scene.cubes.values()]
                movable_paths = [cube.prim_path for cube in scene.cubes.values()
                                 if cube is not held_cube]
                pose_event = injector.maybe_teleport_cubes(
                    movable_paths, all_paths, x_range, y_range, min_dist
                )

            if action is not None:        # a held step has no action to record
                obs_buf.append(obs_t)
                act_buf.append(action)

            scene.step(rt_subframes=args.pose_flush_subframes if pose_event else 4)
            obs_t = scene.get_obs()   # dict {cam_id: (H,W,3)}

            if scene.is_success():
                success = True
                print(f"  ep {ep+1:4d}  SUCCESS at step {step+1}")
                break

        if not success:
            print(f"  ep {ep+1:4d}  FAILED"
                  + (" — recorded" if args.record_failures else " — discarded"))
        if success:
            n_success += 1
        if success or args.record_failures:
            writer.write(obs_buf, act_buf, metadata={
                "disturbed":       int(args.interference_action or args.interference_pose),
                "n_action_events": injector.action_events,
                "n_pose_events":   injector.pose_events,
                "success":         int(success),
                "steps":           len(obs_buf),
                "instruction":     args.instruction,
            })

        elapsed = time.time() - t0
        eta     = elapsed / (ep + 1) * (args.num_episodes - ep - 1)
        success_str = f"success={n_success}/{ep+1} ({100*n_success/(ep+1):.1f}%)"

        if args.scripted:
            phase = getattr(scene, "phase", "?")
            print(f"  ep {ep+1:4d}/{args.num_episodes}  phase={phase}  "
                  f"{success_str}  d_pose={injector.pose_events}  ETA {eta:.0f}s")
        else:
            avg_lat = total_latency / max(total_requests, 1)
            print(f"  ep {ep+1:4d}/{args.num_episodes}  {success_str}  "
                  f"d_action={injector.action_events}  d_pose={injector.pose_events}  "
                  f"avg_vla={avg_lat:.0f}ms  ETA {eta:.0f}s")

    final_rate = 100.0 * n_success / max(args.num_episodes, 1)
    print(f"\n[Collector] Success rate: {n_success}/{args.num_episodes} "
          f"({final_rate:.1f}%)")
    writer.close()
    if client is not None:
        client.close()
    print(f"[Collector] Done. Total time: {time.time()-t0:.0f}s")
