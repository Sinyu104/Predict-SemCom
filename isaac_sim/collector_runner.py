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
import threading
import time

import h5py
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
    """

    def __init__(
        self,
        instruction:  str   = "",
        unnorm_key:   str   = "franka_isaac",
        timeout_s:    float = 10.0,
        jpeg_quality: int   = 85,
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
        self._pending = {}
        self._results = {}
        self._lock    = threading.Lock()

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
        from PIL import Image as PILImage
        pil = PILImage.fromarray(obs_rgb)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def request_action(self, obs_rgb: np.ndarray) -> np.ndarray:
        """Publish obs, block until server replies, return (7,) float32 action."""
        from std_msgs.msg import String

        with self._lock:
            self._seq += 1
            seq        = self._seq
            event      = threading.Event()
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
                f"No response from VLA server within {self.timeout_s}s."
            )

        with self._lock:
            reply = self._results.pop(seq)
            self._pending.pop(seq, None)

        if reply.get("status") != "ok":
            print(f"[VLAClient] Server error: {reply.get('message')}. Using zero action.")
            return np.zeros(7, dtype=np.float32)

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
            import omni.isaac.core.utils.prims as prim_utils
            from pxr import Gf, UsdGeom
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


# ============================================================================ #
#  HDF5 Writer                                                                  #
# ============================================================================ #

class HDF5Writer:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.f  = h5py.File(path, "w")
        self.ep = 0
        print(f"[HDF5Writer] Writing to: {path}")

    def write(self, observations: list, actions: list,
              metadata: dict | None = None):
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
    p.add_argument("--scripted", action="store_true",
                   help="Use the scene's scripted_action() instead of the VLA server")
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
        )

    writer   = HDF5Writer(args.output)
    injector = InterferenceInjector(
        action_prob        = args.interference_prob if args.interference_action else 0.0,
        pose_prob          = args.interference_prob if args.interference_pose   else 0.0,
        action_noise_scale = args.action_noise_scale,
        pose_delta_max     = args.pose_delta_max,
        seed               = args.seed or 0,
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

        obs_t       = scene.get_obs()             # dict {cam_id: (H,W,3)}
        primary_cam = min(obs_t.keys())           # lowest cam id used for VLA

        for step in range(args.episode_length):
            if args.scripted:
                action = scene.scripted_action()
            else:
                t_req  = time.perf_counter()
                action = client.request_action(obs_t[primary_cam])
                total_latency  += (time.perf_counter() - t_req) * 1000
                total_requests += 1
                if args.interference_action:
                    action = injector.maybe_perturb_action(action)
                scene.apply_action(action)

            if args.interference_pose and hasattr(scene, "cubes"):
                # Teleport the first dynamic cube (task-agnostic disturbance)
                first_prim = next(iter(scene.cubes.values()))._prim_path
                injector.maybe_teleport_cube(first_prim)

            obs_buf.append(obs_t)
            act_buf.append(action)

            scene.step()
            obs_t = scene.get_obs()   # dict {cam_id: (H,W,3)}

            if scene.is_success():
                success = True
                print(f"  ep {ep+1:4d}  SUCCESS at step {step+1}")
                break

        if not success:
            print(f"  ep {ep+1:4d}  FAILED — discarded")
        if success:
            n_success += 1
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
                  f"{success_str}  Δp={injector.pose_events}  ETA {eta:.0f}s")
        else:
            avg_lat = total_latency / max(total_requests, 1)
            print(f"  ep {ep+1:4d}/{args.num_episodes}  {success_str}  "
                  f"Δa={injector.action_events}  Δp={injector.pose_events}  "
                  f"avg_vla={avg_lat:.0f}ms  ETA {eta:.0f}s")

    final_rate = 100.0 * n_success / max(args.num_episodes, 1)
    print(f"\n[Collector] Success rate: {n_success}/{args.num_episodes} "
          f"({final_rate:.1f}%)")
    writer.close()
    if client is not None:
        client.close()
    print(f"[Collector] Done. Total time: {time.time()-t0:.0f}s")
