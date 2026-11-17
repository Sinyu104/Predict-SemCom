"""
real_robot/so101_runner.py  —  Real SO-101 analogue of isaac_sim/vla_runner.py.

Runs a trained Pi0-FAST policy on the physical SO-101 arm:

    SO101Follower (serial + 2 cameras)
        -> framed-JSON TCP -> vla_server.py (Pi0-FAST)   [direct, default]
        -> action chunk    -> SO101Follower.send_action()

The same wire protocol is spoken by semcom_server.py, so inserting the SemCom
pipeline later is purely a --server_port change:

    direct  :  runner ------------------> vla_server.py      (default 5555)
    semcom  :  runner -> semcom_server -> vla_server.py      (5556 -> 5557)

No ROS anywhere.  isaac_sim/collector_runner.py routes through ROS2 only
because Isaac Sim's embedded Python cannot host rclpy; vla_server.py itself
imports no ROS and binds a plain TCP socket.

Differences from vla_runner.py that are deliberate, not oversights
------------------------------------------------------------------
* No scene: episodes are reset by a human, success is judged by a human.
  There is no scripted oracle, so no --compare_scripted / --dagger.
* Action space is SO-101's 6 DoF (5 joints + gripper) in lerobot's native
  units: degrees for the joints, 0-100 for the gripper.  It is NOT Franka's
  7-vector and NOT normalised to [-1, 1].  Nothing here zeroes dims 3:6.
* An error reply is never applied to the arm.  Both servers return a
  Franka-shaped zero vector on failure, and in SO-101 degree space "all zeros"
  is a real pose the arm would snap to -- not a no-op.  We hold position.

CAMERA MAPPING -- read before changing
--------------------------------------
The server's second image field is named `jpeg_b64_wrist`, but for THIS dataset
that name is misleading.  convert_to_pi0_hdf5.py built the training file as:

    observations_cam1  <- wrist camera   (--cam1_index, lerobot camera index 0)
    observations_cam2  <- top camera     (--cam2_index, lerobot camera index 1)

and vla_server maps  jpeg_b64 -> cam1  and  jpeg_b64_wrist -> cam2.  Therefore:

    jpeg_b64        MUST carry the WRIST camera
    jpeg_b64_wrist  MUST carry the TOP camera

The field names are just labels; what matters is that this mapping is identical
to training.  Swapping them silently feeds the policy transposed views.

Usage
-----
    python real_robot/so101_runner.py \
        --server_host 192.168.1.50 --server_port 5555 \
        --task pick_blue_cube --episodes 5
"""

import argparse
import base64
import io
import json
import os
import socket
import threading
import time

import numpy as np
from PIL import Image

try:
    import msvcrt            # Windows console key polling (right arrow = task finished)
except ImportError:          # not Windows: the key is unavailable, episodes end by max_steps/Ctrl-C
    msvcrt = None

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# Action vector order.  Must match the `action` feature order the policy was
# fine-tuned on (meta/info.json of the recording dataset).  Declared explicitly
# rather than relying on dict iteration order.
MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
ACTION_DIM = len(MOTOR_ORDER)


# ========================================================================== #
#  Framed-JSON TCP client                                                     #
# ========================================================================== #

class PolicyTCPClient:
    """
    Persistent TCP client for vla_server.py / semcom_server.py.
    Wire format: 4-byte big-endian length prefix + JSON payload.

    Request : {seq, episode, jpeg_b64, jpeg_b64_wrist, instruction, unnorm_key, state}
    Reply   : {seq, action, status[, exec_horizon][, message]}

    `episode` is ignored by vla_server but drives semcom_server's per-episode
    Ctrl-World history reset, so it is always sent.

    `exec_horizon` is absent from vla_server replies (execute the whole chunk)
    and is 1 from semcom_server (its closed loop conditions on the action
    actually taken, so it cannot hand out future actions).
    """

    def __init__(self, host: str, port: int, jpeg_quality: int = 85,
                 timeout_s: float = 30.0):
        self.jpeg_quality = jpeg_quality
        self._seq = 0
        self.episode = 0
        print(f"[SO101Runner] Connecting to policy server at {host}:{port} ...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        self.sock.connect((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("[SO101Runner] Connected.")

    # -- framing ---------------------------------------------------------- #

    def _recv_all(self, n: int) -> bytes:
        chunks, got = [], 0
        while got < n:
            c = self.sock.recv(min(65536, n - got))
            if not c:
                raise ConnectionError("policy server disconnected")
            chunks.append(c)
            got += len(c)
        return b"".join(chunks)

    def _encode(self, rgb: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # -- public ----------------------------------------------------------- #

    def new_episode(self):
        """Advance the episode counter (resets semcom_server's world-model history)."""
        self.episode += 1

    def request_action(self, cam_main_rgb: np.ndarray, cam_aux_rgb: np.ndarray | None,
                       instruction: str,
                       state: list[float] | None = None) -> tuple[np.ndarray | None, int | None, str]:
        """
        cam_main_rgb -> jpeg_b64        (training observations_cam1 == wrist)
        cam_aux_rgb  -> jpeg_b64_wrist  (training observations_cam2 == top)
        state        -> state           (6 raw joint readings in degrees, MOTOR_ORDER,
                                         read at the same moment as the images; the server
                                         normalises/discretises, so nothing is done here)

        Returns (actions, exec_horizon, status).
        `actions` is (N, ACTION_DIM) on success, or None when the server
        reported an error -- the caller must then hold position.
        `exec_horizon` is None when the server did not specify one.
        """
        self._seq += 1
        req = {
            "seq": self._seq,
            "episode": self.episode,
            "jpeg_b64": self._encode(cam_main_rgb),
            "instruction": instruction,
            "unnorm_key": "so101",
        }
        if state is not None:
            req["state"] = [float(v) for v in state]
        if cam_aux_rgb is not None:
            req["jpeg_b64_wrist"] = self._encode(cam_aux_rgb)

        payload = json.dumps(req).encode()
        self.sock.sendall(len(payload).to_bytes(4, "big") + payload)

        n = int.from_bytes(self._recv_all(4), "big")
        reply = json.loads(self._recv_all(n))

        if reply.get("status") != "ok":
            return None, None, str(reply.get("message", "unknown error"))

        actions = np.asarray(reply["action"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]

        if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
            # Most likely a Franka-shaped fallback ([[0.0]*7]) or a server still
            # configured for the 7-DoF policy.  Never push this to the arm.
            return None, None, (
                f"expected action dim {ACTION_DIM}, got shape {actions.shape} "
                f"-- server may still be configured for the 7-DoF Franka policy"
            )

        horizon = reply.get("exec_horizon")
        return actions, (int(horizon) if horizon is not None else None), "ok"

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# ========================================================================== #
#  Robot helpers                                                              #
# ========================================================================== #

def build_robot(args) -> SO101Follower:
    cameras = {
        "main": OpenCVCameraConfig(index_or_path=args.cam_main_index,
                                   width=args.width, height=args.height, fps=args.fps),
        "aux": OpenCVCameraConfig(index_or_path=args.cam_aux_index,
                                  width=args.width, height=args.height, fps=args.fps),
    }
    cfg = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        # Per-step clamp on |goal - present|, same units as the action vector.
        # Sized from the recorded demonstrations: the largest per-step motion in
        # any of the 20 episodes was 3.9 (joints, deg) / 5.4 (gripper), so this
        # leaves ~2x headroom while cutting a runaway command to 8 per step.
        max_relative_target=args.max_relative_target,
    )
    return SO101Follower(cfg)


def to_action_dict(vec: np.ndarray) -> dict[str, float]:
    return {f"{name}.pos": float(vec[i]) for i, name in enumerate(MOTOR_ORDER)}


class FinishKey:
    """
    Background listener for the RIGHT ARROW key, meaning "the task is finished".

    Polls the console (never blocks, so input() prompts keep working afterwards) while an
    episode runs.  On Windows an arrow key arrives as two characters: a prefix (chr 0 or
    chr 224) and then 'M' for right.
    """

    def __init__(self):
        self.pressed = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    @property
    def available(self) -> bool:
        return msvcrt is not None

    def start(self):
        self.pressed.clear()
        self._stop.clear()
        if not self.available:
            return
        while msvcrt.kbhit():          # drop stale keys, e.g. the ENTER that started this episode
            msvcrt.getwch()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in (chr(0), chr(224)) and msvcrt.getwch() == "M":
                    self.pressed.set()
                    return
            else:
                time.sleep(0.02)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def hold_current_pose(robot: SO101Follower) -> None:
    """
    Turn torque back on WITHOUT the arm jumping.

    A Feetech servo drives to its stored Goal_Position the moment torque is enabled.
    After a hand reposition that register still holds the last command of the previous
    episode, so enabling torque directly would whip the arm back there.  Writing the
    present reading into Goal_Position first makes "hold here" the target.
    """
    present = robot.bus.sync_read("Present_Position")
    robot.bus.sync_write("Goal_Position", present)
    robot.bus.enable_torque()


def release_torque(robot: SO101Follower) -> dict[str, str]:
    """
    Disable torque one motor at a time.  Returns {motor: reason} for motors that refused.

    A gripper left clamped under load can latch a Feetech "Overload" fault; that motor then
    refuses the write.  Releasing per motor means one stuck motor no longer stops the others.
    """
    failed = {}
    for name in MOTOR_ORDER:
        try:
            robot.bus.disable_torque(name, num_retry=3)
        except RuntimeError as e:
            failed[name] = str(e).splitlines()[0][:200]
    return failed


def reposition_arm(robot: SO101Follower, next_episode: int) -> None:
    """Between episodes: release torque so the arm can be moved by hand, then re-hold it."""
    print()
    print("[SO101Runner] Episode finished. SUPPORT THE ARM: torque is about to be released "
          "and it will go limp.")
    input("[SO101Runner] Press ENTER to release torque (Ctrl-C to quit) ...")
    failed = release_torque(robot)
    if failed:
        print("[SO101Runner] WARNING: these motors refused to release torque and stay stiff:")
        for name, why in failed.items():
            print(f"    {name}: {why}")
        print("[SO101Runner] An 'Overload error' is a latched fault (e.g. the gripper clamped "
              "on the cube). Power-cycle the arm to clear it; open the gripper before an "
              "episode ends so it is not left squeezing.")
    print("[SO101Runner] Torque released on the other motors. Move the arm to the start pose "
          "and reset the scene.")
    input(f"[SO101Runner] When ready for episode {next_episode}, press ENTER to re-enable "
          f"torque where the arm is now and start (Ctrl-C to quit) ...")
    hold_current_pose(robot)
    print("[SO101Runner] Torque re-enabled at the current pose.")


def read_state(obs: dict) -> list[float]:
    """Raw joint positions (degrees; gripper 0-100) in MOTOR_ORDER from one observation."""
    return [float(obs[f"{name}.pos"]) for name in MOTOR_ORDER]


def load_instruction(task: str, data_dir: str, override: str | None) -> str:
    if override:
        return override
    path = os.path.join(data_dir, task, "task.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"task.json not found at '{path}'. Pass --instruction explicitly "
            f"or check --data_dir."
        )
    with open(path) as f:
        return json.load(f)["instruction"]


# ========================================================================== #
#  Episode loop                                                               #
# ========================================================================== #

def run_episode(robot: SO101Follower, client: PolicyTCPClient, instruction: str,
                args, ep_idx: int) -> dict:
    client.new_episode()
    period = 1.0 / args.max_hz if args.max_hz > 0 else 0.0

    step = 0
    n_errors = 0
    lat_ms = []
    t_start = time.perf_counter()

    print(f"\n[SO101Runner] --- episode {ep_idx} (server episode "
          f"{client.episode}) --- Ctrl-C to end early")
    finish = FinishKey()
    finish.start()
    if finish.available:
        print("[SO101Runner] Press the RIGHT ARROW key when the task is finished "
              "to end this episode.")

    try:
        while step < args.max_steps and not finish.pressed.is_set():
            obs = robot.get_observation()
            cam_main = obs["main"]
            cam_aux = None if args.single_camera else obs["aux"]
            state = read_state(obs)   # same get_observation() call as the images

            t_req = time.perf_counter()
            actions, horizon, status = client.request_action(
                cam_main, cam_aux, instruction, state)
            lat_ms.append((time.perf_counter() - t_req) * 1000)

            if actions is None:
                # Hold position: the servers' zero-vector fallback is a real
                # pose on this arm, not a no-op.
                n_errors += 1
                print(f"  step {step:4d}  SERVER ERROR, holding position: {status}")
                if n_errors >= args.max_consecutive_errors:
                    print(f"[SO101Runner] {n_errors} consecutive errors, aborting episode.")
                    break
                continue
            n_errors = 0

            # No exec_horizon (vla_server) -> execute the whole chunk, which is
            # what makes remote inference viable.  exec_horizon=1 (semcom_server)
            # -> one step per round trip, so the control rate collapses to
            # 1 / round-trip latency.
            n_exec = len(actions) if horizon is None else max(1, min(horizon, len(actions)))

            for k in range(n_exec):
                if step >= args.max_steps or finish.pressed.is_set():
                    break
                t_tick = time.perf_counter()
                if not args.dry_run:
                    robot.send_action(to_action_dict(actions[k]))
                step += 1
                if period:
                    dt = time.perf_counter() - t_tick
                    if dt < period:
                        time.sleep(period - dt)

            if args.verbose:
                a  = ", ".join(f"{v:+7.2f}" for v in actions[0])
                st = ", ".join(f"{v:+7.2f}" for v in state)
                # How far the commanded targets are from where the arm is right now: ~0
                # means the policy is asking the arm to stay put (no visible motion).
                gap0   = np.abs(actions[0]  - np.asarray(state)).max()
                gapmax = np.abs(actions[:n_exec] - np.asarray(state)).max()
                print(f"  step {step:4d}  lat={lat_ms[-1]:6.1f}ms  "
                      f"chunk={len(actions)} exec={n_exec}  a0=[{a}]\n"
                      f"             state=[{st}]  max|a0-state|={gap0:5.1f}  "
                      f"max|chunk-state|={gapmax:5.1f}")

    except KeyboardInterrupt:
        print("\n[SO101Runner] Episode interrupted by user.")
    finally:
        finish.stop()

    finished = finish.pressed.is_set()
    if finished:
        print(f"[SO101Runner] RIGHT ARROW: task marked finished at step {step}.")

    elapsed = time.perf_counter() - t_start
    hz = step / elapsed if elapsed > 0 else 0.0
    mean_lat = float(np.mean(lat_ms)) if lat_ms else float("nan")
    print(f"[SO101Runner] episode {ep_idx}: {step} steps in {elapsed:.1f}s "
          f"({hz:.1f} Hz)  mean round-trip {mean_lat:.0f} ms  "
          f"({len(lat_ms)} requests)")
    return {"steps": step, "hz": hz, "mean_latency_ms": mean_lat,
            "requests": len(lat_ms), "finished": finished}


def main():
    args = parse_args()
    instruction = load_instruction(args.task, args.data_dir, args.instruction)

    print(f"[SO101Runner] Instruction : {instruction!r}")
    print(f"[SO101Runner] Cameras     : jpeg_b64<-index {args.cam_main_index} "
          f"(must be WRIST), jpeg_b64_wrist<-index {args.cam_aux_index} "
          f"(must be TOP)")
    print(f"[SO101Runner] Safety clamp: max_relative_target="
          f"{args.max_relative_target}")
    if args.dry_run:
        print("[SO101Runner] DRY RUN: actions are logged, never sent to the arm.")
    if args.single_camera:
        print("[SO101Runner] WARNING: --single_camera set. The policy was "
              "fine-tuned on two views; its output will be unreliable.")

    robot = build_robot(args)
    # calibrate=False: the calibration must stay exactly as it was during data
    # collection.  Re-running calibration shifts the frame the policy's actions
    # are expressed in and silently invalidates the model.
    robot.connect(calibrate=False)

    client = PolicyTCPClient(args.server_host, args.server_port,
                             jpeg_quality=args.jpeg_quality,
                             timeout_s=args.timeout_s)

    results = []
    try:
        for ep in range(1, args.episodes + 1):
            if ep > 1 and not args.keep_torque:
                reposition_arm(robot, ep)        # release torque -> move arm/scene -> re-hold
            elif ep > 1 or not args.no_prompt:
                input(f"\n[SO101Runner] Reset the scene for episode {ep}, "
                      f"then press ENTER (Ctrl-C to quit) ...")
            results.append(run_episode(robot, client, instruction, args, ep))
    except KeyboardInterrupt:
        print("\n[SO101Runner] Stopped by user.")
    finally:
        client.close()
        try:
            robot.disconnect()
            print("[SO101Runner] Robot disconnected (torque released).")
        except RuntimeError as e:
            print("[SO101Runner] WARNING: could not release torque cleanly on disconnect:")
            print(f"    {str(e).splitlines()[0][:200]}")
            print("[SO101Runner] Power-cycle the arm before the next run.")

    if results:
        print("\n" + "=" * 62)
        print(f"{'episode':>8}{'steps':>8}{'reqs':>8}{'Hz':>8}{'latency ms':>14}{'finished':>10}")
        for i, r in enumerate(results, 1):
            print(f"{i:>8}{r['steps']:>8}{r['requests']:>8}"
                  f"{r['hz']:>8.1f}{r['mean_latency_ms']:>14.0f}{'yes' if r['finished'] else 'no':>10}")
        n_fin = sum(r["finished"] for r in results)
        print(f"Marked finished with the right arrow: {n_fin}/{len(results)}")


# ========================================================================== #
#  CLI                                                                        #
# ========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a Pi0-FAST policy on the real SO-101 over framed-JSON TCP")

    # Robot hardware
    p.add_argument("--robot_port", type=str, default="COM4",
                   help="Serial port of the SO-101 follower arm")
    p.add_argument("--robot_id", type=str, default="my_awesome_follower_arm",
                   help="Calibration id -- MUST match the one used for recording")
    p.add_argument("--cam_main_index", type=int, default=0,
                   help="Camera sent as jpeg_b64. For this dataset that is the "
                        "WRIST camera (training observations_cam1).")
    p.add_argument("--cam_aux_index", type=int, default=1,
                   help="Camera sent as jpeg_b64_wrist. For this dataset that is "
                        "the TOP camera (training observations_cam2).")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--single_camera", action="store_true",
                   help="Send only the main camera (debugging; policy expects two)")

    # Safety
    p.add_argument("--max_relative_target", type=float, default=8.0,
                   help="Per-step clamp on |goal - present|, in action units. "
                        "Demos peaked at 3.9 (joints) / 5.4 (gripper).")
    p.add_argument("--max_consecutive_errors", type=int, default=5)

    # Policy server
    p.add_argument("--server_host", type=str, default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=5555,
                   help="5555 = vla_server.py direct; point at semcom_server.py "
                        "to insert the SemCom pipeline later.")
    p.add_argument("--jpeg_quality", type=int, default=85)
    p.add_argument("--timeout_s", type=float, default=30.0)

    # Task / rollout
    p.add_argument("--task", type=str, default="pick_blue_cube")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--instruction", type=str, default=None,
                   help="Override the instruction from task.json")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=400,
                   help="Steps per episode (recorded demos averaged ~420)")
    p.add_argument("--max_hz", type=float, default=30.0,
                   help="Control rate cap; 0 disables pacing. Recording was 30 Hz.")
    p.add_argument("--keep_torque", action="store_true",
                   help="Keep torque on between episodes (old behaviour). By default torque is "
                        "released after each episode so the arm can be moved back by hand.")
    p.add_argument("--no_prompt", action="store_true",
                   help="Skip the ENTER prompt before the first episode")
    p.add_argument("--dry_run", action="store_true",
                   help="Query the policy and log its actions but never call "
                        "send_action -- the arm does not move. Use this to "
                        "inspect action values before trusting them.")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    main()
