"""
real_robot/preflight_check.py  —  Pre-inference health check for the SO-101.

Run this before so101_runner.py.  Every check here corresponds to a failure we
actually hit during data collection, and each one is silent or misleading if it
reaches inference undetected.

Checks
------
1. Serial bus      : all 6 motors answer a ping.  Loose 3-pin connectors took
                     out motors 2 and 6 twice during recording; the runner's
                     failure mode is a hard crash mid-episode.
2. Torque release  : Torque_Enable can be written.  A gripper left clamped
                     under load latches a Feetech "Overload" fault that refuses
                     this write until the arm is power-cycled.
3. Calibration     : the file exists, covers all 6 motors, and matches a saved
                     reference.  The policy's actions are expressed in the
                     frame this file defines -- recalibrating silently
                     invalidates the model, with no error anywhere.
4. Cameras         : both open at the requested resolution, produce non-black
                     frames, and are genuinely different devices (a stale
                     process holding one camera, or the same index opened
                     twice, both look fine until the policy sees nonsense).
                     Preview PNGs are written so the wrist/top mapping can be
                     confirmed by eye -- the server field names do not match
                     this dataset (see so101_runner.py).
5. Pose in-range   : the arm's current pose lies inside the joint range spanned
                     by the demonstrations.  Starting outside it puts the
                     policy out of distribution from step 0.
6. Policy server   : optional TCP reachability probe.

Nothing here moves the arm.  Torque is only ever disabled, never enabled --
enabling torque makes a Feetech servo jump to its last Goal_Position, which
may differ from where the arm is now.

Usage
-----
    python real_robot/preflight_check.py
    python real_robot/preflight_check.py --server_host vwong-nv1.ece.ubc.ca --server_port 5555
    python real_robot/preflight_check.py --save_reference     # first run only
"""

import argparse
import json
import os
import socket
import sys

import numpy as np
from PIL import Image

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

MOTOR_ORDER = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_SYMBOL = {PASS: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status: str, name: str, detail: str = ""):
        self.rows.append((status, name, detail))
        print(f"  {_SYMBOL[status]}  {name}" + (f"  --  {detail}" if detail else ""))

    @property
    def failed(self) -> bool:
        return any(s == FAIL for s, _, _ in self.rows)

    @property
    def warned(self) -> bool:
        return any(s == WARN for s, _, _ in self.rows)


# ========================================================================== #
#  1-3.  Motors, torque, calibration                                          #
# ========================================================================== #

def check_calibration_file(args, rep: Report) -> dict | None:
    cal_dir = HF_LEROBOT_CALIBRATION / ROBOTS / "so_follower"
    cal_path = cal_dir / f"{args.robot_id}.json"

    if not cal_path.is_file():
        rep.add(FAIL, "calibration file", f"not found at {cal_path}")
        return None

    with open(cal_path) as f:
        cal = json.load(f)

    missing = [m for m in MOTOR_ORDER if m not in cal]
    if missing:
        rep.add(FAIL, "calibration coverage", f"missing motors: {missing}")
        return cal

    rep.add(PASS, "calibration file", str(cal_path))

    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"calibration_reference_{args.robot_id}.json")

    if args.save_reference:
        with open(ref_path, "w") as f:
            json.dump(cal, f, indent=2, sort_keys=True)
        rep.add(PASS, "calibration reference", f"written to {ref_path}")
        return cal

    if not os.path.isfile(ref_path):
        rep.add(WARN, "calibration reference",
                "no reference stored -- cannot detect drift since recording. "
                "Re-run with --save_reference IF the arm is currently calibrated "
                "exactly as it was during data collection.")
        return cal

    with open(ref_path) as f:
        ref = json.load(f)

    drift = []
    for m in MOTOR_ORDER:
        for field in ("homing_offset", "range_min", "range_max", "drive_mode"):
            now, then = cal[m].get(field), ref.get(m, {}).get(field)
            if now != then:
                drift.append(f"{m}.{field}: {then} -> {now}")
    if drift:
        rep.add(FAIL, "calibration drift",
                "calibration CHANGED since reference; the policy's action frame "
                "no longer matches its training data. " + "; ".join(drift[:4])
                + (" ..." if len(drift) > 4 else ""))
    else:
        rep.add(PASS, "calibration unchanged", "matches stored reference")
    return cal


def check_motors_and_torque(robot: SO101Follower, rep: Report):
    # connect() already performs the handshake that pings every expected motor,
    # so reaching this point means all 6 answered.
    rep.add(PASS, "serial bus", f"all {len(MOTOR_ORDER)} motors answered on "
                                f"{robot.config.port}")

    # Torque disable only -- never enable (see module docstring).
    try:
        robot.bus.disable_torque()
        rep.add(PASS, "torque release", "Torque_Enable written on all motors")
    except Exception as e:
        rep.add(FAIL, "torque release",
                f"{type(e).__name__}: {e} -- a latched Overload fault (commonly "
                f"a gripper left clamped) needs a power-cycle of the controller")


# ========================================================================== #
#  4.  Cameras                                                                #
# ========================================================================== #

def check_cameras(args, rep: Report) -> dict[str, np.ndarray]:
    frames = {}
    for label, index, expected in (
        ("main (-> jpeg_b64, must be WRIST)", args.cam_main_index, "wrist"),
        ("aux  (-> jpeg_b64_wrist, must be TOP)", args.cam_aux_index, "top"),
    ):
        cam = None
        try:
            cfg = OpenCVCameraConfig(index_or_path=index, width=args.width,
                                     height=args.height, fps=args.fps, warmup_s=1)
            cam = OpenCVCamera(cfg)
            cam.connect(warmup=True)
            frame = cam.read()
        except Exception as e:
            rep.add(FAIL, f"camera {index} {label}",
                    f"{type(e).__name__}: {e} -- another process may hold it "
                    f"(check for stale python processes)")
            continue
        finally:
            if cam is not None and cam.is_connected:
                cam.disconnect()

        frames[expected] = frame
        h, w = frame.shape[:2]
        if (w, h) != (args.width, args.height):
            rep.add(WARN, f"camera {index} resolution",
                    f"got {w}x{h}, requested {args.width}x{args.height}")
        else:
            rep.add(PASS, f"camera {index} {label}", f"{w}x{h}")

        mean = float(frame.mean())
        if mean < 5.0:
            rep.add(FAIL, f"camera {index} image",
                    f"mean pixel {mean:.1f} -- effectively black (lens cap? "
                    f"no light?)")
        elif mean > 250.0:
            rep.add(WARN, f"camera {index} image",
                    f"mean pixel {mean:.1f} -- saturated/washed out")
        else:
            rep.add(PASS, f"camera {index} image", f"mean pixel {mean:.1f}")

    if len(frames) == 2:
        a, b = frames["wrist"], frames["top"]
        if a.shape == b.shape:
            diff = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
            if diff < 1.0:
                rep.add(FAIL, "camera distinctness",
                        f"the two streams are nearly identical (mean abs diff "
                        f"{diff:.2f}) -- likely the same device opened twice")
            else:
                rep.add(PASS, "camera distinctness", f"mean abs diff {diff:.1f}")

        out_dir = args.preview_dir
        os.makedirs(out_dir, exist_ok=True)
        for name, img in frames.items():
            p = os.path.join(out_dir, f"preflight_{name}.png")
            Image.fromarray(img).save(p)
        rep.add(WARN, "camera identity",
                f"previews written to {out_dir}/preflight_wrist.png and "
                f"preflight_top.png -- CONFIRM BY EYE that they match their "
                f"names; swapping them feeds the policy transposed views")

    return frames


# ========================================================================== #
#  5.  Pose within the demonstration distribution                             #
# ========================================================================== #

def check_pose_in_range(robot: SO101Follower, args, rep: Report):
    if not os.path.isfile(args.demos):
        rep.add(WARN, "pose vs demos",
                f"demos file not found at {args.demos}; skipping "
                f"(pass --demos to enable)")
        return

    try:
        import h5py
    except ImportError:
        rep.add(WARN, "pose vs demos", "h5py not installed; skipping")
        return

    lo = np.full(len(MOTOR_ORDER), np.inf, dtype=np.float32)
    hi = np.full(len(MOTOR_ORDER), -np.inf, dtype=np.float32)
    with h5py.File(args.demos, "r") as f:
        for k in f:
            if "actions" not in f[k]:
                continue
            a = np.asarray(f[k]["actions"], dtype=np.float32)
            lo = np.minimum(lo, a.min(axis=0))
            hi = np.maximum(hi, a.max(axis=0))

    obs = robot.get_observation()
    pose = np.array([obs[f"{m}.pos"] for m in MOTOR_ORDER], dtype=np.float32)

    outside = []
    for i, m in enumerate(MOTOR_ORDER):
        # Allow a small margin: the demos' extremes are not hard limits.
        span = hi[i] - lo[i]
        margin = 0.1 * span if span > 0 else 1.0
        if pose[i] < lo[i] - margin or pose[i] > hi[i] + margin:
            outside.append(f"{m}={pose[i]:.1f} (demos {lo[i]:.1f}..{hi[i]:.1f})")

    if outside:
        rep.add(WARN, "pose vs demos",
                "arm is outside the demonstrated range on: " + "; ".join(outside)
                + " -- the policy will be out of distribution from step 0; "
                  "move the arm to a typical episode start pose")
    else:
        rep.add(PASS, "pose vs demos", "current pose lies inside the demo range")


# ========================================================================== #
#  6.  Policy server reachability                                             #
# ========================================================================== #

def check_server(args, rep: Report):
    if not args.server_host:
        return
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(args.server_timeout)
    try:
        s.connect((args.server_host, args.server_port))
        rep.add(PASS, "policy server",
                f"{args.server_host}:{args.server_port} accepting connections")
    except ConnectionRefusedError:
        rep.add(FAIL, "policy server",
                f"{args.server_host}:{args.server_port} refused -- host is "
                f"reachable but nothing is listening. Is vla_server.py started, "
                f"and bound to --host 0.0.0.0 rather than 127.0.0.1?")
    except socket.timeout:
        rep.add(FAIL, "policy server",
                f"{args.server_host}:{args.server_port} timed out -- firewall "
                f"is dropping packets on this path")
    except Exception as e:
        rep.add(FAIL, "policy server", f"{type(e).__name__}: {e}")
    finally:
        s.close()


# ========================================================================== #
#  Main                                                                       #
# ========================================================================== #

def main():
    args = parse_args()
    rep = Report()

    print("\nSO-101 pre-inference check")
    print("=" * 64)

    print("\n[1/6] calibration")
    check_calibration_file(args, rep)

    print("\n[2/6] serial bus and torque")
    robot = SO101Follower(SO101FollowerConfig(port=args.robot_port,
                                              id=args.robot_id,
                                              cameras={}))
    connected = False
    try:
        robot.connect(calibrate=False)
        connected = True
        check_motors_and_torque(robot, rep)
    except Exception as e:
        rep.add(FAIL, "serial bus", f"{type(e).__name__}: {e}")

    print("\n[3/6] pose vs demonstrations")
    if connected:
        try:
            check_pose_in_range(robot, args, rep)
        except Exception as e:
            rep.add(WARN, "pose vs demos", f"{type(e).__name__}: {e}")
    else:
        rep.add(WARN, "pose vs demos", "skipped (no serial connection)")

    if connected:
        robot.disconnect()

    print("\n[4/6] cameras")
    check_cameras(args, rep)

    print("\n[5/6] policy server")
    if args.server_host:
        check_server(args, rep)
    else:
        rep.add(WARN, "policy server", "skipped (pass --server_host to check)")

    print("\n[6/6] summary")
    print("=" * 64)
    n_fail = sum(1 for s, _, _ in rep.rows if s == FAIL)
    n_warn = sum(1 for s, _, _ in rep.rows if s == WARN)
    if rep.failed:
        print(f"  {n_fail} FAILED, {n_warn} warnings -- do NOT start inference.")
        sys.exit(1)
    if rep.warned:
        print(f"  0 failed, {n_warn} warnings -- review them, then proceed.")
        sys.exit(0)
    print("  All checks passed.")
    sys.exit(0)


def parse_args():
    p = argparse.ArgumentParser(description="Pre-inference health check for the SO-101")
    p.add_argument("--robot_port", type=str, default="COM4")
    p.add_argument("--robot_id", type=str, default="my_awesome_follower_arm")
    p.add_argument("--cam_main_index", type=int, default=0)
    p.add_argument("--cam_aux_index", type=int, default=1)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--demos", type=str,
                   default=r"C:\Users\sinyu104.stu\lerobot\data\pick_blue_cube\demos.hdf5",
                   help="HDF5 used to derive the demonstrated joint ranges")
    p.add_argument("--preview_dir", type=str, default="outputs/preflight")
    p.add_argument("--save_reference", action="store_true",
                   help="Store the CURRENT calibration as the reference to "
                        "compare against on later runs. Only use this when the "
                        "arm is calibrated exactly as it was during recording.")
    p.add_argument("--server_host", type=str, default=None)
    p.add_argument("--server_port", type=int, default=5555)
    p.add_argument("--server_timeout", type=float, default=5.0)
    return p.parse_args()


if __name__ == "__main__":
    main()
