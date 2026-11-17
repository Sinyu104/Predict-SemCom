"""
isaac_sim/vla_runner.py  —  Shared VLA inference / evaluation entry point.

Runs the trained VLA policy on one or more tasks inside Isaac Sim.
Each task is evaluated sequentially (Isaac Sim can only have one World active).

The server returns a chunk of `chunk_size` actions per request.  The runner
executes the entire chunk before issuing the next request, which reduces
communication overhead by a factor of chunk_size.

Usage
-----
  # Single task
  <isaac_sim_root>\\python.bat isaac_sim\\vla_runner.py --tasks pick_red_cube_to_tray

  # Multiple tasks
  <isaac_sim_root>\\python.bat isaac_sim\\vla_runner.py ^
      --tasks pick_red_cube_to_tray pick_blue_cube_to_tray stack_red_on_blue

  # All registered tasks
  <isaac_sim_root>\\python.bat isaac_sim\\vla_runner.py --tasks all

  # Compare VLA actions vs scripted actions
  <isaac_sim_root>\\python.bat isaac_sim\\vla_runner.py --tasks pick_red_cube_to_tray --compare_scripted

  # Evaluate under injected disturbance (actuator noise and/or cube teleports).
  # Each request sent to the server after a disturbance event carries
  # "disturbed": true, so a stateful server (e.g. semcom_server.py) can flag
  # or log the affected observations.
  <isaac_sim_root>\\python.bat isaac_sim\\vla_runner.py --tasks pick_red_cube_to_tray ^
      --interference_action --interference_pose --interference_prob 0.10
"""

import argparse
import json
import os
import random
import sys
import time

import re
import subprocess

# Quiet by default: re-run this script as a child process and show only results and errors.
# Pass --verbose to see everything Kit prints.
# Kit's startup banner, extension list and device tables are written before any Python
# runs, so they can only be removed by filtering the child's output.
_QUIET_KEEP = re.compile(
    r"^\s*ep\s+\d+|\[VLARunner\]|\[VLAClient\]|\[HDF5Writer\]|^=+$|^Summary$"
    r"|^\s+pick_\w+\s+[\d.]+%|Shutting Down|\[py stderr\]|Traceback|\w+Error:|\w+Exception:")
_QUIET_PREFIX = re.compile(r"^\S+Z \[[\d,]+ms\] \[\w+\] \[[^\]]*\] (?:\[py (?:stderr|stdout)\]: )?")
_ANSI = re.compile(chr(27) + r"\[[0-9;]*m")


def _run_quiet_child() -> int:
    env = dict(os.environ, VLA_QUIET_CHILD="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")    # odd bytes in Kit's output must not crash us
    proc = subprocess.Popen([sys.executable] + sys.argv, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")
    try:
        for raw in proc.stdout:
            line = _ANSI.sub("", raw.rstrip("\r\n"))
            if _QUIET_KEEP.search(line):
                print(_QUIET_PREFIX.sub("", line), flush=True)
    except KeyboardInterrupt:
        pass                      # the child received the same Ctrl-C; wait for it to finish
    return proc.wait()


if "--verbose" not in sys.argv and not os.environ.get("VLA_QUIET_CHILD"):
    sys.exit(_run_quiet_child())
if "--verbose" in sys.argv:
    # Kit also understands --verbose, as "log everything" (over 100k lines for a short run).
    # Our own meaning is simply "don't filter", which was decided above, so hide it from Kit.
    sys.argv.remove("--verbose")

import h5py  # must be imported before SimulationApp to avoid DLL conflicts
import numpy as np

# ── SimulationApp must be created before any Isaac Sim imports ───────────── #
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--headless", action="store_true", default=False)
_pre_args, _ = _pre.parse_known_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless":      _pre_args.headless,
    "renderer":      "RayTracedLighting",
    "anti_aliasing": 0,
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tasks.registry import TASK_NAMES, get_scene
from isaac_sim.collector_runner import VLAClient, InterferenceInjector


# ============================================================================ #
#  CLI                                                                          #
# ============================================================================ #

def parse_args():
    p = argparse.ArgumentParser(
        description="VLA inference runner — evaluates a trained policy on Isaac Sim tasks"
    )
    p.add_argument("--tasks", nargs="+", default=["pick_red_cube_to_tray"],
                   help="Task name(s) to evaluate, or 'all' for all registered tasks")
    p.add_argument("--num_episodes",      type=int,   default=50,
                   help="Total episodes; with --start_episode K, episodes K..num_episodes run")
    p.add_argument("--start_episode",     type=int,   default=1,
                   help="1-based episode to start from (default 1). With --seed, the cube "
                        "layouts of episodes K.. are the ones a full seeded run would have; the "
                        "earlier layouts are skipped without simulating them.")
    p.add_argument("--episode_length",    type=int,   default=300)
    p.add_argument("--chunk_size",        type=int,   default=10,
                   help="Action chunk size (must match --chunk_size used on the server)")
    p.add_argument("--timeout_s",         type=float, default=10.0)
    p.add_argument("--jpeg_quality",      type=int,   default=85)
    p.add_argument("--topic_prefix",      type=str,   default="",
                   help="Topic namespace, e.g. /B -> /B/vla/request and /B/vla/response "
                        "(default: /vla/request, /vla/response)")
    p.add_argument("--data_dir",          type=str,   default="data",
                   help="Root data directory containing task.json files")
    p.add_argument("--headless",          action="store_true", default=False)
    p.add_argument("--verbose",           action="store_true", default=False,
                   help="Show everything Kit prints. By default only the episode results, "
                        "summary and errors are shown (the full log is still written to "
                        "Kit's log file).")
    p.add_argument("--quiet",             action="store_true", default=False,
                   help="Accepted for compatibility; quiet is already the default.")
    p.add_argument("--compare_scripted",  action="store_true", default=False,
                   help="Log scripted action, VLA action, and L2 difference at every step")
    p.add_argument("--fallback_scripted", action="store_true", default=False,
                   help="Fall back to scripted action when L1 vs scripted exceeds 0.1")
    p.add_argument("--log_actions",       action="store_true", default=False,
                   help="Record VLA actions per episode and print a cross-episode consistency report")
    p.add_argument("--dagger",            action="store_true", default=False,
                   help="DAgger mode: apply VLA actions but save (obs, scripted_action) to HDF5")
    p.add_argument("--dagger_file",       type=str,   default="dagger.hdf5",
                   help="File name (inside data/<task>/) that --dagger appends to")
    p.add_argument("--collect_file",      type=str,   default="disturbed.hdf5",
                   help="File name (inside data/<task>/) that --collect_rollouts appends to")
    p.add_argument("--collect_rollouts", "--collect_disturbed", dest="collect_disturbed",
                   action="store_true", default=False,
                   help="Save EVERY episode (success and failure): per step the observation and "
                        "the action actually applied (the policy's), plus a per-frame 'disturbed' "
                        "marker (all False without interference). Unlike --dagger, which only "
                        "saves failures with the scripted label. (--collect_disturbed is the "
                        "old name of this option.)")
    p.add_argument("--scripted",          action="store_true", default=False,
                   help="Use the scene's scripted expert instead of querying the VLA/SemCom "
                        "server — no network/server dependency. Combine with --collect_disturbed "
                        "to collect disturbed data without needing the server up.")
    p.add_argument("--interference_action", action="store_true", default=False,
                   help="Randomly perturb the applied action (actuator noise)")
    p.add_argument("--interference_pose",   action="store_true", default=False,
                   help="Randomly teleport cubes mid-episode (sudden visual change)")
    p.add_argument("--interference_prob",   type=float, default=0.10)
    p.add_argument("--action_noise_scale",  type=float, default=0.30)
    p.add_argument("--pose_delta_max",      type=float, default=0.15)
    p.add_argument("--pose_flush_subframes", type=int, default=32,
                   help="Render subframes for the step right after a cube teleport, so the old "
                        "cube position doesn't linger as a faded ghost (normal steps use 4)")
    p.add_argument("--interference_min_gap", type=int,  default=5,
                   help="Minimum steps between two interference events of the same kind "
                        "(action noise / cube teleport). Default 5; 0 = no limit.")
    p.add_argument("--seed", type=int, default=None,
                   help="Fix the random seed. Seeds the cube placement (re-seeded at the start "
                        "of every task) and the interference injector, so runs with the same "
                        "seed see the same sequence of cube layouts. Default: random.")
    p.add_argument("--camera", nargs="+", type=int, default=[1],
                   help="Camera IDs to activate and save (1=fixed, 2=second fixed, "
                        "3=wrist). E.g. --camera 1 2. All active cameras are sent in the "
                        "request as images_b64 {cam_id: JPEG}; one camera uses jpeg_b64.")
    return p.parse_args()


# ============================================================================ #
#  Per-task evaluation loop                                                     #
# ============================================================================ #

def load_task_config(task_name: str, data_dir: str) -> dict:
    path = os.path.join(data_dir, task_name, "task.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"task.json not found at '{path}'. "
            "Create the file or check --data_dir."
        )
    with open(path) as f:
        return json.load(f)


GRIPPER_HOLD_RADIUS = 0.08   # m — cube centre this close to the end effector counts as held


# During chunk 0 ONLY (the first chunk of each episode), the observation after each of these many
# actions (counted from the chunk's start) is sent to the server as its own small "between"
# message (client.send_between), tagged with the absolute step in the episode, so a stateful
# server can warm start. The server ignores them after chunk 0, so later chunks send none.
# Requests themselves stay one-observation-sized.
BETWEEN_STEPS = (5, 10, 15, 20, 25)


def _robot_state(scene) -> dict:
    """Robot state at this instant, world frame (metres): end-effector pose and joint angles."""
    ee_pos, ee_quat = scene.franka.end_effector.get_world_pose()
    return {
        "robot_ee_pos":    np.asarray(ee_pos,  dtype=np.float32),                      # (3,)
        "robot_ee_quat":   np.asarray(ee_quat, dtype=np.float32),                      # (4,) w,x,y,z
        "robot_joint_pos": np.asarray(scene.franka.get_joint_positions(), dtype=np.float32),  # (9,) 7 arm + 2 fingers
    }


def _pose_row(scene) -> list:
    """One pose row for the server: [ee_x, ee_y, ee_z, joint_1..joint_9] (12 floats), the same
    quantities and order as robot_ee_pos + robot_joint_pos in the collected datasets."""
    s = _robot_state(scene)
    return [*s["robot_ee_pos"].tolist(), *s["robot_joint_pos"].tolist()]


def _cubes_near_gripper(scene, radius: float = GRIPPER_HOLD_RADIUS) -> list:
    """Cubes within `radius` of the end effector (in the gripper or about to be grasped)."""
    ee_pos, _ = scene.franka.end_effector.get_world_pose()
    return [cube for cube in scene.cubes.values()
            if np.linalg.norm(cube.get_world_pose()[0] - ee_pos) < radius]


def evaluate_task(task_name: str, args) -> dict:
    cfg = load_task_config(task_name, args.data_dir)

    if args.dagger:
        from isaac_sim.collector_runner import HDF5Writer
        dagger_path = os.path.join(args.data_dir, task_name, args.dagger_file)
        dagger_writer = HDF5Writer(dagger_path, append=True)
    else:
        dagger_writer = None

    if args.collect_disturbed:
        from isaac_sim.collector_runner import HDF5Writer
        disturbed_path = os.path.join(args.data_dir, task_name, args.collect_file)
        disturbed_writer = HDF5Writer(disturbed_path, append=True)
    else:
        disturbed_writer = None

    if args.scripted:
        client = None
        print("[VLARunner] Scripted mode — scene.scripted_action(), no VLA/SemCom server.")
    else:
        client = VLAClient(
            instruction  = cfg["instruction"],
            unnorm_key   = cfg.get("unnorm_key", "franka_isaac"),
            timeout_s    = args.timeout_s,
            jpeg_quality = args.jpeg_quality,
            topic_prefix = args.topic_prefix,
        )

    episode_length = args.episode_length
    if task_name == "sort_two_cubes":
        episode_length = max(episode_length, 240)

    scene = get_scene(task_name, camera_ids=args.camera)

    injector = InterferenceInjector(
        action_prob        = args.interference_prob if args.interference_action else 0.0,
        pose_prob          = args.interference_prob if args.interference_pose   else 0.0,
        action_noise_scale = args.action_noise_scale,
        pose_delta_max     = args.pose_delta_max,
        seed               = args.seed or 0,
        min_gap            = args.interference_min_gap,
    )

    if args.seed is not None:
        # Cube positions come from np.random in the scene's reset(); seed right before the
        # episode loop so scene construction can't shift the sequence.
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"[VLARunner] Seed          : {args.seed}")

    first_ep = max(args.start_episode, 1) - 1
    if first_ep >= args.num_episodes:
        raise SystemExit(f"[VLARunner] --start_episode {args.start_episode} is past "
                         f"--num_episodes {args.num_episodes}: nothing to run.")
    if first_ep:
        if args.seed is not None:
            # The only randomness in a scene reset is the cube layout, drawn from np.random,
            # so drawing the skipped layouts here leaves the stream exactly where a full
            # seeded run would have it at episode K.
            for _ in range(first_ep):
                scene._sample_non_overlapping(len(scene.cubes))
            print(f"[VLARunner] Starting at episode {first_ep + 1}: skipped {first_ep} seeded "
                  f"cube layouts without simulating them.")
        else:
            print("[VLARunner] WARNING: --start_episode without --seed only changes the "
                  "numbering; cube layouts are random anyway.")

    n_success      = 0
    total_latency  = 0.0
    total_requests = 0
    t0             = time.time()

    print(f"\n[VLARunner] Task          : {task_name}")
    print(f"[VLARunner] Instruction   : {cfg['instruction']}")
    print(f"[VLARunner] Episodes      : {args.num_episodes}  "
          f"Length: {episode_length}  chunk_size: {args.chunk_size}")
    print(f"[VLARunner] Compare script: {args.compare_scripted}")
    print(f"[VLARunner] Interference  : action={args.interference_action}  "
          f"pose={args.interference_pose}  p={args.interference_prob}\n")

    if args.compare_scripted:
        print(f"  {'step':>5}  {'scripted_action':^55}  {'vla_action':^55}  {'L1':>8}")
        print(f"  {'-'*5}  {'-'*55}  {'-'*55}  {'-'*8}")

    for ep in range(first_ep, args.num_episodes):
        scene.reset(randomise=True)
        if client is not None:
            client.new_episode()   # stateful servers reset per-episode history on this
        injector.reset()
        obs_t            = scene.get_obs()
        success          = False
        step             = 0
        disturbed_pending = False   # set when interference fired since the last request
        chunk_start       = 0       # step at which the latest chunk started executing
        chunk_idx         = 0       # chunks received so far this episode (1 while executing chunk 0)
        # Pose rows measured since the previous request, one per control step, aligned with the
        # camera frame after that step. Starts with the pose of the first image (request 0).
        pose_rows         = [] if args.scripted else [_pose_row(scene)]

        ep_l1_list = []
        obs_buf    = []
        act_buf    = []
        cobs_buf   = []   # --collect_disturbed: every episode, real applied action
        cact_buf   = []
        cdist_buf  = []   # per-step disturbance marker, aligned with cobs_buf/cact_buf
        crobot_buf = []   # per-step robot state (ee pose + joints), aligned with cobs_buf/cact_buf

        while step < episode_length and not success:
            if args.scripted:
                # No network round trip — one fresh scripted action per "chunk" of 1.
                action_chunk = np.zeros((1, 7), dtype=np.float32)   # placeholder, recomputed below
            else:
                # get_obs() returns {cam_id: array}; the client sends every active camera
                obs_img = obs_t

                # Request a full action chunk from the server
                import omni.log
                omni.log.info(f"[VLARunner] send obs step {step}", channel="VLARunner")
                t_req        = time.perf_counter()
                action_chunk = client.request_action(obs_img, disturbed=disturbed_pending,
                                                     pose=pose_rows)
                disturbed_pending = False   # each request reports the interval since the previous one
                total_latency  += (time.perf_counter() - t_req) * 1000
                total_requests += 1
                omni.log.info(f"[VLARunner] recv obs step {step}", channel="VLARunner")

                if action_chunk is None:
                    # Server error: no action — hold the arm still (no apply_action) and
                    # advance the sim one step so the loop still terminates at episode_length.
                    print(f"  ep {ep+1:4d}  step {step}: no action from server — holding position")
                    scene.step()
                    step += 1
                    obs_t = scene.get_obs()
                    pose_rows.append(_pose_row(scene))   # held steps still count as steps
                    continue

                chunk_start = step   # between frames are counted from here
                chunk_idx  += 1
                pose_rows   = []     # the next request carries only the steps executed after this one

                # action_chunk is (chunk_size, 7) for pi0fast, or (7,) for legacy single-action servers
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk[np.newaxis, :]   # normalise to (1, 7)

            # Execute each action in the chunk sequentially
            for k in range(len(action_chunk)):
                if step >= episode_length:
                    break

                if args.scripted:
                    applied_action = scene.scripted_action()
                else:
                    vla_action = action_chunk[k].copy()
                    vla_action[3:6] = 0.0  # zero rotation dims; scripted policy uses position+gripper only

                    if args.compare_scripted or args.fallback_scripted or args.dagger:
                        scripted_action = scene.scripted_action()
                        l1 = float(np.sum(np.abs(vla_action - scripted_action)))
                        use_script = args.fallback_scripted and l1 > 0.15
                        applied_action = scripted_action if use_script else vla_action
                        if use_script or args.compare_scripted:
                            import omni.log
                            vla_str    = "[" + ",".join(f"{v:+.4f}" for v in vla_action) + "]"
                            script_str = "[" + ",".join(f"{v:+.4f}" for v in scripted_action) + "]"
                            omni.log.warn(
                                f"[VLARunner] ACTION  ep={ep+1} step={step}"
                                f"  L1={l1:.4f}  script={script_str}  vla={vla_str}",
                                channel="VLARunner")
                        if args.compare_scripted:
                            ep_l1_list.append(l1)
                        if args.dagger:
                            obs_buf.append(obs_t)
                            act_buf.append(scripted_action)
                    else:
                        applied_action = vla_action

                action_disturbed_now = False
                if args.interference_action:
                    prev_events    = injector.action_events
                    applied_action = injector.maybe_perturb_action(applied_action)
                    if injector.action_events != prev_events:
                        disturbed_pending    = True
                        action_disturbed_now = True

                if args.collect_disturbed:
                    cobs_buf.append(obs_t)
                    cact_buf.append(applied_action)
                    crobot_buf.append(_robot_state(scene))   # state when the observation was taken

                scene.apply_action(applied_action)

                pose_disturbed_now = False
                if args.interference_pose and hasattr(scene, "cubes"):
                    # Single shared roll: if it fires, all cubes teleport together,
                    # except any cube in (or about to be in) the gripper — teleporting a
                    # cube physics is already holding via the gripper destabilises
                    # the articulation. Held = physically next to the end effector;
                    # scene.phase/grasp_target only follow the scripted expert, which
                    # isn't running when the VLA drives the arm.
                    held_cubes = _cubes_near_gripper(scene)
                    x_range   = getattr(scene, "CUBE_X_RANGE", None)
                    y_range   = getattr(scene, "CUBE_Y_RANGE", None)
                    min_dist  = getattr(scene, "MIN_CUBE_DIST", 0.12)
                    all_paths = [cube.prim_path for cube in scene.cubes.values()]
                    movable_paths = [cube.prim_path for cube in scene.cubes.values()
                                     if cube not in held_cubes]
                    if injector.maybe_teleport_cubes(
                        movable_paths, all_paths, x_range, y_range, min_dist
                    ):
                        disturbed_pending  = True
                        pose_disturbed_now = True

                if args.collect_disturbed:
                    cdist_buf.append(bool(action_disturbed_now or pose_disturbed_now))

                scene.step(rt_subframes=args.pose_flush_subframes if pose_disturbed_now else 4)
                step += 1
                obs_t = scene.get_obs()
                if not args.scripted:
                    row = _pose_row(scene)   # read at the same instant as obs_t
                    pose_rows.append(row)
                    if chunk_idx == 1 and (step - chunk_start) in BETWEEN_STEPS:
                        client.send_between(obs_t, step, pose=[row])
                if scene.is_success():
                    success = True
                    print(f"  ep {ep+1:4d}  SUCCESS at step {step}")
                    break

        if success:
            n_success += 1

        if args.dagger and obs_buf and not success:
            dagger_writer.write(obs_buf, act_buf, metadata={
                "success":         int(success),
                "steps":           len(obs_buf),
                "instruction":     cfg["instruction"],
                "disturbed":       int(args.interference_action or args.interference_pose),
                "n_action_events": injector.action_events,
                "n_pose_events":   injector.pose_events,
            })

        if args.collect_disturbed and cobs_buf:
            # Every episode, success or failure — unlike --dagger this is not filtered.
            disturbed_writer.write(cobs_buf, cact_buf, metadata={
                "success":         int(success),
                "steps":           len(cobs_buf),
                "instruction":     cfg["instruction"],
                "disturbed":       int(args.interference_action or args.interference_pose),
                "n_action_events": injector.action_events,
                "n_pose_events":   injector.pose_events,
            }, disturbed=cdist_buf,
               extras={k: [r[k] for r in crobot_buf] for k in crobot_buf[0]})

        elapsed = time.time() - t0
        done    = ep - first_ep + 1                  # episodes run in this session
        eta     = elapsed / done * (args.num_episodes - ep - 1)
        avg_lat = total_latency / max(total_requests, 1)
        avg_l1  = float(np.mean(ep_l1_list)) if ep_l1_list else float("nan")

        disturb_str = (
            f"  d_action={injector.action_events}  d_pose={injector.pose_events}"
            if (args.interference_action or args.interference_pose) else ""
        )

        if args.compare_scripted:
            print(f"  ep {ep+1:4d}/{args.num_episodes}  "
                  f"success={n_success}/{done} ({100*n_success/done:.1f}%)  "
                  f"avg_vla={avg_lat:.0f}ms  avg_L1={avg_l1:.4f}{disturb_str}  ETA {eta:.0f}s\n")
        else:
            print(f"  ep {ep+1:4d}/{args.num_episodes}  "
                  f"success={n_success}/{done} ({100*n_success/done:.1f}%)  "
                  f"avg_vla={avg_lat:.0f}ms{disturb_str}  ETA {eta:.0f}s")

    if client is not None:
        client.close()
    if dagger_writer:
        dagger_writer.close()
    if disturbed_writer:
        disturbed_writer.close()
    n_run = args.num_episodes - first_ep
    rate = 100.0 * n_success / max(n_run, 1)
    print(f"\n[VLARunner] {task_name}: {n_success}/{n_run} ({rate:.1f}%)")
    return {"task": task_name, "success_rate": rate, "n_success": n_success}


# ============================================================================ #
#  Main                                                                         #
# ============================================================================ #

def main():
    args = parse_args()

    tasks = TASK_NAMES if (len(args.tasks) == 1 and args.tasks[0] == "all") \
            else args.tasks

    for name in tasks:
        if name not in TASK_NAMES:
            print(f"[VLARunner] Warning: unknown task '{name}', skipping.")

    results = []
    for name in tasks:
        if name in TASK_NAMES:
            results.append(evaluate_task(name, args))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r['task']:45s}  {r['success_rate']:5.1f}%")

    simulation_app.close()


if __name__ == "__main__":
    main()
