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
"""

import argparse
import json
import os
import sys
import time

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
from isaac_sim.collector_runner import VLAClient


# ============================================================================ #
#  CLI                                                                          #
# ============================================================================ #

def parse_args():
    p = argparse.ArgumentParser(
        description="VLA inference runner — evaluates a trained policy on Isaac Sim tasks"
    )
    p.add_argument("--tasks", nargs="+", default=["pick_red_cube_to_tray"],
                   help="Task name(s) to evaluate, or 'all' for all registered tasks")
    p.add_argument("--num_episodes",      type=int,   default=50)
    p.add_argument("--episode_length",    type=int,   default=300)
    p.add_argument("--chunk_size",        type=int,   default=10,
                   help="Action chunk size (must match --chunk_size used on the server)")
    p.add_argument("--timeout_s",         type=float, default=10.0)
    p.add_argument("--jpeg_quality",      type=int,   default=85)
    p.add_argument("--data_dir",          type=str,   default="data",
                   help="Root data directory containing task.json files")
    p.add_argument("--headless",          action="store_true", default=False)
    p.add_argument("--compare_scripted",  action="store_true", default=False,
                   help="Log scripted action, VLA action, and L2 difference at every step")
    p.add_argument("--fallback_scripted", action="store_true", default=False,
                   help="Fall back to scripted action when L1 vs scripted exceeds 0.1")
    p.add_argument("--log_actions",       action="store_true", default=False,
                   help="Record VLA actions per episode and print a cross-episode consistency report")
    p.add_argument("--dagger",            action="store_true", default=False,
                   help="DAgger mode: apply VLA actions but save (obs, scripted_action) to HDF5")
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


def evaluate_task(task_name: str, args) -> dict:
    cfg = load_task_config(task_name, args.data_dir)

    if args.dagger:
        from isaac_sim.collector_runner import HDF5Writer
        dagger_path = os.path.join(args.data_dir, task_name, "dagger.hdf5")
        dagger_writer = HDF5Writer(dagger_path, append=True)
    else:
        dagger_writer = None

    client = VLAClient(
        instruction  = cfg["instruction"],
        unnorm_key   = cfg.get("unnorm_key", "franka_isaac"),
        timeout_s    = args.timeout_s,
        jpeg_quality = args.jpeg_quality,
    )

    episode_length = args.episode_length
    if task_name == "sort_two_cubes":
        episode_length = max(episode_length, 240)

    scene = get_scene(task_name)

    n_success      = 0
    total_latency  = 0.0
    total_requests = 0
    t0             = time.time()

    print(f"\n[VLARunner] Task          : {task_name}")
    print(f"[VLARunner] Instruction   : {cfg['instruction']}")
    print(f"[VLARunner] Episodes      : {args.num_episodes}  "
          f"Length: {episode_length}  chunk_size: {args.chunk_size}")
    print(f"[VLARunner] Compare script: {args.compare_scripted}\n")

    if args.compare_scripted:
        print(f"  {'step':>5}  {'scripted_action':^55}  {'vla_action':^55}  {'L1':>8}")
        print(f"  {'-'*5}  {'-'*55}  {'-'*55}  {'-'*8}")

    for ep in range(args.num_episodes):
        scene.reset(randomise=True)
        client.new_episode()   # stateful servers reset per-episode history on this
        obs_t   = scene.get_obs()
        success = False
        step    = 0

        ep_l1_list = []
        obs_buf    = []
        act_buf    = []

        while step < episode_length and not success:
            # get_obs() returns {cam_id: array}; use the lowest cam id for VLA
            obs_img = obs_t[min(obs_t.keys())] if isinstance(obs_t, dict) else obs_t

            # Request a full action chunk from the server
            import omni.log
            omni.log.info(f"[VLARunner] send obs step {step}", channel="VLARunner")
            t_req        = time.perf_counter()
            action_chunk = client.request_action(obs_img)
            total_latency  += (time.perf_counter() - t_req) * 1000
            total_requests += 1
            omni.log.info(f"[VLARunner] recv obs step {step}", channel="VLARunner")

            # action_chunk is (chunk_size, 7) for pi0fast, or (7,) for legacy single-action servers
            if action_chunk.ndim == 1:
                action_chunk = action_chunk[np.newaxis, :]   # normalise to (1, 7)

            # Execute each action in the chunk sequentially
            for k in range(len(action_chunk)):
                if step >= episode_length:
                    break

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

                scene.apply_action(applied_action)
                scene.step()
                step += 1
                obs_t = scene.get_obs()
                if scene.is_success():
                    success = True
                    print(f"  ep {ep+1:4d}  SUCCESS at step {step}")
                    break

        if success:
            n_success += 1

        if args.dagger and obs_buf and not success:
            dagger_writer.write(obs_buf, act_buf, metadata={
                "success":     int(success),
                "steps":       len(obs_buf),
                "instruction": cfg["instruction"],
            })

        elapsed = time.time() - t0
        eta     = elapsed / (ep + 1) * (args.num_episodes - ep - 1)
        avg_lat = total_latency / max(total_requests, 1)
        avg_l1  = float(np.mean(ep_l1_list)) if ep_l1_list else float("nan")

        if args.compare_scripted:
            print(f"  ep {ep+1:4d}/{args.num_episodes}  "
                  f"success={n_success}/{ep+1} ({100*n_success/(ep+1):.1f}%)  "
                  f"avg_vla={avg_lat:.0f}ms  avg_L1={avg_l1:.4f}  ETA {eta:.0f}s\n")
        else:
            print(f"  ep {ep+1:4d}/{args.num_episodes}  "
                  f"success={n_success}/{ep+1} ({100*n_success/(ep+1):.1f}%)  "
                  f"avg_vla={avg_lat:.0f}ms  ETA {eta:.0f}s")

    client.close()
    if dagger_writer:
        dagger_writer.close()
    rate = 100.0 * n_success / max(args.num_episodes, 1)
    print(f"\n[VLARunner] {task_name}: {n_success}/{args.num_episodes} ({rate:.1f}%)")
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
