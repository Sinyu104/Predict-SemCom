"""
isaac_sim/vla_runner.py  —  Shared VLA inference / evaluation entry point.

Runs the trained VLA policy on one or more tasks inside Isaac Sim.
Each task is evaluated sequentially (Isaac Sim can only have one World active).

Usage
-----
  # Single task
  <isaac_sim_root>\\python.bat isaac_sim\\vla_runner.py --tasks pick_red_cube_to_tray

  # Multiple tasks
  <isaac_sim_root>\\python.bat isaac_sim\\vla_runner.py ^
      --tasks pick_red_cube_to_tray pick_blue_cube_to_tray stack_red_on_blue

  # All registered tasks
  <isaac_sim_root>\\python.bat isaac_sim\\vla_runner.py --tasks all
"""

import argparse
import json
import os
import sys
import time

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
    p.add_argument("--num_episodes",   type=int,   default=50)
    p.add_argument("--episode_length", type=int,   default=120)
    p.add_argument("--timeout_s",      type=float, default=10.0)
    p.add_argument("--jpeg_quality",   type=int,   default=85)
    p.add_argument("--data_dir",       type=str,   default="data",
                   help="Root data directory containing task.json files")
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

    print(f"\n[VLARunner] Task: {task_name}")
    print(f"[VLARunner] Instruction: {cfg['instruction']}")
    print(f"[VLARunner] Episodes: {args.num_episodes}  Length: {episode_length}\n")

    for ep in range(args.num_episodes):
        scene.reset(randomise=True)
        obs_t   = scene.get_obs()
        success = False

        for step in range(episode_length):
            t_req  = time.perf_counter()
            action = client.request_action(obs_t)
            total_latency  += (time.perf_counter() - t_req) * 1000
            total_requests += 1

            scene.apply_action(action)
            scene.step()
            obs_t = scene.get_obs()

            if scene.is_success():
                success = True
                print(f"  ep {ep+1:4d}  SUCCESS at step {step+1}")
                break

        if success:
            n_success += 1

        elapsed = time.time() - t0
        eta     = elapsed / (ep + 1) * (args.num_episodes - ep - 1)
        avg_lat = total_latency / max(total_requests, 1)
        print(f"  ep {ep+1:4d}/{args.num_episodes}  "
              f"success={n_success}/{ep+1} ({100*n_success/(ep+1):.1f}%)  "
              f"avg_vla={avg_lat:.0f}ms  ETA {eta:.0f}s")

    client.close()
    rate = 100.0 * n_success / max(args.num_episodes, 1)
    print(f"\n[VLARunner] {task_name}: {n_success}/{args.num_episodes} "
          f"({rate:.1f}%)")
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
