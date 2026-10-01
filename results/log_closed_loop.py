"""Append one closed-loop (robot, in-sim) evaluation run to results/closed_loop.csv.

Tidy, append-only, one row per RUN (see the "Simulation Data & MATLAB Plotting Convention"
in CLAUDE.md).  Every row is stamped with the git sha + an ISO timestamp so a figure can be
regenerated and traced later.  Plot in MATLAB by reading this CSV; never hand-edit it.

New experiment axes become new COLUMNS only if they are genuinely new knobs (keep them stable
once added); new runs are new ROWS.  For the semantic pipeline set --mode/--semantic/--snr_db/
--channel; for a raw-policy control use --mode bypass (snr/channel left blank).

Example (the run this scaffold was created for):
  python results/log_closed_loop.py \
      --task pick_purple_cube_to_tray --policy openvla \
      --ckpt outputs/openvla_dagger_v3/checkpoint_epoch24 \
      --mode bypass --semantic none \
      --n_episodes 30 --episode_length 300 --chunk_size 10 \
      --intf_action false --intf_pose false --intf_p 0.1 \
      --n_success 11 --avg_vla_ms 768 --sim_seconds 8221.9 \
      --success_episodes 7:238,11:230,13:242,15:250,16:254,17:229,18:238,21:208,22:245,23:233,24:193
"""
import argparse, csv, os, subprocess, sys
from datetime import datetime

FIELDS = [
    "timestamp", "git_sha", "run_id",
    "policy", "policy_ckpt", "mode", "semantic", "snr_db", "channel",
    "task", "instruction",
    "n_episodes", "episode_length", "chunk_size",
    "intf_action", "intf_pose", "intf_p", "seed",
    "n_success", "success_rate", "avg_vla_ms", "sim_seconds",
    "success_episodes", "notes",
]


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description="Append a closed-loop eval run to results/closed_loop.csv")
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "closed_loop.csv"))
    # condition axes
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy", default="openvla")
    ap.add_argument("--ckpt", default="", help="policy checkpoint path")
    ap.add_argument("--mode", default="bypass",
                    help="semcom_server mode: bypass / vae / predict / nochannel / full")
    ap.add_argument("--semantic", default="none",
                    help="semantic model used, e.g. none / conv_wz_b1e-4 / legacy")
    ap.add_argument("--snr_db", default="", help="channel SNR in dB (blank for bypass)")
    ap.add_argument("--channel", default="", help="awgn / rayleigh / cdl (blank for bypass)")
    ap.add_argument("--instruction", default="")
    # run settings
    ap.add_argument("--n_episodes", type=int, required=True)
    ap.add_argument("--episode_length", type=int, default=300)
    ap.add_argument("--chunk_size", type=int, default=10)
    ap.add_argument("--intf_action", default="false")
    ap.add_argument("--intf_pose", default="false")
    ap.add_argument("--intf_p", default="")
    ap.add_argument("--seed", default="", help="RNG seed if known")
    # outcomes
    ap.add_argument("--n_success", type=int, required=True)
    ap.add_argument("--avg_vla_ms", default="", help="mean VLA latency per step (ms)")
    ap.add_argument("--sim_seconds", default="", help="total wall-clock of the sim run")
    ap.add_argument("--success_episodes", default="",
                    help="comma list of ep[:step] that succeeded, e.g. 7:238,11:230")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()

    ts = datetime.now().isoformat(timespec="seconds")
    sha = _git_sha()
    rate = a.n_success / a.n_episodes if a.n_episodes else float("nan")
    run_id = f"{ts.replace(':', '').replace('-', '')}_{a.policy}_{a.mode}_{a.task}"

    row = {
        "timestamp": ts, "git_sha": sha, "run_id": run_id,
        "policy": a.policy, "policy_ckpt": a.ckpt, "mode": a.mode, "semantic": a.semantic,
        "snr_db": a.snr_db, "channel": a.channel,
        "task": a.task, "instruction": a.instruction,
        "n_episodes": a.n_episodes, "episode_length": a.episode_length, "chunk_size": a.chunk_size,
        "intf_action": a.intf_action, "intf_pose": a.intf_pose, "intf_p": a.intf_p, "seed": a.seed,
        "n_success": a.n_success, "success_rate": f"{rate:.4f}",
        "avg_vla_ms": a.avg_vla_ms, "sim_seconds": a.sim_seconds,
        "success_episodes": a.success_episodes, "notes": a.notes,
    }

    new = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[log_closed_loop] appended run {run_id}")
    print(f"  {a.task}  {a.policy}/{a.mode}  {a.n_success}/{a.n_episodes} = {rate*100:.1f}%  -> {a.csv}")


if __name__ == "__main__":
    main()
