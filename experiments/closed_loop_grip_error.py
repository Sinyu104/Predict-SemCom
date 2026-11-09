"""Closed-loop gripper / action error of WZ-SC (k=4, CDL-C) per SNR, from the closed-loop dumps scored by score_arms_pi0fast.py
(experiments/closed_loop_dump_arms.py builds the npz: real = true images, vae = received images, demo = executed chunk).
a* = PI0Fast on the TRUE images of the state the robot was actually in.  Per request:
  grip_wrong    fraction of the 30 steps whose executed open/close command differs from a*'s (threshold as offline: midpoint of the
                demo gripper range in outputs/cmp_pose_unf2/nogate)
  GRASP split   requests whose a* chunk switches the gripper (the grasp/release moment)
  a_real_xyz    mean |a_exec - a*| over xyz and the chunk
Also: rescored received-image actions vs executed (sanity: should match), and failed vs successful episodes.
Appends rows to results/wz_closed_loop_action.csv.

    python experiments/closed_loop_grip_error.py
"""
import csv, datetime, os, subprocess
import numpy as np
B = "outputs/cmp_pose_unf2/closed_grip"; CSV = "results/wz_closed_loop_action.csv"
DUMP = {20: "old_nogate_res_20dB_k4_30ep_23", 16: "old_nogate_snr_16dB_k4_30ep_22", 12: "old_nogate_snr_12dB_k4_30ep_23",
        8: "old_nogate_snr_8dB_k4_30ep_17", 4: "old_nogate_snr_4dB_k4_30ep_11", 0: "old_nogate_snr_0dB_k4_30ep_11"}
ref = np.concatenate([np.load(f"outputs/cmp_pose_unf2/nogate/scores_part{i}_actions.npz")["demo"] for i in (0, 1)])
thr = (ref[..., 6].min() + ref[..., 6].max()) / 2

def success_eps(snr):
    """successful episode ids of the k=4 nogate_snr run at this SNR (closed_loop.csv: success_episodes 'ep:steps,...')"""
    best = None
    for r in csv.DictReader(open("results/exp1_main_closed_loop/data/closed_loop.csv")):
        if "nogate" in r["semantic"] and "k4" in r["semantic"] and r["snr_db"] not in ("", None) and float(r["snr_db"]) == snr \
                and int(r["n_success"]) == int(DUMP[snr].split("_")[-1]):
            best = r
    if best is None: return None
    return {int(x.split(":")[0]) for x in best["success_episodes"].replace('"', "").split(",") if ":" in x}

sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ts = datetime.datetime.now().isoformat(timespec="seconds"); rows = []
new = not os.path.isfile(CSV)
print(f"{'SNR':>4s} {'succ':>5s} {'n':>4s} | {'grip wrong':>10s} {'GRASP grip':>10s} {'nG':>4s} | {'xyz':>7s} {'GRASP xyz':>9s} | "
      f"{'rx-vs-exec':>10s} | grip wrong: fail eps / success eps")
for snr in (20, 16, 12, 8, 4, 0):
    s = np.load(f"{B}/{snr}dB_actions.npz"); m = np.load(f"{B}/{snr}dB.npz", allow_pickle=True)
    ex, star, rx, ep = s["demo"], s["real"], s["vae"], m["ep"]
    cl = star[..., 6] > thr; G = cl.all(1) != cl.any(1)
    gw = ((ex[..., 6] > thr) != cl).mean(1); xyz = np.abs(ex[..., :3] - star[..., :3]).mean((1, 2))
    rxx = np.abs(rx[..., :3] - ex[..., :3]).mean()
    se = success_eps(snr); fail = np.array([e not in se for e in ep]) if se is not None else np.zeros(len(ep), bool)
    nsucc = int(DUMP[snr].split("_")[-1])
    print(f"{snr:4d} {nsucc:3d}/30 {len(ep):4d} | {100 * gw.mean():9.1f}% {100 * gw[G].mean():9.1f}% {G.sum():4d} | {xyz.mean():7.4f} "
          f"{xyz[G].mean():9.4f} | {rxx:10.4f} | {100 * gw[fail].mean():5.1f}% / {100 * gw[~fail].mean():5.1f}%")
    for split, msk in (("ALL", np.ones(len(ep), bool)), ("GRASP", G), ("FAIL_EPS", fail), ("SUCCESS_EPS", ~fail)):
        if msk.sum() == 0: continue
        for met, v in (("grip_wrong", gw[msk]), ("a_real_xyz", xyz[msk])):
            rows.append(["wz", 4, snr, "CDL-C", split, met, f"{v.mean():.6f}", int(msk.sum()), 42, f"outputs/online_ft/{DUMP[snr]}", sha, ts])
            rows.append(["wz", 4, snr, "CDL-C", split, met + "_se", f"{v.std(ddof=1) / np.sqrt(len(v)):.6f}", int(msk.sum()), 42,
                         f"outputs/online_ft/{DUMP[snr]}", sha, ts])
with open(CSV, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow("arm,k,snr_db,channel,split,metric,value,n,seed,ckpt,git_sha,timestamp".split(","))
    w.writerows(rows)
print(f"appended {len(rows)} rows to {CSV}")
