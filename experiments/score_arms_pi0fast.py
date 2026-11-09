"""Stage B of the Pi0Fast sensitivity test (pi0fast env, needs a whole free 15 GB GPU).

Feeds each image arm from experiments/make_arms_pi0fast.py to the primary PI0Fast policy
(full-weights base + transition LoRA, two cameras, chunk 30) and compares action chunks.
Only the base camera changes between arms; the wrist camera is always the true frame,
and the proprio state is zeros (as in server/vla_server.py).

  real = true frame (reference), vae = VAE round trip of the true frame,
  pred = decoded +30 prediction (unit-noise start, blurry mean), predS = sigma-scaled start (sharp sample), copy = decoded last history frame (stale)

Metrics, per arm (mean over samples and the 30 steps; units of the denormalised action):
  |a-demo|  : xyz error vs the demo actions that follow the target frame
  |a-real|  : xyz difference vs the chunk the policy gives on the TRUE frame
  |a-vae|   : xyz difference vs the chunk on the VAE round trip (decode-artifact reference)
  grip      : fraction of steps whose gripper command disagrees with the demo (sign of a[:,6]-0.5... see below)

    CUDA_VISIBLE_DEVICES=1 /ubc/ece/home/vw/grads/syhuang/miniconda3/envs/pi0fast/bin/python \
        experiments/score_arms_pi0fast.py --in outputs/stage1_chunk30/pi0fast_arms.npz
"""
import argparse, io, json, os, pathlib, sys

import numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "server"))

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="outputs/stage1_chunk30/pi0fast_arms.npz")
ap.add_argument("--out", default="outputs/stage1_chunk30/pi0fast_scores.json")
ap.add_argument("--full_weights",
                default="outputs/pi0fast_5colors_scratch_dagger/checkpoint_epoch5/full_weights.pt")
ap.add_argument("--lora_path",
                default="outputs/pi0fast_5colors_lora_transition/checkpoint_epoch7/lora_weights.pt")
ap.add_argument("--model_dir", default="lerobot/pi0fast-base")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

from vla_server import Pi0FastServer   # noqa: E402

d = np.load(os.path.join(REPO, a.inp), allow_pickle=True)
n = len(d["real"]) if a.limit is None else min(a.limit, len(d["real"]))
NEW = any(k.startswith("img_") for k in d.files)          # decode_arms_wz_chunk30.py format
ARMS = (["real"] + [k[4:] for k in d.files if k.startswith("img_")]) if NEW else ["real", "vae", "pred", "predS", "copy"]
TWO = (not NEW) and "pred_w" in d.files      # make_arms_pi0fast.py --views 2: arms carry a predicted wrist image (*_w)
if TWO:
    ARMS.append("predS_basonly")             # predicted base + TRUE wrist (isolates the cost of predicting the wrist)
def WRIST(k, i):
    if NEW and "wimg_" + k in d.files: return d["wimg_" + k][i]        # 2-cam arms carry their own wrist image
    if TWO and k + "_w" in d.files: return d[k + "_w"][i]
    return d["wrist"][i]
IMG = (lambda k, i: d["real"][i] if k == "real" else d["img_" + k][i]) if NEW else \
      (lambda k, i: d[k.replace("_basonly", "")][i])

srv = Pi0FastServer(model_dir=a.model_dir,
                    lora_path=os.path.join(REPO, a.lora_path),
                    full_weights=os.path.join(REPO, a.full_weights),
                    chunk_size=30, state_dim=6, action_dim=7, device="cuda:0")

acts = {k: [] for k in ARMS}
for i in range(n):
    for k in ARMS:
        acts[k].append(np.array(srv.predict(IMG(k, i), str(d["task"][i]), WRIST(k, i)), dtype=np.float32))
    print(f"[score] {i + 1}/{n}", flush=True)
acts = {k: np.stack(v) for k, v in acts.items()}            # (n, 30, 7)
demo = d["demo_a"][:n].astype(np.float32)                    # (n, 30, 7)


def xyz(x, y):
    return np.abs(x[..., :3] - y[..., :3]).mean(axis=(1, 2))   # per-sample mean abs error


def grip(x, y):          # gripper command disagreement (binary, threshold at the midpoint)
    thr = 0.5 * (demo[..., 6].max() + demo[..., 6].min())
    return ((x[..., 6] > thr) != (y[..., 6] > thr)).mean(axis=1)


rows = {}
print(f"\n{n} samples; mean abs xyz error (and gripper disagreement) per arm")
print(f"{'arm':6s} {'|a-demo|':>9s} {'|a-real|':>9s} {'|a-vae|':>9s} {'grip vs demo':>13s} {'grip vs real':>13s}")
for k in ARMS:
    r = dict(demo=xyz(acts[k], demo), real=xyz(acts[k], acts["real"]), vae=xyz(acts[k], acts["vae"]),
             grip_demo=grip(acts[k], demo), grip_real=grip(acts[k], acts["real"]))
    rows[k] = {m: float(v.mean()) for m, v in r.items()}
    rows[k]["_per_sample"] = {m: v.tolist() for m, v in r.items()}
    print(f"{k:6s} {r['demo'].mean():9.4f} {r['real'].mean():9.4f} {r['vae'].mean():9.4f} "
          f"{r['grip_demo'].mean():13.3f} {r['grip_real'].mean():13.3f}")

if NEW:     # split by how much the prediction misses (n_obj = oracle-gate cells); calm = lowest third
    no = d["n_obj"][:n]; lo, hi = np.quantile(no, [1 / 3, 2 / 3])
    splits = {"ALL": np.ones(n, bool), "CALM": no <= lo, "DISTURBED": no >= hi}
    print(f"\nsplits by n_obj: CALM <= {lo:.0f} ({splits['CALM'].sum()}), DISTURBED >= {hi:.0f} ({splits['DISTURBED'].sum()})")
    print(f"{'arm':9s}" + "".join(f"{s_:>11s}" for s_ in ["ALL|demo", "CALM|demo", "DIST|demo", "ALL|real", "CALM|real", "DIST|real"]))
    for k in ARMS:
        ps = rows[k]["_per_sample"]
        print(f"{k:9s}" + "".join(f"{np.mean(np.array(ps[m_])[splits[s_]]):11.4f}"
                                  for m_ in ["demo", "real"] for s_ in ["ALL", "CALM", "DISTURBED"]))
    rows["_splits"] = {s_: v.tolist() for s_, v in splits.items()}
# paired: is pred closer to the real-frame policy output than the stale copy is?
dc = np.array(rows["copy"]["_per_sample"]["real"]) if not NEW else None
for k in (["pred", "predS"] + (["predS_basonly"] if TWO else []) if not NEW else []):
    dp = np.array(rows[k]["_per_sample"]["real"])
    print(f"{k:5s} closer to the real-frame action chunk than copy-last in {float(np.mean(dp < dc)):.0%} of samples")
json.dump({"n": n, "ckpt": str(d["ckpt"]), "stage1": str(d["stage1"]) if "stage1" in d.files else None, "arms": rows}, open(os.path.join(REPO, a.out), "w"))
np.savez_compressed(os.path.join(REPO, a.out.replace(".json", "_actions.npz")), demo=demo, **acts)
print("saved", a.out, "DONE")
