"""Stage B of the OFFLINE VLA action-space eval: score decoded arms with OpenVLA.

Runs in the VLA env (worldmodel). Loads OpenVLA-7B + LoRA (+FiLM) IN-PROCESS via
server.openvla_server.OpenVLAServer -- no socket, no running server. Reads the arm
images decode_arms.py saved and reports, per arm, the VLA action error vs the demo
action, split calm vs disturbed (flag+1).

    |a-demo|   L2 of (VLA action - ground-truth demo action) over the 7-dim action
    |a-vae|    L2 vs the vae-arm action = reconstruction-induced action change
               (removes the policy's own imitation error)
    grip|err|  |dim6 - demo dim6|; a grasp task is lost on the gripper axis

Read: vae is the floor. On DISTURBED frames, if full < zhat the channel is helping
the policy act correctly; if full ~ zhat the refiner is not using the channel.

    conda run -n worldmodel python experiments/stage2_twophase/score_arms.py \
        --in outputs/vla_offline_arms.npz \
        --lora_dir outputs/openvla_dagger_v3/checkpoint_epoch24
"""
import os, sys, pathlib, argparse
import numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="src", default="outputs/vla_offline_arms.npz")
ap.add_argument("--lora_dir", default="outputs/openvla_dagger_v3/checkpoint_epoch24")
ap.add_argument("--model_dir", default="openvla/openvla-7b")
a = ap.parse_args()

from server.openvla_server import OpenVLAServer            # noqa: E402
srv = OpenVLAServer(lora_dir=os.path.join(REPO, a.lora_dir), model_dir=a.model_dir)

d = np.load(a.src, allow_pickle=True)
# auto-detect arms from img_* keys; keep vae first (it is the |a-vae| reference), then a stable order
_found = [k[4:] for k in d.files if k.startswith("img_")]
_pref = [ar for ar in ["vae", "zhat", "full", "oracle_z"] if ar in _found]
ARMS = _pref + sorted(ar for ar in _found if ar not in _pref)
assert "vae" in ARMS, "need img_vae as the |a-vae| reference"
imgs = {ar: d[f"img_{ar}"] for ar in ARMS}
a_demo = d["a_demo"]; label = d["label"]; instr = d["instruction"]
N = len(a_demo)
print(f"[data] {N} frames  disturbed={int(label.sum())} calm={int(N-label.sum())}  "
      f"snr={d['snr_db']}dB {d['channel']}  stage2={d['stage2']}", flush=True)

# predict every arm for every frame; cache vae action per frame for |a-vae|
acts = {ar: np.zeros((N, 7), np.float32) for ar in ARMS}
for i in range(N):
    for ar in ARMS:
        acts[ar][i] = np.asarray(srv.predict(imgs[ar][i], str(instr[i])), np.float32)[:7]
    if (i + 1) % 25 == 0:
        print(f"  scored {i+1}/{N}", flush=True)

def stats(ar, mask):
    m = mask
    if m.sum() == 0:
        return (float("nan"),) * 3
    dda = np.linalg.norm(acts[ar][m] - a_demo[m][:, :7], axis=1).mean()
    dva = np.linalg.norm(acts[ar][m] - acts["vae"][m], axis=1).mean()
    grp = np.abs(acts[ar][m][:, 6] - a_demo[m][:, 6]).mean()
    return dda, dva, grp

dist_m = label == 1; calm_m = label == 0; all_m = np.ones(N, bool)
for name, m in [("ALL", all_m), ("CALM", calm_m), ("DISTURBED (flag+1)", dist_m)]:
    print(f"\n=== {name}  (n={int(m.sum())}) ===")
    print(f"{'arm':<10} {'|a-demo|':>9} {'|a-vae|':>9} {'grip|err|':>10}")
    for ar in ARMS:
        dda, dva, grp = stats(ar, m)
        print(f"{ar:<10} {dda:9.4f} {dva:9.4f} {grp:10.4f}")

print("\nChannel effect on DISTURBED: compare full vs zhat (lower |a-demo| = channel helps the policy).")
print("If oracle_z ~ vae, the refiner is fine given a good prior and Stage 1 is the limit.")
print("DONE")
