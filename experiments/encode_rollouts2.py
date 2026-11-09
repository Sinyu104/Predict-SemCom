"""VAE-encode policy_rollouts2.hdf5 (blue/red/yellow) to a 2-camera latent cache, one file per call.

Same format as outputs/stage1_chunk30_2cam_ft/rollout_latents.pt (Z: (T,4,28,56) fp16, cam1|cam2 concatenated along
width; A: (T,7) fp16; F: file index) plus the episode keys, the pose (ee pos 3 + joint pos 9) and the disturbed flags.
The file may still be growing: episodes whose cam1/cam2/actions/pose lengths disagree are skipped.

    CUDA_VISIBLE_DEVICES=0 python experiments/encode_rollouts2.py --task blue
"""
import argparse, os, pathlib, sys
import h5py, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from config import CONFIG              # noqa: E402
from vae_wrapper import VAEWrapper     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--file", default="policy_rollouts2.hdf5")
ap.add_argument("--out", default="outputs/rollouts2")
a = ap.parse_args()
p = os.path.join(REPO, f"data/pick_{a.task}_cube_to_tray/{a.file}")
vae = VAEWrapper(CONFIG["vae_model_name"]).to("cuda")


def enc(g, name):
    x = torch.from_numpy(np.array(g[name])).float().div_(255.).permute(0, 3, 1, 2)
    with torch.no_grad():
        return torch.cat([vae.encode(x[j:j + 16].cuda()).cpu() for j in range(0, len(x), 16)])


out = {"Z": [], "A": [], "P": [], "D": [], "key": [], "task": a.task}
with h5py.File(p, "r", locking=False) as f:
    keys = [k for k in f if "observations_cam1" in f[k]]
    print(f"{a.task}: {len(keys)} episodes in file", flush=True)
    for i, k in enumerate(keys):
        g = f[k]
        try:
            n1, n2, na = g["observations_cam1"].shape[0], g["observations_cam2"].shape[0], g["actions"].shape[0]
            pe, pj = np.array(g["robot_ee_pos"]), np.array(g["robot_joint_pos"])
            if not (n1 == n2 == na == len(pe) == len(pj)):
                print(f"  skip {k}: lengths {n1},{n2},{na},{len(pe)},{len(pj)}"); continue
            z = torch.cat([enc(g, "observations_cam1"), enc(g, "observations_cam2")], dim=-1)
            out["Z"].append(z.half()); out["A"].append(torch.from_numpy(np.array(g["actions"], dtype=np.float32)).half())
            out["P"].append(torch.from_numpy(np.concatenate([pe, pj], 1)).float())
            out["D"].append(torch.from_numpy(np.array(g["disturbed"]).astype(np.int8))); out["key"].append(k)
        except Exception as e:
            print(f"  skip {k}: {e}"); continue
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(keys)}", flush=True)
os.makedirs(os.path.join(REPO, a.out), exist_ok=True)
torch.save(out, os.path.join(REPO, a.out, f"latents_{a.task}.pt"))
print("saved", len(out["Z"]), "episodes", "DONE")
