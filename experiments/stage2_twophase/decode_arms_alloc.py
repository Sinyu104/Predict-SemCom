"""Stage A of the OFFLINE VLA action-space eval: decode the four arms to images.

Runs in the `predictor` env (needs diffusers/VAE/Ctrl-World). Produces images only;
scoring by the VLA is Stage B (score_arms.py, worldmodel env) so the two incompatible
transformers versions never share a process -- no server, fully offline.

Arms (same as vla_attribution.py):
    vae       dec(z_t)                 ceiling (perfect image)
    zhat      dec(z_hat)               Stage 1 alone (predictor)
    full      dec(refine(z_hat, s~))   Stage 1 + Stage 2 (deployed path)
    oracle_z  dec(refine(z_t,   s~))   Stage 2 with a perfect prior

z_hat is the SAME single-step clean-history prediction the gate was trained on
(num_pred=1, stride=1), so the eval matches training. Disturbance labels are
aligned +1 (DIST[t] = disturbed[t-1]) -- the moved cube appears one frame after the
flag; see project_disturbance_label_offby1. Calm vs disturbed frames are sampled
balanced so the VLA numbers are not swamped by the 94% calm majority.

    python experiments/stage2_twophase/decode_arms.py \
        --stage2_ckpt outputs/stage2_gate_aligned/phase2.pt --disagree_gate \
        --n_dist 150 --n_calm 150 --out outputs/vla_offline_arms.npz --device cuda:0
"""
import os, sys, pathlib, argparse
import numpy as np, torch, h5py

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from vae_wrapper import VAEWrapper               # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage1_ckpt", default="outputs/stage1_rollout_ft/stage1_rollout_ep1.pt")
ap.add_argument("--stage2_ckpt", default="outputs/stage2_gate_aligned/phase2.pt")
ap.add_argument("--disagree_gate", action="store_true", help="ckpt has the gate_proj head")
ap.add_argument("--refine_arch", choices=["residual", "fusion"], default="residual",
                help="must match the checkpoint's Stage-2 architecture")
ap.add_argument("--demos_file", default="disturbed.hdf5")
ap.add_argument("--history_stride", type=int, default=1)
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--snr_db", type=float, default=20.0)
ap.add_argument("--channel", default="awgn")
ap.add_argument("--mask_thr", type=float, default=0.08,
                help="oracle mask: latent cells with per-cell ||z_t-z_hat||^2 > thr use the direct "
                     "reconstruction (cube); the rest keep z_hat (background)")
ap.add_argument("--n_dist", type=int, default=150, help="target disturbed (flag+1) frames")
ap.add_argument("--n_calm", type=int, default=150, help="target calm frames")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", default="outputs/vla_offline_arms.npz")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--tasks", nargs="+",
                default=["pick_blue_cube_to_tray", "pick_orange_cube_to_tray",
                         "pick_purple_cube_to_tray", "pick_red_cube_to_tray",
                         "pick_yellow_cube_to_tray"])
a = ap.parse_args()

DEV = a.device
T_H = CONFIG["num_history"]; M = a.history_stride; SPAN = (T_H - 1) * M + 1
CONFIG["snr_db"] = a.snr_db; CONFIG["channel_type"] = a.channel
CONFIG["refine_disagreement_gate"] = a.disagree_gate
CONFIG["refine_arch"] = a.refine_arch
COLOR = lambda t: t.split("_")[1]
INSTR = lambda t: f"pick up the {COLOR(t)} cube and place it on the tray"
rng = np.random.default_rng(a.seed)

# ── build balanced candidate list (task, ep, t, is_dist) with +1 alignment ──── #
dist_c, calm_c = [], []
for task in a.tasks:
    p = os.path.join(REPO, "data", task, a.demos_file)
    if not os.path.isfile(p):
        print(f"skip {task}: no {a.demos_file}", flush=True); continue
    with h5py.File(p, "r") as f:
        for e in sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit()))):
            g = f[e]
            if "disturbed" not in g:
                continue
            d = np.array(g["disturbed"]).astype(int)
            n = len(g["observations_cam1"])
            for t in range(SPAN, n):
                (dist_c if d[t - 1] == 1 else calm_c).append((task, e, t))   # +1 alignment
print(f"candidates: disturbed(flag+1)={len(dist_c)}  calm={len(calm_c)}", flush=True)

def sample(cands, k):
    if len(cands) <= k:
        return cands
    idx = rng.choice(len(cands), size=k, replace=False)
    return [cands[i] for i in sorted(idx)]

picks = sample(dist_c, a.n_dist) + sample(calm_c, a.n_calm)
labels = {(t, e, ti): (1 if i < min(a.n_dist, len(dist_c)) else 0)
          for i, (t, e, ti) in enumerate(picks)}
# group by (task, ep) so each episode is encoded once
by_ep = {}
for task, e, t in picks:
    by_ep.setdefault((task, e), []).append(t)
print(f"sampled: {sum(1 for v in labels.values() if v)} disturbed + "
      f"{sum(1 for v in labels.values() if not v)} calm across {len(by_ep)} episodes", flush=True)

# ── models (MASKED-DIRECT: direct JSCC autoencoder + oracle mask) ─────────────── #
from models import JsccEncoder, FadingChannel    # noqa: E402
import torch.nn as nn
class Dec(nn.Module):
    def __init__(s, Din):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(Din,1024), nn.GELU(), nn.Linear(1024,2048), nn.GELU(),
                              nn.Linear(2048,4*28*28))
    def forward(s, x): return s.net(x).view(-1,4,28,28)
_dk = torch.load(os.path.join(REPO, a.stage2_ckpt), map_location="cpu")
_dc = _dk["cfg"]; D_jscc=_dc["D_jscc"]; THR=_dc["thr"]
enc = JsccEncoder(4,28,28,D_jscc,spatial=False).to(DEV); enc.load_state_dict(_dk["enc"]); enc.eval()
dec = Dec(D_jscc).to(DEV); dec.load_state_dict(_dk["dec"]); dec.eval()
chan = FadingChannel(a.snr_db, channel_type=a.channel)
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=1,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.stage1_ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck; cw.eval()
print(f"[alloc] loaded {a.stage2_ckpt}  D_jscc={D_jscc}  change-thr={THR}", flush=True)

def hidx(t):
    return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]

def dec_uint8(z):
    im = vae.decode(z.to(DEV).float()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (im * 255).astype(np.uint8)

IMG = {ar: [] for ar in ["vae", "zhat", "full", "oracle_z"]}
A_DEMO, LAB, TASKID, INSTRS = [], [], [], []
done = 0
for (task, e) in by_ep:
    p = os.path.join(REPO, "data", task, a.demos_file)
    with h5py.File(p, "r") as f:
        g = f[e]; n = len(g["observations_cam1"])
        x = torch.from_numpy(np.asarray(g["observations_cam1"], np.float32) / 255.).permute(0, 3, 1, 2)
        act = torch.from_numpy(np.asarray(g["actions"], np.float32))
    with torch.no_grad():
        Z = torch.cat([vae.encode(x[j:j + 8].to(DEV)).cpu() for j in range(0, len(x), 8)])
    for t in by_ep[(task, e)]:
        with torch.no_grad():
            H = torch.stack([Z[j] for j in hidx(t)], 0).unsqueeze(0).to(DEV)
            aw = torch.cat([act[torch.tensor(hidx(t))], act[t:t + 1]], 0).unsqueeze(0).to(DEV)
            zh = cw.predict_next_latent(H, aw, n_steps=a.ddim).float()[:, 0]      # (1,C,H,W)
            z_t = Z[t].unsqueeze(0).to(DEV)
            z_pv = Z[t - 1].unsqueeze(0).to(DEV)                                   # z_{t-1} (transmitter, WZ-clean)
            m = (((z_t - z_pv) ** 2).mean(1, keepdim=True) > THR).float()          # change mask (cube cells)
            _, _, s = enc(z_t * m, sample=False); st = chan(s)                     # encode only the changed cells
            rec = dec(st)
            lat = {"vae": z_t, "zhat": zh,
                   "full": m * rec + (1.0 - m) * zh,                               # CHANGE-MASKED ALLOC: cube from s~, bg from z_hat
                   "oracle_z": m * z_t + (1.0 - m) * zh}                           # ceiling: TRUE cube composited (perfect channel)
        for ar in IMG:
            IMG[ar].append(dec_uint8(lat[ar]))
        A_DEMO.append(act[t].numpy()); LAB.append(labels[(task, e, t)])
        TASKID.append(task); INSTRS.append(INSTR(task))
        done += 1
        if done % 25 == 0:
            print(f"  decoded {done}/{len(picks)}", flush=True)

out = {f"img_{ar}": np.stack(IMG[ar]).astype(np.uint8) for ar in IMG}
out.update(a_demo=np.stack(A_DEMO).astype(np.float32),
           label=np.array(LAB, np.int64),
           task=np.array(TASKID), instruction=np.array(INSTRS),
           snr_db=a.snr_db, channel=a.channel,
           stage2=a.stage2_ckpt, disagree_gate=a.disagree_gate)
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
np.savez_compressed(a.out, **out)
print(f"saved {a.out}  frames={done}  disturbed={int(np.sum(LAB))} calm={int(len(LAB)-np.sum(LAB))}",
      flush=True)
