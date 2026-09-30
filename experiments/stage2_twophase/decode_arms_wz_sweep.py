"""Multi-k decode for the WZ conv codec: reconstruct at EVERY channel level in one pass.

Ctrl-World's z_hat is the expensive part and is independent of k, so we compute it once per
frame and reconstruct the composite at k in {2,4,8,16} -- giving the full bandwidth-distortion
axis for a single beta model from one ~40-min decode. Combine several beta checkpoints to trace
the rate-distortion-action frontier (project_adaptive_rate_conv).

Arms saved: vae (ceiling), zhat (predictor), oracle_z (true cube composited), and k2/k4/k8/k16
(WZ codec: send first k channels, receiver fills the rest with the prior mean mu_p(z_hat), then
the disagreement composite z~ = g*rec + (1-g)*z_hat).

    python experiments/stage2_twophase/decode_arms_wz_sweep.py \
        --stage2_ckpt outputs/masked_alloc_wz_b1e-4/masked_alloc.pt \
        --n_dist 150 --n_calm 150 --snr_db 20 --out outputs/vla_wz_sweep_b1e-4.npz --device cuda:0
"""
import os, sys, pathlib, argparse
import numpy as np, torch, h5py

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from vae_wrapper import VAEWrapper               # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402
from models import FadingChannel                 # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conv_codec import ConvWZCodec              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage1_ckpt", default="outputs/stage1_rollout_ft/stage1_rollout_ep1.pt")
ap.add_argument("--stage2_ckpt", required=True)
ap.add_argument("--demos_file", default="disturbed.hdf5")
ap.add_argument("--history_stride", type=int, default=1)
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--snr_db", type=float, default=20.0)
ap.add_argument("--channel", default="awgn")
ap.add_argument("--klevels", type=int, nargs="+", default=[2, 4, 8, 16])
ap.add_argument("--n_dist", type=int, default=150)
ap.add_argument("--n_calm", type=int, default=150)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--tasks", nargs="+",
                default=["pick_blue_cube_to_tray", "pick_orange_cube_to_tray",
                         "pick_purple_cube_to_tray", "pick_red_cube_to_tray",
                         "pick_yellow_cube_to_tray"])
a = ap.parse_args()

DEV = a.device
T_H = CONFIG["num_history"]; M = a.history_stride; SPAN = (T_H - 1) * M + 1
COLOR = lambda t: t.split("_")[1]
INSTR = lambda t: f"pick up the {COLOR(t)} cube and place it on the tray"
rng = np.random.default_rng(a.seed)

dist_c, calm_c = [], []
for task in a.tasks:
    p = os.path.join(REPO, "data", task, a.demos_file)
    if not os.path.isfile(p):
        print(f"skip {task}: no {a.demos_file}", flush=True); continue
    with h5py.File(p, "r") as f:
        for e in sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit()))):
            g = f[e]
            if "disturbed" not in g: continue
            d = np.array(g["disturbed"]).astype(int); n = len(g["observations_cam1"])
            for t in range(SPAN, n):
                (dist_c if d[t - 1] == 1 else calm_c).append((task, e, t))    # +1 alignment
print(f"candidates: disturbed(flag+1)={len(dist_c)}  calm={len(calm_c)}", flush=True)

def sample(cands, k):
    if len(cands) <= k: return cands
    return [cands[i] for i in sorted(rng.choice(len(cands), size=k, replace=False))]

picks = sample(dist_c, a.n_dist) + sample(calm_c, a.n_calm)
labels = {(t, e, ti): (1 if i < min(a.n_dist, len(dist_c)) else 0) for i, (t, e, ti) in enumerate(picks)}
by_ep = {}
for task, e, t in picks:
    by_ep.setdefault((task, e), []).append(t)
print(f"sampled: {sum(labels.values())} disturbed + {len(labels)-sum(labels.values())} calm "
      f"across {len(by_ep)} episodes", flush=True)

# ── models ──────────────────────────────────────────────────────────────────── #
_dk = torch.load(os.path.join(REPO, a.stage2_ckpt), map_location="cpu"); _dc = _dk["cfg"]
assert _dc["arch"] == "conv_wz", "this script is for conv_wz checkpoints"
THR = _dc["thr"]; RECV_THR = _dc["recv_thr"]; DCH = _dc["conv_dch"]
net = ConvWZCodec(4, _dc["conv_w"], DCH, _dc["conv_nb"]).to(DEV); net.load_state_dict(_dk["net"]); net.eval()
chan = FadingChannel(a.snr_db, channel_type=a.channel)
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=1,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.stage1_ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck; cw.eval()
print(f"[wz-sweep] beta={_dc.get('beta')}  Dch={DCH}  klevels={a.klevels}", flush=True)

def hidx(t): return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]
def dec_uint8(z):
    im = vae.decode(z.to(DEV).float()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (im * 255).astype(np.uint8)

def compose(rec, m, zh):
    g = m * (((rec - zh) ** 2).mean(1, keepdim=True) > RECV_THR).float()
    return g * rec + (1.0 - g) * zh

ARMS = ["vae", "zhat", "oracle_z"] + [f"k{k}" for k in a.klevels]
IMG = {ar: [] for ar in ARMS}
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
            zh = cw.predict_next_latent(H, aw, n_steps=a.ddim).float()[:, 0]
            z_t = Z[t].unsqueeze(0).to(DEV); z_pv = Z[t - 1].unsqueeze(0).to(DEV)
            m = (((z_t - z_pv) ** 2).mean(1, keepdim=True) > THR).float()
            mu_e, _ = net.encode(z_t * m); mu_p, _ = net.prior(zh)
            B, Dc, Hh, Ww = mu_e.shape
            s_t = chan(mu_e.flatten(1)).view(B, Dc, Hh, Ww)                     # code through the channel (once)
            g_or = m * (((z_t - zh) ** 2).mean(1, keepdim=True) > RECV_THR).float()
            lat = {"vae": z_t, "zhat": zh, "oracle_z": g_or * z_t + (1.0 - g_or) * zh}
            for k in a.klevels:
                keep = (torch.arange(DCH, device=DEV) < k).float()[None, :, None, None]
                rec = net.decode(keep * s_t + (1 - keep) * mu_p)                # fill dropped channels w/ prior
                lat[f"k{k}"] = compose(rec, m, zh)
        for ar in ARMS:
            IMG[ar].append(dec_uint8(lat[ar]))
        A_DEMO.append(act[t].numpy()); LAB.append(labels[(task, e, t)])
        TASKID.append(task); INSTRS.append(INSTR(task)); done += 1
        if done % 25 == 0: print(f"  decoded {done}/{len(picks)}", flush=True)

out = {f"img_{ar}": np.stack(IMG[ar]).astype(np.uint8) for ar in ARMS}
out.update(a_demo=np.stack(A_DEMO).astype(np.float32), label=np.array(LAB, np.int64),
           task=np.array(TASKID), instruction=np.array(INSTRS),
           klevels=np.array(a.klevels), snr_db=a.snr_db, channel=a.channel,
           stage2=a.stage2_ckpt, beta=float(_dc.get("beta", 0.0)))
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
np.savez_compressed(a.out, **out)
print(f"saved {a.out}  frames={done}  arms={ARMS}", flush=True)
