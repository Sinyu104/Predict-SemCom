"""Reconstructed-history cache for fine-tuning the 2-camera chunk-30 predictor on what the receiver really feeds it.

For frame f (t = f-29): predict f from CLEAN history ending at t-1 (+ the chunk's actions), send z_f*m through
the codec (k channels, AWGN), composite -> z~_f.  z~_f is what the live loop stores as the boundary frame.
Training then uses  history = z~ at f = t-1-5i  (i=0..5), target = TRUE latents  (stage1_rollout_finetune.py
--recon_hist).  Only frames on the grid f = 0 (mod 5) are built: a window with t = 1 (mod 5) needs exactly those.

    for i in 2 3; do CUDA_VISIBLE_DEVICES=$i python experiments/make_recon_history_2cam.py --shard $((i-2)) --nshards 2 \
        --out outputs/recon_hist_2cam/shard$((i-2)).pt & done; wait
"""
import argparse, os, pathlib, sys, time
import numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402
from models import FadingChannel                 # noqa: E402
sys.path.insert(0, os.path.join(REPO, "experiments", "stage2_twophase"))
from conv_codec import ConvWZCodec               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--latents", default="outputs/stage1_chunk30_2cam/latents.pt")
ap.add_argument("--stage1_ckpt", default="outputs/stage1_chunk30_2cam/stage1_rollout_ep3.pt")
ap.add_argument("--stage2_ckpt", default="outputs/masked_alloc_wz_2cam_joint_mix_unw/masked_alloc.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--n_eps", type=int, default=0, help="use every (498/n_eps)-th episode (0 = all)")
ap.add_argument("--ks", type=int, nargs="+", default=[2, 2, 2, 4, 16], help="k drawn uniformly per frame")
ap.add_argument("--snr_db", type=float, default=20.0)
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--pred_bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

DEV = "cuda:0"; T_H, M, PS, CHUNK = CONFIG["num_history"], 5, 5, 30
K = CHUNK // PS; SPAN = (T_H - 1) * M + 1
CONFIG["num_pred"] = K
torch.manual_seed(a.seed + a.shard); rng = np.random.default_rng(a.seed + a.shard)

d = torch.load(os.path.join(REPO, a.latents), map_location="cpu")
Zs, As = d["Z"], d["A"]
eps = list(range(len(Zs)))
if a.n_eps: eps = eps[::max(1, len(Zs) // a.n_eps)][:a.n_eps]
jobs = []                                  # (episode, f)
for e in eps:
    for f in range(SPAN + CHUNK - 1, len(Zs[e]), M):          # f = 0 (mod 5), t = f-29 >= SPAN
        jobs.append((e, f))
jobs = jobs[a.shard::a.nshards]
print(f"[shard {a.shard}] {len(jobs)} frames from {len(eps)} episodes", flush=True)

_dk = torch.load(os.path.join(REPO, a.stage2_ckpt), map_location="cpu"); _dc = _dk["cfg"]
THR, RECV_THR, DCH = _dc["thr"], _dc["recv_thr"], _dc["conv_dch"]
net = ConvWZCodec(4, _dc["conv_w"], DCH, _dc["conv_nb"]).to(DEV); net.load_state_dict(_dk["net"]); net.eval()
chan = FadingChannel(a.snr_db, channel_type="awgn")
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K, svd_path=CONFIG["svd_model_name"],
                      freeze_unet=True, finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.stage1_ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck; cw.eval()

def hidx(t): return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]

out = {"e": [], "f": [], "k": [], "zr": [], "zh": []}
t0 = time.perf_counter()
for i in range(0, len(jobs), a.pred_bs):
    jb = jobs[i:i + a.pred_bs]
    ts = [f - CHUNK + 1 for _, f in jb]
    H = torch.stack([Zs[e].float()[hidx(t)] for (e, _), t in zip(jb, ts)]).to(DEV)
    aw = torch.stack([torch.cat([As[e].float()[hidx(t)],
                                 torch.stack([As[e].float()[t + PS * (j - 1): t + PS * j].mean(0) for j in range(1, K + 1)])], 0)
                      for (e, _), t in zip(jb, ts)]).to(DEV)
    with torch.no_grad():
        zh = cw.predict_next_latent(H, aw, n_steps=a.ddim).float()[:, -1]
        z_t = torch.stack([Zs[e][f].float() for e, f in jb]).to(DEV)
        z_pv = torch.stack([Zs[e][t - 1].float() for (e, _), t in zip(jb, ts)]).to(DEV)
        m = (((z_t - z_pv) ** 2).mean(1, keepdim=True) > THR).float()
        mu_e, _ = net.encode(z_t * m); mu_p, _ = net.prior(zh)
        B, Dc, Hh, Ww = mu_e.shape
        s_t = chan(mu_e.flatten(1)).view(B, Dc, Hh, Ww)
        ks = rng.choice(a.ks, size=B)
        keep = (torch.arange(DCH, device=DEV)[None] < torch.tensor(ks, device=DEV)[:, None]).float()[:, :, None, None]
        rec = net.decode(keep * s_t + (1 - keep) * mu_p)
        g = m * (((rec - zh) ** 2).mean(1, keepdim=True) > RECV_THR).float()
        zr = g * rec + (1 - g) * zh
    out["e"] += [e for e, _ in jb]; out["f"] += [f for _, f in jb]; out["k"] += ks.tolist()
    out["zr"].append(zr.half().cpu()); out["zh"].append(zh.half().cpu())
    if (i // a.pred_bs) % 10 == 0:
        n = i + len(jb); el = time.perf_counter() - t0
        zt_ = z_t; print(f"[shard {a.shard}] {n}/{len(jobs)} {el/n:.2f}s/frame eta {el/n*(len(jobs)-n)/60:.0f}min  "
                         f"|zhat-z|^2 {((zh-zt_)**2).mean():.4f} |z~-z|^2 {((zr-zt_)**2).mean():.4f}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
zr_all = torch.cat(out["zr"]); zh_all = torch.cat(out["zh"])
err = lambda x: float(np.mean([((x[j].float() - Zs[e][f].float()) ** 2).mean().item()
                               for j, (e, f) in enumerate(zip(out["e"], out["f"]))]))
torch.save({"e": torch.tensor(out["e"]), "f": torch.tensor(out["f"]), "k": torch.tensor(out["k"]),
            "zr": zr_all, "zh": zh_all,
            "meta": {"stage1": a.stage1_ckpt, "stage2": a.stage2_ckpt, "snr_db": a.snr_db, "ks": a.ks, "ddim": a.ddim}}, a.out)
print(f"[shard {a.shard}] saved {a.out}: {len(zr_all)} frames  mse(zhat,z)={err(zh_all):.4f}  mse(z~,z)={err(zr_all):.4f}", flush=True)
