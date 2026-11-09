"""Offline closed-loop check of a 2-camera chunk-30 predictor (mirrors server _step_wz2cam, demo actions).

Chunk 0 is predicted from TRUE strided history (t0 = SPAN); every later chunk uses the loop's own history
(previous chunk's predicted slots, last slot replaced by the codec composite z~).  Per chunk n we report
mse(zhat, z_f) and mse(z~, z_f) against the true boundary frame f = t-1+30 -- a flat curve = no drift.

    CUDA_VISIBLE_DEVICES=0 python experiments/eval_loop_rollout_2cam.py --stage1_ckpt <ckpt> --tag ep3
"""
import argparse, json, os, pathlib, sys
import numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402
from models import FadingChannel                 # noqa: E402
sys.path.insert(0, os.path.join(REPO, "experiments", "stage2_twophase"))
from conv_codec import ConvWZCodec               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--latents", default="outputs/stage1_chunk30_2cam/latents.pt")
ap.add_argument("--stage1_ckpt", required=True)
ap.add_argument("--stage2_ckpt", default="outputs/masked_alloc_wz_2cam_joint_mix_unw/masked_alloc.pt")
ap.add_argument("--n_eps", type=int, default=24)
ap.add_argument("--chunks", type=int, default=6)
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--snr_db", type=float, default=20.0)
ap.add_argument("--channel", default="awgn", choices=["awgn", "cdl"])
ap.add_argument("--anchor", action="store_true", help="start like the server: history = frame 0 repeated, first target = frame 30")
ap.add_argument("--live_last", action="store_true", help="last future slot uses only the 4 actions the server knows")
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

DEV = "cuda:0"; T_H, M, PS, CHUNK = CONFIG["num_history"], 5, 5, 30
K = CHUNK // PS; SPAN = (T_H - 1) * M + 1
CONFIG["num_pred"] = K
torch.manual_seed(a.seed)

d = torch.load(os.path.join(REPO, a.latents), map_location="cpu"); Zs, As = d["Z"], d["A"]
need = SPAN + CHUNK * a.chunks + 1
# episodes NOT used for the reconstructed-history cache (it takes every 4th episode) and long enough
eps = [e for e in range(len(Zs)) if e % 4 == 2 and len(Zs[e]) >= need][:a.n_eps]
print(f"[{a.tag}] {len(eps)} episodes, {a.chunks} chunks, k={a.k}", flush=True)

_dk = torch.load(os.path.join(REPO, a.stage2_ckpt), map_location="cpu"); _dc = _dk["cfg"]
THR, RECV_THR, DCH = _dc["thr"], _dc["recv_thr"], _dc["conv_dch"]
net = ConvWZCodec(4, _dc["conv_w"], DCH, _dc["conv_nb"]).to(DEV); net.load_state_dict(_dk["net"]); net.eval()
chan = FadingChannel(a.snr_db, channel_type=a.channel,
                     cdl_kwargs={"model": "C", "delay_spread": 100e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0} if a.channel == "cdl" else None)
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K, svd_path=CONFIG["svd_model_name"],
                      freeze_unet=True, finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.stage1_ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck; cw.eval()

def hidx(t): return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]

eh = np.zeros((len(eps), a.chunks)); ec = np.zeros_like(eh); ecopy = np.zeros_like(eh)
slot = np.zeros((len(eps), a.chunks, K))   # error of each predicted slot (t-1+5(j+1)) vs the true frame, BEFORE the codec replaces the last
ehc = np.zeros((len(eps), a.chunks, 2)); ecc = np.zeros((len(eps), a.chunks, 2))   # zhat / z~ error per camera half (0 = base, 1 = wrist)
ehm = np.zeros_like(eh); ehu = np.zeros_like(eh); fm = np.zeros_like(eh)   # zhat error inside / outside the change mask, mask fraction
for b0 in range(0, len(eps), a.bs):
    ids = eps[b0:b0 + a.bs]
    T0 = 1 if a.anchor else SPAN
    if a.anchor:
        Hh = torch.stack([Zs[e].float()[0:1].repeat(T_H, 1, 1, 1) for e in ids]).to(DEV)    # anchor: frame 0 repeated
        zprev = torch.stack([Zs[e][0].float() for e in ids]).to(DEV)
    else:
        Hh = torch.stack([Zs[e].float()[hidx(SPAN)] for e in ids]).to(DEV)        # true history for chunk 0
        zprev = torch.stack([Zs[e][SPAN - 1].float() for e in ids]).to(DEV)
    for n in range(a.chunks):
        t = T0 + CHUNK * n; f = t - 1 + CHUNK
        fa = lambda e, j: As[e].float()[t + PS * (j - 1): t + PS * j - (1 if (a.live_last and j == K) else 0)].mean(0)
        aw = torch.stack([torch.cat([As[e].float()[hidx(t)],
                                     torch.stack([fa(e, j) for j in range(1, K + 1)])], 0)
                          for e in ids]).to(DEV)
        with torch.no_grad():
            F = cw.predict_next_latent(Hh, aw, n_steps=a.ddim).float()         # (B,K,4,28,56)
            zh = F[:, -1]
            z_t = torch.stack([Zs[e][f].float() for e in ids]).to(DEV)
            m = (((z_t - zprev) ** 2).mean(1, keepdim=True) > THR).float()
            mu_e, _ = net.encode(z_t * m); mu_p, _ = net.prior(zh)
            B, Dc, H_, W_ = mu_e.shape
            s_t = chan(mu_e.flatten(1)).view(B, Dc, H_, W_)
            keep = (torch.arange(DCH, device=DEV) < a.k).float()[None, :, None, None]
            rec = net.decode(keep * s_t + (1 - keep) * mu_p)
            g = m * (((rec - zh) ** 2).mean(1, keepdim=True) > RECV_THR).float()
            zt_ = g * rec + (1 - g) * zh
        for j in range(len(ids)):
            i = b0 + j
            eh[i, n] = ((zh[j] - z_t[j]) ** 2).mean().item(); ec[i, n] = ((zt_[j] - z_t[j]) ** 2).mean().item()
            ecopy[i, n] = ((Hh[j, -1] - z_t[j]) ** 2).mean().item()
            cell = ((zh[j] - z_t[j]) ** 2).mean(0); mj = m[j, 0] > 0.5            # per-cell zhat error
            for c in range(2):
                sl = slice(28 * c, 28 * (c + 1))
                ehc[i, n, c] = ((zh[j][..., sl] - z_t[j][..., sl]) ** 2).mean().item()
                ecc[i, n, c] = ((zt_[j][..., sl] - z_t[j][..., sl]) ** 2).mean().item()
            fm[i, n] = mj.float().mean().item()
            for jj in range(K):
                slot[i, n, jj] = ((F[j, jj] - Zs[ids[j]][t - 1 + PS * (jj + 1)].float().to(DEV)) ** 2).mean().item()
            ehm[i, n] = cell[mj].mean().item() if mj.any() else 0.0; ehu[i, n] = cell[~mj].mean().item()
        F[:, -1] = zt_; Hh = F; zprev = z_t
    print(f"  batch {b0 // a.bs + 1}/{(len(eps) + a.bs - 1) // a.bs} done", flush=True)

res = {"tag": a.tag, "ckpt": a.stage1_ckpt, "n_eps": len(eps), "k": a.k,
       "zhat": eh.mean(0).round(4).tolist(), "ztilde": ec.mean(0).round(4).tolist(), "copy_last": ecopy.mean(0).round(4).tolist(),
       "zhat_in_mask": ehm.mean(0).round(4).tolist(), "zhat_out_mask": ehu.mean(0).round(4).tolist(), "mask_frac": fm.mean(0).round(3).tolist(),
       "zhat_base": ehc[:, :, 0].mean(0).round(4).tolist(), "zhat_wrist": ehc[:, :, 1].mean(0).round(4).tolist(),
       "ztilde_base": ecc[:, :, 0].mean(0).round(4).tolist(), "ztilde_wrist": ecc[:, :, 1].mean(0).round(4).tolist(),
       "slot_err_chunk1": slot[:, 0].mean(0).round(4).tolist(), "slot_err_chunk2": slot[:, 1].mean(0).round(4).tolist(),
       "slot_err_chunk6": slot[:, -1].mean(0).round(4).tolist(), "slot_err_mean_over_chunks": slot.mean((0, 1)).round(4).tolist()}
print(json.dumps(res), flush=True)
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"))
