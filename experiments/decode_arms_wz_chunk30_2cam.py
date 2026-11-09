"""Stage A (predictor env): TWO-CAMERA chunk-30 Wyner-Ziv decode arms for PI0Fast scoring.

Same as decode_arms_wz_chunk30.py, but the 2-cam predictor (4x28x56 width-concat latents)
predicts base AND wrist; each view has its own codec (--stage2_ckpt <base> <wrist>) and both go
through the channel.  Saves img_<arm> (base) and wimg_<arm> (wrist) for every arm; `real` and
`wrist` stay the true frames.  Only episodes with index %% holdout_mod == 0 are sampled (the cache
builder skipped those).


Chunk-30 analogue of stage2_twophase/decode_arms_wz_sweep.py + make_arms_pi0fast.py.  For a
random frame t of a demo episode: strided history ends at t-1, the chunk's 30 actions give
z_hat for the target frame f = t-1+30 (the receiver's once-per-chunk prediction).  The
transmitter sends the conv WZ code of z_f * m, m = 1[mean_c(z_f - z_{t-1})^2 > thr]; the
receiver fills dropped channels with mu_p(z_hat) and composites with the disagreement gate.

Arms (base camera only; the wrist camera is always the true frame):
  real = true frame, vae = VAE round trip, zhat = decoded prediction, oracle_z = true object
  composited onto z_hat, k{K} = deployed WZ path sending the first K code channels.
Also saved per sample: n_obj = #cells of the oracle gate (how much the prediction misses),
n_mask = #changed cells in m (what the router would see).
Stage B: experiments/score_arms_pi0fast.py --in <npz>   (pi0fast env)

    python experiments/decode_arms_wz_chunk30.py \
        --stage2_ckpt outputs/masked_alloc_wz_chunk30_b1e-4_unw/masked_alloc.pt \
        --stage1_ckpt outputs/stage1_chunk30_h5/stage1_rollout_ep2.pt --n 160 \
        --out outputs/masked_alloc_wz_chunk30_b1e-4_unw/pi0fast_arms.npz
"""
import argparse, glob, json, os, pathlib, sys
import h5py, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from vae_wrapper import VAEWrapper               # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402
from models import FadingChannel                 # noqa: E402
sys.path.insert(0, os.path.join(REPO, "experiments", "stage2_twophase"))
from conv_codec import ConvWZCodec               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage1_ckpt", default="outputs/stage1_chunk30_2cam/stage1_rollout_ep3.pt")
ap.add_argument("--stage2_ckpt", required=True, nargs="+", metavar="CKPT", help="ONE joint codec (trained with --cam 0) or two: base then wrist")
ap.add_argument("--holdout_mod", type=int, default=5)
ap.add_argument("--demos", default="data/pick_*_cube_to_tray/demos_disturbed.hdf5")
ap.add_argument("--n", type=int, default=160)
ap.add_argument("--klevels", type=int, nargs="+", default=[2, 4, 8, 16])
ap.add_argument("--snr_db", type=float, default=20.0)
ap.add_argument("--channel", default="awgn")
ap.add_argument("--digital_bits", type=int, default=0, help=">0: uncoded digital link (quantize to this many bits, Gray, QAM, hard demap)")
ap.add_argument("--digital_nbps", type=int, default=4)
ap.add_argument("--cdl_model", default="C"); ap.add_argument("--cdl_delay_spread", type=float, default=100e-9)
ap.add_argument("--cdl_carrier_freq", type=float, default=3.5e9); ap.add_argument("--cdl_max_speed", type=float, default=3.0)
ap.add_argument("--legacy_noise", action="store_true", help="old behaviour: channel over all Dch channels (noise set by their power) instead of the k sent ones, as the server does")
ap.add_argument("--history_stride", type=int, default=5)
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

DEV = "cuda"
T_H, M, PS, CHUNK = CONFIG["num_history"], a.history_stride, 5, 30
K = CHUNK // PS; SPAN = (T_H - 1) * M + 1
CONFIG["num_pred"] = K
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

index = []     # (file, episode, t)
for p in sorted(glob.glob(os.path.join(REPO, a.demos))):
    with h5py.File(p, "r") as f:
        for e in f:
            if "observations_cam2" not in f[e]: continue
            if int("".join(c for c in e if c.isdigit())) % a.holdout_mod != 0: continue
            T = len(f[e]["observations_cam1"])
            index += [(p, e, t) for t in range(SPAN, T - 2 * CHUNK + 2)]      # demo actions after the target frame
picks = [index[i] for i in sorted(rng.choice(len(index), size=min(a.n, len(index)), replace=False))]
print(f"{len(index)} candidate frames, sampled {len(picks)}", flush=True)

NETS = []
for pth in a.stage2_ckpt:
    _dk = torch.load(os.path.join(REPO, pth), map_location="cpu"); _dc = _dk["cfg"]; assert _dc["arch"] == "conv_wz"
    n_ = ConvWZCodec(4, _dc["conv_w"], _dc["conv_dch"], _dc["conv_nb"]).to(DEV); n_.load_state_dict(_dk["net"]); n_.eval()
    NETS.append((n_, _dc["thr"], _dc["recv_thr"], _dc["conv_dch"]))
chan = FadingChannel(a.snr_db, channel_type=a.channel, cdl_kwargs={"model": a.cdl_model, "delay_spread": a.cdl_delay_spread,
                     "carrier_frequency": a.cdl_carrier_freq, "max_speed": a.cdl_max_speed})
if a.digital_bits > 0:
    from digital_channel import DigitalQAMChannel
    chan = DigitalQAMChannel(a.snr_db, bits=a.digital_bits, nbps=a.digital_nbps, channel_type=a.channel, group=28 * 56, device=DEV,
                             cdl_kwargs={"model": a.cdl_model, "delay_spread": a.cdl_delay_spread, "carrier_frequency": a.cdl_carrier_freq, "max_speed": a.cdl_max_speed})
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.stage1_ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck; cw.eval()
print(f"[wz-chunk30-2cam] codecs={a.stage2_ckpt} klevels={a.klevels}", flush=True)

def hidx(t): return [max(0, t - 1 - i * M) for i in range(T_H)][::-1]
def u8(x): return (x.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
def compose(rec, m, zh, recv_thr):
    g = m * (((rec - zh) ** 2).mean(1, keepdim=True) > recv_thr).float()
    return g * rec + (1.0 - g) * zh

ARMS = ["vae", "zhat", "oracle_z"] + [f"k{k}" for k in a.klevels]
out = {f"img_{ar}": [] for ar in ARMS}; out.update({f"wimg_{ar}": [] for ar in ARMS})
out.update(real=[], wrist=[], demo_a=[], task=[], ep=[], t=[], n_obj=[], n_mask=[], n_obj_w=[], n_mask_w=[])
instr = {}
by_ep = {}
for p, e, t in picks: by_ep.setdefault((p, e), []).append(t)
done = 0
for (p, e), ts in by_ep.items():
    with h5py.File(p, "r") as f:
        g = f[e]
        x = torch.from_numpy(np.asarray(g["observations_cam1"], np.float32) / 255.).permute(0, 3, 1, 2)
        cam1 = np.asarray(g["observations_cam1"]); cam2 = np.asarray(g["observations_cam2"])
        act = torch.from_numpy(np.asarray(g["actions"], np.float32))
    tdir = os.path.dirname(p)
    if tdir not in instr: instr[tdir] = json.load(open(os.path.join(tdir, "task.json")))["instruction"]
    with torch.no_grad():
        enc = lambda xx: torch.cat([vae.encode(xx[j:j + 8].to(DEV)).cpu() for j in range(0, len(xx), 8)])
        Z = torch.cat([enc(x), enc(torch.from_numpy(cam2.astype(np.float32) / 255.).permute(0, 3, 1, 2))], dim=-1)
    for t in ts:
        f_t = t - 1 + CHUNK
        with torch.no_grad():
            H = Z[hidx(t)].unsqueeze(0).to(DEV)                                  # (1,T_H,4,28,56)
            fut = torch.stack([act[t + PS * (j - 1): t + PS * j].mean(0) for j in range(1, K + 1)])
            aw = torch.cat([act[hidx(t)], fut], 0).unsqueeze(0).to(DEV)
            zh_all = cw.predict_next_latent(H, aw, n_steps=a.ddim).float()[:, -1]      # (1,4,28,56)
            lats, nobj, nmask = [], [], []
            JOINT = len(NETS) == 1                     # one codec on the whole 28x56 grid
            for v, (net, THR, RECV_THR, DCH) in enumerate(NETS):
                sl = slice(0, 56) if JOINT else (slice(0, 28) if v == 0 else slice(28, 56))
                zh = zh_all[..., sl]
                z_t = Z[f_t][..., sl].unsqueeze(0).to(DEV); z_pv = Z[t - 1][..., sl].unsqueeze(0).to(DEV)
                m = (((z_t - z_pv) ** 2).mean(1, keepdim=True) > THR).float()
                g_or = m * (((z_t - zh) ** 2).mean(1, keepdim=True) > RECV_THR).float()
                mu_e, _ = net.encode(z_t * m); mu_p, _ = net.prior(zh)
                B, Dc, Hh, Ww = mu_e.shape
                s_t = chan(mu_e.flatten(1)).view(B, Dc, Hh, Ww)
                lat = {"vae": z_t, "zhat": zh, "oracle_z": g_or * z_t + (1.0 - g_or) * zh}
                for k in a.klevels:
                    keep = (torch.arange(DCH, device=DEV) < k).float()[None, :, None, None]
                    if a.legacy_noise: mix = keep * s_t + (1 - keep) * mu_p
                    else: mix = torch.cat([chan(mu_e[:, :k].flatten(1)).view(B, k, Hh, Ww), mu_p[:, k:]], 1)    # == server: only the k sent channels use the channel
                    lat[f"k{k}"] = compose(net.decode(mix), m, zh, RECV_THR)
                lats.append(lat)
                if JOINT: nobj += [int(g_or[..., :28].sum().item()), int(g_or[..., 28:].sum().item())]; nmask += [int(m[..., :28].sum().item()), int(m[..., 28:].sum().item())]
                else: nobj.append(int(g_or.sum().item())); nmask.append(int(m.sum().item()))
            for ar in ARMS:
                halves = [lats[0][ar][..., :28], lats[0][ar][..., 28:]] if JOINT else [lats[0][ar], lats[1][ar]]
                for key, hv in zip(["img_", "wimg_"], halves):
                    out[f"{key}{ar}"].append(u8(vae.decode(hv.float().to(DEV)).clamp(0, 1)[0].cpu()))
        out["real"].append(cam1[f_t]); out["wrist"].append(cam2[f_t])
        out["demo_a"].append(act[f_t:f_t + CHUNK].numpy()); out["task"].append(instr[tdir])
        out["ep"].append(f"{os.path.basename(tdir)}/{e}"); out["t"].append(t)
        out["n_obj"].append(nobj[0]); out["n_mask"].append(nmask[0]); out["n_obj_w"].append(nobj[1]); out["n_mask_w"].append(nmask[1])
        done += 1
        if done % 20 == 0: print(f"  decoded {done}/{len(picks)}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
np.savez_compressed(os.path.join(REPO, a.out) if not os.path.isabs(a.out) else a.out,
                    **{k: np.array(v) for k, v in out.items()}, klevels=np.array(a.klevels),
                    snr_db=a.snr_db, ckpt=" + ".join(a.stage2_ckpt), stage1=a.stage1_ckpt)
n_obj = np.array(out["n_obj"]); n_mask = np.array(out["n_mask"])
print(f"saved {a.out} frames={done}  n_obj: median {np.median(n_obj):.0f} (zero in {np.mean(n_obj == 0):.0%})  "
      f"n_mask: median {np.median(n_mask):.0f} mean {n_mask.mean():.0f}", flush=True)
