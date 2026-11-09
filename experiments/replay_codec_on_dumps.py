"""Replay test: re-encode the states a closed-loop run actually visited with another codec, then let diag_policy_on_dumps.py score them.
For every coded request of the K_unf2 20 dB k=4 run (episodes 1..30) it takes the logged true latent lat_zt, the logged prediction lat_zh and the
logged transmitter mask lat_mask (the mask depends only on true frames, so it is the same for any codec), sends the first k code channels through
CDL-C (seeded), fills the rest from the prior, decodes, gates (g = m * 1[mean_c(rec - zhat)^2 > recv_thr]), VAE-decodes, and writes a dump-like dir.
Also prints the cube-cell latent error (cube cells from the TRUE images, saturated-blue detector).
predictor env:
    CUDA_VISIBLE_DEVICES=1 .../envs/predictor/bin/python experiments/replay_codec_on_dumps.py --ckpt outputs/wz2cam_retrain/<run>/masked_alloc.pt --tag <name>
"""
import argparse, glob, os, sys, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/stage2_twophase")
from models import FadingChannel
from conv_codec import ConvWZCodec
from vae_wrapper import VAEWrapper
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--src", default="outputs/online_ft/old_K_unf2_20dB_k4_100ep_56"); ap.add_argument("--k", type=int, default=4)
ap.add_argument("--snr", type=float, default=20.0); ap.add_argument("--recv_thr", type=float, default=None)
a = ap.parse_args()
dev = "cuda:0"; DST = f"outputs/diag_policy/replay/{a.tag}"; os.makedirs(DST, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu"); c = ck["cfg"]
net = ConvWZCodec(4, c["conv_w"], c["conv_dch"], c["conv_nb"], residual=bool(c.get("residual", False)), mask_cond=bool(c.get("mask_cond", False)), snr_cond=bool(c.get("snr_cond", False))).to(dev).eval()
net.load_state_dict(ck["net"]); thr = c["recv_thr"] if a.recv_thr is None else a.recv_thr
NOGATE = c.get("mask_mode") == "none"            # full-z encoder + residual decoder: z~ = rec = z_hat + dec([s~, z_hat]), no mask, no gate
ch = FadingChannel(a.snr, channel_type="cdl", cdl_kwargs=dict(model="C", carrier_frequency=3.5e9, max_speed=3.0, delay_spread=50e-9))
vae = VAEWrapper("stabilityai/sd-vae-ft-mse").to(dev).eval()
u8 = lambda x: (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
def cubecells(im):
    R, G, B = (im[..., i].astype(int) for i in range(3)); b = (B > 120) & (R < 60) & (B - G > 50); return b.reshape(28, 8, 28, 8).sum((1, 3)) >= 4
fs = [f for f in sorted(glob.glob(a.src + "/*.npz")) if int(os.path.basename(f).split("_")[0]) <= 30]
torch.manual_seed(0); EC, EF, n = [], [], 0
with torch.no_grad():
    for f in fs:
        x = dict(np.load(f))
        if "lat_zh" not in x: continue
        zt, zh = (torch.from_numpy(x[k].astype(np.float32))[None].to(dev) for k in ("lat_zt", "lat_zh"))
        if c.get("no_side"): zh = torch.zeros_like(zh)          # control codec: no side information anywhere at the receiver
        m = torch.from_numpy(x["lat_mask"].astype(np.float32))[None, None].to(dev)
        if NOGATE: m = torch.ones_like(m)
        mu, _ = net.encode(zt * m); mp, _ = net.prior(zh); B = 1; k = a.k
        st = ch(mu[:, :k].flatten(1)).view(B, k, 28, 56)
        rec = net.decode(torch.cat([st, mp[:, k:]], 1), zh if c.get("residual") else None, m if c.get("mask_cond") else None, a.snr)
        g = torch.ones_like(m) if NOGATE else m * (((rec - zh) ** 2).mean(1, keepdim=True) > thr).float(); zz = g * rec + (1 - g) * zh
        y = dict(x); y["out1"], y["out2"] = u8(vae.decode(zz[..., :28])), u8(vae.decode(zz[..., 28:]))
        y["lat_ztilde"] = zz[0].half().cpu().numpy(); y["lat_rec"] = rec[0].half().cpu().numpy(); y["lat_gate"] = g[0, 0].cpu().numpy().astype(np.uint8)
        np.savez(f"{DST}/{os.path.basename(f)}", **y)
        e = ((zz - zt) ** 2).mean(1)[0].cpu().numpy(); EF.append(e.mean())
        for cam, off in (("1", 0), ("2", 28)):
            cc = cubecells(x["obs" + cam])
            if cc.any(): EC += list(e[:, off:off + 28][cc])
        n += 1
EC = np.array(EC)
print(f"[replay {a.tag}] {n} requests  frame z~ err {np.mean(EF):.4f}  cube-cell err mean {EC.mean():.4f} median {np.median(EC):.4f} frac>0.1 {np.mean(EC > 0.1):.3f}  -> {DST}")
