"""Wyner-Ziv conv codec: restores L = distortion + beta * KL(q(s|z_t) || p(s|z_hat)).

Combines the two threads: the conv codec that beats zhat for the VLA
([[project_jscc_codec_overfits]]) + the original WZ rate term. The per-channel KL against
a side-info prior built from z_hat concentrates rate on the unpredictable cube (q != p) and
vanishes on predictable regions (q ~= p) -- the "make the encoder concentrate" goal, done by
backprop within the WZ loss (no teacher, no change-mask weighting).

  encoder (WZ-blind):  z_t*m -> (mu_e, logvar_e)                        (B,Dch,H,W)
  prior   (receiver):  z_hat -> (mu_p, logvar_p)      = p(s~|z_hat)
  channel:             s~ = channel(reparam(mu_e,logvar_e)),  q_var = sigma_e^2 + sigma_n^2
  nested dropout:      random level k; channels >= k are NOT sent -> filled with mu_p at the
                       receiver (WZ-legal: receiver has z_hat). Trains any-prefix decoding and
                       the prior fallback used at deploy.
  rate (transmitted):  sum over the sent channels of KL(q||p)
  loss:                dist-weighted cube MSE(rec, z_t)  +  beta * rate

Deploy (decode_arms_alloc.py, arch=conv_wz): router picks k from the WZ-clean change-cell
count (the transmitter cannot compute KL -- it has no z_hat); the receiver fills channels>=k
with mu_p. beta sets the rate-distortion operating point at train time.

    python experiments/stage2_twophase/train_masked_alloc_wz.py \
        --pre outputs/rollout_disturbed_clean_s2_moe --out outputs/masked_alloc_wz_b1e-4 \
        --beta 1e-4 --thr 0.05 --recv_thr 0.10 --epochs 45
"""
import argparse, glob, os, pathlib, sys
import torch

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                            # noqa: E402
from models import FadingChannel                      # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conv_codec import ConvWZCodec                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=45)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--snr", type=float, default=20.0)
ap.add_argument("--conv_w", type=int, default=96)
ap.add_argument("--conv_dch", type=int, default=16)
ap.add_argument("--conv_nb", type=int, default=3)
ap.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8, 16])
ap.add_argument("--thr", type=float, default=0.05)
ap.add_argument("--recv_thr", type=float, default=0.10)
ap.add_argument("--cube_w", type=float, default=4.0)
ap.add_argument("--dist_weight", type=float, default=15.0)
ap.add_argument("--beta", type=float, default=1e-4, help="weight on the WZ rate term (per-frame summed KL)")
a = ap.parse_args()
DEV = "cuda:0"; os.makedirs(a.out, exist_ok=True)
SIG_N = 1.0 / (10.0 ** (a.snr / 10.0))
DCH = a.conv_dch

ZT = []; ZP = []; ZH = []; DIST = []
for f in sorted(glob.glob(os.path.join(a.pre, "shard*.pt"))):
    d = torch.load(f)
    assert "ZPREV" in d, "need ZPREV cache (rollout_disturbed_clean_s2_moe)"
    ZT.append(d["ZT"].float()); ZP.append(d["ZPREV"].float()); ZH.append(d["ZH"].float())
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
ZT = torch.cat(ZT).reshape(-1, 4, 28, 28); ZP = torch.cat(ZP).reshape(-1, 4, 28, 28)
ZH = torch.cat(ZH).reshape(-1, 4, 28, 28); DIST = torch.cat(DIST); N = len(ZT)
print(f"[data] {N} frames disturbed={int(DIST.sum())}  beta={a.beta} levels={a.levels} sig_n={SIG_N:.4f}", flush=True)


def change_mask(z, zp):
    return (((z - zp) ** 2).mean(1, keepdim=True) > a.thr).float()


def cube_region(z, h):
    return (((z - h) ** 2).mean(1, keepdim=True) > a.recv_thr).float()


net = ConvWZCodec(4, a.conv_w, a.conv_dch, a.conv_nb).to(DEV)
ch = FadingChannel(a.snr, channel_type="awgn")
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
va = perm[:int(N * 0.1)]; tr = perm[int(N * 0.1):]
LV = torch.tensor(a.levels, device=DEV)


def run(z, zp, h, k, sample):
    """Encode z*m, transmit first k channels, fill the rest with the prior mean, decode."""
    m = change_mask(z, zp)
    mu_e, lv_e = net.encode(z * m)
    mu_p, lv_p = net.prior(h)
    s = mu_e + torch.randn_like(mu_e) * (0.5 * lv_e).exp() if sample else mu_e
    B, Dc, H, W = s.shape
    s_t = ch(s.flatten(1)).view(B, Dc, H, W)
    keep = (torch.arange(DCH, device=DEV) < k.view(-1, 1)).float()[:, :, None, None]   # (B,Dch,1,1)
    s_mix = keep * s_t + (1 - keep) * mu_p                                              # fill dropped channels w/ prior
    rec = net.decode(s_mix)
    kl = net.kl_elem(mu_e, lv_e, mu_p, lv_p, SIG_N)                                     # (B,Dch,H,W)
    return rec, m, mu_p, kl, keep


@torch.no_grad()
def evaluate():
    net.eval(); out = {}
    for lv in a.levels:
        nc = dc = nnc = nnd = 0.0
        for i in range(0, len(va), 256):
            sel = va[i:i + 256]; z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); h = ZH[sel].to(DEV); dm = DIST[sel].to(DEV)
            rec, _, _, _, _ = run(z, zp, h, torch.full((len(z),), lv, device=DEV), sample=False)
            o = cube_region(z, h); wo = o.expand_as(z); err = ((rec - z) ** 2) * wo
            cm = (dm < 0.5); dmk = (dm > 0.5)
            nc += err[cm].sum().item(); nnc += wo[cm].sum().item()
            dc += err[dmk].sum().item(); nnd += wo[dmk].sum().item()
        out[lv] = (nc / max(nnc, 1), dc / max(nnd, 1))
    # per-channel mean KL (concentration signal), disturbed frames
    kacc = torch.zeros(DCH, device=DEV); n = 0
    for i in range(0, len(va), 256):
        sel = va[i:i + 256]; z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); h = ZH[sel].to(DEV); dm = DIST[sel].to(DEV)
        d = dm > 0.5
        if d.sum() == 0: continue
        _, _, _, kl, _ = run(z[d], zp[d], h[d], torch.full((int(d.sum()),), DCH, device=DEV), sample=False)
        kacc += kl.sum(dim=(0, 2, 3)); n += int(d.sum())
    return out, (kacc / max(n, 1)).cpu()


for e in range(1, a.epochs + 1):
    net.train(); order = tr[torch.randperm(len(tr))]; L = 0.0; RA = 0.0; nb = 0
    for i in range(0, len(order), a.bs):
        sel = order[i:i + a.bs]
        z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); h = ZH[sel].to(DEV); dm = DIST[sel].to(DEV)
        k = LV[torch.randint(0, len(a.levels), (len(z),), device=DEV)]                  # nested dropout level
        rec, m, mu_p, kl, keep = run(z, zp, h, k, sample=True)
        o = cube_region(z, h); w = m + a.cube_w * o
        per = (((rec - z) ** 2) * w.expand_as(z)).sum(dim=(1, 2, 3)) / w.expand_as(z).sum(dim=(1, 2, 3)).clamp(min=1)
        fw = 1.0 + a.dist_weight * dm
        distortion = (fw * per).sum() / fw.sum()
        rate = (keep * kl).sum(dim=(1, 2, 3)).mean()                                    # transmitted WZ rate (nats/frame)
        loss = distortion + a.beta * rate
        opt.zero_grad(); loss.backward(); opt.step()
        L += distortion.item(); RA += rate.item(); nb += 1
    rd, kch = evaluate()
    curve = "  ".join(f"k{lv}:c{rd[lv][0]:.3f}/d{rd[lv][1]:.3f}" for lv in a.levels)
    ktop = "/".join(f"{v:.0f}" for v in kch[:8])
    print(f"ep{e:>2}/{a.epochs} dist {L/nb:.4f} rate {RA/nb:.1f}  val cube {curve}  KL/ch[0:8] {ktop}", flush=True)
    torch.save({"net": net.state_dict(),
                "cfg": {"arch": "conv_wz", "conv_w": a.conv_w, "conv_dch": a.conv_dch, "conv_nb": a.conv_nb,
                        "levels": a.levels, "thr": a.thr, "recv_thr": a.recv_thr, "snr": a.snr, "beta": a.beta}},
               os.path.join(a.out, "masked_alloc.pt"))
print("DONE", flush=True)
