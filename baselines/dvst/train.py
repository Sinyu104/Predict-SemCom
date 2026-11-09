"""Train the DVST baseline on the VAE-latent cache (progressive strategy, paper Sec. III-B).

  phase 1 (--pre_epochs): nonlinear transform + entropy model only, uniform-noise quantisation, no channel
          L = D(z_tr, z) + lam_r * (bits_y + bits_z) / source_reals                    (Eq. 20/21)
  phase 2 (--epochs):     full pipeline with the channel and the rate allocator
          L = D(z_hat, z) + lam * CBR + D(z_tr, z)                                     (Eq. 23/24)
  CBR = sum_i k_{t,i} / (C_vae*H*W)  -- reals sent per source real. lam trades CBR against distortion.

  python baselines/dvst/train.py --pre outputs/rollout_disturbed_clean_s2_moe \
      --out outputs/dvst_lam0.1 --lam 0.1 --snr_min 0 --snr_max 20 --channel awgn
Run in the `predictor` env (torch only; no diffusers needed).
"""
import argparse, os, pathlib, sys, torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from model import DVST, MaskedChannel          # noqa: E402
from data import load_cache, split              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True, nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--pre_epochs", type=int, default=10)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--lam", type=float, default=0.1, help="weight on CBR (phase 2); sweep to trace the RD curve")
ap.add_argument("--lam_r", type=float, default=0.02, help="weight on bits/real in phase 1")
ap.add_argument("--eta_prime", type=float, default=1.0, help="JSCC efficiency factor eta' (Eq. 10)")
ap.add_argument("--ref", default="mix", choices=["clean", "recon", "mix"], help="previous-frame reference during training")
ap.add_argument("--p_intra", type=float, default=0.1, help="prob. of training a frame with no reference (I-frame mode)")
ap.add_argument("--snr", type=float, default=20.0)
ap.add_argument("--snr_min", type=float, default=None); ap.add_argument("--snr_max", type=float, default=None)
ap.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh", "cdl"])
ap.add_argument("--cdl_model", default="C"); ap.add_argument("--cdl_delay_spread", type=float, default=100e-9)
ap.add_argument("--cdl_carrier_freq", type=float, default=3.5e9); ap.add_argument("--cdl_max_speed", type=float, default=3.0)
ap.add_argument("--cy", type=int, default=32); ap.add_argument("--cs", type=int, default=16)
ap.add_argument("--w", type=int, default=96); ap.add_argument("--nb", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
DEV = "cuda:0"  # pick the GPU with CUDA_VISIBLE_DEVICES; os.makedirs(a.out, exist_ok=True)

ZT, ZP, DIST = load_cache(a.pre)
tr, va = split(len(ZT))
SRC_REALS = ZT.shape[1] * ZT.shape[2] * ZT.shape[3]
net = DVST(ZT.shape[1], a.cy, a.cs, a.w, a.nb, eta_prime=a.eta_prime).to(DEV)
ch = MaskedChannel(a.snr, a.channel, {"model": a.cdl_model, "delay_spread": a.cdl_delay_spread,
                   "carrier_frequency": a.cdl_carrier_freq, "max_speed": a.cdl_max_speed}).to(DEV)
RANDOM_SNR = a.snr_min is not None and a.snr_max is not None
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
total = (a.pre_epochs + a.epochs) * ((len(tr) + a.bs - 1) // a.bs)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
print(f"[data] N={len(ZT)} train={len(tr)} val={len(va)} disturbed={int(DIST.sum())} lam={a.lam}", flush=True)


def refs(z_prev, phase2):
    """(ref_z, ref_y) for a batch; None => intra."""
    if torch.rand(()) < a.p_intra:
        return None, None
    rz, ry = net.intra_reference(z_prev)
    if a.ref == "clean" or (a.ref == "mix" and torch.rand(()) < 0.5):
        rz = z_prev
    return rz, ry


def step(z, zp, dm, phase2, snr):
    ch.snr_db = snr
    rz, ry = refs(zp, phase2)
    o = net(z, rz, ry, ch, snr)
    d_tr = ((o["z_tr"] - z) ** 2).mean()
    cbr = o["k"].sum(dim=(1, 2, 3)).mean() / SRC_REALS
    bits = (o["bits"].sum(dim=(1, 2, 3)) + o["zbits"]).mean() / SRC_REALS
    if phase2:
        d = ((o["z_hat"] - z) ** 2).mean()
        loss = d + a.lam * cbr + d_tr
    else:
        d = d_tr; loss = d_tr + a.lam_r * bits
    return loss, d, cbr, bits


@torch.no_grad()
def evaluate(snr):
    net.eval(); acc = {"all": [0, 0], "calm": [0, 0], "dist": [0, 0]}; cb = 0.0; n = 0
    for i in range(0, len(va), 256):
        sel = va[i:i + 256]; z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); dm = DIST[sel].to(DEV)
        ch.snr_db = snr; rz, ry = net.intra_reference(zp)
        o = net(z, rz, ry, ch, snr)
        e = ((o["z_hat"] - z) ** 2).mean(dim=(1, 2, 3))
        for key, msk in (("all", torch.ones_like(dm) > 0), ("calm", dm < 0.5), ("dist", dm > 0.5)):
            acc[key][0] += e[msk].sum().item(); acc[key][1] += int(msk.sum())
        cb += o["k_q"].sum(dim=(1, 2, 3)).sum().item() / SRC_REALS; n += len(z)
    net.train()
    return {k: v[0] / max(v[1], 1) for k, v in acc.items()}, cb / n


best = float("inf")
for e in range(1, a.pre_epochs + a.epochs + 1):
    phase2 = e > a.pre_epochs
    net.entropy.ste = phase2                       # round-STE once the rate allocator is live (train/test rate match)
    net.train(); order = tr[torch.randperm(len(tr))]; L = D = C = 0.0; nb = 0
    for i in range(0, len(order), a.bs):
        sel = order[i:i + a.bs]
        z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); dm = DIST[sel].to(DEV)
        snr = float(torch.empty(()).uniform_(a.snr_min, a.snr_max)) if RANDOM_SNR else a.snr
        loss, d, cbr, bits = step(z, zp, dm, phase2, snr)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); L += loss.item(); D += d.item(); C += cbr.item() if phase2 else bits.item(); nb += 1
    if phase2:
        mse, cb = evaluate(a.snr)
        print(f"ep{e:>3} [P2] loss {L/nb:.4f} D {D/nb:.4f} cbr {C/nb:.3f} | val@{a.snr:g}dB mse all {mse['all']:.4f} calm {mse['calm']:.4f} dist {mse['dist']:.4f} cbr {cb:.3f}", flush=True)
        score = mse["all"] + a.lam * cb
        ck = {"net": net.state_dict(), "cfg": {"arch": "dvst", "cy": a.cy, "cs": a.cs, "w": a.w, "nb": a.nb, "C_vae": ZT.shape[1],
              "eta_prime": a.eta_prime, "lam": a.lam, "snr": a.snr, "snr_range": [a.snr_min, a.snr_max], "channel": a.channel}}
        torch.save(ck, os.path.join(a.out, "dvst_last.pt"))
        if score < best:
            best = score; torch.save(ck, os.path.join(a.out, "dvst.pt")); print(f"  [best] ep{e} {score:.4f}", flush=True)
    else:
        print(f"ep{e:>3} [P1] loss {L/nb:.4f} D {D/nb:.4f} bits/real {C/nb:.3f}", flush=True)
print("DONE", flush=True)
