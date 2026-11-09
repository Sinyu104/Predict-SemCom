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
ap.add_argument("--pre", required=True, nargs="+", help="one or more cache dirs (shard*.pt each); frames are concatenated")
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
ap.add_argument("--fisher_w", type=float, default=0.0, help="if >0: add fisher_w * F/mean(F) to the training weight (F from fisher_shard*.pt, experiments/fisher_weights.py); trains only on frames that have F")
ap.add_argument("--fisher_only", action="store_true", help="loss weight = F/mean(F) + fisher_eps only (no m, o, cube_w): the second-order task loss with damping; per-frame loss is the cell MEAN (frames the policy is more sensitive to count more)")
ap.add_argument("--fisher_eps", type=float, default=0.05)
ap.add_argument("--fisher_cap_q", type=float, default=0.999, help="cap F at this quantile (variance reduction of the one-sample Fisher estimate; 18%% of F mass sits above p99.9); 1 = no cap")
ap.add_argument("--obj_w", type=float, default=0.0, help="if >0: add obj_w * (object/cube cell mask) to the training weight; mask from obj_<shard>.pt (experiments/make_obj_mask.py)")
ap.add_argument("--no_m_loss", action="store_true", help="loss weight w = cube_w*o only (drop the m term)")
ap.add_argument("--soft_cube", type=float, default=0.0, help="if >0: training weight o = mean_c(z-zhat)^2 / soft_cube (continuous, no threshold) instead of the 0/1 cube_region indicator")
ap.add_argument("--dist_weight", type=float, default=15.0)
ap.add_argument("--cam", type=int, default=0, help="2-view caches (N,4,28,56): 0 = ONE joint codec on the whole width-concat grid (both views in, both out); 1 / 2 = base / wrist half only")
ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip (0 = off)")
ap.add_argument("--beta", type=float, default=1e-4, help="weight on the WZ rate term (per-frame summed KL)")
ap.add_argument("--residual", type=int, default=0, help="1: decoder predicts a correction to z_hat (rec = z_hat + dec([s~, z_hat]))")
ap.add_argument("--grad_w", type=float, default=0.0, help="weight of the spatial-gradient (sharpness) loss on the weighted region")
ap.add_argument("--mask_mode", default="prev", choices=["prev", "pred", "net", "none"], help="prev: m = 1[|z_t - z_prev|^2 > thr]; pred: m = 1[|z_t - z_hat|^2 > thr] (transmitter needs z_hat or a proxy); none: m = 1 (encode the full z_t; with --residual 1 the decoder output is z~, no gate)")
ap.add_argument("--no_side", action="store_true", help="control without side information: the prior and the decoder get zeros instead of z_hat (plain JSCC at the same rate); z_hat is still used for the training-only loss weight o")
ap.add_argument("--snr_cond", type=int, default=0, help="1: the decoder also gets the channel SNR (receiver CSI) as an input map snr_db/20 (zero-init when warm-started)")
ap.add_argument("--mask_cond", type=int, default=0, help="1: the decoder also gets the change mask m as an input channel (zero-init when warm-started)")
ap.add_argument("--sparse", action="store_true", help="transmit only the cells with m=1 (the rest = prior mean mu_p); channel noise set by the power of the SENT cells; KL counts sent cells only")
ap.add_argument("--mask_ckpt", default=None, help="mask_mode net: frozen MaskNet checkpoint (net, tau)")
ap.add_argument("--init", default=None, help="warm start from this checkpoint (same arch)")
ap.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh", "cdl"])
ap.add_argument("--snr_min", type=float, default=None, help="with --snr_max: SNR (dB) drawn uniformly per batch in training; eval averages 0/10/20 dB")
ap.add_argument("--snr_max", type=float, default=None)
ap.add_argument("--kl_snr", type=float, default=None, help="SNR (dB) assumed in the KL noise term; default = the SNR of the batch")
ap.add_argument("--digital_bits", type=int, default=0, help=">0: uncoded digital link (b-bit quantiser -> Gray -> QAM -> channel -> hard demap), straight-through gradient")
ap.add_argument("--digital_nbps", type=int, default=2, help="bits per QAM symbol for --digital_bits (2=QPSK, 4=16-QAM, 6=64-QAM)")
ap.add_argument("--cdl_model", default="C"); ap.add_argument("--cdl_delay_spread", type=float, default=100e-9)
ap.add_argument("--cdl_carrier_freq", type=float, default=3.5e9); ap.add_argument("--cdl_max_speed", type=float, default=3.0)
a = ap.parse_args()
DEV = "cuda:0"; os.makedirs(a.out, exist_ok=True)
SIG_N = 1.0 / (10.0 ** (a.snr / 10.0))
DCH = a.conv_dch

ZT = []; ZP = []; ZH = []; DIST = []; OBJ = []
SRC = []
for f in sorted(f for d_ in a.pre for f in glob.glob(os.path.join(d_, "shard*.pt"))):
    d = torch.load(f)
    assert "ZPREV" in d, "need ZPREV cache (rollout_disturbed_clean_s2_moe)"
    ZT.append(d["ZT"].float()); ZP.append(d["ZPREV"].float()); ZH.append(d["ZH"].float())
    if a.obj_w > 0: OBJ.append(torch.load(os.path.join(os.path.dirname(f), "obj_" + os.path.basename(f)))["OBJ"].float())
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
    SRC.append(torch.full((d["ZT"].shape[0],), a.pre.index(os.path.dirname(f)) if os.path.dirname(f) in a.pre else 0))
def _view(x):
    x = torch.cat(x)
    if x.dim() == 4 and x.shape[-1] == 56 and a.cam in (1, 2):    # per-camera codec: pick its half
        x = x[..., :28] if a.cam == 1 else x[..., 28:]
    return x.reshape(-1, 4, 28, x.shape[-1] if x.dim() == 4 else 28).contiguous()
ZT = _view(ZT); ZP = _view(ZP)
ZH = _view(ZH); DIST = torch.cat(DIST); N = len(ZT)
OBJ = torch.cat(OBJ) if a.obj_w > 0 else None
FISH = None
if a.fisher_w > 0 or a.fisher_only:
    FISH = torch.full((N, 1, 28, 56), float("nan")); off_ = 0
    for f in sorted(f for d_ in a.pre for f in glob.glob(os.path.join(d_, "shard*.pt"))):
        n_ = torch.load(f)["ZT"].shape[0]; si = os.path.basename(f)[5:-3]
        for g_ in glob.glob(os.path.join(os.path.dirname(f), f"fisher_shard{si}_p*.pt")):
            fd = torch.load(g_)
            if not g_.endswith(("_lim3.pt",)): FISH[off_ + fd["idx"]] = fd["F"].float()
        off_ += n_
    HAS = ~torch.isnan(FISH[:, 0, 0, 0]); FISH[~HAS] = 0
    if a.fisher_cap_q < 1:
        fl_ = FISH[HAS].flatten(); cap_ = torch.quantile(fl_[torch.randperm(len(fl_), generator=torch.Generator().manual_seed(0))[:4000000]], a.fisher_cap_q)
        FISH = FISH.clamp(max=cap_)
    FISH = FISH / FISH[HAS].mean(); FISH[~HAS] = 0
    print(f"[fisher] {int(HAS.sum())}/{N} frames have F; top-5% cells hold {torch.topk(FISH[HAS].flatten(1), 78).values.sum(1).div(FISH[HAS].flatten(1).sum(1)).mean():.2f} of each frame's F", flush=True)
if OBJ is not None: assert OBJ.shape[0] == N and a.cam not in (1, 2); print(f"[obj] object cells {OBJ.mean().item():.4f} of grid, obj_w {a.obj_w}", flush=True)
SRC = torch.cat(SRC)
print("[data] per --pre dir:", {d_: int((SRC == i).sum()) for i, d_ in enumerate(a.pre)}, flush=True)
print(f"[data] {N} frames disturbed={int(DIST.sum())}  beta={a.beta} levels={a.levels} sig_n={SIG_N:.4f}", flush=True)


MASKNET = None
if a.mask_mode == "net":
    from conv_codec import MaskNet
    _mk = torch.load(a.mask_ckpt, map_location=DEV); MASKNET = MaskNet().to(DEV).eval(); MASKNET.load_state_dict(_mk["net"]); MASK_TAU = _mk["tau"]
    for p_ in MASKNET.parameters(): p_.requires_grad_(False)
    print(f"[mask] frozen MaskNet {a.mask_ckpt} tau={MASK_TAU:.4f}", flush=True)


def change_mask(z, zp, h=None):
    if MASKNET is not None:
        with torch.no_grad(): return (torch.sigmoid(MASKNET(z)) > MASK_TAU).float()
    if a.mask_mode == "none": return torch.ones_like(z[:, :1])
    if a.mask_mode == "pred": zp = h
    return (((z - zp) ** 2).mean(1, keepdim=True) > a.thr).float()


def cube_region(z, h):
    return (((z - h) ** 2).mean(1, keepdim=True) > a.recv_thr).float()


net = ConvWZCodec(4, a.conv_w, a.conv_dch, a.conv_nb, residual=bool(a.residual), mask_cond=bool(a.mask_cond), snr_cond=bool(a.snr_cond)).to(DEV)
ch = FadingChannel(a.snr, channel_type=a.channel, cdl_kwargs={"model": a.cdl_model, "delay_spread": a.cdl_delay_spread,
                   "carrier_frequency": a.cdl_carrier_freq, "max_speed": a.cdl_max_speed})
if a.digital_bits > 0:
    from digital_channel import DigitalQAMChannel
    ch = DigitalQAMChannel(a.snr, a.digital_bits, a.digital_nbps, a.channel, {"model": a.cdl_model, "delay_spread": a.cdl_delay_spread,
                           "carrier_frequency": a.cdl_carrier_freq, "max_speed": a.cdl_max_speed}, group=28 * 56, device=DEV)
if a.init:
    sd = torch.load(a.init, map_location=DEV)["net"]
    if a.residual or a.mask_cond or a.snr_cond:       # decoder input grows by the 4 z_hat channels: zero-init them and the output conv, so training starts from rec = z_hat
        w0 = net.dec[0].weight.data.clone(); w0.zero_(); w0[:, :a.conv_dch] = sd["dec.0.weight"]; sd["dec.0.weight"] = w0
        if a.residual:
            last = max(int(k.split(".")[1]) for k in sd if k.startswith("dec."))
            sd[f"dec.{last}.weight"] = torch.zeros_like(sd[f"dec.{last}.weight"]); sd[f"dec.{last}.bias"] = torch.zeros_like(sd[f"dec.{last}.bias"])
    net.load_state_dict(sd); print("[init]", a.init, flush=True)
RANDOM_SNR = a.snr_min is not None and a.snr_max is not None
EVAL_SNRS = ([10.0, 15.0, 20.0] if a.digital_bits > 0 else [0.0, 10.0, 20.0]) if RANDOM_SNR else [a.snr]
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
va = perm[:int(N * 0.1)]; tr = perm[int(N * 0.1):]
if FISH is not None: va = va[HAS[va]]; tr = tr[HAS[tr]]; print(f"[fisher] train {len(tr)} / val {len(va)} frames", flush=True)
LV = torch.tensor(a.levels, device=DEV)
steps_per_ep = (len(tr) + a.bs - 1) // a.bs
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * steps_per_ep)
best = float("inf")


def run(z, zp, h, k, sample, snr=None):
    """Encode z*m, transmit first k channels, fill the rest with the prior mean, decode."""
    m = change_mask(z, zp, h)
    if a.no_side: h = torch.zeros_like(h)
    mu_e, lv_e = net.encode(z * m)
    mu_p, lv_p = net.prior(h)
    s = mu_e + torch.randn_like(mu_e) * (0.5 * lv_e).exp() if sample else mu_e
    snr = a.snr if snr is None else snr
    ch.snr_db = snr; sig_n = 1.0 / (10.0 ** ((snr if a.kl_snr is None else a.kl_snr) / 10.0))                              # nominal noise; fading adds sigma_n^2/|h|^2 on top
    B, Dc, H, W = s.shape
    # noise level is set by the power of the SENT channels only (as at deployment: wz_channel(mu_e[:, :k])), one group per level
    s_t = torch.zeros_like(s)
    for lv in k.unique().tolist():
        sel = (k == lv).nonzero(as_tuple=True)[0]
        x_tx = s[sel, :int(lv)].flatten(1)
        if a.sparse:      # only m=1 cells are sent; unsent slots get dummy symbols of the sent power so the channel noise level (relative to the vector power) is right; they are discarded below
            mx = m[sel].expand(-1, int(lv), -1, -1).flatten(1)
            pw = (x_tx.detach() ** 2 * mx).sum(1, keepdim=True) / mx.sum(1, keepdim=True).clamp(min=1)
            x_in = torch.where(mx > 0, x_tx, torch.randn_like(x_tx) * pw.sqrt())
            x_rx = ch(x_in)
        else:
            x_rx = ch(x_tx)
        if a.digital_bits > 0:
            x_rx = x_tx + (x_rx - x_tx).detach()                                      # straight-through: quantiser + bit errors are non-differentiable
        s_t[sel, :int(lv)] = x_rx.view(len(sel), int(lv), H, W)
    keep = (torch.arange(DCH, device=DEV) < k.view(-1, 1)).float()[:, :, None, None]   # (B,Dch,1,1)
    if a.sparse: keep = keep * m                                                        # (B,Dch,H,W): sent = first k channels AND m=1
    s_mix = keep * s_t + (1 - keep) * mu_p                                              # fill dropped channels w/ prior
    rec = net.decode(s_mix, h, m, snr)
    kl = net.kl_elem(mu_e, lv_e, mu_p, lv_p, sig_n)                                     # (B,Dch,H,W)
    return rec, m, mu_p, kl, keep


@torch.no_grad()
def _evaluate(snr):
    net.eval(); out = {}
    for lv in a.levels:
        nc = dc = nnc = nnd = 0.0
        for i in range(0, len(va), 256):
            sel = va[i:i + 256]; z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); h = ZH[sel].to(DEV); dm = DIST[sel].to(DEV)
            rec, _, _, _, _ = run(z, zp, h, torch.full((len(z),), lv, device=DEV), sample=False, snr=snr)
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
        _, _, _, kl, _ = run(z[d], zp[d], h[d], torch.full((int(d.sum()),), DCH, device=DEV), sample=False, snr=snr)
        kacc += kl.sum(dim=(0, 2, 3)); n += int(d.sum())
    return out, (kacc / max(n, 1)).cpu()


@torch.no_grad()
def evaluate():
    rs = [_evaluate(sn) for sn in EVAL_SNRS]                  # random-SNR training: average the val error over EVAL_SNRS
    out = {lv: tuple(sum(r[0][lv][j] for r in rs) / len(rs) for j in range(2)) for lv in a.levels}
    return out, sum(r[1] for r in rs) / len(rs)


for e in range(1, a.epochs + 1):
    net.train(); order = tr[torch.randperm(len(tr))]; L = 0.0; RA = 0.0; nb = 0
    for i in range(0, len(order), a.bs):
        sel = order[i:i + a.bs]
        z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); h = ZH[sel].to(DEV); dm = DIST[sel].to(DEV)
        k = LV[torch.randint(0, len(a.levels), (len(z),), device=DEV)]                  # nested dropout level
        snr = float(torch.empty(()).uniform_(a.snr_min, a.snr_max)) if RANDOM_SNR else a.snr
        rec, m, mu_p, kl, keep = run(z, zp, h, k, sample=True, snr=snr)
        o = (((z - h) ** 2).mean(1, keepdim=True) / a.soft_cube) if a.soft_cube > 0 else cube_region(z, h); w = (0 if a.no_m_loss else m) + a.cube_w * o
        if OBJ is not None: w = w + a.obj_w * OBJ[sel].to(DEV)
        if FISH is not None: w = (FISH[sel].to(DEV) + a.fisher_eps) if a.fisher_only else (w + a.fisher_w * FISH[sel].to(DEV))
        per = (((rec - z) ** 2) * w.expand_as(z)).sum(dim=(1, 2, 3)) / w.expand_as(z).sum(dim=(1, 2, 3)).clamp(min=1)
        if a.fisher_only: per = (((rec - z) ** 2) * w.expand_as(z)).mean(dim=(1, 2, 3))
        fw = 1.0 + a.dist_weight * dm
        distortion = (fw * per).sum() / fw.sum()
        if a.grad_w > 0:                                                                # sharpness: match the spatial gradients of z_t where the weight is on
            d_ = rec - z
            gx = (d_[..., :, 1:] - d_[..., :, :-1]) ** 2 * w[..., :, 1:]; gy = (d_[..., 1:, :] - d_[..., :-1, :]) ** 2 * w[..., 1:, :]
            gl = (gx.sum(dim=(1, 2, 3)) + gy.sum(dim=(1, 2, 3))) / (w.sum(dim=(1, 2, 3)).clamp(min=1) * 4)
            distortion = distortion + a.grad_w * (fw * gl).sum() / fw.sum()
        rate = (keep * kl).sum(dim=(1, 2, 3)).mean()                                    # transmitted WZ rate (nats/frame)
        loss = distortion + a.beta * rate
        opt.zero_grad(); loss.backward()
        if a.clip > 0: torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        opt.step(); sched.step()
        L += distortion.item(); RA += rate.item(); nb += 1
    rd, kch = evaluate()
    curve = "  ".join(f"k{lv}:c{rd[lv][0]:.3f}/d{rd[lv][1]:.3f}" for lv in a.levels)
    ktop = "/".join(f"{v:.0f}" for v in kch[:8])
    print(f"ep{e:>2}/{a.epochs} dist {L/nb:.4f} rate {RA/nb:.1f}  val cube {curve}  KL/ch[0:8] {ktop}", flush=True)
    ck = {"net": net.state_dict(),
          "cfg": {"arch": "conv_wz", "conv_w": a.conv_w, "conv_dch": a.conv_dch, "conv_nb": a.conv_nb,
                  "levels": a.levels, "thr": a.thr, "recv_thr": a.recv_thr, "snr": a.snr, "beta": a.beta, "channel": a.channel, "snr_range": [a.snr_min, a.snr_max],
                  "digital_bits": a.digital_bits, "digital_nbps": a.digital_nbps, "residual": bool(a.residual), "mask_mode": a.mask_mode, "mask_ckpt": a.mask_ckpt, "grad_w": a.grad_w, "sparse": bool(a.sparse), "mask_cond": bool(a.mask_cond), "snr_cond": bool(a.snr_cond), "no_side": bool(a.no_side), "obj_w": a.obj_w, "cube_w": a.cube_w, "fisher_w": a.fisher_w, "fisher_only": bool(a.fisher_only), "fisher_eps": a.fisher_eps, "fisher_cap_q": a.fisher_cap_q}}
    torch.save(ck, os.path.join(a.out, "masked_alloc_last.pt"))
    score = sum(rd[lv][0] + rd[lv][1] for lv in a.levels)        # val cube error, all levels (calm + disturbed)
    if score < best:                                              # best-by-val is the deployed ckpt
        best = score; torch.save(ck, os.path.join(a.out, "masked_alloc.pt")); print(f"  [best] ep{e} score {score:.4f}", flush=True)
print("DONE", flush=True)
