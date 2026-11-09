"""Fine-tune DVST with the paper's GOP training: unrolled I,P,..,P sequences, tx/rx reference mismatch (Eq. 2, step 5).

Why (see README): the single-pair trainer never saw its own accumulated errors nor the transmitter-simulation vs
receiver-actual reference split, so online P-frames drifted (x5 MSE per request).  Here a chain is unrolled:
  I-frame: intra (no context);  tx reference = noise-free simulated receiver output, rx reference = the actually
  received (channel-corrupted) reconstruction -- both carried to the next frame; no gradient from I to P (paper),
  gradients flow P->P through the unroll.   L = (1/(L+1)) sum_frames [ D(z_hat) + lam*CBR (+ D(z_tr) on P) ].

  python baselines/dvst/train_gop.py --pre outputs/rollout_disturbed_2cam outputs/rollout_calm_2cam \
      --init outputs/dvst_v2_cdl_10_20_lam0.1/dvst.pt --out outputs/dvst_gop_cdl_10_20 --snr_min 10 --snr_max 20 --snr 15
"""
import argparse, os, pathlib, sys, torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from model import DVST, MaskedChannel              # noqa: E402
from data import load_ordered, build_chains, chain_batch, split   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True, nargs="+"); ap.add_argument("--out", required=True)
ap.add_argument("--init", default=None, help="warm start from a single-pair DVST checkpoint (same arch)")
ap.add_argument("--L", type=int, default=4, help="P-frames per GOP (GOP size = L+1)")
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lam", type=float, default=0.1)
ap.add_argument("--snr", type=float, default=15.0); ap.add_argument("--snr_min", type=float, default=None); ap.add_argument("--snr_max", type=float, default=None)
ap.add_argument("--channel", default="cdl", choices=["awgn", "rayleigh", "cdl"])
ap.add_argument("--link_thr", type=float, default=0.15)
ap.add_argument("--cy", type=int, default=32); ap.add_argument("--cs", type=int, default=16); ap.add_argument("--w", type=int, default=96); ap.add_argument("--nb", type=int, default=3)
ap.add_argument("--chg_w", type=float, default=0.0, help="WZ-SC-style weighted distortion: w = 1 + chg_w*m, m = change mask vs the previous frame (P-frames only)")
ap.add_argument("--chg_thr", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); torch.manual_seed(a.seed); DEV = "cuda:0"; os.makedirs(a.out, exist_ok=True)

ZT, ZP, DIST, ORIG, DS = load_ordered(a.pre)
starts = build_chains(ZT, ZP, DS, a.L, thr=a.link_thr)
tr_rows, va_rows = split(len(ZT))                                     # the original row split (same seed as everything else)
is_va = torch.zeros(len(ZT), dtype=torch.bool); is_va[va_rows] = True
va_s = starts[is_va[ORIG[starts]]]; tr_s = starts[~is_va[ORIG[starts]]]
SRC = ZT.shape[1] * ZT.shape[2] * ZT.shape[3]
print(f"[data] rows {len(ZT)}  chains(L={a.L}) {len(starts)}  train {len(tr_s)}  val {len(va_s)}", flush=True)
assert len(tr_s) > 100, "too few chains; relax --link_thr or reduce --L"

net = DVST(ZT.shape[1], a.cy, a.cs, a.w, a.nb).to(DEV)
if a.init:
    ck0 = torch.load(a.init, map_location=DEV); net.load_state_dict(ck0["net"]); print("[init]", a.init, flush=True)
    eta = ck0["cfg"]["eta_prime"]; net.eta_prime = eta
net.entropy.ste = True
ch = MaskedChannel(a.snr, a.channel, {"model": "C", "delay_spread": 100e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0}).to(DEV)
RAND = a.snr_min is not None and a.snr_max is not None
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * ((len(tr_s) + a.bs - 1) // a.bs))


def run_gop(fr, snr):
    """fr: (B, L+1, C, H, W). Returns per-frame (mse, cbr) lists and the training loss."""
    ch.snr_db = snr
    mses, cbrs, loss = [], [], 0.0
    tx_ref = rx_ref = None
    for k in range(fr.shape[1]):
        z = fr[:, k]
        c_tx, d_tx = net.contexts(*tx_ref) if tx_ref is not None else (None, None)
        c_rx, d_rx = net.contexts(*rx_ref) if rx_ref is not None else (None, None)
        e = net.encode(z, c_tx, d_tx, snr)
        s_tx = e["s"] * e["keep"]
        y_sim, z_sim = net.decode(s_tx, e["idx"], c_tx, d_tx)                 # transmitter's noise-free simulation
        y_rx, z_rx = net.decode(ch(s_tx, e["keep"]), e["idx"], c_rx, d_rx)    # receiver, with the channel and ITS reference
        mse = ((z_rx - z) ** 2).mean(); cbr = e["k"].sum(dim=(1, 2, 3)).mean() / SRC
        if k > 0 and a.chg_w > 0:
            w = 1.0 + a.chg_w * (((z - fr[:, k - 1]) ** 2).mean(1, keepdim=True) > a.chg_thr).float()
            wmse = lambda r: (((r - z) ** 2).mean(1, keepdim=True) * w).sum() / w.sum()
        else:
            wmse = lambda r: ((r - z) ** 2).mean()
        l = wmse(z_rx) + a.lam * cbr
        if k > 0:
            z_tr = net.g_s(torch.cat([e["yb"], c_tx], 1)); l = l + wmse(z_tr)
        loss = loss + l
        mses.append(mse.item()); cbrs.append(e["k_q"].sum(dim=(1, 2, 3)).mean().item() / SRC)
        if k == 0:                                                            # no gradient from the I-frame into the P-frames
            tx_ref = (z_sim.detach(), y_sim.detach()); rx_ref = (z_rx.detach(), y_rx.detach())
        else:
            tx_ref = (z_sim, y_sim); rx_ref = (z_rx, y_rx)
    return loss / fr.shape[1], mses, cbrs


@torch.no_grad()
def evaluate(snr):
    net.eval(); M = torch.zeros(a.L + 1); R = torch.zeros(a.L + 1); n = 0
    for i in range(0, len(va_s), 64):
        fr = chain_batch(ZT, ZP, va_s[i:i + 64], a.L).to(DEV)
        _, m, r = run_gop(fr, snr); M += torch.tensor(m) * len(fr); R += torch.tensor(r) * len(fr); n += len(fr)
    net.train(); return M / n, R / n


best = float("inf")
for e in range(1, a.epochs + 1):
    net.train(); order = tr_s[torch.randperm(len(tr_s))]; Ls = 0.0; nb = 0
    for i in range(0, len(order), a.bs):
        fr = chain_batch(ZT, ZP, order[i:i + a.bs], a.L).to(DEV)
        snr = float(torch.empty(()).uniform_(a.snr_min, a.snr_max)) if RAND else a.snr
        loss, _, _ = run_gop(fr, snr)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        Ls += loss.item(); nb += 1
    M, R = evaluate(a.snr)
    print(f"ep{e:>3} loss {Ls/nb:.4f} | val@{a.snr:g}dB mse I,P1..P{a.L}: " + " ".join(f"{v:.4f}" for v in M) +
          " | cbr " + " ".join(f"{v:.2f}" for v in R), flush=True)
    ck = {"net": net.state_dict(), "cfg": {"arch": "dvst", "cy": a.cy, "cs": a.cs, "w": a.w, "nb": a.nb, "C_vae": ZT.shape[1],
          "eta_prime": net.eta_prime, "lam": a.lam, "snr": a.snr, "snr_range": [a.snr_min, a.snr_max], "channel": a.channel, "gop_L": a.L}}
    torch.save(ck, os.path.join(a.out, "dvst_last.pt"))
    score = M.mean().item() + a.lam * R.mean().item()
    if score < best:
        best = score; torch.save(ck, os.path.join(a.out, "dvst.pt")); print(f"  [best] ep{e} {score:.4f}", flush=True)
print("DONE", flush=True)
