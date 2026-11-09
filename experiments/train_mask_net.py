"""Train the transmitter-side MaskNet (z_t only) to predict o = 1[mean_c(z_t - z_hat)^2 > 0.05]; tau chosen so the train-set mask density is --dens.
    CUDA_VISIBLE_DEVICES=0 python experiments/train_mask_net.py --out outputs/wz2cam_retrain/mask_net.pt
"""
import argparse, glob, sys, torch, torch.nn as nn
sys.path.insert(0, "experiments/stage2_twophase")
from conv_codec import MaskNet
ap = argparse.ArgumentParser(); ap.add_argument("--pre", default="outputs/rollout_pose_unf4"); ap.add_argument("--out", required=True)
ap.add_argument("--thr", type=float, default=0.05); ap.add_argument("--dens", type=float, default=0.20); ap.add_argument("--epochs", type=int, default=30); a = ap.parse_args()
dev = "cuda"; D = [torch.load(f) for f in sorted(glob.glob(a.pre + "/shard*.pt"))]; cat = lambda k: torch.cat([d[k].float() for d in D])
ZT, ZH, HELD = cat("ZT"), cat("ZH"), cat("HELD").bool()
O = (((ZT - ZH) ** 2).mean(1, keepdim=True) > a.thr).float(); tr = (~HELD).nonzero().squeeze(1); te = HELD.nonzero().squeeze(1)
n = MaskNet().to(dev); opt = torch.optim.AdamW(n.parameters(), 1e-3, weight_decay=1e-4); bs = 64
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * ((len(tr) + bs - 1) // bs))
for e in range(a.epochs):
    p = tr[torch.randperm(len(tr))]
    for i in range(0, len(p), bs):
        s = p[i:i + bs]; l = nn.functional.binary_cross_entropy_with_logits(n(ZT[s].to(dev)), O[s].to(dev)); opt.zero_grad(); l.backward(); opt.step(); sch.step()
n.eval()
@torch.no_grad()
def P(idx): return torch.cat([torch.sigmoid(n(ZT[idx[i:i + 256]].to(dev))).cpu() for i in range(0, len(idx), 256)])
tau = torch.quantile(P(tr).flatten()[::7], 1 - a.dens).item(); m = (P(te) > tau).float(); o = O[te]
print(f"tau {tau:.4f}  test density {m.mean():.3f}  teacher {o.mean():.3f}  recall {(m*o).sum()/o.sum():.3f}  precision {(m*o).sum()/m.sum():.3f}")
torch.save({"net": n.state_dict(), "tau": tau, "thr": a.thr, "dens": a.dens}, a.out)
