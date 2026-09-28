"""Change-masked allocation JSCC with a translation-equivariant CONV codec.

Transmitter (WZ-clean, uses z_{t-1}, NOT z_hat):
  m     = per-cell (||z_t - z_{t-1}||^2 > thr)     # changed region (cube + moving arm)
  z_in  = z_t * m                                  # mask the encoder input to the changed cells
  s     = conv_enc(z_in) -> channel -> s~          # spatial code, flattened only for the channel
  rec   = conv_dec(s~)                             # reconstruction on the transmitted region

Receiver (has z_hat -> WZ-legal disagreement composite, applied at DECODE):
  g     = m * (||rec - z_hat||^2 > recv_thr)       # keep z_hat where rec agrees (arm), use rec on the cube
  z~    = g * rec + (1 - g) * z_hat

Why conv, not the dense JsccEncoder: the dense codec MEMORIZES cube positions (train
0.02 / val 0.24) and shrinking it makes val worse. A fully-conv codec is translation
equivariant -> held-out cube MSE 0.24 -> 0.084 (W96). See project_jscc_codec_overfits.

Training supervises `rec` DIRECTLY against z_t on the transmitted region m (with the
cube region up-weighted), so rec is faithful on BOTH the arm (rec~=z_hat -> gate stays
off) and the cube (rec~=z_t -> gate fires). The hard gate is applied at inference only.

    python experiments/stage2_twophase/train_masked_alloc.py \
        --pre outputs/rollout_disturbed_clean_s2_moe --out outputs/masked_alloc_conv \
        --thr 0.05 --recv_thr 0.10 --conv_w 96 --conv_dch 16 --conv_nb 3 --epochs 40
"""
import argparse, glob, os, pathlib, sys
import torch

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from models import FadingChannel                 # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conv_codec import ConvCodec                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--snr", type=float, default=20.0)
ap.add_argument("--conv_w", type=int, default=96)
ap.add_argument("--conv_dch", type=int, default=16)
ap.add_argument("--conv_nb", type=int, default=3)
ap.add_argument("--thr", type=float, default=0.05, help="transmitter change-mask threshold (||z_t-z_{t-1}||^2 per cell)")
ap.add_argument("--recv_thr", type=float, default=0.10,
                help="RECEIVER disagreement threshold used at decode; also defines the cube region up-weighted in training")
ap.add_argument("--cube_w", type=float, default=4.0, help="extra weight on the cube (disagreement) region within m")
ap.add_argument("--dist_weight", type=float, default=15.0)
a = ap.parse_args()
DEV = "cuda:0"; os.makedirs(a.out, exist_ok=True)

ZT = []; ZP = []; ZH = []; DIST = []
for f in sorted(glob.glob(os.path.join(a.pre, "shard*.pt"))):
    d = torch.load(f)
    assert "ZPREV" in d, "need ZPREV cache (rollout_disturbed_clean_s2_moe)"
    ZT.append(d["ZT"].float()); ZP.append(d["ZPREV"].float()); ZH.append(d["ZH"].float())
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
ZT = torch.cat(ZT).reshape(-1, 4, 28, 28); ZP = torch.cat(ZP).reshape(-1, 4, 28, 28)
ZH = torch.cat(ZH).reshape(-1, 4, 28, 28); DIST = torch.cat(DIST); N = len(ZT)
print(f"[data] {N} frames disturbed={int(DIST.sum())}  thr={a.thr} recv_thr={a.recv_thr} "
      f"conv W{a.conv_w}/Dch{a.conv_dch}/nb{a.conv_nb}", flush=True)


def change_mask(z, zp):
    return (((z - zp) ** 2).mean(1, keepdim=True) > a.thr).float()          # transmitter (WZ-clean)


def cube_region(z, h):
    return (((z - h) ** 2).mean(1, keepdim=True) > a.recv_thr).float()      # disagreement w/ z_hat


net = ConvCodec(4, a.conv_w, a.conv_dch, a.conv_nb).to(DEV)
ch = FadingChannel(a.snr, channel_type="awgn")
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
va = perm[:int(N * 0.1)]; tr = perm[int(N * 0.1):]
print(f"[data] train {len(tr)}  val {len(va)}  params {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)


def cube_val():
    net.eval(); num = den = 0.0
    with torch.no_grad():
        for i in range(0, len(va), 256):
            sel = va[i:i + 256]; z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); h = ZH[sel].to(DEV)
            rec = net(z * change_mask(z, zp), ch)
            o = cube_region(z, h); wo = o.expand_as(z)
            num += (((rec - z) ** 2) * wo).sum().item(); den += wo.sum().item()
    return num / max(den, 1)


for e in range(1, a.epochs + 1):
    net.train(); order = tr[torch.randperm(len(tr))]; L = 0.0; nb = 0; msz = 0.0
    for i in range(0, len(order), a.bs):
        sel = order[i:i + a.bs]
        z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); h = ZH[sel].to(DEV); dm = DIST[sel].to(DEV)
        m = change_mask(z, zp); o = cube_region(z, h)
        rec = net(z * m, ch)
        # supervise rec directly on the transmitted region, cube up-weighted (arm keeps rec~=z_t~=z_hat -> gate off)
        w = m + a.cube_w * o                                  # (B,1,H,W)
        per = (((rec - z) ** 2) * w.expand_as(z)).sum(dim=(1, 2, 3)) / w.expand_as(z).sum(dim=(1, 2, 3)).clamp(min=1)
        fw = 1.0 + a.dist_weight * dm
        loss = (fw * per).sum() / fw.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        L += loss.item(); nb += 1; msz += m.mean().item()
    print(f"ep{e:>2}/{a.epochs}  loss {L/nb:.4f}  mask frac {msz/nb:.3f}  val cube MSE {cube_val():.4f}", flush=True)
    torch.save({"net": net.state_dict(),
                "cfg": {"arch": "conv", "conv_w": a.conv_w, "conv_dch": a.conv_dch, "conv_nb": a.conv_nb,
                        "thr": a.thr, "recv_thr": a.recv_thr, "snr": a.snr}},
               os.path.join(a.out, "masked_alloc.pt"))
print("DONE", flush=True)
