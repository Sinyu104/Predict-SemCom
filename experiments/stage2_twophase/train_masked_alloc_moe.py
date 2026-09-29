"""Adaptive-rate conv codec, PHASE A: train a NESTED-CHANNEL conv code.

Same conv codec that beats zhat for the VLA (project_jscc_codec_overfits), trained so that
any PREFIX of its 16 code channels is a valid lower-rate code (nested dropout: each frame
sees a random level, channels beyond it zeroed). This decouples the codec from the router
-- joint training collapsed the gumbel router to a flat 8 channels and the codec only
learned that one length. With a robust nested codec, the router is a separate, simple
choice (phase B: threshold on the WZ-clean change statistic, or a small learned head).

Why nest over CHANNELS, not spatial cells: the conv codec's wide receptive field spreads
cube info across space, so a spatial subset needs an ~80%-of-frame halo (no savings).
A channel prefix keeps the full spatial extent (full coding gain) at a fraction of the rate.

  transmitter:  m = 1[||z_t - z_{t-1}||^2 > thr];   code = conv_enc(z_t*m)   (B,16,28,28)
  nested:       sample level k in {2,4,8,16}; send code[:, :k] (zero the rest) -> s~
  receiver:     rec = conv_dec(s~);  g = m*1[||rec-z_hat||^2>recv_thr];  z~ = g*rec+(1-g)*z_hat

Prints the per-level val cube MSE (calm/disturbed) each epoch -- the rate-distortion curve
that tells us how few channels calm can tolerate before the cube degrades.

    python experiments/stage2_twophase/train_masked_alloc_moe.py \
        --pre outputs/rollout_disturbed_clean_s2_moe --out outputs/masked_alloc_moe \
        --thr 0.05 --recv_thr 0.10 --epochs 45
"""
import argparse, glob, os, pathlib, sys
import torch

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                            # noqa: E402
from models import FadingChannel                      # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from conv_codec import ConvCodec                      # noqa: E402

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
ap.add_argument("--levels", type=int, nargs="+", default=[2, 4, 8, 16], help="nested channel counts (<= conv_dch)")
ap.add_argument("--thr", type=float, default=0.05)
ap.add_argument("--recv_thr", type=float, default=0.10)
ap.add_argument("--cube_w", type=float, default=4.0)
ap.add_argument("--dist_weight", type=float, default=15.0)
a = ap.parse_args()
DEV = "cuda:0"; os.makedirs(a.out, exist_ok=True)
assert max(a.levels) == a.conv_dch, "top level must send all conv_dch channels"

ZT = []; ZP = []; ZH = []; DIST = []
for f in sorted(glob.glob(os.path.join(a.pre, "shard*.pt"))):
    d = torch.load(f)
    assert "ZPREV" in d, "need ZPREV cache (rollout_disturbed_clean_s2_moe)"
    ZT.append(d["ZT"].float()); ZP.append(d["ZPREV"].float()); ZH.append(d["ZH"].float())
    DIST.append(d["DIST"].reshape(-1) if "DIST" in d else torch.zeros(d["ZT"].shape[0]))
ZT = torch.cat(ZT).reshape(-1, 4, 28, 28); ZP = torch.cat(ZP).reshape(-1, 4, 28, 28)
ZH = torch.cat(ZH).reshape(-1, 4, 28, 28); DIST = torch.cat(DIST); N = len(ZT)
print(f"[data] {N} frames disturbed={int(DIST.sum())}  levels={a.levels} (nested-dropout)", flush=True)


def change_mask(z, zp):
    return (((z - zp) ** 2).mean(1, keepdim=True) > a.thr).float()


def cube_region(z, h):
    return (((z - h) ** 2).mean(1, keepdim=True) > a.recv_thr).float()


net = ConvCodec(4, a.conv_w, a.conv_dch, a.conv_nb).to(DEV)
ch = FadingChannel(a.snr, channel_type="awgn")
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
va = perm[:int(N * 0.1)]; tr = perm[int(N * 0.1):]
DCH = a.conv_dch
LV = torch.tensor(a.levels, device=DEV)


def reconstruct(z, zp, k):
    """Encode z*m, keep the first k conv-code channels, channel, decode."""
    m = change_mask(z, zp)
    code = net.enc(z * m)                                                # (B,DCH,28,28)
    keep = (torch.arange(DCH, device=DEV) < k.view(-1, 1)).float()       # (B,DCH) channel prefix
    code = code * keep[:, :, None, None]
    B, Dc, H, W = code.shape
    return net.dec(ch(code.flatten(1)).view(B, Dc, H, W)), m


@torch.no_grad()
def rd_curve():
    """Per-level held-out cube MSE, split calm/disturbed -> the rate-distortion curve."""
    net.eval(); out = {}
    dmv = DIST[va].to(DEV)
    for lv in a.levels:
        num_c = den_c = num_d = den_d = 0.0
        for i in range(0, len(va), 256):
            sel = va[i:i + 256]; z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); h = ZH[sel].to(DEV); dm = DIST[sel].to(DEV)
            rec, _ = reconstruct(z, zp, torch.full((len(z),), lv, device=DEV))
            o = cube_region(z, h); wo = o.expand_as(z); err = ((rec - z) ** 2) * wo
            cmask = (dm < 0.5); dmask = (dm > 0.5)
            num_c += err[cmask].sum().item(); den_c += wo[cmask].sum().item()
            num_d += err[dmask].sum().item(); den_d += wo[dmask].sum().item()
        out[lv] = (num_c / max(den_c, 1), num_d / max(den_d, 1))
    return out


for e in range(1, a.epochs + 1):
    net.train(); order = tr[torch.randperm(len(tr))]; L = 0.0; nb = 0
    for i in range(0, len(order), a.bs):
        sel = order[i:i + a.bs]
        z = ZT[sel].to(DEV); zp = ZP[sel].to(DEV); h = ZH[sel].to(DEV); dm = DIST[sel].to(DEV)
        # nested dropout: each frame a random level, so every channel prefix is a valid code
        k = LV[torch.randint(0, len(a.levels), (len(z),), device=DEV)]
        rec, m = reconstruct(z, zp, k)
        o = cube_region(z, h); w = m + a.cube_w * o
        per = (((rec - z) ** 2) * w.expand_as(z)).sum(dim=(1, 2, 3)) / w.expand_as(z).sum(dim=(1, 2, 3)).clamp(min=1)
        fw = 1.0 + a.dist_weight * dm
        loss = (fw * per).sum() / fw.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        L += loss.item(); nb += 1
    rd = rd_curve()
    curve = "  ".join(f"k{lv}:c{rd[lv][0]:.3f}/d{rd[lv][1]:.3f}" for lv in a.levels)
    print(f"ep{e:>2}/{a.epochs}  loss {L/nb:.4f}  val cube (calm/dist per level)  {curve}", flush=True)
    torch.save({"net": net.state_dict(),
                "cfg": {"arch": "conv_nested", "conv_w": a.conv_w, "conv_dch": a.conv_dch, "conv_nb": a.conv_nb,
                        "levels": a.levels, "thr": a.thr, "recv_thr": a.recv_thr, "snr": a.snr}},
               os.path.join(a.out, "masked_alloc.pt"))
print("DONE", flush=True)
