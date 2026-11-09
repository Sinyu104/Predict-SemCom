"""Stage A (predictor env): DVST decode arms for the offline PI0Fast check (score with experiments/score_arms_pi0fast.py).

Same sample picks as experiments/decode_arms_wz_chunk30_2cam.py (same index, seed, holdout) so the arms pair with the
WZ npz.  The target frame is f = t-1+30.  Per sample and per DVST checkpoint, two arms (both over the channel):
  <name>_intra : f coded as an I-frame (no reference)         -- the first frame of each GOP / `--dvst_gop 1` online
  <name>_gop   : closed-loop chain of earlier targets f-30*(n-1) ... f (n = min(L+1, f//30+1) frames, first one intra),
                 tx reference = noise-free simulation, rx reference = what was actually received (Eq. 2), exactly as the
                 server does -- the last frame of the chain is the arm (worst case drift: the final P-frame of a GOP)
No world model is used (DVST has no side information); the wrist image is the decoded wrist half of the same latent.

    python baselines/dvst/decode_arms.py --ckpt outputs/dvst_gop_cdl_10_20/dvst.pt --name gop --alloc_snr 15 \
        --out outputs/dvst_offline/arms_gop.npz
"""
import argparse, glob, json, os, pathlib, sys
import h5py, numpy as np, torch

HERE = pathlib.Path(__file__).resolve().parent; REPO = str(HERE.parents[1])
sys.path.insert(0, REPO); sys.path.insert(0, str(HERE))
from config import CONFIG                        # noqa: E402
from vae_wrapper import VAEWrapper               # noqa: E402
from model import DVST, MaskedChannel            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, nargs="+"); ap.add_argument("--name", required=True, nargs="+")
ap.add_argument("--alloc_snr", type=float, nargs="+", default=None, help="per checkpoint (default: the run SNR)")
ap.add_argument("--L", type=int, default=4, help="P-frames after the I-frame in the chain")
ap.add_argument("--holdout_mod", type=int, default=5)
ap.add_argument("--demos", default="data/pick_*_cube_to_tray/demos_disturbed.hdf5")
ap.add_argument("--n", type=int, default=160); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--snr_db", type=float, default=20.0); ap.add_argument("--channel", default="cdl")
ap.add_argument("--out", required=True)
a = ap.parse_args()
assert len(a.ckpt) == len(a.name)
DEV = "cuda"; CHUNK = 30; T_H, M = CONFIG["num_history"], 5; SPAN = (T_H - 1) * M + 1
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

index = []
for p in sorted(glob.glob(os.path.join(REPO, a.demos))):
    with h5py.File(p, "r") as f:
        for e in f:
            if "observations_cam2" not in f[e]: continue
            if int("".join(c for c in e if c.isdigit())) % a.holdout_mod != 0: continue
            T = len(f[e]["observations_cam1"])
            index += [(p, e, t) for t in range(SPAN, T - 2 * CHUNK + 2)]
picks = [index[i] for i in sorted(rng.choice(len(index), size=min(a.n, len(index)), replace=False))]
print(f"{len(index)} candidate frames, sampled {len(picks)}", flush=True)

NETS = []
for pth in a.ckpt:
    ck = torch.load(os.path.join(REPO, pth), map_location=DEV); c = ck["cfg"]
    net = DVST(c["C_vae"], c["cy"], c["cs"], c["w"], c["nb"], eta_prime=c["eta_prime"]).to(DEV)
    net.load_state_dict(ck["net"]); net.eval(); net.entropy.ste = True; NETS.append(net)
for net, s in zip(NETS, a.alloc_snr or [None] * len(NETS)): net.alloc_snr = s
ch = MaskedChannel(a.snr_db, a.channel, {"model": "C", "delay_spread": 100e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0}).to(DEV)
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
SRC = 4 * 28 * 56


def u8(x): return (x.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


@torch.no_grad()
def code_chain(net, frames):
    """frames: list of (1,4,28,56) latents; frame 0 intra.  Returns (z_hat, frac16) of the last frame."""
    tx = rx = None
    for z in frames:
        c_tx, d_tx = net.contexts(*tx) if tx is not None else (None, None)
        c_rx, d_rx = net.contexts(*rx) if rx is not None else (None, None)
        e = net.encode(z, c_tx, d_tx, a.snr_db); s_tx = e["s"] * e["keep"]
        ys, zs = net.decode(s_tx, e["idx"], c_tx, d_tx)
        yh, zh = net.decode(ch(s_tx, e["keep"]), e["idx"], c_rx, d_rx)
        tx, rx = (zs, ys), (zh, yh)
    return zh, float(e["k_q"].sum().item()) / (16 * 28 * 56)


ARMS = ["vae"] + [f"{n}_{k}" for n in a.name for k in ("intra", "gop")]
out = {f"img_{ar}": [] for ar in ARMS}; out.update({f"wimg_{ar}": [] for ar in ARMS})
out.update(real=[], wrist=[], demo_a=[], task=[], ep=[], t=[], mse={f"{n}_{k}": [] for n in a.name for k in ("intra", "gop")},
           frac16={f"{n}_{k}": [] for n in a.name for k in ("intra", "gop")})
instr, by_ep, done = {}, {}, 0
for p, e, t in picks: by_ep.setdefault((p, e), []).append(t)
for (p, e), ts in by_ep.items():
    with h5py.File(p, "r") as f:
        g = f[e]
        cam1 = np.asarray(g["observations_cam1"]); cam2 = np.asarray(g["observations_cam2"])
        act = torch.from_numpy(np.asarray(g["actions"], np.float32))
    tdir = os.path.dirname(p)
    if tdir not in instr: instr[tdir] = json.load(open(os.path.join(tdir, "task.json")))["instruction"]
    with torch.no_grad():
        enc = lambda xx: torch.cat([vae.encode(xx[j:j + 8].to(DEV)).cpu() for j in range(0, len(xx), 8)])
        tt = lambda c: torch.from_numpy(c.astype(np.float32) / 255.).permute(0, 3, 1, 2)
        Z = torch.cat([enc(tt(cam1)), enc(tt(cam2))], dim=-1)
    for t in ts:
        f_t = t - 1 + CHUNK
        npf = min(a.L, f_t // CHUNK)                              # P-frames available before f_t
        idxs = [f_t - CHUNK * (npf - j) for j in range(npf + 1)]
        zt = Z[f_t].unsqueeze(0).to(DEV)
        lats = {"vae": zt}
        for net, nm in zip(NETS, a.name):
            for kind, fr in (("intra", [zt]), ("gop", [Z[i].unsqueeze(0).to(DEV) for i in idxs])):
                zh, fr16 = code_chain(net, fr); lats[f"{nm}_{kind}"] = zh
                out["mse"][f"{nm}_{kind}"].append(float(((zh - zt) ** 2).mean())); out["frac16"][f"{nm}_{kind}"].append(fr16)
        with torch.no_grad():
            for ar in ARMS:
                for key, hv in zip(["img_", "wimg_"], [lats[ar][..., :28], lats[ar][..., 28:]]):
                    out[f"{key}{ar}"].append(u8(vae.decode(hv.float().to(DEV)).clamp(0, 1)[0].cpu()))
        out["real"].append(cam1[f_t]); out["wrist"].append(cam2[f_t]); out["demo_a"].append(act[f_t:f_t + CHUNK].numpy())
        out["task"].append(instr[tdir]); out["ep"].append(f"{os.path.basename(tdir)}/{e}"); out["t"].append(t)
        done += 1
        if done % 20 == 0: print(f"  decoded {done}/{len(picks)}", flush=True)

mse, fr16 = out.pop("mse"), out.pop("frac16")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
n_obj = np.zeros(len(out["t"]), int)                              # score_arms splits by n_obj; DVST has no gate -> use WZ's if pairing
np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, n_obj=n_obj, snr_db=a.snr_db, ckpt=" + ".join(a.ckpt), stage1="none")
for k in mse: print(f"{k:14s} latent mse {np.mean(mse[k]):.4f}  frac16 {np.mean(fr16[k]):.3f}")
print("saved", a.out, "DONE", flush=True)
