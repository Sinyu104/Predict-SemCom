"""Stage A (predictor env): DVST arms on the SAME 200 held-out windows as experiments/decode_arms_pose_nogate.py, one arm per SNR,
scored with experiments/score_arms_pi0fast.py so DVST and K_nogate_snr pair sample by sample.

Rate-matched to WZ k=4 (frac16 ~0.25) with the closed-loop settings: SNR >= 10 dB -> 10-20 dB ckpt, alloc 20 dB, eta x0.86;
SNR < 10 dB -> 0-10 dB ckpt, alloc 8 dB, eta x0.43.  GOP 5 as on the server (--dvst_gop 5, one request every 30 steps): the target
f = t-1+30 is coded as the ((f//30) % 5)-th P-frame of a chain starting with an I-frame at f - 30*((f//30) % 5); tx reference = noise-free
simulation, rx reference = what was received (Eq. 2).  CDL-C 50 ns.

    CUDA_VISIBLE_DEVICES=0 python baselines/dvst/decode_arms_pose.py --out outputs/cmp_pose_unf2/dvst/arms.npz
"""
import argparse, glob, json, os, pathlib, sys
import h5py, numpy as np, torch

HERE = pathlib.Path(__file__).resolve().parent; REPO = str(HERE.parents[1])
sys.path.insert(0, REPO); sys.path.insert(0, str(HERE))
from config import CONFIG                        # noqa: E402
from vae_wrapper import VAEWrapper               # noqa: E402
from model import DVST, MaskedChannel            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hi", default="outputs/dvst_gop_cdl_10_20_lam0.01_w4/dvst.pt"); ap.add_argument("--hi_alloc", type=float, default=20); ap.add_argument("--hi_eta", type=float, default=0.86)
ap.add_argument("--lo", default="outputs/dvst_gop_cdl_0_10_lam0.01_w4/dvst.pt");  ap.add_argument("--lo_alloc", type=float, default=8);  ap.add_argument("--lo_eta", type=float, default=0.43)
ap.add_argument("--snrs", type=float, nargs="+", default=[20, 16, 12, 8, 4, 0])
ap.add_argument("--gop", type=int, default=5)
ap.add_argument("--dir", default="outputs/cmp_pose_unf2")
ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
DEV = "cuda"; T_H, M, PS, CH = 6, 5, 5, 30; SPAN = (T_H - 1) * M + 1

# --- window list and picks: identical to experiments/decode_arms_pose_nogate.py ------------------------------------------------------- #
d = torch.load(os.path.join(REPO, "outputs/stage1_chunk30_2cam_ft/rollout_latents.pt"), map_location="cpu")
Z = d["Z"]; FILE = list(d["F"]); n_ep = len(Z)
rs = np.random.default_rng(123); held = set()
for fi in sorted(set(FILE)):
    idx = [e for e in range(n_ep) if FILE[e] == fi]; held |= set(int(x) for x in rs.choice(idx, min(10, len(idx)), replace=False))
EPS = []
for p_ in sorted(glob.glob(os.path.join(REPO, "data/pick_*_cube_to_tray/policy_rollouts.hdf5"))):
    with h5py.File(p_, "r") as f_:
        EPS += [(p_, e_) for e_ in f_ if "observations_cam1" in f_[e_]]
assert len(EPS) == n_ep
allidx = [(e, t) for e in range(n_ep) for t in range(SPAN + 10, len(Z[e]) - CH + 1, 20) if e in held]
W = [w for i in range(3) for w in allidx[i::3]]
ZT = torch.cat([torch.load(f)["ZT"].float() for f in sorted(glob.glob(f"{REPO}/{a.dir}/new/shard*.pt"))])
assert len(W) == len(ZT) and all(torch.equal(Z[e][t - 1 + CH].float(), ZT[i]) for i, (e, t) in enumerate(W[:50])), "window list mismatch"
nlen = {}
def eplen(e):
    if e not in nlen:
        with h5py.File(EPS[e][0], "r") as f_: nlen[e] = len(f_[EPS[e][1]]["actions"])
    return nlen[e]
ok = [i for i, (e, t) in enumerate(W) if t - 1 + CH + CH <= eplen(e)]
rng = np.random.default_rng(a.seed); picks = sorted(rng.choice(ok, size=min(a.n, len(ok)), replace=False).tolist())
print(f"{len(W)} windows, {len(ok)} with actions after the target, using {len(picks)}", flush=True)

def load(path, alloc, eta):
    ck = torch.load(os.path.join(REPO, path), map_location=DEV); c = ck["cfg"]
    net = DVST(c["C_vae"], c["cy"], c["cs"], c["w"], c["nb"], eta_prime=c["eta_prime"]).to(DEV)
    net.load_state_dict(ck["net"]); net.eval(); net.alloc_snr = alloc; net.eta_prime *= eta; return net
HI, LO = load(a.hi, a.hi_alloc, a.hi_eta), load(a.lo, a.lo_alloc, a.lo_eta)
ch = MaskedChannel(20.0, "cdl", {"model": "C", "delay_spread": 50e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0}).to(DEV)
torch.manual_seed(a.seed)
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()

@torch.no_grad()
def code_chain(net, frames, snr):
    """frames: list of (1,4,28,56) latents, frame 0 intra.  Returns (z_hat, frac16) of the last frame (same as server _step_dvst)."""
    ch.snr_db = snr; tx = rx = None
    for z in frames:
        c_tx, d_tx = net.contexts(*tx) if tx is not None else (None, None)
        c_rx, d_rx = net.contexts(*rx) if rx is not None else (None, None)
        e = net.encode(z, c_tx, d_tx, snr); s_tx = e["s"] * e["keep"]
        ys, zs = net.decode(s_tx, e["idx"], c_tx, d_tx)
        yh, zh = net.decode(ch(s_tx, e["keep"]), e["idx"], c_rx, d_rx)
        tx, rx = (zs, ys), (zh, yh)
    return zh, float(e["k_q"].sum().item()) / (16 * 28 * 56)

u8 = lambda x: (x.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
tag = lambda s: f"dvst_{s:g}dB"
ARMS = ["vae"] + [tag(s) for s in a.snrs]
out = {f"img_{ar}": [] for ar in ARMS}; out.update({f"wimg_{ar}": [] for ar in ARMS})
out.update(real=[], wrist=[], demo_a=[], task=[], ep=[], t=[], n_obj=[], n_mask=[], lat_err=[], frac16=[])
ZH = torch.cat([torch.load(f)["ZH"].float() for f in sorted(glob.glob(f"{REPO}/{a.dir}/new/shard*.pt"))])
tasks = {}
for j, i in enumerate(picks):
    e, t = W[i]; p, key = EPS[e]; f_t = t - 1 + CH
    with h5py.File(p, "r") as f_:
        g_ = f_[key]; c1 = np.asarray(g_["observations_cam1"][f_t]); c2 = np.asarray(g_["observations_cam2"][f_t]); act = np.asarray(g_["actions"][f_t:f_t + CH], np.float32)
    td = os.path.dirname(p)
    if td not in tasks: tasks[td] = json.load(open(os.path.join(td, "task.json")))["instruction"]
    z = ZT[i:i + 1].to(DEV); ph = (f_t // CH) % a.gop
    frames = [Z[e][f_t - CH * (ph - q)][None].float().to(DEV) for q in range(ph + 1)]
    assert torch.equal(frames[-1], z)
    lat, fr = {"vae": z}, []
    for s in a.snrs:
        lat[tag(s)], f16 = code_chain(HI if s >= 10 else LO, frames, s); fr.append(f16)
    for ar in ARMS:
        for key_, hv in (("img_", lat[ar][..., :28]), ("wimg_", lat[ar][..., 28:])):
            out[key_ + ar].append(u8(vae.decode(hv.float()).clamp(0, 1)[0].cpu()))
    no = int((((z - ZH[i:i + 1].to(DEV)) ** 2).mean(1) > 0.05).sum())             # same CALM/DISTURBED key as the WZ npz
    out["real"].append(c1); out["wrist"].append(c2); out["demo_a"].append(act); out["task"].append(tasks[td])
    out["ep"].append(f"{os.path.basename(td)}/{key}"); out["t"].append(t); out["n_obj"].append(no); out["n_mask"].append(0)
    out["lat_err"].append([((lat[ar] - z) ** 2).mean().item() for ar in ARMS]); out["frac16"].append(fr)
    if (j + 1) % 20 == 0: print(f"  decoded {j + 1}/{len(picks)}", flush=True)

le = np.array(out["lat_err"]).mean(0); fr = np.array(out["frac16"]).mean(0)
print("latent mse vs z_t: " + "  ".join(f"{ar}={v:.4f}" for ar, v in zip(ARMS, le)), flush=True)
print("frac16: " + "  ".join(f"{tag(s)}={v:.3f}" for s, v in zip(a.snrs, fr)), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, arms=np.array(ARMS), snr_db=np.array(a.snrs),
                    ckpt=f"{a.hi} (alloc {a.hi_alloc}, eta x{a.hi_eta}) | {a.lo} (alloc {a.lo_alloc}, eta x{a.lo_eta})", channel="CDL-C 50 ns", gop=a.gop)
print("saved", a.out, flush=True)
