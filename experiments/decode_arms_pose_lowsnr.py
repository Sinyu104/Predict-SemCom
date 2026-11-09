"""Stage A of the low-SNR PI0Fast policy check (unf2 predictor, CDL-C 50 ns): zhat alone vs Klow50_masknet (learned mask) vs Klow50_r05 (plain mask) at k=2,4.
    CUDA_VISIBLE_DEVICES=0 python experiments/decode_arms_pose_lowsnr.py --snr_db 4 --out <npz>
"""
import argparse, glob, json, os, pathlib, sys
import h5py, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments", "stage2_twophase"))
from config import CONFIG                     # noqa: E402
from models import FadingChannel              # noqa: E402
from vae_wrapper import VAEWrapper            # noqa: E402
from conv_codec import ConvWZCodec, MaskNet   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--MN", default="outputs/wz2cam_retrain/Klow50_masknet/masked_alloc.pt")
ap.add_argument("--PL", default="outputs/wz2cam_retrain/Klow50_r05/masked_alloc.pt")
ap.add_argument("--mask", default="outputs/wz2cam_retrain/mask_net_unf2.pt")
ap.add_argument("--dir", default="outputs/cmp_pose_unf2")
ap.add_argument("--snr_db", type=float, default=12.0)
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
DEV = "cuda"; T_H, M, PS, CH = 6, 5, 5, 30; SPAN = (T_H - 1) * M + 1
cdl = {"model": "C", "delay_spread": 100e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0}

# --- episode table (same order as the latent cache) and the window list --------------------------------------------------------------- #
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
W = [w for i in range(3) for w in allidx[i::3]]                                   # shard order of the cache
load = lambda sub: {k: torch.cat([torch.load(f)[k].float() for f in sorted(glob.glob(f"{REPO}/{a.dir}/{sub}/shard*.pt"))]) for k in ("ZT", "ZH", "ZPREV")}
new, old = load("new"), load("old")
assert len(W) == len(new["ZT"]) and all(torch.equal(Z[e][t - 1 + CH].float(), new["ZT"][i]) for i, (e, t) in enumerate(W[:50])), "window list mismatch"

# --- pick windows that have 30 policy actions after the target frame ---------------------------------------------------------------- #
nlen = {}
def eplen(e):
    if e not in nlen:
        with h5py.File(EPS[e][0], "r") as f_: nlen[e] = len(f_[EPS[e][1]]["actions"])
    return nlen[e]
ok = [i for i, (e, t) in enumerate(W) if t - 1 + CH + CH <= eplen(e)]
rng = np.random.default_rng(a.seed); picks = sorted(rng.choice(ok, size=min(a.n, len(ok)), replace=False).tolist())
print(f"{len(W)} windows, {len(ok)} with actions after the target, using {len(picks)}", flush=True)

def codec(path):
    ck = torch.load(os.path.join(REPO, path), map_location="cpu"); c = ck["cfg"]
    net = ConvWZCodec(4, c["conv_w"], c["conv_dch"], c["conv_nb"], residual=c.get("residual", False)).to(DEV).eval(); net.load_state_dict(ck["net"]); return net, c
NM, cM = codec(a.MN); NP, cP = codec(a.PL)
_mk = torch.load(os.path.join(REPO, a.mask), map_location=DEV); MN = MaskNet().to(DEV).eval(); MN.load_state_dict(_mk["net"]); TAU = _mk["tau"]
CHAN = FadingChannel(a.snr_db, channel_type="cdl", cdl_kwargs={**cdl, "delay_spread": 50e-9}); torch.manual_seed(a.seed)
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()

@torch.no_grad()
def zt_arm(net, c, z, zp, h, k, learned=False):
    m = (torch.sigmoid(MN(z)) > TAU).float() if learned else (((z - zp) ** 2).mean(1, keepdim=True) > c["thr"]).float()
    mu, _ = net.encode(z * m); mp, _ = net.prior(h)
    rec = net.decode(torch.cat([CHAN(mu[:, :k].flatten(1)).view(1, k, 28, 56), mp[:, k:]], 1), h)
    g = m * (((rec - h) ** 2).mean(1, keepdim=True) > c["recv_thr"]).float()
    return g * rec + (1 - g) * h, int(m.sum()), int((m * (((z - h) ** 2).mean(1, keepdim=True) > c["recv_thr"]).float()).sum())

u8 = lambda x: (x.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
ARMS = ["vae", "zhat_new", "mask_k2", "mask_k4", "plain_k2", "plain_k4"]
out = {f"img_{ar}": [] for ar in ARMS}; out.update({f"wimg_{ar}": [] for ar in ARMS})
out.update(real=[], wrist=[], demo_a=[], task=[], ep=[], t=[], n_obj=[], n_mask=[])
tasks = {}
for j, i in enumerate(picks):
    e, t = W[i]; p, key = EPS[e]; f_t = t - 1 + CH
    with h5py.File(p, "r") as f_:
        g_ = f_[key]; c1 = np.asarray(g_["observations_cam1"][f_t]); c2 = np.asarray(g_["observations_cam2"][f_t]); act = np.asarray(g_["actions"][f_t:f_t + CH], np.float32)
    td = os.path.dirname(p)
    if td not in tasks: tasks[td] = json.load(open(os.path.join(td, "task.json")))["instruction"]
    z, zp, ho, hn = [x[i:i + 1].to(DEV) for x in (new["ZT"], new["ZPREV"], old["ZH"], new["ZH"])]
    lat = {"vae": z, "zhat_new": hn}
    for k in (2, 4):
        lat[f"mask_k{k}"], nm, no = zt_arm(NM, cM, z, zp, hn, k, learned=True); lat[f"plain_k{k}"] = zt_arm(NP, cP, z, zp, hn, k)[0]
    for ar in ARMS:
        for key_, hv in (("img_", lat[ar][..., :28]), ("wimg_", lat[ar][..., 28:])):
            out[key_ + ar].append(u8(vae.decode(hv.float()).clamp(0, 1)[0].cpu()))
    out["real"].append(c1); out["wrist"].append(c2); out["demo_a"].append(act); out["task"].append(tasks[td])
    out["ep"].append(f"{os.path.basename(td)}/{key}"); out["t"].append(t); out["n_obj"].append(no); out["n_mask"].append(nm)
    if (j + 1) % 20 == 0: print(f"  decoded {j + 1}/{len(picks)}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, snr_db=a.snr_db, ckpt=f"{a.MN} | {a.PL} | {a.mask}", stage1="unfreeze2 ep5")
print("saved", a.out, flush=True)
