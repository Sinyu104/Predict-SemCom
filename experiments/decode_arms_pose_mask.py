"""Stage A of the offline PI0Fast check for codec F + old predictor  vs  codec K + new (pose) predictor.

Uses the windows cached by precompute_pose_rollout.py --held_only --t_offset 10 (outputs/cmp_pose_unf4/{new,old}); the window list is rebuilt
exactly as in that script (episode, t) and checked against the cached ZT.  Writes the npz format score_arms_pi0fast.py reads (img_<arm>,
wimg_<arm>, real, wrist, demo_a, task, n_obj).  demo_a = the policy-rollout actions that follow the target frame.

    CUDA_VISIBLE_DEVICES=0 python experiments/decode_arms_pose_cmp.py --snr_db 12 --n 200 --out outputs/cmp_pose_unf4/arms_12dB.npz
    CUDA_VISIBLE_DEVICES=0 pi0fast/bin/python experiments/score_arms_pi0fast.py --in <npz> --out <json>
"""
import argparse, glob, json, os, pathlib, sys
import h5py, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments", "stage2_twophase"))
from config import CONFIG                     # noqa: E402
from models import FadingChannel              # noqa: E402
from vae_wrapper import VAEWrapper            # noqa: E402
from conv_codec import ConvWZCodec, MaskNet   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--F", default="outputs/wz2cam_retrain/F_cdl_10_20/masked_alloc.pt")
ap.add_argument("--K", default="outputs/wz2cam_retrain/K_pose_unf4_cdl_12_20_k12/masked_alloc.pt")
ap.add_argument("--M", default="outputs/wz2cam_retrain/K_pose_unf4_masknet/masked_alloc.pt")
ap.add_argument("--mask", default="outputs/wz2cam_retrain/mask_net.pt")
ap.add_argument("--dir", default="outputs/cmp_pose_unf4")
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
    net = ConvWZCodec(4, c["conv_w"], c["conv_dch"], c["conv_nb"]).to(DEV).eval(); net.load_state_dict(ck["net"]); return net, c
NK, cK = codec(a.K); NM, cM = codec(a.M)
_mk = torch.load(os.path.join(REPO, a.mask), map_location=DEV); MN = MaskNet().to(DEV).eval(); MN.load_state_dict(_mk["net"]); TAU = _mk["tau"]
chan = FadingChannel(a.snr_db, channel_type="cdl", cdl_kwargs=cdl); torch.manual_seed(a.seed)
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()

@torch.no_grad()
def zt_arm(net, c, z, zp, h, k, learned=False):
    m = (torch.sigmoid(MN(z)) > TAU).float() if learned else (((z - zp) ** 2).mean(1, keepdim=True) > c["thr"]).float()
    mu, _ = net.encode(z * m); mp, _ = net.prior(h)
    rec = net.decode(torch.cat([chan(mu[:, :k].flatten(1)).view(1, k, 28, 56), mp[:, k:]], 1))     # == server: only the k sent channels see the channel
    g = m * (((rec - h) ** 2).mean(1, keepdim=True) > c["recv_thr"]).float()
    return g * rec + (1 - g) * h, int(m.sum()), int((m * (((z - h) ** 2).mean(1, keepdim=True) > c["recv_thr"]).float()).sum())

u8 = lambda x: (x.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
ARMS = ["vae", "zhat_new", "K_new_k1", "K_new_k2", "K_new_k4", "M_new_k1", "M_new_k2", "M_new_k4"]
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
    lat = {"vae": z, "zhat_new": hn}; nm = no = 0
    for k in (1, 2, 4):
        lat[f"K_new_k{k}"], nm, no = zt_arm(NK, cK, z, zp, hn, k); lat[f"M_new_k{k}"] = zt_arm(NM, cM, z, zp, hn, k, learned=True)[0]
    for ar in ARMS:
        for key_, hv in (("img_", lat[ar][..., :28]), ("wimg_", lat[ar][..., 28:])):
            out[key_ + ar].append(u8(vae.decode(hv.float()).clamp(0, 1)[0].cpu()))
    out["real"].append(c1); out["wrist"].append(c2); out["demo_a"].append(act); out["task"].append(tasks[td])
    out["ep"].append(f"{os.path.basename(td)}/{key}"); out["t"].append(t); out["n_obj"].append(no); out["n_mask"].append(nm)
    if (j + 1) % 20 == 0: print(f"  decoded {j + 1}/{len(picks)}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, snr_db=a.snr_db, ckpt=f"{a.K} | {a.M} | {a.mask}", stage1="old=stage1_chunk30_2cam_ft ep1, new=pose_unfreeze ep4")
print("saved", a.out, flush=True)
