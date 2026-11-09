"""Stage A of the offline PI0Fast SNR sweep for the deployed codec K_nogate_snr (no mask, no gate, residual decoder with SNR input):
z~ = zhat + Dec([s~ (ch 0..k-1), mu_p (ch k..15)], zhat, SNR/20), CDL-C 50 ns, one arm per SNR.  Same 200 held-out windows as
decode_arms_pose_lowsnr.py (unf2 ep5 predictor, true history, cache outputs/cmp_pose_unf2/new).

    CUDA_VISIBLE_DEVICES=0 python experiments/decode_arms_pose_nogate.py --out outputs/cmp_pose_unf2/nogate/arms.npz
    CUDA_VISIBLE_DEVICES=0 pi0fast/bin/python experiments/score_arms_pi0fast.py --in <npz> --out <json>
"""
import argparse, glob, json, os, pathlib, sys
import h5py, numpy as np, torch

REPO = str(pathlib.Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments", "stage2_twophase"))
from config import CONFIG                     # noqa: E402
from models import FadingChannel              # noqa: E402
from vae_wrapper import VAEWrapper            # noqa: E402
from conv_codec import ConvWZCodec            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--codec", default="outputs/wz2cam_retrain/K_nogate_snr/masked_alloc.pt")
ap.add_argument("--dir", default="outputs/cmp_pose_unf2")
ap.add_argument("--snrs", type=float, nargs="+", default=[20, 16, 12, 8, 4, 0])
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--ks", type=int, nargs="+", default=None, help="sweep k (arms wz_<snr>dB_k<k>); default: --k only")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
DEV = "cuda"; T_H, M, PS, CH = 6, 5, 5, 30; SPAN = (T_H - 1) * M + 1
cdl = {"model": "C", "delay_spread": 50e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0}

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
new = {k: torch.cat([torch.load(f)[k].float() for f in sorted(glob.glob(f"{REPO}/{a.dir}/new/shard*.pt"))]) for k in ("ZT", "ZH")}
assert len(W) == len(new["ZT"]) and all(torch.equal(Z[e][t - 1 + CH].float(), new["ZT"][i]) for i, (e, t) in enumerate(W[:50])), "window list mismatch"

# --- same window picks as decode_arms_pose_lowsnr.py ---------------------------------------------------------------------------------- #
nlen = {}
def eplen(e):
    if e not in nlen:
        with h5py.File(EPS[e][0], "r") as f_: nlen[e] = len(f_[EPS[e][1]]["actions"])
    return nlen[e]
ok = [i for i, (e, t) in enumerate(W) if t - 1 + CH + CH <= eplen(e)]
rng = np.random.default_rng(a.seed); picks = sorted(rng.choice(ok, size=min(a.n, len(ok)), replace=False).tolist())
print(f"{len(W)} windows, {len(ok)} with actions after the target, using {len(picks)}", flush=True)

ck = torch.load(os.path.join(REPO, a.codec), map_location="cpu"); c = ck["cfg"]
assert c.get("mask_mode") == "none" and c.get("residual") and c.get("snr_cond"), "expects the no-mask residual SNR-conditioned codec"
net = ConvWZCodec(4, c["conv_w"], c["conv_dch"], c["conv_nb"], residual=True, snr_cond=True).to(DEV).eval(); net.load_state_dict(ck["net"])
CHAN = FadingChannel(20.0, channel_type="cdl", cdl_kwargs=cdl); torch.manual_seed(a.seed)
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()

@torch.no_grad()
def wz(z, h, snr, k):
    CHAN.snr_db = snr
    mu, _ = net.encode(z); mp, _ = net.prior(h)
    return net.decode(torch.cat([CHAN(mu[:, :k].flatten(1)).view(1, k, 28, 56), mp[:, k:]], 1), h, None, snr)

u8 = lambda x: (x.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
KS = a.ks or [a.k]
tag = (lambda s, k: f"wz_{s:g}dB") if a.ks is None else (lambda s, k: f"wz_{s:g}dB_k{k}")
ARMS = ["vae", "zhat"] + [tag(s, k) for s in a.snrs for k in KS]
out = {f"img_{ar}": [] for ar in ARMS}; out.update({f"wimg_{ar}": [] for ar in ARMS})
out.update(real=[], wrist=[], demo_a=[], task=[], ep=[], t=[], n_obj=[], n_mask=[], lat_err=[])
tasks = {}
for j, i in enumerate(picks):
    e, t = W[i]; p, key = EPS[e]; f_t = t - 1 + CH
    with h5py.File(p, "r") as f_:
        g_ = f_[key]; c1 = np.asarray(g_["observations_cam1"][f_t]); c2 = np.asarray(g_["observations_cam2"][f_t]); act = np.asarray(g_["actions"][f_t:f_t + CH], np.float32)
    td = os.path.dirname(p)
    if td not in tasks: tasks[td] = json.load(open(os.path.join(td, "task.json")))["instruction"]
    z, h = new["ZT"][i:i + 1].to(DEV), new["ZH"][i:i + 1].to(DEV)
    lat = {"vae": z, "zhat": h}
    for s in a.snrs:
        for k in KS: lat[tag(s, k)] = wz(z, h, s, k)
    for ar in ARMS:
        for key_, hv in (("img_", lat[ar][..., :28]), ("wimg_", lat[ar][..., 28:])):
            out[key_ + ar].append(u8(vae.decode(hv.float()).clamp(0, 1)[0].cpu()))
    no = int((((z - h) ** 2).mean(1) > 0.05).sum())                                 # cells where the prediction is wrong (CALM/DISTURBED split)
    out["real"].append(c1); out["wrist"].append(c2); out["demo_a"].append(act); out["task"].append(tasks[td])
    out["ep"].append(f"{os.path.basename(td)}/{key}"); out["t"].append(t); out["n_obj"].append(no); out["n_mask"].append(0)
    out["lat_err"].append([((lat[ar] - z) ** 2).mean().item() for ar in ARMS])
    if (j + 1) % 20 == 0: print(f"  decoded {j + 1}/{len(picks)}", flush=True)

le = np.array(out["lat_err"]).mean(0); print("latent mse vs z_t: " + "  ".join(f"{ar}={v:.4f}" for ar, v in zip(ARMS, le)), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, arms=np.array(ARMS), snr_db=np.array(a.snrs), k=np.array(KS),
                    ckpt=a.codec, stage1="unfreeze2 ep5", channel="CDL-C 50 ns")
print("saved", a.out, flush=True)
