"""Build the interactive-demo assets (docs/static/demo/) from an OPEN-LOOP replay of one recorded policy rollout.  Demo only, not a paper result.

Why open loop: the closed-loop dumps hold one frame per policy request, so the robot's motion during the 30 action steps is not recorded.
A recorded rollout (data/pick_blue_cube_to_tray/policy_rollouts.hdf5) has every frame.  Nothing is re-predicted here: the world-model
predictions come from the existing caches (pose-conditioned predictor "unfreeze2 ep5", true history)
    outputs/rollout_pose_unf2/shard*.pt      windows t = 26 + 20 n   (all episodes)
    outputs/cmp_pose_unf2/new/shard*.pt      windows t = 36 + 20 n   (predictor-held-out episodes)
which together give every 10th step, hence consecutive 30-step chunks, for the held-out episodes.  Only the codec (K_nogate_snr, K = 4,
CDL-C) and the VAE decoder are run, once per chunk and SNR.

Time slot 0 is the anchor (frame 25): no prediction yet, the image is sent uncoded.  Time slot n >= 1: t = 26 + 30 (n - 1), captured frame
f = t - 1 + 30; prediction = cached z_hat of window t; reconstruction = codec(z_f, z_hat).
Outputs
    ol<snr>_<n>.jpg   4 x 2 tiles of 224 px: input | prediction | reconstruction | reconstruction with the corrected region spotlit (rows = cameras)
    olmove_<n>.jpg    the recorded frames f, f+2, ..., f+30 (the robot executing the next chunk), one column per frame, rows = cameras
    demo.json         per-slot metadata, the next action chunk (dx, dy, dz, gripper) and the end-effector path (x, z)

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 ~/miniconda3/envs/predictor/bin/python docs/build_demo_openloop.py [--list] [--ep E]
"""
import argparse, glob, json, os, sys
import h5py, numpy as np, torch
from PIL import Image, ImageFilter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(REPO)
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments", "stage2_twophase"))
ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true", help="list the candidate episodes and exit")
ap.add_argument("--ep", type=int, default=None, help="episode index in the latent cache (default: the candidate whose prediction is furthest off)")
ap.add_argument("--snrs", type=float, nargs="+", default=[20, 12, 8, 4])
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--codec", default="outputs/wz2cam_retrain/K_nogate_snr/masked_alloc.pt")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
T_H, M, CH = 6, 5, 30; SPAN = (T_H - 1) * M + 1; OUT = "docs/static/demo"; PAPER = np.array([255, 255, 255], np.float32)

# --- episode table and the two cached window grids (same bookkeeping as experiments/decode_arms_pose_nogate.py) ------------------------- #
d = torch.load("outputs/stage1_chunk30_2cam_ft/rollout_latents.pt", map_location="cpu"); Z = d["Z"]; FILE = list(d["F"]); n_ep = len(Z)
rs = np.random.default_rng(123); held = set()
for fi in sorted(set(FILE)):
    idx = [e for e in range(n_ep) if FILE[e] == fi]; held |= set(int(x) for x in rs.choice(idx, min(10, len(idx)), replace=False))
EPS = []
for p_ in sorted(glob.glob("data/pick_*_cube_to_tray/policy_rollouts.hdf5")):
    with h5py.File(p_, "r") as f_: EPS += [(p_, e_) for e_ in f_ if "observations_cam1" in f_[e_]]
assert len(EPS) == n_ep
ZH = {}
for pat, off, only_held in (("outputs/rollout_pose_unf2/shard*.pt", 0, False), ("outputs/cmp_pose_unf2/new/shard*.pt", 10, True)):
    allidx = [(e, t) for e in range(n_ep) for t in range(SPAN + off, len(Z[e]) - CH + 1, 20) if (e in held or not only_held)]
    W = [w for i in range(3) for w in allidx[i::3]]; sh = [torch.load(f, map_location="cpu") for f in sorted(glob.glob(pat)) if "obj_" not in f and "fisher" not in f]
    zt = torch.cat([s["ZT"] for s in sh]); zh = torch.cat([s["ZH"] for s in sh])
    assert len(W) == len(zt) and all(torch.equal(Z[e][t - 1 + CH].float(), zt[i].float()) for i, (e, t) in enumerate(W[:50])), "window list mismatch"
    for i, (e, t) in enumerate(W):
        if e in held: ZH[(e, t)] = zh[i]
slots_of = lambda e: [t for t in range(SPAN, len(Z[e]) - CH + 1, CH) if (e, t) in ZH]
cand = []
for e in sorted(held):
    p, key = EPS[e]
    if "pick_blue" not in p: continue
    with h5py.File(p, "r") as f_: ok = int(f_[key].attrs.get("success", 0))
    ts = slots_of(e); err = [((Z[e][t - 1 + CH].float() - ZH[(e, t)].float()) ** 2).mean().item() for t in ts]
    cand.append((e, key, len(Z[e]), ok, len(ts), max(err) if err else 0)); print(f"ep {e:3d} {key:12s} len {len(Z[e]):3d} success {ok} slots {len(ts)} max|z-zhat|^2 {max(err) if err else 0:.4f}  " + " ".join(f"{v:.3f}" for v in err))
if a.list: sys.exit()
E = a.ep if a.ep is not None else max((c for c in cand if c[3] and c[4] >= 5), key=lambda c: c[5])[0]
p, key = EPS[E]; TS = slots_of(E); print(f"episode {E} ({p}:{key}), {len(TS)} time slots", flush=True)

from config import CONFIG                     # noqa: E402
from models import FadingChannel              # noqa: E402
from vae_wrapper import VAEWrapper            # noqa: E402
from conv_codec import ConvWZCodec            # noqa: E402
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(a.codec, map_location="cpu"); c = ck["cfg"]
assert c.get("mask_mode") == "none" and c.get("residual") and c.get("snr_cond"), "expects the no-mask residual SNR-conditioned codec"
net = ConvWZCodec(4, c["conv_w"], c["conv_dch"], c["conv_nb"], residual=True, snr_cond=True).to(DEV).eval(); net.load_state_dict(ck["net"])
CHAN = FadingChannel(20.0, channel_type="cdl", cdl_kwargs={"model": "C", "delay_spread": 50e-9, "carrier_frequency": 3.5e9, "max_speed": 3.0}); torch.manual_seed(a.seed)
vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()


@torch.no_grad()
def wz(z, h, snr):
    CHAN.snr_db = snr; mu, _ = net.encode(z); mp, _ = net.prior(h)
    return net.decode(torch.cat([CHAN(mu[:, :a.k].flatten(1)).view(1, a.k, 28, 56), mp[:, a.k:]], 1), h, None, snr)


@torch.no_grad()
def dec(lat):                                 # width-concatenated 2-camera latent -> (cam1, cam2) uint8 images
    u8 = lambda x: (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return u8(vae.decode(lat[..., :28].float())), u8(vae.decode(lat[..., 28:].float()))


def spotlight(rec, pred):
    """Reconstruction with the region where it departs from the prediction kept vivid and the rest washed out."""
    dd = np.abs(rec.astype(np.float32) - pred.astype(np.float32)).mean(2)
    w = Image.fromarray(np.clip(dd * 6, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
    w = np.clip(np.asarray(w, np.float32) / 255 * 2.2, 0, 1)[..., None]; al = 0.22 + 0.78 * w
    return (rec * al + PAPER * (1 - al)).round().astype(np.uint8)


os.makedirs(OUT, exist_ok=True)
for f_old in glob.glob(f"{OUT}/ol*.jpg"): os.remove(f_old)
with h5py.File(p, "r") as f_:
    g = f_[key]; C1 = np.asarray(g["observations_cam1"]); C2 = np.asarray(g["observations_cam2"]); ACT = np.asarray(g["actions"], np.float32); EE = np.asarray(g["robot_ee_pos"], np.float32)
    ok = bool(g.attrs.get("success", 0)); L = len(ACT)
eps = {s: [] for s in a.snrs}; slots = []
blank = np.full((224, 224, 3), 255, np.uint8)
# slot 0 is the anchor (frame t0 - 1): no prediction exists yet, the image is sent uncoded (VAE round trip), as server/semcom_server.py does for the first request
for n, t in enumerate([None] + TS):
    f = TS[0] - 1 if t is None else t - 1 + CH; z = Z[E][f][None].float().to(DEV)
    if t is not None: h = ZH[(E, t)][None].float().to(DEV); pr = dec(h)
    for s in a.snrs:
        rc = dec(z) if t is None else dec(wz(z, h, s)); sheet = np.zeros((448, 896, 3), np.uint8)
        for r in range(2):
            o, u = (C1, C2)[r][f], rc[r]
            sheet[r * 224:(r + 1) * 224] = np.concatenate([o, blank, u, u] if t is None else [o, pr[r], u, spotlight(u, pr[r])], 1)
        Image.fromarray(sheet).save(f"{OUT}/ol{s:g}_{n}.jpg", quality=86)
    fr = list(range(f, min(L - 1, f + CH) + 1, 2))                                   # the robot executing the chunk that follows this capture
    if len(fr) < 2: fr = [f, f]
    Image.fromarray(np.concatenate([np.concatenate([C1[i] for i in fr], 1), np.concatenate([C2[i] for i in fr], 1)], 0)).save(f"{OUT}/olmove_{n}.jpg", quality=80)
    rows = ACT[f + 1:f + 1 + CH:2] if f + 1 < L else ACT[-1:]
    if len(rows) < 2: rows = np.concatenate([rows, rows])
    slots.append(dict(pred=t is not None, step=int(f), mv=len(fr), act=[[round(float(v), 3) for v in (r[0], r[1], r[2], r[6])] for r in rows],
                      ee=[[round(float(EE[i][0]), 4), round(float(EE[i][2]), 4)] for i in fr]))
    print(f"  slot {n}: {'anchor' if t is None else f't={t}'} frame {f}, {len(fr)} motion frames", flush=True)
json.dump(dict(source="open-loop replay of a recorded policy rollout (see docs/build_demo_openloop.py); demo only", mode="open_loop", v=int(__import__("time").time()), tile=224, move="olmove",
               cols=["input", "prediction", "reconstruction", "spotlight"], act_dims=["dx", "dy", "dz", "gripper"], episode=f"{os.path.basename(os.path.dirname(p))}/{key}",
               episodes=[dict(name=f"ol{s:g}", snr=s, ok=ok, slots=slots) for s in a.snrs]), open(f"{OUT}/demo.json", "w"), separators=(",", ":"))
print("wrote", f"{OUT}/demo.json")
