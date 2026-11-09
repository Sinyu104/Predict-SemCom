"""Stage A (predictor env): SSCC (H.264 + NR LDPC/QAM) arms on the SAME 200 held-out windows as experiments/decode_arms_pose_nogate.py,
scored with experiments/score_arms_pi0fast.py.  Same config as the closed-loop SSCC runs: each camera its own H.264 4:4:4 800 kbps CBR
stream (10 fps, GOP 10), NR fixed MCS 16 (one transmission, no HARQ), no keyframe request after a loss, CDL-C (NRLink: 100 ns).

One request every 30 steps = one stream frame.  The target f = t-1+30 is frame ((f//30) % GOP) of its GOP, so each camera's stream is run
from the GOP's keyframe at f - 30*phase up to f, and the decoder shows what it would have shown at f (drift concealment, as the server).
A window COUNTS at an SNR only if the target frame of BOTH cameras decoded (the edge acts only on a complete input, as on the server);
otherwise it is an outage and is not scored -- ok_<arm> holds the flags, and the arm images of lost windows are the true frames (ignored).
Arms whose every window was lost are dropped.

    CUDA_VISIBLE_DEVICES=0 python baselines/traditional/decode_arms_pose.py --out outputs/cmp_pose_unf2/sscc/arms.npz
"""
import argparse, glob, json, os, pathlib, sys, time
import h5py, numpy as np, torch

HERE = pathlib.Path(__file__).resolve().parent; REPO = str(HERE.parents[1]); sys.path.insert(0, REPO)
from baselines.traditional.nr_link import NRLink                              # noqa: E402
from baselines.traditional.stream_codec import StreamEncoder, StreamDecoder   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--snrs", type=float, nargs="+", default=[20, 16, 12, 8, 4, 0])
ap.add_argument("--mcs", type=int, default=16)
ap.add_argument("--kbps", type=float, default=800.0); ap.add_argument("--fps", type=int, default=10); ap.add_argument("--gop", type=int, default=10)
ap.add_argument("--dir", default="outputs/cmp_pose_unf2")
ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
T_H, M, PS, CH = 6, 5, 5, 30; SPAN = (T_H - 1) * M + 1

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
SH = sorted(glob.glob(f"{REPO}/{a.dir}/new/shard*.pt"))
ZT = torch.cat([torch.load(f)["ZT"].float() for f in SH]); ZH = torch.cat([torch.load(f)["ZH"].float() for f in SH])
assert len(W) == len(ZT) and all(torch.equal(Z[e][t - 1 + CH].float(), ZT[i]) for i, (e, t) in enumerate(W[:50])), "window list mismatch"
nlen = {}
def eplen(e):
    if e not in nlen:
        with h5py.File(EPS[e][0], "r") as f_: nlen[e] = len(f_[EPS[e][1]]["actions"])
    return nlen[e]
ok = [i for i, (e, t) in enumerate(W) if t - 1 + CH + CH <= eplen(e)]
rng = np.random.default_rng(a.seed); picks = sorted(rng.choice(ok, size=min(a.n, len(ok)), replace=False).tolist())
picks = picks[a.shard::a.nshards]                                                       # the scorer re-pairs by (ep, t)
print(f"{len(W)} windows, {len(ok)} with actions after the target, shard {a.shard}/{a.nshards}: {len(picks)}", flush=True)

tb = int(round(a.kbps * 1000 / a.fps / 8))
LINKS = {s: NRLink("cdl", s, tb_bytes=tb, re_budget=None, max_tx=1, bler_target=0.1, cdl_model="C", fixed_mcs=a.mcs, seed=a.seed + a.shard) for s in a.snrs}
print(f"NR MCS {a.mcs}, TB {tb} B/frame/camera, {a.kbps:g} kbps H.264 4:4:4 GOP {a.gop}", flush=True)

tag = lambda s: f"sscc_{s:g}dB"
ARMS = ["vae"] + [tag(s) for s in a.snrs]
out = {f"img_{ar}": [] for ar in ARMS}; out.update({f"wimg_{ar}": [] for ar in ARMS})
out.update({f"ok_{tag(s)}": [] for s in a.snrs}); out.update({f"clean_{tag(s)}": [] for s in a.snrs})
out.update(real=[], wrist=[], demo_a=[], task=[], ep=[], t=[], n_obj=[], n_mask=[], n_symbols=[])
from vae_wrapper import VAEWrapper   # noqa: E402  (only for the vae reference arm, same as the other npz files)
from config import CONFIG            # noqa: E402
vae = VAEWrapper(CONFIG["vae_model_name"]).to("cuda").eval()
u8 = lambda x: (x.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
tasks, t0 = {}, time.perf_counter()
for j, i in enumerate(picks):
    e, t = W[i]; p, key = EPS[e]; f_t = t - 1 + CH; ph = (f_t // CH) % a.gop
    fidx = [f_t - CH * (ph - q) for q in range(ph + 1)]                                   # keyframe ... target
    with h5py.File(p, "r") as f_:
        g_ = f_[key]; cams = [np.asarray(g_[f"observations_cam{c}"][fidx]) for c in (1, 2)]
        act = np.asarray(g_["actions"][f_t:f_t + CH], np.float32)
    td = os.path.dirname(p)
    if td not in tasks: tasks[td] = json.load(open(os.path.join(td, "task.json")))["instruction"]
    pk = []                                                                                # encoder is link-independent (no keyframe request)
    for c in range(2):
        enc = StreamEncoder("h264", gop=a.gop, fps=a.fps, kbps=a.kbps, chroma="444"); pk.append([enc.encode(fr) for fr in cams[c]])
    z = ZT[i:i + 1].cuda()
    with torch.no_grad():
        out["img_vae"].append(u8(vae.decode(z[..., :28])[0])); out["wimg_vae"].append(u8(vae.decode(z[..., 28:])[0]))
    for s in a.snrs:
        imgs, tgt_ok, clean = [], True, True
        for c in range(2):
            dec = StreamDecoder("h264", conceal="drift")
            for q, pkt in enumerate(pk[c]):
                rx, info = LINKS[s].transmit(pkt); y = dec.decode(rx)
                clean &= bool(info["ok"])
                if q == len(pk[c]) - 1: tgt_ok &= bool(info["ok"])
            imgs.append(y)
        if not tgt_ok: imgs = [cams[0][-1], cams[1][-1]]                                  # outage: not scored
        out[f"img_{tag(s)}"].append(imgs[0]); out[f"wimg_{tag(s)}"].append(imgs[1])
        out[f"ok_{tag(s)}"].append(tgt_ok); out[f"clean_{tag(s)}"].append(clean)
    no = int((((z - ZH[i:i + 1].cuda()) ** 2).mean(1) > 0.05).sum())
    out["real"].append(cams[0][-1]); out["wrist"].append(cams[1][-1]); out["demo_a"].append(act); out["task"].append(tasks[td])
    out["ep"].append(f"{os.path.basename(td)}/{key}"); out["t"].append(t); out["n_obj"].append(no); out["n_mask"].append(0)
    if (j + 1) % 10 == 0:
        print(f"  decoded {j + 1}/{len(picks)}  {(time.perf_counter() - t0) / (j + 1):.1f} s/window  ok: "
              + " ".join(f"{s:g}dB={np.mean(out[f'ok_{tag(s)}']):.2f}" for s in a.snrs), flush=True)

dead = [s for s in a.snrs if not any(out[f"ok_{tag(s)}"])]
for s in dead:                                                                            # never decoded: nothing to score
    for k in (f"img_{tag(s)}", f"wimg_{tag(s)}"): out.pop(k)
print("received fraction (both cameras, target frame): " + "  ".join(f"{s:g}dB={np.mean(out[f'ok_{tag(s)}']):.3f}" for s in a.snrs)
      + f"   dropped (all lost): {dead}", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
np.savez_compressed(a.out, **{k: np.array(v) for k, v in out.items()}, snr_db=np.array(a.snrs), mcs=a.mcs, tb_bytes=tb,
                    ckpt=f"sscc h264 444 {a.kbps:g}kbps gop{a.gop} MCS{a.mcs}", cfg=f"h264 444 {a.kbps:g}kbps gop{a.gop} fps{a.fps}, NR MCS{a.mcs} no HARQ, no kf request, CDL-C 100ns")
print("saved", a.out, flush=True)
