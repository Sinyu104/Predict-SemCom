"""Stage I vs Stage II attribution IN ACTION SPACE, via the live VLA.

Cube geometry (recall/contrast) is a pixel proxy.  The real target is whether the
reconstruction lets the VLA act correctly.  This decodes each arm's image, sends
it to the running policy server, and compares the action the VLA produces against
the ground-truth demo action -- so the metric is the thing that actually drives
the robot, not a colour mask.

Arms, and what each isolates:

    vae       dec(z_t)                 ceiling: the VLA on an essentially perfect
                                       image.  Its error vs the demo action is the
                                       policy's OWN imitation error -- the floor no
                                       reconstruction can beat.
    zhat      dec(z_hat)               Stage 1 alone (the predictor).
    full      dec(refine(z_hat, s~))   Stage 1 + Stage 2 (the deployed path).
    oracle_z  dec(refine(z_t,   s~))   Stage 2 handed a PERFECT prior.  If this is
                                       good, the refiner is not the bottleneck and
                                       the fault is upstream in Stage 1.  If this is
                                       bad, Stage 2 damages even a perfect input.

Attribution:
    full bad  &  oracle_z good  &  zhat bad   ->  STAGE 1 (predictor)
    full bad  &  oracle_z bad                 ->  STAGE 2 (refiner)

Reported vs the demo action AND vs the vae-arm action (reconstruction-induced
action change, which removes the policy's own imitation error).  The gripper axis
(dim 6) is split out: a grasp task is lost on that one dimension.

Runs in the `predictor` env; the VLA is a separate process (worldmodel env) on TCP
5559 -- the same framed-JSON protocol semcom_server.PolicyClient uses.

    python experiments/stage2_twophase/vla_attribution.py \
        --stage1_ckpt outputs/stage1_stride_k8/stage1_rollout_ep1.pt \
        --stage2_ckpt outputs/stage2_lam0_bs32/phase2.pt \
        --eps_per_task 1 --n_frames 24 --device cuda:2
"""
import os, sys, io, json, socket, base64, pathlib, argparse
import numpy as np, torch, h5py
from PIL import Image

REPO = str(pathlib.Path(__file__).resolve().parents[2]); sys.path.insert(0, REPO)
from config import CONFIG                        # noqa: E402
from models import SemComSystem                  # noqa: E402
from vae_wrapper import VAEWrapper               # noqa: E402
from ctrl_world_wrapper import CtrlWorldWrapper  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage1_ckpt", default="outputs/stage1_stride_k8/stage1_rollout_ep1.pt")
ap.add_argument("--stage2_ckpt", default="outputs/stage2_lam0_bs32/phase2.pt")
ap.add_argument("--stride", type=int, default=8)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--ddim", type=int, default=10)
ap.add_argument("--eps_per_task", type=int, default=1)
ap.add_argument("--n_frames", type=int, default=24, help="frames sampled across each episode")
ap.add_argument("--snr_db", type=float, default=20.0)
ap.add_argument("--channel", default="awgn")
ap.add_argument("--policy_host", default="127.0.0.1")
ap.add_argument("--policy_port", type=int, default=5559)
ap.add_argument("--unnorm_key", default="franka_isaac")
ap.add_argument("--device", default="cuda:2")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--delta_scales", nargs="+", type=float, default=None,
                help="Residual-clamp sweep. The refiner returns base+delta; this rescales "
                     "delta by each alpha, so full@0 = the prior untouched and full@1 = the "
                     "trained refiner. Tests whether the refiner OVER-corrects: if a smaller "
                     "alpha lowers the VLA action error, a residual penalty is the lever. "
                     "Default (None) = the original vae/zhat/full/oracle_z arms.")
ap.add_argument("--tasks", nargs="+",
                default=["pick_blue_cube_to_tray", "pick_orange_cube_to_tray",
                         "pick_purple_cube_to_tray", "pick_red_cube_to_tray",
                         "pick_yellow_cube_to_tray"])
a = ap.parse_args()

DEV = a.device
T_H = CONFIG["num_history"]; M = a.stride; K = a.K
SEED = (T_H - 1) * M + 1
CONFIG["snr_db"] = a.snr_db; CONFIG["channel_type"] = a.channel
COLOR = lambda task: task.split("_")[1]
INSTR = lambda task: f"pick up the {COLOR(task)} cube and place it on the tray"


# ── minimal framed-JSON VLA client (mirrors semcom_server.PolicyClient) ────── #
class VLA:
    def __init__(self, host, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.seq = 0

    def _recv(self, n):
        b = b""
        while len(b) < n:
            c = self.s.recv(min(65536, n - len(b)))
            if not c:
                raise ConnectionError("policy server disconnected")
            b += c
        return b

    def predict(self, img_rgb, instruction):
        buf = io.BytesIO(); Image.fromarray(img_rgb).save(buf, format="JPEG", quality=95)
        self.seq += 1
        payload = json.dumps({
            "seq": self.seq,
            "jpeg_b64": base64.b64encode(buf.getvalue()).decode(),
            "instruction": instruction, "unnorm_key": a.unnorm_key,
        }).encode()
        self.s.sendall(len(payload).to_bytes(4, "big") + payload)
        n = int.from_bytes(self._recv(4), "big")
        r = json.loads(self._recv(n))
        if r.get("status") != "ok":
            raise RuntimeError(f"policy error: {r.get('message')}")
        return np.asarray(r["action"], dtype=np.float32)


print(f"[cfg] stage1={a.stage1_ckpt}\n      stage2={a.stage2_ckpt}\n"
      f"      SEED={SEED} K={K} stride={M} snr={a.snr_db}dB {a.channel}", flush=True)
vla = VLA(a.policy_host, a.policy_port)
print(f"[vla] connected {a.policy_host}:{a.policy_port}", flush=True)

vae = VAEWrapper(CONFIG["vae_model_name"]).to(DEV).eval()
cw = CtrlWorldWrapper(action_dim=CONFIG["action_dim"], num_history=T_H, num_pred=K,
                      svd_path=CONFIG["svd_model_name"], freeze_unet=True,
                      finetune_cross_attn=CONFIG.get("finetune_unet_cross_attn", False),
                      dtype=torch.float16).to(DEV)
ck = torch.load(os.path.join(REPO, a.stage1_ckpt), map_location="cpu")
cw.action_encoder.load_state_dict(ck["action_encoder_state"], strict=False)
cw.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"]); del ck; cw.eval()

sysm = SemComSystem(CONFIG).to(DEV)
sk = torch.load(os.path.join(REPO, a.stage2_ckpt), map_location="cpu")
sysm.load_state_dict(sk.get("system_state", sk), strict=False); sysm.eval()

def dec_uint8(z):
    im = vae.decode(z.to(DEV).float()).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (im * 255).astype(np.uint8)

if a.delta_scales:
    # vae + prior + a residual-scale sweep on both bases (zhat = deployed prior,
    # z_t = perfect prior). full@1.0 == the trained refiner; @0.0 == base untouched.
    ARMS = (["vae", "zhat"]
            + [f"full@{s:g}" for s in a.delta_scales]
            + [f"orc@{s:g}"  for s in a.delta_scales])
else:
    ARMS = ["vae", "zhat", "full", "oracle_z"]
PHASES = ["early", "mid", "late"]
# per (arm, phase): lists of L2-to-demo, L2-to-vae, gripper |err| to demo
rec = {ar: {ph: {"d": [], "v": [], "g": []} for ph in PHASES} for ar in ARMS}
rng = np.random.default_rng(a.seed)

for task in a.tasks:
    p = os.path.join(REPO, "data", task, "demos_disturbed.hdf5")
    if not os.path.isfile(p):
        print(f"  skip {task}"); continue
    instr = INSTR(task)
    with h5py.File(p, "r") as f:
        eps = sorted(f.keys(), key=lambda k: int("".join(c for c in k if c.isdigit())))
        picked = 0
        for e in eps:
            g = f[e]; n_ep = len(g["observations_cam1"])
            if n_ep < SEED + K + 2:
                continue
            imgs = np.array(g["observations_cam1"][:n_ep])
            act = torch.from_numpy(np.array(g["actions"][:n_ep], dtype=np.float32))
            # hdf5 actions have exactly n_ep rows; the last chunk's window reaches
            # t+K, so pad by repeating the final action (a held pose) to n_ep+K.
            act = torch.cat([act, act[-1:].repeat(K, 1)], 0)
            x = torch.from_numpy(imgs).float().div_(255.).permute(0, 3, 1, 2)
            with torch.no_grad():
                Z = torch.cat([vae.encode(x[j:j + 4].to(DEV)) for j in range(0, len(x), 4)])
                # clean-history chunk walk -> z_hat for every frame (deployment horizons)
                zhat = {}
                for t in range(SEED, n_ep, K):
                    hidx = [max(0, t - 1 - i * M) for i in range(T_H)][::-1]
                    H = torch.stack([Z[j] for j in hidx], 0).unsqueeze(0)
                    a_win = torch.cat([act[torch.tensor(hidx)], act[t:t + K]], 0).unsqueeze(0).to(DEV)
                    zp = cw.predict_next_latent(H, a_win, n_steps=a.ddim).float()
                    for i in range(min(K, n_ep - t)):
                        zhat[t + i] = zp[:, i]
            frames = sorted(rng.choice(np.arange(SEED, n_ep), replace=False,
                                       size=min(a.n_frames, n_ep - SEED)).tolist())
            for t in frames:
                z_t = Z[t].unsqueeze(0); zh = zhat[t]
                with torch.no_grad():
                    _, _, s = sysm.jscc_encoder(z_t, sample=False); st = sysm.channel(s)
                    t0 = torch.zeros(1, dtype=torch.long, device=DEV)
                    rd = sysm.refinement_diffusion
                    lat = {"vae": z_t, "zhat": zh}
                    if a.delta_scales:
                        d_full = rd._denoise_x0(zh,  t0, zh,  st) - zh      # residual on deployed prior
                        d_orc  = rd._denoise_x0(z_t, t0, z_t, st) - z_t     # residual on perfect prior
                        for sc in a.delta_scales:
                            lat[f"full@{sc:g}"] = zh  + sc * d_full
                            lat[f"orc@{sc:g}"]  = z_t + sc * d_orc
                    else:
                        lat["full"]     = rd._denoise_x0(zh,  t0, zh,  st)
                        lat["oracle_z"] = rd._denoise_x0(z_t, t0, z_t, st)
                acts = {ar: vla.predict(dec_uint8(lat[ar]), instr) for ar in ARMS}
                a_demo = act[t].numpy()
                ph = PHASES[min(int((t - SEED) / (n_ep - SEED) * 3), 2)]
                for ar in ARMS:
                    rec[ar][ph]["d"].append(float(np.linalg.norm(acts[ar] - a_demo)))
                    rec[ar][ph]["v"].append(float(np.linalg.norm(acts[ar] - acts["vae"])))
                    rec[ar][ph]["g"].append(float(abs(acts[ar][6] - a_demo[6])))
            picked += 1
            print(f"  {task}/{e}: {len(frames)} frames scored", flush=True)
            if picked >= a.eps_per_task:
                break


def agg(ar, key, phs):
    xs = [x for ph in phs for x in rec[ar][ph][key]]
    return float(np.mean(xs)) if xs else float("nan")

print(f"\n=== VLA action error by arm  (L2 over 7-dim action; gripper is dim 6) ===")
print(f"{'arm':<10} {'|a-demo|':>9} {'|a-vae|':>9} {'grip|err|':>10}   "
      f"{'mid |a-demo|':>13} {'mid |a-vae|':>12}")
for ar in ARMS:
    print(f"{ar:<10} {agg(ar,'d',PHASES):9.4f} {agg(ar,'v',PHASES):9.4f} "
          f"{agg(ar,'g',PHASES):10.4f}   {agg(ar,'d',['mid']):13.4f} {agg(ar,'v',['mid']):12.4f}")

print(f"\n=== by episode phase  (|a - demo|) ===")
print(f"{'arm':<10} " + " ".join(f"{ph:>9}" for ph in PHASES))
for ar in ARMS:
    print(f"{ar:<10} " + " ".join(f"{agg(ar,'d',[ph]):9.4f}" for ph in PHASES))

print("\nRead: vae is the floor (policy imitation error). If full >> vae and oracle_z ~ vae,")
print("Stage 1 is the fault. If oracle_z >> vae too, Stage 2 damages even a perfect prior.")
print("DONE")
