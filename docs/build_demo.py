"""Build the interactive-demo assets (docs/static/demo/) from closed-loop dumps.  Nothing is simulated here: every image is what was
logged during a closed-loop run (one dump per policy request = one time slot = one 30-step action chunk).

For each episode and time slot it writes one sprite sheet  docs/static/demo/<name>_<i>.jpg  of 4 columns x 2 rows of 224 px tiles
(row = camera):   input image | world-model prediction | reconstruction | reconstruction with the corrected region spotlit
and docs/static/demo/demo.json with the per-slot metadata and the action chunk the policy returned (dx, dy, dz, gripper).

The prediction is decoded from the logged latent lat_zh with the same SD-VAE (CPU is fine):
    CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/predictor/bin/python docs/build_demo.py
"""
import glob, json, os, sys
import numpy as np, torch
from PIL import Image, ImageFilter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
from vae_wrapper import VAEWrapper  # noqa: E402

OUT = "docs/static/demo"; os.makedirs(OUT, exist_ok=True)
O = "outputs/online_ft"
PAPER = np.array([244, 239, 230], np.float32)                      # page background, the spotlight fades the rest toward it
dev = "cuda" if torch.cuda.is_available() else "cpu"
vae = VAEWrapper("stabilityai/sd-vae-ft-mse").to(dev).eval()

# same episodes as docs/build_media.py
EPISODES = [dict(name="wz20", snr=20, dir=f"{O}/old_nogate_res_20dB_k4_30ep_23", ep=2, ok=True),
            dict(name="wz12", snr=12, dir=f"{O}/old_nogate_snr_12dB_k4_30ep_23", ep=4, ok=True),
            dict(name="wz8", snr=8, dir=f"{O}/old_nogate_snr_8dB_k4_30ep_17", ep=3, ok=True),
            dict(name="wz4", snr=4, dir=f"{O}/old_nogate_snr_4dB_k4_30ep_11", ep=2, ok=False)]


@torch.no_grad()
def dec(lat):
    z = torch.from_numpy(lat.astype(np.float32))[None].to(dev)
    u8 = lambda x: (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return u8(vae.decode(z[..., :28])), u8(vae.decode(z[..., 28:]))


def spotlight(rec, pred):
    """Reconstruction with the region where it departs from the prediction kept vivid and the rest washed out."""
    d = np.abs(rec.astype(np.float32) - pred.astype(np.float32)).mean(2)
    w = Image.fromarray(np.clip(d * 6, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
    w = np.clip(np.asarray(w, np.float32) / 255 * 2.2, 0, 1)[..., None]
    a = 0.22 + 0.78 * w
    return (rec * a + PAPER * (1 - a)).round().astype(np.uint8), float(d.mean())


meta = []
for E in EPISODES:
    fs = sorted(glob.glob(f"{E['dir']}/{E['ep']}_*.npz"), key=lambda f: int(os.path.basename(f)[:-4].split("_")[1]))
    slots = []
    for i, f in enumerate(fs):
        x = dict(np.load(f)); has = "lat_zh" in x
        sheet = np.zeros((448, 896, 3), np.uint8) + PAPER.astype(np.uint8)
        preds = dec(x["lat_zh"]) if has else (None, None); corr = []
        for r, (o, p, u) in enumerate([(x["obs1"], preds[0], x["out1"]), (x["obs2"], preds[1], x["out2"])]):
            sheet[r * 224:(r + 1) * 224, 0:224] = o; sheet[r * 224:(r + 1) * 224, 448:672] = u
            if has:
                s, c = spotlight(u, p); corr.append(c)
                sheet[r * 224:(r + 1) * 224, 224:448] = p; sheet[r * 224:(r + 1) * 224, 672:896] = s
        Image.fromarray(sheet).save(f"{OUT}/{E['name']}_{i}.jpg", quality=86)
        a = x["action_chunk"].astype(np.float32)
        slots.append(dict(pred=has, step=30 * i, corr=round(float(np.mean(corr)), 2) if corr else None,
                          act=[[round(float(v), 3) for v in (row[0], row[1], row[2], row[6])] for row in a[::2]]))     # every 2nd control step
    meta.append(dict(name=E["name"], snr=E["snr"], ok=E["ok"], slots=slots)); print(E["name"], len(slots), "slots", flush=True)
json.dump(dict(source="closed-loop dumps under outputs/online_ft (see docs/build_demo.py)", tile=224, cols=["input", "prediction", "reconstruction", "spotlight"],
               act_dims=["dx", "dy", "dz", "gripper"], episodes=meta), open(f"{OUT}/demo.json", "w"), separators=(",", ":"))
print("wrote", f"{OUT}/demo.json")
