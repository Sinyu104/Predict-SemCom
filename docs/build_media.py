"""Build the project-page media (docs/static/) from closed-loop dumps and the results CSV.  Nothing on the page is typed by hand:
re-run this after new runs and the videos + results.json update.

Each dump (one per policy request = one 30-step action chunk) holds the true camera frames (obs1/obs2) and what the policy saw (out1/out2);
our WZ dumps also hold the prediction latent lat_zh, decoded here with the same SD-VAE.

predictor env (needs the VAE for the prediction column):
    CUDA_VISIBLE_DEVICES=0 .../envs/predictor/bin/python docs/build_media.py
"""
import csv, glob, json, os, subprocess, sys
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
from vae_wrapper import VAEWrapper  # noqa: E402

OUT = "docs/static"; os.makedirs(f"{OUT}/videos", exist_ok=True); os.makedirs(f"{OUT}/images", exist_ok=True)
O = "outputs/online_ft"
FONT = next((p for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"] if os.path.exists(p)), None)
font = lambda s: ImageFont.truetype(FONT, s) if FONT else ImageFont.load_default()
dev = "cuda" if torch.cuda.is_available() else "cpu"
vae = VAEWrapper("stabilityai/sd-vae-ft-mse").to(dev).eval()


def episode(d, ep):
    fs = sorted(glob.glob(f"{d}/{ep}_*.npz"), key=lambda f: int(os.path.basename(f)[:-4].split("_")[1]))
    return [dict(np.load(f)) for f in fs]


@torch.no_grad()
def dec(lat):
    z = torch.from_numpy(lat.astype(np.float32))[None].to(dev)
    u8 = lambda x: (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return u8(vae.decode(z[..., :28])), u8(vae.decode(z[..., 28:]))


def frame(cols, title, step, status=None):
    """cols = [(label, (img_cam1, img_cam2)), ...] -> one video frame: a header, the column labels, cam1 row over cam2 row."""
    W, Hh, Lh = 224, 34, 26
    im = Image.new("RGB", (W * len(cols), Hh + Lh + 2 * W), (255, 255, 255)); dr = ImageDraw.Draw(im)
    dr.text((8, 7), title, fill=(20, 20, 20), font=font(17))
    st = f"step {step}" + (f"   {status}" if status else "")
    dr.text((W * len(cols) - 8 - dr.textlength(st, font=font(15)), 9), st, fill=(90, 90, 90), font=font(15))
    for i, (lab, (a, b)) in enumerate(cols):
        dr.text((i * W + (W - dr.textlength(lab, font=font(14))) / 2, Hh + 4), lab, fill=(50, 50, 50), font=font(14))
        im.paste(Image.fromarray(a), (i * W, Hh + Lh)); im.paste(Image.fromarray(b), (i * W, Hh + Lh + W))
    return np.array(im)


def write(frames, name, fps=2, hold=3):
    frames = frames + [frames[-1]] * hold
    h, w = frames[0].shape[:2]; h += h % 2; w += w % 2
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", f"{OUT}/videos/{name}.mp4"], stdin=subprocess.PIPE)
    for f in frames:
        g = np.zeros((h, w, 3), np.uint8) + 255; g[:f.shape[0], :f.shape[1]] = f; p.stdin.write(g.tobytes())
    p.stdin.close(); p.wait()
    Image.fromarray(frames[len(frames) // 2]).save(f"{OUT}/images/{name}.jpg", quality=85)       # poster
    print("wrote", name, len(frames), "frames", flush=True)


def ours_video(run_dir, ep, name, title, status, with_pred=True):
    E = episode(run_dir, ep); F = []
    for i, x in enumerate(E):
        cols = [("Robot camera (true)", (x["obs1"], x["obs2"]))]
        if with_pred:
            cols.append(("Prediction only", dec(x["lat_zh"]) if "lat_zh" in x else (x["obs1"], x["obs2"])))
        cols.append(("Policy input (ours)", (x["out1"], x["out2"])))
        F.append(frame(cols, title, 30 * i, status if i == len(E) - 1 else None))
    write(F, name)


def pair_video(dvst_dir, dvst_ep, ours_dir, ours_ep, name, title):
    D, W_ = episode(dvst_dir, dvst_ep), episode(ours_dir, ours_ep); F = []
    for i in range(max(len(D), len(W_))):
        d, w = D[min(i, len(D) - 1)], W_[min(i, len(W_) - 1)]
        F.append(frame([("DVST: true", (d["obs1"], d["obs2"])), ("DVST: policy input", (d["out1"], d["out2"])),
                        ("Ours: true", (w["obs1"], w["obs2"])), ("Ours: policy input", (w["out1"], w["out2"]))], title, 30 * i))
    write(F, name)


# -- videos ------------------------------------------------------------------------------------------------------------- #
# teaser: CDL-C 8 dB, same rate (~0.25).  DVST 0/30 there; ours 17/30 (dump ep n = client ep n+4 in that run, see closed_loop.csv notes)
pair_video("outputs/dvst_online_0_10_r25_snr8/log", 7, f"{O}/old_nogate_snr_8dB_k4_30ep_17", 3, "teaser_8dB", "CDL-C 8 dB, same rate (0.25)")
ours_video(f"{O}/old_nogate_res_20dB_k4_30ep_23", 2, "ours_20dB", "Ours, CDL-C 20 dB", "SUCCESS")
ours_video(f"{O}/old_nogate_snr_12dB_k4_30ep_23", 4, "ours_12dB", "Ours, CDL-C 12 dB", "SUCCESS")
ours_video(f"{O}/old_nogate_snr_8dB_k4_30ep_17", 3, "ours_8dB", "Ours, CDL-C 8 dB", "SUCCESS")
ours_video(f"{O}/old_nogate_snr_4dB_k4_30ep_11", 2, "ours_4dB_fail", "Ours, CDL-C 4 dB (failure)", "FAILED")

# -- results.json from the tidy CSV --------------------------------------------------------------------------------------- #
rows = list(csv.DictReader(open("results/exp1_main_closed_loop/data/closed_loop.csv")))
pick = lambda f: [r for r in rows if f(r) and r["seed"] == "42" and r["task"] == "pick_blue_cube_to_tray"]
pt = lambda r, extra=None: dict(snr=float(r["snr_db"]) if r["snr_db"] else None, succ=int(r["n_success"]), n=int(r["n_episodes"]), run=r["run_id"], **(extra or {}))
ours = [pt(r) for r in pick(lambda r: r["mode"] == "wz2cam" and r["semantic"] == "K_nogate_snr_k4_analog")]
ours += [pt(r) for r in pick(lambda r: r["mode"] == "wz2cam" and r["semantic"] == "K_nogate_res_k4_analog") if not any(p["snr"] == float(r["snr_db"]) for p in ours)]
dvst = [pt(r, {"rate_matched": "eta" in r["semantic"]}) for r in pick(lambda r: r["mode"] == "dvst" and "lam0.01" in r["semantic"])]
dvst = sorted({p["snr"]: p for p in sorted(dvst, key=lambda p: p["rate_matched"])}.values(), key=lambda p: p["snr"])   # prefer the rate-matched row per SNR
pred = [pt(r) for r in pick(lambda r: r["semantic"].startswith("predictor_only"))]
json.dump({"ours": sorted(ours, key=lambda p: p["snr"]), "dvst": dvst, "predictor_only": pred[-1] if pred else None,
           "source": "results/exp1_main_closed_loop/data/closed_loop.csv"}, open(f"{OUT}/results.json", "w"), indent=1)
print("wrote results.json", len(ours), "ours,", len(dvst), "dvst")
