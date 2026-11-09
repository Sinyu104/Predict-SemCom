"""Object (cube) cell mask for a Stage-2 latent cache: decode each true latent ZT with the frozen VAE (both cameras), find the
saturated non-green pixels (the cubes; the floor is low-saturation blue, the arm grey, the tray green), mark a latent cell when
>= 4 of its 8x8 pixels are cube pixels, dilate by one cell (cube edges).  Writes obj_shard<i>.pt = {"OBJ": (N,1,28,56) uint8}
next to each shard<i>.pt (the name does not match the trainer's shard*.pt glob).  --viz saves a check image.
    CUDA_VISIBLE_DEVICES=1 .../envs/predictor/bin/python experiments/make_obj_mask.py --pre outputs/rollout_pose_unf2 --viz <png>
"""
import argparse, glob, os, sys, numpy as np, torch
import torch.nn.functional as Fn
sys.path.insert(0, ".")
from vae_wrapper import VAEWrapper
ap = argparse.ArgumentParser(); ap.add_argument("--pre", required=True); ap.add_argument("--viz", default=""); ap.add_argument("--sat", type=float, default=0.7); ap.add_argument("--vmin", type=float, default=0.5); ap.add_argument("--viz_only", action="store_true"); a = ap.parse_args(); SAT, VMIN = a.sat, a.vmin
vae = VAEWrapper("stabilityai/sd-vae-ft-mse").to("cuda:0").eval()

def cube_px(img):                         # img (B,3,H,W) in [0,1] -> (B,H,W) bool
    mx, mn = img.max(1).values, img.min(1).values; sat = (mx - mn) / mx.clamp(min=1e-3)
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    green = (g > r) & (g > b)              # tray
    return (sat > SAT) & (mx > VMIN) & ~green

shown = []
for f in sorted(glob.glob(os.path.join(a.pre, "shard*.pt"))):
    Z = torch.load(f)["ZT"].float()[::300] if a.viz_only else torch.load(f)["ZT"].float(); OBJ = torch.zeros(len(Z), 1, 28, 56, dtype=torch.uint8)
    with torch.no_grad():
        for i in range(0, len(Z), 32):
            z = Z[i:i + 32].cuda(); cells = []
            for off in (0, 28):
                img = vae.decode(z[..., off:off + 28]).clamp(0, 1)
                px = cube_px(img).float()
                c = (Fn.avg_pool2d(px[:, None], 8) * 64 >= 4).float()
                cells.append(c)
                if a.viz and len(shown) < 6 and off == 0:
                    im = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy(); m = px[0].cpu().numpy() > 0
                    im2 = im.copy(); im2[m] = [255, 0, 0]; shown.append(np.concatenate([im, im2], 1))
            c = torch.cat(cells, -1); c = (Fn.max_pool2d(c, 3, 1, 1) > 0)      # one-cell dilation (within each camera's 28x28 is close enough)
            OBJ[i:i + 32] = c.to(torch.uint8).cpu()
    out = os.path.join(os.path.dirname(f), "obj_" + os.path.basename(f))
    if a.viz_only: print(f, "object cells", OBJ.float().mean().item()); continue
    torch.save({"OBJ": OBJ}, out)
    print(f"{out}: {len(Z)} frames, object cells {OBJ.float().mean().item():.4f} of grid", flush=True)
if a.viz:
    from PIL import Image; Image.fromarray(np.concatenate(shown, 0)).save(a.viz)
