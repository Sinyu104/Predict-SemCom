"""
inference.py  —  Evaluation for the VAE-latent Predictive SemCom System.

Metrics
-------
  L_distortion  MSE(z̃_t, z_t) in VAE latent space
  L_rate        KL(q(ŝ_t|z_t) || p(ŝ_t|ẑ_t))
  PSNR          Peak signal-to-noise ratio of decoded image x̃_t vs x_t
  Action MSE    ||â_t − a_t||² (first action of predicted chunk vs GT)
"""

import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import SemComSystem


def run_inference(
    config:             dict,
    data_path:          str,
    ckpt_path:          str,
    device:             torch.device,
    vae,
    policy,
    max_batches:        int  = 100,
    save_figures:       bool = True,
):
    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    system = SemComSystem(config).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    system.load_state_dict(ckpt["system_state"], strict=False)
    system.eval()
    print(f"[inference] Checkpoint : {ckpt_path}")
    print(f"[inference] snr_db={config['snr_db']}dB  D_jscc={config['D_jscc']}")

    from dataset import GNMTrajectoryDataset
    dataset = GNMTrajectoryDataset(
        data_path, config["obs_height"], config["obs_width"], config["obs_channels"]
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"],
                        shuffle=False, num_workers=0)
    print(f"[inference] Dataset: {len(dataset)} samples")

    metrics = {"distortion": [], "rate": [], "psnr": [], "action_mse": []}

    @torch.no_grad()
    def encode_vae(x):
        if vae is None:
            C, H, W = config["C_vae"], config["H_vae"], config["W_vae"]
            return torch.zeros(x.size(0), C, H, W, device=device)
        return vae.encode(x.to(device))

    @torch.no_grad()
    def decode_vae(z):
        if vae is None:
            B = z.size(0)
            return torch.zeros(B, 3, config["obs_height"], config["obs_width"], device=device)
        return vae.decode(z)

    with torch.no_grad():
        for bi, (obs_t, act_t, obs_tp1) in enumerate(loader):
            if bi >= max_batches:
                break
            obs_t   = obs_t.to(device)
            act_t   = act_t.to(device)
            obs_tp1 = obs_tp1.to(device)

            z_prev = encode_vae(obs_t)    # (B, C_vae, Hv, Wv)
            z_curr = encode_vae(obs_tp1)

            # World model side information
            z_pred = system.predictor(z_prev, act_t)

            # JSCC pipeline
            mu_enc, log_var_enc, s_t = system.jscc_encoder(z_curr, sample=False)
            s_hat                    = system.channel(s_t)
            mu_prior, log_var_prior  = system.side_info_encoder(z_pred)
            z_tilde                  = system.denoising_decoder(s_hat, z_pred)

            # Latent metrics
            L_dist = F.mse_loss(z_tilde, z_curr).item()
            L_rate = SemComSystem.rate_loss(
                mu_enc, log_var_enc, mu_prior, log_var_prior, config["snr_db"]
            ).item()

            # Image-level PSNR
            x_tilde = decode_vae(z_tilde)
            mse_pix = F.mse_loss(x_tilde, obs_tp1).item()
            psnr    = 10.0 * math.log10(1.0 / (mse_pix + 1e-8))

            # Action prediction (first action of chunk vs ground truth)
            chunk   = policy.predict_action_chunk(x_tilde.cpu())   # (B, K, 7)
            a_hat   = chunk[:, 0].to(act_t.device)                 # (B, 7)
            a_mse   = F.mse_loss(a_hat.float(), act_t.float()).item()

            metrics["distortion"].append(L_dist)
            metrics["rate"].append(L_rate)
            metrics["psnr"].append(psnr)
            metrics["action_mse"].append(a_mse)

    def avg(lst):
        return sum(lst) / max(len(lst), 1)

    print("\n" + "=" * 60)
    print(f"  INFERENCE  SNR={config['snr_db']}dB  λ={config.get('lambda_rate', 0.01)}")
    print("=" * 60)
    print(f"  L_distortion (latent MSE) : {avg(metrics['distortion']):.6f}")
    print(f"  L_rate (KL)               : {avg(metrics['rate']):.6f}")
    print(f"  PSNR  (image quality)     : {avg(metrics['psnr']):.2f} dB")
    print(f"  Action MSE (chunk[0])     : {avg(metrics['action_mse']):.6f}")
    print("=" * 60)

    if save_figures:
        _save_timeline(metrics, config["snr_db"], out_dir)

    return metrics


def _save_timeline(metrics: dict, snr_db: float, out_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        keys   = ["distortion", "rate", "psnr", "action_mse"]
        titles = ["Latent MSE (z̃_t vs z_t)", "Wyner-Ziv Rate (KL)",
                  "Decoded Image PSNR (dB)", "Action Chunk MSE (step 0)"]
        colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

        fig, axes = plt.subplots(len(keys), 1, figsize=(12, 3 * len(keys)), sharex=True)
        for ax, k, title, c in zip(axes, keys, titles, colors):
            ax.plot(metrics[k], color=c, lw=1.5)
            ax.set_ylabel(k)
            ax.set_title(title)
        axes[-1].set_xlabel("Batch")
        plt.suptitle(f"SNR = {snr_db} dB", fontsize=13)
        plt.tight_layout()
        path = os.path.join(out_dir, f"inference_snr{snr_db}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[inference] Timeline → {path}")
    except Exception as e:
        print(f"[inference] Could not save timeline: {e}")
