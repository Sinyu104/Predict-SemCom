"""
inference.py  —  Evaluation and figure generation.

Metrics
-------
  L_distortion   MSE(ẑ_t, z_t) in latent space
  L_rate         KL(q(ŝ_t|z_t) || p(ŝ_t|ẑ_t^{pred}))
  Action MSE     ||â_t − a_t||² comparing predicted vs ground-truth actions
  Task loss      Cross-entropy of LLM action tokens (lower = better)

Figures saved to output_dir
----------------------------
  latent_snr{snr}.png    L_distortion and L_rate over time
"""

import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import SemComSystem


# ========================================================================== #
#  Main inference function                                                    #
# ========================================================================== #

def run_inference(
    config:             dict,
    data_path:          str,
    ckpt_path:          str,
    device:             torch.device,
    agent,
    max_batches:        int  = 100,
    save_figures:       bool = True,
    interference_steps: list | None = None,
):
    """
    Parameters
    ----------
    config       : CONFIG dict
    data_path    : path to HDF5 trajectory file
    ckpt_path    : checkpoint to evaluate
    device       : torch device
    agent        : OpenVLAAgent or OpenVLAStub
    max_batches  : cap evaluation to this many batches
    save_figures : save PNG result figures to output_dir
    """
    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Build and load system ─────────────────────────────────────────── #
    system = SemComSystem(config).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    system.load_state_dict(ckpt["system_state"], strict=False)
    system.eval()
    print(f"[inference] Checkpoint : {ckpt_path}")
    print(f"[inference] snr_db={config['snr_db']} dB  "
          f"latent_dim={config['latent_dim']}  D_jscc={config['D_jscc']}")

    # ── Dataset ───────────────────────────────────────────────────────── #
    from dataset import GNMTrajectoryDataset
    dataset = GNMTrajectoryDataset(
        data_path,
        config["obs_height"],
        config["obs_width"],
        config["obs_channels"],
    )
    loader = DataLoader(
        dataset,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = 0,
        pin_memory  = device.type == "cuda",
    )
    print(f"[inference] Dataset size: {len(dataset)} samples")

    instruction = config.get("openvla_instruction", "")

    # ── Tracking ─────────────────────────────────────────────────────── #
    metrics = {k: [] for k in ["distortion", "rate", "action_mse", "task_loss"]}

    with torch.no_grad():
        for bi, (obs_t, act_t, obs_tp1) in enumerate(loader):
            if bi >= max_batches:
                break
            obs_t   = obs_t.to(device)
            act_t   = act_t.to(device)
            obs_tp1 = obs_tp1.to(device)

            # Encode both frames (frozen ViT + TokenEncoder)
            tok_prev = agent.encode_image(obs_t).to(device)
            tok_curr = agent.encode_image(obs_tp1).to(device)
            z_prev   = system.token_encoder(tok_prev)
            z_t      = system.token_encoder(tok_curr)

            # Predictor side information
            z_pred_si = system.predictor(z_prev, act_t)

            # JSCC pipeline
            mu_enc, log_var_enc, s_t = system.jscc_encoder(z_t, sample=False)
            s_hat                    = system.channel(s_t)
            mu_prior, log_var_prior  = system.side_info_encoder(z_pred_si)
            z_hat                    = system.jscc_decoder(s_hat, z_pred_si)

            # Latent metrics
            L_dist = F.mse_loss(z_hat, z_t).item()
            L_rate = SemComSystem.rate_loss(
                mu_enc, log_var_enc, mu_prior, log_var_prior, config["snr_db"]
            ).item()

            # Action prediction from reconstructed latent
            tokens_hat = system.token_decoder(z_hat)
            a_hat      = agent.predict_action_from_tokens(tokens_hat, instruction)
            a_mse      = F.mse_loss(a_hat, act_t.cpu()).item()

            # Task loss (cross-entropy)
            task_loss = 0.0
            if hasattr(agent, "task_loss_from_tokens"):
                try:
                    task_loss = agent.task_loss_from_tokens(
                        tokens_hat, instruction, act_t.cpu()
                    ).item()
                except Exception:
                    pass

            metrics["distortion"].append(L_dist)
            metrics["rate"].append(L_rate)
            metrics["action_mse"].append(a_mse)
            metrics["task_loss"].append(task_loss)

    # ── Summary ───────────────────────────────────────────────────────── #
    def avg(lst):
        return sum(lst) / max(len(lst), 1)

    print("\n" + "=" * 60)
    print(f"  INFERENCE  SNR={config['snr_db']}dB  λ={config.get('lambda_rate', 0.01)}")
    print("=" * 60)
    print(f"  L_distortion (latent MSE) : {avg(metrics['distortion']):.6f}")
    print(f"  L_rate (KL)               : {avg(metrics['rate']):.6f}")
    print(f"  Action MSE                : {avg(metrics['action_mse']):.6f}")
    print(f"  Task loss (cross-entropy) : {avg(metrics['task_loss']):.5f}")
    print("=" * 60)

    # ── Figure ────────────────────────────────────────────────────────── #
    if save_figures:
        tag = f"snr{config['snr_db']}"
        _save_timeline(
            metrics["distortion"],
            metrics["rate"],
            metrics["action_mse"],
            interference_steps,
            os.path.join(out_dir, f"latent_{tag}.png"),
        )

    return metrics


# ========================================================================== #
#  Figure helper                                                              #
# ========================================================================== #

def _save_timeline(
    distortions:        list,
    rates:              list,
    action_mses:        list,
    interference_steps: list | None,
    path: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(len(distortions)))
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(steps, distortions, color="steelblue", lw=1.5)
        axes[0].set_ylabel("L_distortion")
        axes[0].set_title("Latent MSE (ẑ_t vs z_t)")

        axes[1].plot(steps, rates, color="darkorange", lw=1.5)
        axes[1].set_ylabel("L_rate (KL)")
        axes[1].set_title("Wyner-Ziv Rate")

        axes[2].plot(steps, action_mses, color="crimson", lw=1.5)
        axes[2].set_ylabel("Action MSE")
        axes[2].set_xlabel("Time step")
        axes[2].set_title("Action Prediction Error")

        if interference_steps:
            for s in interference_steps:
                for ax in axes:
                    ax.axvline(s, color="red", ls=":", lw=1.5, alpha=0.7)

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[inference] Timeline → {path}")
    except Exception as e:
        print(f"[inference] Could not save timeline: {e}")
