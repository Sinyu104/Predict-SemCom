"""
inference.py  —  Evaluation and figure generation.

Metrics
-------
  PSNR          reconstruction quality of y_rec vs x_t
  SSIM          structural similarity
  Task loss     OpenVLA action prediction error (lower = better)
  nnz / bits    transmission cost per step
  delta_z_norm  ||Δz|| per step (twin gap)

Figures saved to output_dir
----------------------------
  recon_tau{tau}_snr{snr}.png      original vs reconstructed grid
  timeline_tau{tau}_snr{snr}.png   ||Δz||, bits, task_loss over time
"""

import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import PredictiveSemComSystem


# ========================================================================== #
#  Metric helpers                                                             #
# ========================================================================== #

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def ssim_simple(pred: torch.Tensor, target: torch.Tensor) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = pred.mean(), target.mean()
    s1  = ((pred   - mu1) ** 2).mean()
    s2  = ((target - mu2) ** 2).mean()
    s12 = ((pred - mu1) * (target - mu2)).mean()
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return (num / den).item()


# ========================================================================== #
#  Main inference function                                                    #
# ========================================================================== #

def run_inference(
    config:              dict,
    data_path:           str,
    ckpt_path:           str,
    device:              torch.device,
    agent,
    max_batches:         int  = 100,
    save_figures:        bool = True,
    interference_steps:  list | None = None,
):
    """
    Parameters
    ----------
    config             : CONFIG dict
    data_path          : path to HDF5 trajectory file
    ckpt_path          : path to the checkpoint to evaluate
    device             : torch device
    agent              : OpenVLAAgent or OpenVLAStub instance
    max_batches        : cap evaluation to this many batches
    save_figures       : save PNG result figures to output_dir
    interference_steps : optional list of step indices where interference
                         was injected (annotated on the timeline figure)
    """
    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Build and load system ─────────────────────────────────────────── #
    system = PredictiveSemComSystem(config, agent=agent).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    system.load_state_dict(ckpt["system_state"], strict=False)
    system.eval()
    print(f"[inference] Checkpoint : {ckpt_path}")
    print(f"[inference] tau_mode={config['tau_mode']}  "
          f"tau={config['tau']}  snr_db={config['snr_db']} dB")

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

    baseline_bits = config["latent_dim"] * 16   # full float16

    # ── Tracking ─────────────────────────────────────────────────────── #
    metrics = {k: [] for k in [
        "psnr", "ssim", "task_loss",
        "nnz", "bits", "compression", "delta_z_norm",
    ]}
    first_orig  = None
    first_recon = None

    with torch.no_grad():
        z_prev = a_prev = h = None

        for bi, (obs_t, act_t, _) in enumerate(loader):
            if bi >= max_batches:
                break
            obs_t = obs_t.to(device)
            act_t = act_t.to(device)
            B     = obs_t.size(0)

            if z_prev is None or z_prev.size(0) != B:
                z_prev = torch.zeros(B, config["latent_dim"], device=device)
                a_prev = torch.zeros(B, config["action_dim"],  device=device)
                h      = None

            out = system(
                obs_t, z_prev, a_prev, h,
                bypass_predictor=False,
                sparsify=True,
                sample_z=False,
                hard=True,
            )

            z_prev = out["z_t"].detach()
            a_prev = act_t
            h      = out["h_next"]

            y_rec  = out["y_rec"]

            # Metrics
            metrics["psnr"].append(psnr(y_rec, obs_t))
            metrics["ssim"].append(ssim_simple(y_rec, obs_t))
            metrics["task_loss"].append(
                agent.task_loss_forward(y_rec, act_t).item()
                if hasattr(agent, "task_loss_forward") else 0.0
            )
            metrics["nnz"].append(out["nnz"].item())
            metrics["bits"].append(out["bits_per_step"].item())
            metrics["compression"].append(
                out["bits_per_step"].item() / baseline_bits
            )
            metrics["delta_z_norm"].append(
                out["delta_z"].norm(dim=-1).mean().item()
            )

            if bi == 0:
                first_orig  = obs_t[:8].cpu()
                first_recon = y_rec[:8].cpu()

    # ── Summary ───────────────────────────────────────────────────────── #
    def avg(lst):
        return sum(lst) / max(len(lst), 1)

    ld = config["latent_dim"]
    print("\n" + "=" * 65)
    print(f"  INFERENCE  tau={config['tau']}  SNR={config['snr_db']}dB")
    print("=" * 65)
    print(f"  PSNR           : {avg(metrics['psnr']):.2f} dB")
    print(f"  SSIM           : {avg(metrics['ssim']):.4f}")
    print(f"  Task loss      : {avg(metrics['task_loss']):.5f}")
    print(f"  Avg nnz        : {avg(metrics['nnz']):.1f}/{ld} dims "
          f"({100*avg(metrics['nnz'])/ld:.1f}% transmitted)")
    print(f"  Bits/step      : {avg(metrics['bits']):.1f}  "
          f"(baseline={baseline_bits})")
    print(f"  Compression    : {avg(metrics['compression']):.3f}×  "
          f"({100*(1-avg(metrics['compression'])):.1f}% savings)")
    print(f"  ||Δz|| mean    : {avg(metrics['delta_z_norm']):.4f}")
    print("=" * 65)

    # ── Figures ───────────────────────────────────────────────────────── #
    if save_figures:
        tag = f"tau{config['tau']}_snr{config['snr_db']}"
        _save_recon_grid(
            first_orig, first_recon,
            os.path.join(out_dir, f"recon_{tag}.png"),
        )
        _save_timeline(
            metrics["delta_z_norm"],
            metrics["bits"],
            metrics["task_loss"],
            baseline_bits,
            interference_steps,
            os.path.join(out_dir, f"timeline_{tag}.png"),
        )

    return metrics


# ========================================================================== #
#  Figure helpers                                                             #
# ========================================================================== #

def _save_recon_grid(originals, recons, path: str):
    try:
        import torchvision.utils as vutils
        import torchvision.transforms.functional as TF

        n    = min(8, originals.size(0))
        grid = vutils.make_grid(
            torch.cat([originals[:n], recons[:n]], dim=0),
            nrow=n, padding=2, normalize=False,
        )
        TF.to_pil_image(grid.clamp(0, 1)).save(path)
        print(f"[inference] Reconstruction grid → {path}")
    except Exception as e:
        print(f"[inference] Could not save reconstruction grid: {e}")


def _save_timeline(
    delta_z_norms: list,
    bits:          list,
    task_losses:   list,
    baseline_bits: int,
    interference_steps: list | None,
    path: str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(len(delta_z_norms)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Top panel: ||Δz|| and bits
        ax1b = ax1.twinx()
        ax1.plot(steps, delta_z_norms, color="steelblue",
                 lw=1.5, label=r"$\|\Delta z_t\|$")
        ax1b.plot(steps, bits, color="darkorange",
                  lw=1.2, alpha=0.8, label="bits/step")
        ax1b.axhline(baseline_bits, color="grey", ls="--",
                     lw=1.0, label=f"baseline ({baseline_bits} bits)")
        ax1.set_ylabel(r"$\|\Delta z_t\|$", color="steelblue")
        ax1b.set_ylabel("Bits transmitted", color="darkorange")

        # Bottom panel: task loss
        ax2.plot(steps, task_losses, color="crimson",
                 lw=1.2, label="task loss")
        ax2.set_ylabel("Task loss")
        ax2.set_xlabel("Time step")

        # Interference markers
        if interference_steps:
            for s in interference_steps:
                ax1.axvline(s, color="red", ls=":", lw=1.5, alpha=0.7)
                ax2.axvline(s, color="red", ls=":", lw=1.5, alpha=0.7)
            ax1.axvline(-1, color="red", ls=":", lw=1.5,
                        alpha=0.7, label="interference")

        for ax, axb in [(ax1, ax1b)]:
            l1, lb1 = ax.get_legend_handles_labels()
            l2, lb2 = axb.get_legend_handles_labels()
            ax.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

        fig.suptitle("Innovation Timeline — Twin Gap & Transmission Cost")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[inference] Timeline → {path}")
    except Exception as e:
        print(f"[inference] Could not save timeline: {e}")
