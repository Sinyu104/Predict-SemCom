"""
trainer.py  —  Multi-GPU training pipeline (4 × T4 on Linux server).

Launch with torchrun, NOT python:
    torchrun --nproc_per_node=4 main.py --train --stage 2 ...

DDP design
-----------
1. Only trainable SemCom modules are DDP-wrapped per stage.
2. The OpenVLAAgent is NEVER DDP-wrapped — only rank 0 has the real VLA.
3. encode_image is called on rank 0 and tokens are broadcast to all ranks
   before the forward pass (only rank 0 has the real ViT).
4. Checkpoints are saved only from rank 0.

Stage overview
--------------
Stage 1  Predictor                     L_world = MSE(ẑ_t^{pred}, sg(tokens_t))
Stage 2  JsccEncoder + JsccDecoder +   L = L_distortion + λ·L_rate
         SideInfoEncoder
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime

from models import SemComSystem
from dataset import GNMTrajectoryDataset, ClipDataset


# ========================================================================== #
#  DDP helpers                                                                #
# ========================================================================== #

def init_distributed() -> tuple[int, int, bool]:
    if not dist.is_available() or not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank       = int(os.environ.get("RANK",       0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    else:
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count()
    return rank, world_size, (rank == 0)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / dist.get_world_size()
    return tensor


def build_ddp_loaders(
    config: dict, hdf5_path: str, rank: int, world_size: int,
    clip_length: int = 1,
) -> tuple[DataLoader, DataLoader]:
    from torch.utils.data import random_split

    if clip_length > 1:
        full_ds = ClipDataset(
            hdf5_path    = hdf5_path,
            obs_height   = config["obs_height"],
            obs_width    = config["obs_width"],
            obs_channels = config["obs_channels"],
            clip_length  = clip_length,
            stride       = config.get("clip_stride", 1),
        )
    else:
        full_ds = GNMTrajectoryDataset(
            hdf5_path    = hdf5_path,
            obs_height   = config["obs_height"],
            obs_width    = config["obs_width"],
            obs_channels = config["obs_channels"],
        )

    if full_ds._action_dim is not None:
        expected = config.get("action_dim", 7)
        if full_ds._action_dim != expected:
            raise ValueError(
                f"action_dim mismatch: HDF5={full_ds._action_dim} config={expected}"
            )

    n_total = len(full_ds)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    nw = config.get("num_workers", 0)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], sampler=train_sampler,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], sampler=val_sampler,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )
    if rank == 0:
        print(
            f"[dataset] total={n_total}  train={n_train}  val={n_val}  "
            f"per-GPU train batches ≈ {len(train_loader)}"
        )
    return train_loader, val_loader



# ========================================================================== #
#  Base Trainer                                                               #
# ========================================================================== #

class BaseTrainer:
    def __init__(
        self,
        config:     dict,
        data_path:  str,
        device:     torch.device,
        agent,
        rank:       int,
        world_size: int,
    ):
        self.config     = config
        self.device     = device
        self.agent      = agent
        self.rank       = rank
        self.world_size = world_size
        self.is_main    = (rank == 0)
        self.out_dir    = config["output_dir"]

        if self.is_main:
            os.makedirs(self.out_dir, exist_ok=True)

        self.system = SemComSystem(config).to(device)

        self.train_loader, self.val_loader = self._build_loaders(
            config, data_path, rank, world_size
        )
        self.writer = None
        if self.is_main:
            self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, "tb_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

        self.mse = nn.MSELoss()

    def _build_loaders(self, config, data_path, rank, world_size):
        return build_ddp_loaders(config, data_path, rank, world_size)

    def _wrap_ddp(self, module: nn.Module) -> nn.Module:
        if self.world_size > 1:
            return DDP(
                module,
                device_ids             = [self.device.index],
                output_device          = self.device.index,
                find_unused_parameters = self.config.get("ddp_find_unused", False),
            )
        return module

    def _unwrap(self, module) -> nn.Module:
        return module.module if isinstance(module, DDP) else module

    def _encode_all_ranks(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode images to ViT tokens across DDP ranks, then layer-normalise.

        Only rank 0 has the real VLA.  All ranks gather their obs to rank 0,
        rank 0 encodes the full batch, then tokens are broadcast back.
        Returns detached, layer-normalised tokens on self.device.

        LayerNorm is applied per patch token across D_vit dimensions so that
        raw DINOv2 and SigLIP features (which have very different scales and
        spiky distributions) are brought to zero mean / unit variance before
        the predictor or JSCC modules see them.
        """
        if not dist.is_initialized() or self.world_size == 1:
            tokens = self.agent.encode_image(obs).to(self.device)
        else:
            B = obs.size(0)
            all_obs_buf = [torch.empty_like(obs) for _ in range(self.world_size)]
            dist.all_gather(all_obs_buf, obs.contiguous())

            N = self.config["N_patches"]
            D = self.config["D_vit"]
            all_tokens = torch.zeros(
                self.world_size * B, N, D, device=self.device, dtype=obs.dtype
            )
            if self.rank == 0:
                all_obs    = torch.cat(all_obs_buf, dim=0).to(self.device)
                all_tokens = self.agent.encode_image(all_obs).to(self.device)

            dist.broadcast(all_tokens, src=0)
            tokens = all_tokens[self.rank * B : (self.rank + 1) * B]

        # Normalise each patch token across D_vit so the predictor sees a
        # consistent scale regardless of DINOv2 / SigLIP magnitude differences.
        tokens = F.layer_norm(tokens, [tokens.shape[-1]])
        return tokens.detach()

    def save_checkpoint(self, filename: str, extra: dict | None = None):
        if not self.is_main:
            return
        path = os.path.join(self.out_dir, filename)
        # Strip DDP '.module.' so checkpoints load cleanly without DDP
        raw_state = self._unwrap(self.system).state_dict()
        clean_state = {k.replace(".module.", "."): v for k, v in raw_state.items()}
        payload = {"system_state": clean_state}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"  [ckpt] Saved → {path}")

    def load_checkpoint(self, path: str) -> dict:
        ckpt  = torch.load(path, map_location="cpu")
        model = self._unwrap(self.system)
        model_keys = set(model.state_dict().keys())
        state = ckpt["system_state"]

        # save_checkpoint strips ".module." from DDP-wrapped submodule keys.
        # If a submodule has since been DDP-wrapped, its keys now contain
        # ".module." and won't match the saved keys directly.  Remap by
        # inserting ".module." after the first path component when needed.
        remapped = {}
        for k, v in state.items():
            if k in model_keys:
                remapped[k] = v
            else:
                parts = k.split('.', 1)
                if len(parts) == 2:
                    candidate = f"{parts[0]}.module.{parts[1]}"
                    remapped[candidate if candidate in model_keys else k] = v
                else:
                    remapped[k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if self.is_main:
            if missing:
                print(f"  [ckpt] WARNING: {len(missing)} missing keys — "
                      f"e.g. {missing[:3]}")
            print(f"  [ckpt] Loaded ← {path}")
        return ckpt

    def _load_resume(self, path: str) -> tuple[int, float]:
        ckpt        = self.load_checkpoint(path)
        start_epoch = ckpt.get("epoch", 0) + 1
        best        = ckpt.get("best",  math.inf)
        if "optimizer_state" in ckpt and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.is_main:
            print(f"  [ckpt] Checkpoint at epoch {start_epoch - 1}  "
                  f"→ resuming at epoch {start_epoch}  best={best:.6f}")
        del ckpt
        torch.cuda.empty_cache()
        return start_epoch, best

    def _log(self, tag: str, values: dict, step: int):
        if self.is_main and self.writer is not None:
            for k, v in values.items():
                self.writer.add_scalar(f"{tag}/{k}", v, step)


# ========================================================================== #
#  Stage 2  —  Predictor (action-conditioned V-JEPA 2)                       #
# ========================================================================== #

class Stage1Trainer(BaseTrainer):
    """
    Train the Predictor following V-JEPA 2-AC (paper §3.1).

    Training uses T-frame clips (default T=16 at 4 fps = 4 sec).
    Two losses are combined (both L1):
      Eq. 2 — teacher-forcing : (1/T) Σ_k ||ẑ_{k+1} - z_{k+1}||_1
      Eq. 3 — rollout (2-step): ||ẑ_3^{AR} - z_3||_1
      Total : L = L_tf + L_rollout

    The frozen ViT encodes each frame independently.
    Only the Predictor is trained.
    """

    def __init__(
        self,
        config:      dict,
        data_path:   str,
        device:      torch.device,
        agent,
        rank:        int,
        world_size:  int,
        resume_ckpt: str | None = None,
    ):
        self.clip_length = config.get("clip_length", 16)
        super().__init__(config, data_path, device, agent, rank, world_size)

        self.system.predictor = self._wrap_ddp(self.system.predictor)
        self.optimizer = optim.AdamW(
            self._unwrap(self.system.predictor).parameters(),
            lr=config["learning_rate"], weight_decay=0.05,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=1e-5
        )
        self.start_epoch = 1
        self.best        = math.inf
        if resume_ckpt:
            self.start_epoch, self.best = self._load_resume(resume_ckpt)

    def _build_loaders(self, config, data_path, rank, world_size):
        return build_ddp_loaders(config, data_path, rank, world_size,
                                 clip_length=self.clip_length)

    def _encode_clip(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, C, H, W)
        Returns tokens : (B, T, N, D_vit)  — each frame encoded independently.
        """
        B, T, C, H, W = frames.shape
        flat   = frames.reshape(B * T, C, H, W)
        tokens = self._encode_all_ranks(flat)                # (B*T, N, D_vit)
        return tokens.reshape(B, T, tokens.size(1), tokens.size(2))

    def _run_epoch(self, loader, train: bool, epoch: int) -> dict:
        self._unwrap(self.system.predictor).train(train)
        if train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        totals = {"tf": 0.0, "rollout": 0.0, "cosine": 0.0, "cosine_orig": 0.0,
                  "total": 0.0, "tf_plain": 0.0, "rollout_plain": 0.0}
        n      = 0
        phase  = "train" if train else "val"
        pbar   = tqdm(loader, desc=f"  {phase} ep {epoch}",
                      disable=not self.is_main, leave=False, dynamic_ncols=True)

        with torch.set_grad_enabled(train):
            for frames, actions, poses in pbar:
                frames  = frames.to(self.device)              # (B, T, C, H, W)
                actions = actions.to(self.device)             # (B, T, action_dim)
                # poses not used when pose_dim=0

                # Encode all T frames with frozen ViT
                tokens = self._encode_clip(frames).detach()   # (B, T, N, D_vit)
                B, T   = tokens.shape[:2]

                if self.is_main:
                    if not hasattr(self, '_diff_step'):
                        self._diff_step = 0
                    self._diff_step += 1
                if self.is_main and self.config.get("verbose", False) and self._diff_step % 100 == 1:
                        diff  = (tokens[:, 1:] - tokens[:, :-1]).abs()
                        pdiff = (frames[:, 1:] - frames[:, :-1]).abs()
                        lp = torch.quantile(diff.flatten().float(),
                                            torch.tensor([.25,.50,.75,.90,.99],
                                                         device=diff.device))
                        near0 = (diff < 0.01).float().mean().item() * 100
                        diff2  = (tokens[:, 2:] - tokens[:, :-2]).abs()
                        diff3  = (tokens[:, 3:] - tokens[:, :-3]).abs()
                        lp2    = torch.quantile(diff2.flatten().float(),
                                                torch.tensor([.25,.50,.75,.90,.99],
                                                             device=diff2.device))
                        lp3    = torch.quantile(diff3.flatten().float(),
                                                torch.tensor([.25,.50,.75,.90,.99],
                                                             device=diff3.device))
                        near0_2 = (diff2 < 0.01).float().mean().item() * 100
                        near0_3 = (diff3 < 0.01).float().mean().item() * 100
                        print(f"\n[frame-diff step={self._diff_step}]"
                              f"\n  latent 1-step : mean={diff.mean().item():.5f}  "
                              f"max={diff.max().item():.5f}  "
                              f"near-zero(<0.01)={near0:.1f}%"
                              f"\n                 p25={lp[0]:.5f}  p50={lp[1]:.5f}  "
                              f"p75={lp[2]:.5f}  p90={lp[3]:.5f}  p99={lp[4]:.5f}"
                              f"\n  latent 2-step : mean={diff2.mean().item():.5f}  "
                              f"max={diff2.max().item():.5f}  "
                              f"near-zero(<0.01)={near0_2:.1f}%"
                              f"\n                 p25={lp2[0]:.5f}  p50={lp2[1]:.5f}  "
                              f"p75={lp2[2]:.5f}  p90={lp2[3]:.5f}  p99={lp2[4]:.5f}"
                              f"\n  latent 3-step : mean={diff3.mean().item():.5f}  "
                              f"max={diff3.max().item():.5f}  "
                              f"near-zero(<0.01)={near0_3:.1f}%"
                              f"\n                 p25={lp3[0]:.5f}  p50={lp3[1]:.5f}  "
                              f"p75={lp3[2]:.5f}  p90={lp3[3]:.5f}  p99={lp3[4]:.5f}"
                              f"\n  pixel         : mean={pdiff.mean().item():.5f}  "
                              f"max={pdiff.max().item():.5f}  "
                              f"(×255: mean={pdiff.mean().item()*255:.2f})")


                pred = self._unwrap(self.system.predictor)

                # ── Teacher-forcing loss (Eq. 2, weighted L1) ───────── #
                preds_tf = pred.forward_clip(
                    tokens, actions                           # (B, T-1, action_dim)
                )                                             # (B, T-1, N, D_vit)
                gamma    = self.config.get("gamma_delta", 10.0)
                targets  = (gamma * (tokens[:, 1:] - tokens[:, :-1])).detach()  # (B, T-1, N, D_vit)

                if self.is_main and self.config.get("verbose", False) and self._diff_step % 100 == 1:
                    with torch.no_grad():
                        pd = (preds_tf / gamma).abs().float()           # predicted Δ
                        pp = torch.quantile(pd.flatten(),
                                            torch.tensor([.25,.50,.75,.90,.99],
                                                         device=pd.device))
                        pnear0 = (pd < 0.01).float().mean().item() * 100
                        print(f"  pred Δ : mean={pd.mean().item():.5f}  "
                              f"max={pd.max().item():.5f}  "
                              f"near-zero(<0.01)={pnear0:.1f}%"
                              f"\n           p25={pp[0]:.5f}  p50={pp[1]:.5f}  "
                              f"p75={pp[2]:.5f}  p90={pp[3]:.5f}  p99={pp[4]:.5f}")

                w_tf     = (tokens[:, 1:] - tokens[:, :-1]).pow(2).sum(dim=-1, keepdim=True).detach()
                w_tf     = w_tf / (w_tf.mean() + 1e-8)
                L_tf     = (w_tf * (preds_tf - targets).pow(2)).mean()

                # ── Rollout loss (Eq. 3, weighted MSE, 3-step AR) ────── #
                z_hat_3, delta_4 = pred.rollout_3step(tokens, actions[:, :3], gamma=gamma)
                tgt_roll = (gamma * (tokens[:, 3] - z_hat_3)).detach()
                w_roll   = (tokens[:, 3] - tokens[:, 0]).pow(2).sum(dim=-1, keepdim=True).detach()
                w_roll   = w_roll / (w_roll.mean() + 1e-8)
                L_roll   = (w_roll * (delta_4 - tgt_roll).pow(2)).mean()

                lambda_tf   = self.config.get("lambda_tf",   1.0)
                lambda_roll = self.config.get("lambda_roll", 1.0)
                loss = lambda_tf * L_tf + lambda_roll * L_roll

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
                    self.optimizer.step()

                with torch.no_grad():
                    preds_abs = tokens[:, :-1] + preds_tf.detach() / gamma
                    cos_sim = F.cosine_similarity(
                        preds_abs.float(),
                        tokens[:, 1:].float(), dim=-1
                    ).mean().item()
                    cos_sim_orig = F.cosine_similarity(
                        preds_abs.float(),
                        tokens[:, :-1].float(), dim=-1
                    ).mean().item()

                totals["tf"]          += L_tf.item()
                totals["rollout"]     += L_roll.item()
                totals["cosine"]      += cos_sim
                totals["cosine_orig"] += cos_sim_orig
                totals["total"]       += loss.item()
                if not train:
                    with torch.no_grad():
                        totals["tf_plain"]     += F.mse_loss(preds_tf.detach(), targets).item()
                        totals["rollout_plain"] += F.mse_loss(delta_4.detach(), tgt_roll).item()
                n += 1
                pbar.set_postfix(
                    tf=f"{L_tf.item():.4f}",
                    ro=f"{L_roll.item():.4f}",
                    cos=f"{cos_sim:.3f}",
                    cos_orig=f"{cos_sim_orig:.3f}",
                )

        avg = {}
        for k, v in totals.items():
            t      = torch.tensor(v / max(n, 1), device=self.device)
            avg[k] = reduce_mean(t).item()
        return avg

    def validate(self):
        if self.is_main:
            print(f"\n[Stage1] Running validation only  T={self.clip_length} frames")
        vl = self._run_epoch(self.val_loader, train=False, epoch=0)
        if self.is_main:
            print(
                f"[Stage1] val_tf={vl['tf']:.4f} (plain={vl['tf_plain']:.4f})"
                f"  val_ro={vl['rollout']:.4f} (plain={vl['rollout_plain']:.4f})"
                f"  val_cos={vl['cosine']:.3f}"
            )

    def train(self):
        best   = self.best
        epochs = self.config["epochs"]
        if self.is_main:
            print(f"\n[Stage1] V-JEPA 2-AC predictor training  "
                  f"T={self.clip_length} frames  {epochs} epochs  "
                  f"{self.world_size} GPU(s)")

        for ep in range(self.start_epoch, epochs + 1):
            tr = self._run_epoch(self.train_loader, True,  ep)
            vl = self._run_epoch(self.val_loader,   False, ep)
            self.scheduler.step()

            if self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                self._log("Stage1", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage1", {"val_"   + k: v for k, v in vl.items()}, ep)
                self._log("Stage1", {"lr": lr}, ep)
                print(
                    f"[Stage1] ep {ep:3d}/{epochs}  "
                    f"val_tf={vl['tf']:.4f} (plain={vl['tf_plain']:.4f})  "
                    f"val_ro={vl['rollout']:.4f} (plain={vl['rollout_plain']:.4f})  "
                    f"val_cos={vl['cosine']:.3f}  lr={lr:.2e}"
                )
                if vl["total"] < best:
                    best = vl["total"]
                    self.save_checkpoint("stage1_best.pt", {
                        "epoch": ep, "best": best,
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        **vl,
                    })

        self.save_checkpoint("stage1_final.pt", {
            "epoch": epochs, "best": best,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        })
        if self.is_main and self.writer:
            self.writer.close()
        if self.is_main:
            print(f"[Stage1] Done. Best val_total={best:.4f}")


# ========================================================================== #
#  Stage 3  —  JSCC (Wyner-Ziv)                                              #
# ========================================================================== #

class Stage2Trainer(BaseTrainer):
    """
    Train JsccEncoder, JsccDecoder, and SideInfoEncoder.

    Predictor is frozen and provides side information ẑ_t^{pred}.

    Loss: L = MSE(token_hat, tokens_t) + λ · KL(q(ŝ_t|tokens_t) || p(ŝ_t|ẑ_t^{pred}))
    """

    def __init__(
        self,
        config:      dict,
        data_path:   str,
        stage1_ckpt: str,
        device:      torch.device,
        agent,
        rank:        int,
        world_size:  int,
        resume_ckpt: str | None = None,
    ):
        super().__init__(config, data_path, device, agent, rank, world_size)
        self.load_checkpoint(stage1_ckpt)

        for p in self.system.predictor.parameters():
            p.requires_grad_(False)
        if self.is_main:
            print("[Stage3] Predictor frozen.")

        self.system.jscc_encoder      = self._wrap_ddp(self.system.jscc_encoder)
        self.system.jscc_decoder      = self._wrap_ddp(self.system.jscc_decoder)
        self.system.side_info_encoder = self._wrap_ddp(self.system.side_info_encoder)

        params = (
            list(self._unwrap(self.system.jscc_encoder).parameters())      +
            list(self._unwrap(self.system.jscc_decoder).parameters())      +
            list(self._unwrap(self.system.side_info_encoder).parameters())
        )
        self.optimizer = optim.Adam(params, lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=1e-6
        )
        self.lambda_rate = config.get("lambda_rate", 0.01)
        self.snr_db      = config["snr_db"]

        self.start_epoch = 1
        self.best        = math.inf
        if resume_ckpt:
            self.start_epoch, self.best = self._load_resume(resume_ckpt)

    def _run_epoch(self, loader, train: bool, epoch: int) -> dict:
        self._unwrap(self.system.jscc_encoder).train(train)
        self._unwrap(self.system.jscc_decoder).train(train)
        self._unwrap(self.system.side_info_encoder).train(train)
        if train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        totals = {"distortion": 0.0, "rate": 0.0, "total": 0.0}
        n      = 0
        phase  = "train" if train else "val"
        pbar   = tqdm(loader, desc=f"  {phase} ep {epoch}",
                      disable=not self.is_main, leave=False, dynamic_ncols=True)

        with torch.set_grad_enabled(train):
            for obs_t, act_t, obs_tp1 in pbar:
                obs_t   = obs_t.to(self.device)
                act_t   = act_t.to(self.device)
                obs_tp1 = obs_tp1.to(self.device)

                tok_prev = self._encode_all_ranks(obs_t)    # (B, N, D_vit)
                tok_curr = self._encode_all_ranks(obs_tp1)  # (B, N, D_vit)

                # Frozen predictor provides side information
                with torch.no_grad():
                    z_pred_si = self.system.predictor(tok_prev, act_t)  # (B, N, D_vit)

                # JSCC encoding (trainable)
                mu_enc, log_var_enc, s_t = self.system.jscc_encoder(
                    tok_curr, sample=train
                )
                s_hat = self.system.channel(s_t)

                # Conditional prior (trainable, loss only)
                mu_prior, log_var_prior = self.system.side_info_encoder(z_pred_si)

                # JSCC decoding (trainable)
                token_hat = self.system.jscc_decoder(s_hat, z_pred_si)

                # Losses
                L_distortion = self.mse(token_hat, tok_curr.detach())
                L_rate       = SemComSystem.rate_loss(
                    mu_enc, log_var_enc, mu_prior, log_var_prior, self.snr_db
                )
                loss = L_distortion + self.lambda_rate * L_rate

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self._unwrap(self.system.jscc_encoder).parameters())      +
                        list(self._unwrap(self.system.jscc_decoder).parameters())      +
                        list(self._unwrap(self.system.side_info_encoder).parameters()),
                        max_norm=5.0,
                    )
                    self.optimizer.step()

                totals["distortion"] += L_distortion.item()
                totals["rate"]       += L_rate.item()
                totals["total"]      += loss.item()
                n += 1
                pbar.set_postfix(
                    dist=f"{L_distortion.item():.5f}",
                    rate=f"{L_rate.item():.5f}",
                )

        avg = {}
        for k, v in totals.items():
            t      = torch.tensor(v / max(n, 1), device=self.device)
            avg[k] = reduce_mean(t).item()
        return avg

    def train(self):
        best   = self.best
        epochs = self.config["epochs"]
        if self.is_main:
            print(
                f"\n[Stage2] JSCC (Wyner-Ziv) training for {epochs} epochs "
                f"on {self.world_size} GPU(s) …  λ={self.lambda_rate}"
            )

        for ep in range(self.start_epoch, epochs + 1):
            tr = self._run_epoch(self.train_loader, True,  ep)
            vl = self._run_epoch(self.val_loader,   False, ep)
            self.scheduler.step()

            if self.is_main:
                self._log("Stage2", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage2", {"val_"   + k: v for k, v in vl.items()}, ep)
                print(
                    f"[Stage2] ep {ep:3d}/{epochs}  "
                    f"val_dist={vl['distortion']:.5f}  "
                    f"val_rate={vl['rate']:.5f}  "
                    f"val_total={vl['total']:.5f}"
                )
                if vl["total"] < best:
                    best = vl["total"]
                    self.save_checkpoint("stage2_best.pt", {
                        "epoch": ep, "best": best,
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        **vl,
                    })

        self.save_checkpoint("stage2_final.pt", {
            "epoch": epochs, "best": best,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        })
        if self.is_main and self.writer:
            self.writer.close()
        if self.is_main:
            print(f"[Stage2] Done. Best val={best:.5f}")
