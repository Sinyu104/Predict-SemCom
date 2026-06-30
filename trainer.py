"""
trainer.py  —  Training pipeline for the VAE-latent Wyner-Ziv SemCom System.

Stage 1 — Ctrl-World (SVD-based, first diffusion)
    Goal:   Fine-tune Action_encoder2 of CtrlWorldWrapper to predict the next
            VAE latent from history + action using DDPM x₀-prediction.
    Loss:   L_world = E[|| ẑ_0(x_{t'}, t', c) − z_t ||²]
    Launch: torchrun --nproc_per_node=4 main.py --train --stage 1 \\
                --svd_path stabilityai/stable-video-diffusion-img2vid \\
                --stored_data data/train.hdf5

Stage 2 — JSCC + Refinement Diffusion (second diffusion)
    Goal:   Train JsccEncoder, SideInfoEncoder, and RefinementDiffusion jointly,
            with frozen Ctrl-World providing ẑ_t as Wyner-Ziv side information.
    Loss:   L = L_distortion + β·L_rate
            L_distortion = RefinementDiffusion DDPM x₀-prediction loss (MSE)
            L_rate       = KL( q(s_t|z_t) || p(s_t|ẑ_t) )
    Launch: torchrun --nproc_per_node=4 main.py --train --stage 2 \\
                --svd_path stabilityai/stable-video-diffusion-img2vid \\
                --stored_data data/train.hdf5
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
from dataset import ClipDataset, GNMTrajectoryDataset


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


# ========================================================================== #
#  DataLoader builders                                                        #
# ========================================================================== #

def _has_latents(loader: DataLoader) -> bool:
    """Return True if the underlying dataset has pre-computed VAE latents."""
    from torch.utils.data import Subset
    ds = loader.dataset
    if isinstance(ds, Subset):
        ds = ds.dataset
    return getattr(ds, "has_latents", False)


def build_ddp_loaders(
    config: dict, hdf5_path, rank: int, world_size: int,
    clip_length: int = 1,
) -> tuple[DataLoader, DataLoader]:
    from torch.utils.data import random_split, ConcatDataset

    # hdf5_path may be a single path (str) or a list of paths (multi-task).
    paths = [hdf5_path] if isinstance(hdf5_path, str) else list(hdf5_path)

    if clip_length > 1:
        sub_ds = [
            ClipDataset(
                hdf5_path    = p,
                obs_height   = config["obs_height"],
                obs_width    = config["obs_width"],
                obs_channels = config["obs_channels"],
                clip_length  = clip_length,
                stride       = config.get("clip_stride", 1),
                obs_key      = config.get("obs_key"),
                max_episodes = config.get("max_episodes_per_task"),
            )
            for p in paths
        ]
        full_ds = sub_ds[0] if len(sub_ds) == 1 else ConcatDataset(sub_ds)
        # Propagate has_latents so the trainer's online/precomputed switch works.
        full_ds.has_latents = all(d.has_latents for d in sub_ds)
    else:
        full_ds = GNMTrajectoryDataset(
            hdf5_path    = paths[0],
            obs_height   = config["obs_height"],
            obs_width    = config["obs_width"],
            obs_channels = config["obs_channels"],
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
        print(f"[dataset] total={n_total}  train={n_train}  val={n_val}  "
              f"per-GPU train batches ≈ {len(train_loader)}")
    return train_loader, val_loader


def build_single_loaders(config: dict, hdf5_path: str, clip_length: int = 1):
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

    n_total = len(full_ds)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    nw = config.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )
    print(f"[dataset] total={n_total}  train={n_train}  val={n_val}  "
          f"train batches={len(train_loader)}")
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
        vae,
        rank:       int,
        world_size: int,
    ):
        self.config     = config
        self.device     = device
        self.vae        = vae
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
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.out_dir, "tb_logs", ts)
            )
        self.mse = nn.MSELoss()

    def _build_loaders(self, config, data_path, rank, world_size):
        return build_ddp_loaders(config, data_path, rank, world_size)

    def _wrap_ddp(self, module: nn.Module) -> nn.Module:
        params = list(module.parameters())
        if self.world_size > 1 and params:
            return DDP(
                module,
                device_ids             = [self.device.index],
                output_device          = self.device.index,
                find_unused_parameters = self.config.get("ddp_find_unused", False),
            )
        return module

    def _unwrap(self, module) -> nn.Module:
        return module.module if isinstance(module, DDP) else module

    @torch.no_grad()
    def _encode_vae(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, C, H, W) or (B, T, C, H, W)
        Returns VAE latents of same leading shape with (C_vae, H_vae, W_vae) appended.
        Always detached — VAE is frozen throughout training.
        """
        if self.vae is None:
            C_vae = self.config.get("C_vae", 4)
            H_vae = self.config.get("H_vae", 28)
            W_vae = self.config.get("W_vae", 28)
            shape  = frames.shape[:-3] + (C_vae, H_vae, W_vae)
            return torch.zeros(*shape, device=self.device, dtype=torch.float32)

        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            flat    = frames.reshape(B * T, C, H, W).to(self.device)
            latents = self.vae.encode(flat)
            return latents.reshape(B, T, *latents.shape[1:]).detach()
        else:
            return self.vae.encode(frames.to(self.device)).detach()

    def save_checkpoint(self, filename: str, extra: dict | None = None):
        if not self.is_main:
            return
        path      = os.path.join(self.out_dir, filename)
        raw_state = self._unwrap(self.system).state_dict()
        clean     = {k.replace(".module.", "."): v for k, v in raw_state.items()}
        payload   = {"system_state": clean}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"  [ckpt] Saved → {path}")

    def load_checkpoint(self, path: str) -> dict:
        ckpt       = torch.load(path, map_location="cpu")
        model      = self._unwrap(self.system)
        model_keys = set(model.state_dict().keys())
        state      = ckpt.get("system_state", {})
        remapped   = {}
        for k, v in state.items():
            if k in model_keys:
                remapped[k] = v
            else:
                parts = k.split(".", 1)
                if len(parts) == 2:
                    cand = f"{parts[0]}.module.{parts[1]}"
                    remapped[cand if cand in model_keys else k] = v
                else:
                    remapped[k] = v
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if self.is_main:
            if missing:
                print(f"  [ckpt] WARNING: {len(missing)} missing keys — e.g. {missing[:3]}")
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
            print(f"  [ckpt] Resuming at epoch {start_epoch}  best={best:.6f}")
        del ckpt
        torch.cuda.empty_cache()
        return start_epoch, best

    def _log(self, tag: str, values: dict, step: int):
        if self.is_main and self.writer is not None:
            for k, v in values.items():
                self.writer.add_scalar(f"{tag}/{k}", v, step)


# ========================================================================== #
#  Stage 1  —  Ctrl-World (SVD-based first diffusion)                        #
# ========================================================================== #

class Stage1Trainer(BaseTrainer):
    """
    Fine-tune Ctrl-World's Action_encoder2 to predict the next VAE latent
    from history + action using DDPM x₀-prediction loss.

    ctrl_world must be either CtrlWorldWrapper (full SVD model) or StubCtrlWorld
    (lightweight stand-in for testing).  The SemComSystem created by BaseTrainer
    is not used in Stage 1.
    """

    def __init__(
        self,
        config:      dict,
        data_path:   str,
        device:      torch.device,
        vae,
        rank:        int,
        world_size:  int,
        ctrl_world,                  # CtrlWorldWrapper or StubCtrlWorld (required)
        resume_ckpt: str | None = None,
    ):
        self.clip_length = config.get("num_history", 6) + config.get("num_pred", 1)
        self.ctrl_world  = ctrl_world
        super().__init__(config, data_path, device, vae, rank, world_size)

        self.ctrl_world = self.ctrl_world.to(device)
        # Wrap only the trainable action_encoder with DDP — not the entire 1.5B frozen UNet.
        # DDP gradient hooks attach to the module's parameters directly, so calling
        # ctrl_world.forward_ddpm() will still trigger all-reduce on action_encoder grads.
        if self.world_size > 1 and hasattr(self.ctrl_world, "action_encoder"):
            self.ctrl_world.action_encoder = DDP(
                self.ctrl_world.action_encoder,
                device_ids             = [self.device.index],
                output_device          = self.device.index,
                find_unused_parameters = False,
            )
        trainable_params = list(self.ctrl_world.trainable_parameters())

        if self.is_main:
            n = sum(p.numel() for p in trainable_params)
            print(f"[Stage1] Ctrl-World: {n/1e6:.2f}M trainable params (Action_encoder2)")

        if trainable_params:
            self.optimizer = optim.AdamW(
                trainable_params,
                lr           = config["learning_rate"],
                weight_decay = 0.05,
            )
            epochs = config.get("epochs", 30)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-5
            )
        else:
            # StubCtrlWorld has no params — create a no-op optimizer
            self.optimizer = None
            self.scheduler = None

        self.scaler      = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        self.start_epoch = 1
        self.best        = math.inf
        if resume_ckpt:
            self.start_epoch, self.best = self._load_resume(resume_ckpt)

    def _build_loaders(self, config, data_path, rank, world_size):
        return build_ddp_loaders(config, data_path, rank, world_size,
                                 clip_length=self.clip_length)

    # ------------------------------------------------------------------ #
    #  Checkpoint — saves Action_encoder2 only (UNet stays frozen)        #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, filename: str, extra: dict | None = None):
        if not self.is_main:
            return
        path    = os.path.join(self.out_dir, filename)
        ae_state = {}
        if hasattr(self.ctrl_world, "action_encoder"):
            ae = self.ctrl_world.action_encoder
            ae_state = (ae.module if isinstance(ae, DDP) else ae).state_dict()
        payload = {"action_encoder_state": ae_state}
        if getattr(self.ctrl_world, "finetune_cross_attn", False):
            payload["unet_cross_attn_state"] = self.ctrl_world.unet_cross_attn_state_dict()
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"  [ckpt] Saved → {path}")

    def _load_resume(self, path: str) -> tuple[int, float]:
        ckpt = torch.load(path, map_location="cpu")
        if "action_encoder_state" in ckpt and hasattr(self.ctrl_world, "action_encoder"):
            ae = self.ctrl_world.action_encoder
            (ae.module if isinstance(ae, DDP) else ae).load_state_dict(
                ckpt["action_encoder_state"], strict=False
            )
            if self.is_main:
                print(f"  [ckpt] Loaded Action_encoder ← {path}")
        if "unet_cross_attn_state" in ckpt and hasattr(self.ctrl_world, "load_unet_cross_attn_state_dict"):
            self.ctrl_world.load_unet_cross_attn_state_dict(ckpt["unet_cross_attn_state"])
            if self.is_main:
                print(f"  [ckpt] Loaded UNet cross-attn ← {path}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best        = ckpt.get("best",  math.inf)
        if "optimizer_state" in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.is_main:
            print(f"  [ckpt] Resuming from epoch {start_epoch}  best={best:.6f}")
        del ckpt
        torch.cuda.empty_cache()
        return start_epoch, best

    # ------------------------------------------------------------------ #
    #  Visualisation — decode predicted vs GT next frame with SD-VAE      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _visualize_predictions(self, epoch: int, n_samples: int = 4):
        if not self.is_main:
            return
        if not hasattr(self.ctrl_world, "predict_next_latent"):
            return   # StubCtrlWorld doesn't implement full inference

        import torchvision
        from PIL import Image as PILImage

        model = self.ctrl_world
        model.eval()

        frames_b, actions_b, latents_b = [], [], []
        collected = 0
        for frames, actions, _poses, latents in self.val_loader:
            frames_b.append(frames);  actions_b.append(actions);  latents_b.append(latents)
            collected += frames.shape[0]
            if collected >= n_samples:
                break
        frames  = torch.cat(frames_b,  0)[:n_samples]
        actions = torch.cat(actions_b, 0)[:n_samples]
        latents = torch.cat(latents_b, 0)[:n_samples]

        # When the dataset has no pre-computed latents it returns zeros; encode the
        # frames with the (frozen) VAE so the viz reflects real observations.
        if not _has_latents(self.val_loader):
            latents = self._encode_vae(frames).cpu().float()

        T_h = model.num_history
        z_history = latents[:, :T_h].to(self.device)
        T_p   = self.config.get("num_pred", 1)

        a_pad  = torch.zeros(n_samples, 1, actions.shape[-1], device=self.device)
        a_full = torch.cat([actions.to(self.device), a_pad], dim=1)

        n_steps = self.config.get("ctrl_world_n_steps", 10)
        z_pred  = model.predict_next_latent(z_history, a_full, n_steps=n_steps)
        z_pred = z_pred[:, -1].cpu().float()

        from vae_wrapper import VAEWrapper
        vae_name = self.config.get("vae_model_name", "stabilityai/sd-vae-ft-mse")
        vae_cpu  = VAEWrapper(vae_name)

        # Cols 1-2 = raw ground-truth observations (no VAE round-trip); col 3 = decoded prediction.
        imgs_hist = frames[:, T_h - 1].cpu().float()           # last history frame (raw)
        imgs_gt   = frames[:, T_h + T_p - 1].cpu().float()     # target frame (raw)
        imgs_pred = vae_cpu.decode(z_pred)
        del vae_cpu

        tiles = []
        for i in range(n_samples):
            tiles += [imgs_hist[i], imgs_gt[i], imgs_pred[i]]

        grid = torchvision.utils.make_grid(
            torch.stack(tiles), nrow=3, padding=4, pad_value=0.5
        )
        viz_dir  = os.path.join(self.out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        out_path = os.path.join(viz_dir, f"ep{epoch:03d}.png")
        arr = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        PILImage.fromarray(arr).save(out_path)
        print(f"  [viz] ep{epoch:03d} → {out_path}  (cols: last_hist | gt_next | pred_next)")

    def _run_epoch(self, loader, train: bool, epoch: int) -> dict:
        self.ctrl_world.train(train)
        if train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        use_precomp_lats = _has_latents(loader)
        if self.is_main and epoch == (self.start_epoch if train else 0):
            src = "pre-computed HDF5 latents" if use_precomp_lats else "online VAE encode"
            print(f"  [Stage1] latent source: {src}")

        totals = {"pred": 0.0}
        n      = 0
        phase  = "train" if train else "val"
        pbar   = tqdm(loader, desc=f"  {phase} ep {epoch}",
                      disable=not self.is_main, leave=False, dynamic_ncols=True)

        with torch.set_grad_enabled(train):
            for frames, actions, _poses, latents_precomp in pbar:
                frames  = frames.to(self.device)
                actions = actions.to(self.device)

                if use_precomp_lats:
                    z_clip = latents_precomp.to(self.device)
                else:
                    z_clip = self._encode_vae(frames)           # (B, T, 4, Hv, Wv)

                B, T = z_clip.shape[:2]
                a_pad  = torch.zeros(B, 1, actions.shape[-1], device=self.device)
                a_full = torch.cat([actions, a_pad], dim=1)     # (B, T, action_dim)

                L_pred = self.ctrl_world.forward_ddpm(z_clip, a_full)

                if train and self.optimizer is not None:
                    if not torch.isfinite(L_pred):
                        pbar.set_postfix(pred="NaN-skip")
                        continue
                    self.optimizer.zero_grad()
                    self.scaler.scale(L_pred).backward()
                    self.scaler.unscale_(self.optimizer)
                    # Cross-attn params live in the UNet (not DDP-wrapped), so we
                    # manually all_reduce their gradients across GPUs.
                    if self.world_size > 1 and getattr(self.ctrl_world, "finetune_cross_attn", False):
                        for p in self.ctrl_world.cross_attn_parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                                p.grad.div_(self.world_size)
                    nn.utils.clip_grad_norm_(
                        list(self.ctrl_world.trainable_parameters()), 1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                totals["pred"] += L_pred.item()
                n += 1
                pbar.set_postfix(pred=f"{L_pred.item():.5f}")

        avg = {}
        for k, v in totals.items():
            t      = torch.tensor(v / max(n, 1), device=self.device)
            avg[k] = reduce_mean(t).item()
        return avg

    def validate(self):
        vl = self._run_epoch(self.val_loader, train=False, epoch=0)
        if self.is_main:
            print(f"[Stage1] val_pred={vl['pred']:.5f}")

    def train(self):
        epochs = self.config.get("epochs", 30)
        if self.is_main:
            print(f"\n[Stage1] Ctrl-World  {epochs} epochs  clip_length={self.clip_length}")
        best = self.best
        for ep in range(self.start_epoch, epochs + 1):
            tr = self._run_epoch(self.train_loader, True,  ep)
            vl = self._run_epoch(self.val_loader,   False, ep)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.is_main:
                lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
                self._log("Stage1", {"train_pred": tr["pred"], "val_pred": vl["pred"]}, ep)
                print(f"[Stage1] ep {ep:3d}/{epochs}  "
                      f"train_pred={tr['pred']:.5f}  val_pred={vl['pred']:.5f}  lr={lr:.2e}")
                if vl["pred"] < best:
                    best = vl["pred"]
                    self.save_checkpoint("stage1_best.pt", {
                        "epoch": ep, "best": best,
                        **({"optimizer_state": self.optimizer.state_dict()} if self.optimizer else {}),
                        **({"scheduler_state": self.scheduler.state_dict()} if self.scheduler else {}),
                        **vl,
                    })
            self._visualize_predictions(ep)
        self.save_checkpoint("stage1_final.pt", {"epoch": epochs, "best": best})
        if self.is_main and self.writer:
            self.writer.close()
        if self.is_main:
            print(f"[Stage1] Done. Best val_pred={best:.5f}")


# ========================================================================== #
#  Stage 2  —  JSCC + Refinement Diffusion (Wyner-Ziv)                       #
# ========================================================================== #

class Stage2Trainer(BaseTrainer):
    """
    Train JsccEncoder, SideInfoEncoder, and RefinementDiffusion jointly.

    Ctrl-World is loaded from the Stage-1 checkpoint and kept frozen.
    It provides ẑ_t = predict_next_latent(z_history, a_t) as Wyner-Ziv side information.

    Loss:  L = MSE(z̃_t, z_t) + β · KL( q(s_t|z_t) || p(s_t|ẑ_t) )
    """

    def __init__(
        self,
        config:      dict,
        data_path:   str,
        stage1_ckpt: str,
        device:      torch.device,
        vae,
        rank:        int,
        world_size:  int,
        ctrl_world   = None,         # CtrlWorldWrapper or StubCtrlWorld
        resume_ckpt: str | None = None,
    ):
        # clip_length = num_history + 1 (history frames + target frame)
        self.clip_length   = config.get("num_history", 6) + 1
        self.ctrl_world    = ctrl_world
        super().__init__(config, data_path, device, vae, rank, world_size)

        # ── Load Stage-1 action encoder into ctrl_world ───────────────────
        if self.ctrl_world is not None:
            self.ctrl_world = self.ctrl_world.to(device)
            if stage1_ckpt and os.path.exists(stage1_ckpt):
                ckpt = torch.load(stage1_ckpt, map_location="cpu")
                if "action_encoder_state" in ckpt and hasattr(self.ctrl_world, "action_encoder"):
                    self.ctrl_world.action_encoder.load_state_dict(
                        ckpt["action_encoder_state"], strict=False
                    )
                    if self.is_main:
                        print(f"[Stage2] Loaded Action_encoder from {stage1_ckpt}")
                del ckpt
            self.ctrl_world.requires_grad_(False)
            if self.is_main:
                print("[Stage2] Ctrl-World frozen.")
        elif self.is_main:
            print("[Stage2] No Ctrl-World provided — using previous frame as ẑ_t (fallback).")

        # ── Wrap Stage-2 trainable modules with DDP ──────────────────────
        self.system.jscc_encoder         = self._wrap_ddp(self.system.jscc_encoder)
        self.system.side_info_encoder    = self._wrap_ddp(self.system.side_info_encoder)
        self.system.refinement_diffusion = self._wrap_ddp(self.system.refinement_diffusion)

        params = (
            list(self._unwrap(self.system.jscc_encoder).parameters())           +
            list(self._unwrap(self.system.side_info_encoder).parameters())      +
            list(self._unwrap(self.system.refinement_diffusion).parameters())
        )
        self.optimizer   = optim.AdamW(params, lr=config["learning_rate"], weight_decay=1e-4)
        self.scheduler   = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=1e-6
        )
        self.beta_rate   = config.get("beta_rate", config.get("lambda_rate", 0.01))
        self.snr_db      = config["snr_db"]
        self.noise_level = config.get("noise_level_second", 250)
        self.start_epoch = 1
        self.best        = math.inf

        if resume_ckpt:
            self.start_epoch, self.best = self._load_resume(resume_ckpt)

    def _build_loaders(self, config, data_path, rank, world_size):
        return build_ddp_loaders(config, data_path, rank, world_size,
                                 clip_length=self.clip_length)

    def _get_z_hat(
        self,
        z_clip:     torch.Tensor,    # (B, T, 4, Hv, Wv)  — clip latents
        actions:    torch.Tensor,    # (B, T-1, action_dim)
    ) -> torch.Tensor:
        """
        Compute ẑ_t (Wyner-Ziv side information) using frozen Ctrl-World.

        z_clip[:, :-1] = z_history  (num_history frames)
        z_clip[:, -1]  = z_t        (target frame)

        Returns ẑ_t : (B, 4, Hv, Wv)
        """
        B         = z_clip.shape[0]
        num_hist  = self.clip_length - 1
        z_history = z_clip[:, :num_hist]                  # (B, T-1, 4, Hv, Wv)
        action_dim = actions.shape[-1]

        if self.ctrl_world is not None:
            # Pad actions to (B, num_hist+1, action_dim) for predict_next_latent
            a_pad   = torch.zeros(B, 1, action_dim, device=self.device)
            a_full  = torch.cat([actions, a_pad], dim=1)  # (B, num_hist+1, D)
            n_steps = self.config.get("ctrl_world_n_steps", 10)
            with torch.no_grad():
                z_hat = self.ctrl_world.predict_next_latent(z_history, a_full, n_steps=n_steps)
            return z_hat[:, 0]                             # (B, 4, Hv, Wv)
        else:
            # Fallback: use last history frame as a trivial side-info estimate
            return z_clip[:, -2].detach()

    def _run_epoch(self, loader, train: bool, epoch: int) -> dict:
        jscc  = self._unwrap(self.system.jscc_encoder)
        side  = self._unwrap(self.system.side_info_encoder)
        refine = self._unwrap(self.system.refinement_diffusion)
        jscc.train(train);  side.train(train);  refine.train(train)

        if train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        use_precomp_lats = _has_latents(loader)
        if self.is_main and epoch == (self.start_epoch if train else 0):
            src = "pre-computed HDF5 latents" if use_precomp_lats else "online VAE encode"
            print(f"  [Stage2] latent source: {src}")

        totals = {"distortion": 0.0, "rate": 0.0, "total": 0.0}
        n      = 0
        phase  = "train" if train else "val"
        pbar   = tqdm(loader, desc=f"  {phase} ep {epoch}",
                      disable=not self.is_main, leave=False, dynamic_ncols=True)

        with torch.set_grad_enabled(train):
            for frames, actions, _poses, latents_precomp in pbar:
                frames  = frames.to(self.device)
                actions = actions.to(self.device)

                if use_precomp_lats:
                    z_clip = latents_precomp.to(self.device)   # (B, T, 4, Hv, Wv)
                else:
                    z_clip = self._encode_vae(frames)

                z_t   = z_clip[:, -1]                          # (B, 4, Hv, Wv) — target
                z_hat = self._get_z_hat(z_clip, actions)       # (B, 4, Hv, Wv) — side info ẑ_t

                # JSCC Encoder: z_t → q(s_t | z_t)
                mu_enc, log_var_enc, s_t = jscc(z_t, sample=train)
                s_tilde = self.system.channel(s_t)             # (B, D_jscc) — received signal

                # Side Info Encoder: ẑ_t → p(s_t | ẑ_t)   [rate loss only]
                mu_prior, log_var_prior = side(z_hat)

                # Refinement Diffusion: DDPM x₀-prediction loss
                L_distortion = refine.forward_ddpm(z_t, z_hat, s_tilde, self.noise_level)

                # Rate: KL( q(s_t|z_t) || p(s_t|ẑ_t) )
                L_rate = SemComSystem.rate_loss(
                    mu_enc, log_var_enc, mu_prior, log_var_prior, self.snr_db
                )
                loss = L_distortion + self.beta_rate * L_rate

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(jscc.parameters()) + list(side.parameters()) + list(refine.parameters()),
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
            print(f"\n[Stage2] JSCC + Refinement Diffusion  {epochs} epochs  "
                  f"β={self.beta_rate}  SNR={self.snr_db}dB  t''={self.noise_level}")

        for ep in range(self.start_epoch, epochs + 1):
            tr = self._run_epoch(self.train_loader, True,  ep)
            vl = self._run_epoch(self.val_loader,   False, ep)
            self.scheduler.step()

            if self.is_main:
                self._log("Stage2", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage2", {"val_"   + k: v for k, v in vl.items()}, ep)
                print(f"[Stage2] ep {ep:3d}/{epochs}  "
                      f"val_dist={vl['distortion']:.5f}  "
                      f"val_rate={vl['rate']:.5f}  "
                      f"val_total={vl['total']:.5f}")
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


# re-export for backward compat
from models import SemComSystem
