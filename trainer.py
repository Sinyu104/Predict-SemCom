"""
trainer.py  —  Training pipeline for the Predictive Semantic Communication System.

Stage 1 — single-process, device_map="auto"
    VLA (7B) is loaded with device_map="auto", spreading layers across all 4 GPUs
    (~3.5 GB each).  Predictor lives on cuda:0.  No torchrun needed.
    Launch: python main.py --train --stage 1 --stored_data <path>

Stage 2 — multi-GPU DDP (torchrun)
    Launch: torchrun --nproc_per_node=4 main.py --train --stage 2 ...

Stage overview
--------------
Stage 1  Predictor + LoRA VLA          Phase1: L = λ_pred·L_pred + λ_sig·SIGReg(Z)
                                       Phase2: L = L_CE
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
#  SIGReg — Sketched Isotropic Gaussian Regularizer (LeWorldModel)           #
# ========================================================================== #

def sigreg(Z: torch.Tensor, M: int = 1024, a: float = 1.0) -> torch.Tensor:
    """
    Measures deviation of Z's distribution from an isotropic Gaussian via M
    random 1-D projections, each tested with a differentiable Epps-Pulley
    (MMD²) statistic against N(0,1).

    Z : (N, D)  —  batch of embeddings for one timestep
    M : number of random projections (paper default 1024)
    a : Gaussian kernel bandwidth (a=1 works well)
    """
    N, D = Z.shape
    a2 = a * a

    # Normalise Z per-feature so projections live on the N(0,1) scale.
    # Doing this on Z (not on H after projecting) keeps the MMD sensitive to
    # variance collapse: if all embeddings become identical, H stays near-zero
    # and the test statistic grows, providing a corrective gradient.
    Z = Z - Z.mean(0, keepdim=True)
    Z = Z / (Z.std(0, keepdim=True) + 1e-6)

    u = F.normalize(torch.randn(D, M, device=Z.device, dtype=Z.dtype), dim=0)  # (D, M)
    H = Z @ u                                                                    # (N, M) — no further normalisation

    # MMD²(empirical, N(0,1)) with Gaussian kernel k(x,y)=exp(-(x-y)²/(2a²))
    sq_dist = (H.unsqueeze(1) - H.unsqueeze(0)).pow(2)           # (N, N, M)
    A = torch.exp(-sq_dist / (2 * a2)).mean(dim=(0, 1))           # (M,)  empirical vs empirical
    B = (a / (1 + a2) ** 0.5) * torch.exp(                        # (M,)  empirical vs Gaussian
            -H.pow(2) / (2 * (1 + a2))).mean(0)
    C = a / (2 + a2) ** 0.5                                        # scalar Gaussian vs Gaussian

    return (A - 2 * B + C).mean()


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


def build_single_loaders(config: dict, hdf5_path: str, clip_length: int = 1):
    """Non-distributed dataloaders for single-process Stage 1 training."""
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

    nw = config.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
    )
    print(
        f"[dataset] total={n_total}  train={n_train}  val={n_val}  "
        f"train batches={len(train_loader)}"
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
    Two-phase training (single-process, device_map="auto" across all GPUs).

    Phase 1 — reshape latent space to be temporally predictable:
      Trainable : ViT LoRA + Predictor
      Loss      : L_pred (weighted MSE on γ-scaled token deltas)

    Phase 2 — adapt action head to the new latent space:
      Trainable : LLM LoRA only  (ViT + Predictor frozen)
      Loss      : L_CE (cross-entropy on action tokens)
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
        super().__init__(config, data_path, device, agent, rank=0, world_size=1)

        # Apply LoRA to both ViT and LLM upfront (one GPU dispatch).
        self.agent.add_lora(
            r                  = config.get("lora_r",     32),
            r_vit              = config.get("lora_r_vit", None),
            alpha              = config.get("lora_alpha", 32),
            dropout            = config.get("lora_dropout", 0.05),
            llm_target_modules = config.get("lora_llm_target_modules", ["q_proj", "v_proj"]),
            vit_target_modules = config.get("lora_vit_target_modules", ["qkv", "proj"]),
        )

        # Start in Phase 1: ViT LoRA + Predictor trainable, LLM LoRA frozen.
        self.agent.set_trainable_lora(vit=True, llm=False)
        self._build_phase1_optimizer()

        self.start_epoch = 1
        self.best_p1     = math.inf
        self.best_p2     = math.inf
        if resume_ckpt:
            self.start_epoch, self.best_p1 = self._load_resume(resume_ckpt)

    def _build_phase1_optimizer(self):
        pred_params = list(self.system.predictor.parameters())
        vit_lora    = self.agent.lora_parameters()   # only ViT LoRA active now
        param_groups = [{"params": pred_params,
                         "lr": self.config["learning_rate"], "weight_decay": 0.05}]
        if vit_lora:
            param_groups.append({
                "params":       vit_lora,
                "lr":           self.config.get("lora_lr", self.config["learning_rate"]),
                "weight_decay": 0.0,
            })
        p1_epochs       = self.config.get("phase1_epochs", 5)
        self.optimizer  = optim.AdamW(param_groups)
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=p1_epochs, eta_min=1e-5
        )

    def _build_phase2_optimizer(self):
        self.agent.set_projector_trainable(True)
        lora_lr   = self.config.get("lora_lr", self.config["learning_rate"])
        p2_epochs = self.config.get("phase2_epochs", 5)
        self.optimizer = optim.AdamW([
            {"params": self.agent.lora_parameters(),      "lr": lora_lr,        "weight_decay": 0.0},
            {"params": self.agent.projector_parameters(), "lr": lora_lr * 0.1,  "weight_decay": 1e-4},
        ])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=p2_epochs, eta_min=1e-5
        )

    def _build_loaders(self, config, data_path, rank, world_size):
        return build_single_loaders(config, data_path, self.clip_length)

    def _encode_clip(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, C, H, W)
        Returns tokens : (B, T, N, D_vit) — detached, layer-normalised.
        """
        B, T, C, H, W = frames.shape
        flat   = frames.reshape(B * T, C, H, W)
        tokens = self.agent.encode_image(flat).to(self.device)   # (B*T, N, D_vit)
        tokens = F.layer_norm(tokens, [tokens.shape[-1]])
        return tokens.reshape(B, T, tokens.size(1), tokens.size(2)).detach()

    def _encode_clip_rank0_grad(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode rank 0's local frames WITH gradients so LoRA ViT can be trained.
        Only call on rank 0 during training.

        frames : (B, T, C, H, W)
        Returns tokens : (B, T, N, D_vit) with grad_fn pointing to LoRA ViT.
        """
        B, T, C, H, W = frames.shape
        flat   = frames.reshape(B * T, C, H, W)
        tokens = self.agent.encode_image_train(flat)         # (B*T, N, D_vit) with grad
        tokens = F.layer_norm(tokens, [tokens.shape[-1]])
        return tokens.reshape(B, T, tokens.size(1), tokens.size(2))

    def _run_epoch(self, loader, train: bool, epoch: int, mode: str = "pred") -> dict:
        """
        mode='pred' : Phase 1 — encode with ViT grad, loss = L_pred only.
        mode='ce'   : Phase 2 — encode frozen ViT, loss = L_CE only.
        """
        pred = self.system.predictor
        pred.train(train and mode == "pred")

        if getattr(self.agent, "_loaded", False) and self.agent._vla is not None:
            self.agent._vla.train(train)
            # Refreeze base weights; also enforce phase-specific LoRA trainability.
            for name, p in self.agent._vla.named_parameters():
                if "lora_" not in name:
                    p.requires_grad_(False)
                elif "vision_backbone" in name:
                    p.requires_grad_(train and mode == "pred")
                elif "language_model" in name:
                    p.requires_grad_(train and mode == "ce")

        totals = {"pred": 0.0, "ce": 0.0, "sig": 0.0, "cosine": 0.0, "total": 0.0}
        n, step = 0, 0
        phase_tag = ("train" if train else "val") + f"/{mode}"
        pbar = tqdm(loader, desc=f"  {phase_tag} ep {epoch}",
                    leave=False, dynamic_ncols=True)

        gamma = self.config.get("gamma_delta", 10.0)
        # Early stopping for phase 1 training: stop once pred is consistently low
        es_threshold = self.config.get("early_stop_pred", 0.01)
        es_patience  = self.config.get("early_stop_patience", 200)
        pred_history: list[float] = []

        # Encoder drift tracking: fixed reference batch stored on first step
        drift_freq   = self.config.get("drift_log_freq", 100)
        ref_frames   = None
        drift        = 1.0

        for frames, actions, poses in pbar:
            frames  = frames.to(self.device)
            actions = actions.to(self.device)
            B, T = frames.shape[0], frames.shape[1]

            # ── Phase 1: ViT grad + L_pred ─────────────────────────── #
            if mode == "pred":
                # Store reference batch and compute encoder drift periodically
                if train:
                    if ref_frames is None:
                        ref_frames = frames[:1, 0].detach().cpu()  # (1, C, H, W)
                    if step % drift_freq == 0 and hasattr(self.agent, "encoder_drift"):
                        drift = self.agent.encoder_drift(
                            ref_frames.to(self.device)
                        )

                if train:
                    tokens_grad = self._encode_clip_rank0_grad(frames)
                    tokens      = tokens_grad.detach()
                else:
                    tokens      = self._encode_clip(frames)
                    tokens_grad = tokens

                # Predictor runs on detached context: its Jacobian is noisy early
                # in training and would corrupt the ViT if allowed through.
                # The ViT instead receives gradient through the *targets*:
                #   dL/d(delta_z) pushes delta_z toward what the predictor predicts,
                #   naturally keeping consecutive frames close (temporal smoothness).
                preds_tf = pred.forward_clip(tokens, actions)          # context detached
                targets  = gamma * (tokens_grad[:, 1:] - tokens_grad[:, :-1])  # NOT detached
                w_tf     = targets.pow(2).sum(dim=-1, keepdim=True).detach()
                w_tf     = w_tf / (w_tf.mean() + 1e-8)
                L_pred   = (w_tf * (preds_tf - targets).pow(2)).mean()
                L_ce     = torch.tensor(0.0, device=self.device)

                # SIGReg: pool patch tokens → (B, T, D), apply per timestep
                emb  = tokens_grad.mean(2)          # (B, T, D)
                T_   = emb.shape[1]
                L_sig = sum(
                    sigreg(emb[:, t, :],
                           M=self.config.get("sigreg_M", 1024),
                           a=self.config.get("sigreg_a", 1.0))
                    for t in range(T_)
                ) / T_

                loss = (self.config.get("lambda_pred", 1.0) * L_pred
                        + self.config.get("lambda_sig",  0.1) * L_sig)

            # ── Phase 2: frozen ViT + L_CE ─────────────────────────── #
            else:
                tokens = self._encode_clip(frames)
                N, D   = tokens.shape[2], tokens.shape[3]

                with torch.no_grad():
                    preds_tf = pred.forward_clip(tokens, actions)
                    targets  = gamma * (tokens[:, 1:] - tokens[:, :-1])
                    w_tf     = targets.pow(2).sum(dim=-1, keepdim=True)
                    w_tf     = w_tf / (w_tf.mean() + 1e-8)
                    L_pred   = (w_tf * (preds_tf - targets).pow(2)).mean()

                tok_flat  = tokens[:, :-1].reshape(B * (T - 1), N, D)
                stride    = self.config.get("clip_stride", 1)
                act_flat  = actions.reshape(B * (T - 1), actions.shape[2]) / stride
                L_ce      = self.agent.task_loss_from_tokens(
                    tok_flat, self.agent.instruction, act_flat
                )
                loss  = L_ce
                L_sig = torch.tensor(0.0, device=self.device)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                if mode == "pred":
                    nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
                    # ViT LoRA updates less frequently so predictor learns first
                    vit_freq = self.config.get("vit_update_freq", 1)
                    vit_lora_params = [
                        p for p in self.agent.lora_parameters()
                        if p.requires_grad
                    ] if hasattr(self.agent, "_vla") and self.agent._vla is not None else []
                    vit_start = self.config.get("vit_update_start", 0)
                    if step < vit_start or (vit_freq > 1 and step % vit_freq != 0):
                        for p in vit_lora_params:
                            if p.grad is not None:
                                p.grad.zero_()
                    elif vit_lora_params:
                        nn.utils.clip_grad_norm_(vit_lora_params, 1.0)
                self.optimizer.step()

            with torch.no_grad():
                preds_abs = tokens[:, :-1] + preds_tf.detach() / gamma
                cos_sim   = F.cosine_similarity(
                    preds_abs.float(), tokens[:, 1:].float(), dim=-1
                ).mean().item()

            totals["pred"]   += L_pred.item()
            totals["ce"]     += L_ce.item()
            totals["sig"]    += L_sig.item()
            totals["cosine"] += cos_sim
            totals["total"]  += loss.item()
            n    += 1
            step += 1
            postfix = dict(
                pred=f"{L_pred.item():.4f}",
                sig=f"{L_sig.item():.4f}",
                ce=f"{L_ce.item():.4f}",
                cos=f"{cos_sim:.3f}",
            )
            if mode == "pred" and train:
                postfix["drift"] = f"{drift:.4f}"
            pbar.set_postfix(postfix)

            # Per-step TensorBoard logging
            if train and self.is_main and step % self.config.get("log_interval", 10) == 0:
                tag = "Stage1/P1" if mode == "pred" else "Stage1/P2"
                metrics = {"loss": loss.item(), "pred": L_pred.item(),
                           "ce": L_ce.item(), "sig": L_sig.item(), "cosine": cos_sim}
                if mode == "pred":
                    metrics["drift"] = drift
                self._log(tag, metrics, step)

            # Early stop: phase 1 train only — break once cos is stably high
            if train and mode == "pred":
                pred_history.append(cos_sim)
                if len(pred_history) > es_patience:
                    pred_history.pop(0)
                if (len(pred_history) == es_patience
                        and sum(pred_history) / es_patience > es_threshold):
                    if self.is_main:
                        print(
                            f"\n[Stage1/P1] Early stop at step {step}: "
                            f"rolling cos={sum(pred_history)/es_patience:.4f} "
                            f"> threshold {es_threshold}"
                        )
                    break

        return {k: v / max(n, 1) for k, v in totals.items()}

    def save_checkpoint(self, filename: str, extra: dict | None = None):
        super().save_checkpoint(filename, extra)
        if getattr(self.agent, "_loaded", False) and self.agent._vla is not None:
            lora_path = os.path.join(
                self.out_dir, filename.replace(".pt", "_lora.pt")
            )
            torch.save(self.agent.lora_state_dict(), lora_path)
            print(f"  [ckpt] LoRA saved → {lora_path}")

    def _load_resume(self, path: str) -> tuple[int, float]:
        start_epoch, best = super()._load_resume(path)
        if self.is_main:
            lora_path = path.replace(".pt", "_lora.pt")
            if os.path.exists(lora_path):
                self.agent.load_lora_state_dict(
                    torch.load(lora_path, map_location="cpu")
                )
        return start_epoch, best

    def validate(self):
        if self.is_main:
            print(f"\n[Stage1] Running validation only  T={self.clip_length} frames")
        vl = self._run_epoch(self.val_loader, train=False, epoch=0, mode="pred")
        if self.is_main:
            print(
                f"[Stage1] val_pred={vl['pred']:.4f}  "
                f"val_sig={vl['sig']:.4f}  "
                f"val_ce={vl['ce']:.4f}  "
                f"val_cos={vl['cosine']:.3f}"
            )

    def train(self, skip_phase1: bool = False):
        p1 = self.config.get("phase1_epochs", 5)
        p2 = self.config.get("phase2_epochs", 5)
        r  = self.config.get("lora_r", 16)
        print(f"\n[Stage1] T={self.clip_length} frames  "
              f"Phase1={p1}ep (ViT+Pred, L_pred)  "
              f"Phase2={p2}ep (LLM, L_CE)  LoRA r={r}")

        # ── Phase 1 ──────────────────────────────────────────────────── #
        if skip_phase1:
            p1_ckpt = os.path.join(self.out_dir, "stage1p1_final.pt")
            if not os.path.exists(p1_ckpt):
                raise FileNotFoundError(
                    f"--skip_phase1 requires '{p1_ckpt}'. Run Phase 1 first."
                )
            self._load_resume(p1_ckpt)
            print(f"\n[Stage1/P1] Skipped — loaded checkpoint from {p1_ckpt}")
        else:
            print(f"\n[Stage1/P1] Reshaping latent space  ({p1} epochs)")
            best_p1 = self.best_p1
            for ep in range(1, p1 + 1):
                tr = self._run_epoch(self.train_loader, True,  ep, mode="pred")
                self.scheduler.step()
                if self.is_main:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self._log("Stage1/P1", {"train_" + k: v for k, v in tr.items()}, ep)
                    print(
                        f"[Stage1/P1] ep {ep:2d}/{p1}  "
                        f"train_pred={tr['pred']:.4f}  train_sig={tr['sig']:.4f}  "
                        f"train_cos={tr['cosine']:.3f}  lr={lr:.2e}"
                    )
            self.save_checkpoint("stage1p1_final.pt", {"epoch": p1})

        # ── Transition ───────────────────────────────────────────────── #
        print(f"\n[Stage1] Transitioning to Phase 2 — freezing ViT LoRA + Predictor")
        self.agent.set_trainable_lora(vit=False, llm=True)
        for p in self.system.predictor.parameters():
            p.requires_grad_(False)
        self.agent._vla.train()   # enable train mode so LoRA dropout + projector grads work
        self._build_phase2_optimizer()

        # ── Phase 2 ──────────────────────────────────────────────────── #
        print(f"\n[Stage1/P2] Adapting LLM action head  ({p2} epochs)")
        best_p2 = self.best_p2
        for ep in range(1, p2 + 1):
            tr = self._run_epoch(self.train_loader, True,  ep, mode="ce")
            vl = self._run_epoch(self.val_loader,   False, ep, mode="ce")
            self.scheduler.step()
            if self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                self._log("Stage1/P2", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage1/P2", {"val_"   + k: v for k, v in vl.items()}, ep)
                print(
                    f"[Stage1/P2] ep {ep:2d}/{p2}  "
                    f"val_ce={vl['ce']:.4f}  val_cos={vl['cosine']:.3f}  lr={lr:.2e}"
                )
                if vl["ce"] < best_p2:
                    best_p2 = vl["ce"]
                    self.save_checkpoint("stage1_best.pt", {
                        "epoch": p1 + ep, "best": best_p2,
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        **vl,
                    })
        self.save_checkpoint("stage1_final.pt", {"epoch": p1 + p2})
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
