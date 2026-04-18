"""
trainer.py  —  Multi-GPU training pipeline (4 × T4 on Linux server).

Launch with torchrun, NOT python:
    torchrun --nproc_per_node=4 main.py --train --stage 1 ...

Or use server/launch_training.sh which sets all env vars correctly.

DDP design decisions
---------------------
1. Only the SemCom modules (Encoder, Reshaper, Predictor, Sparsifier) are
   wrapped in DistributedDataParallel.  The OpenVLA agent is NEVER wrapped
   — it is stored as a plain attribute on the system and accessed directly.
   DDP would try to synchronise OpenVLA's 7B parameters across GPUs, which
   would require ~56 GB of inter-GPU bandwidth per backward pass.

2. Each GPU gets its own DataLoader shard via DistributedSampler.
   Effective batch size = batch_size × num_gpus = 16 × 4 = 64.

3. Checkpoints are saved only from rank 0 to avoid file conflicts.

4. TensorBoard logging is rank-0 only.

5. The agent (OpenVLA or stub) is constructed on rank 0 and shared via
   the system object — it is not replicated across GPUs.  Each GPU's
   trainer calls agent.task_loss_forward() which may cross to the GPU
   where OpenVLA lives (device_map="auto" handles this transparently).
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
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from models import (
    PredictiveSemComSystem,
    JsccEncoder,
)
from dataset import GNMTrajectoryDataset, build_dataloaders


# ========================================================================== #
#  DDP initialisation helpers                                                 #
# ========================================================================== #

def init_distributed() -> tuple[int, int, bool]:
    """
    Initialise the distributed process group.

    Returns
    -------
    rank      : int   global rank of this process (0 = primary)
    world_size: int   total number of processes / GPUs
    is_main   : bool  True only for rank 0
    """
    if not dist.is_available() or not dist.is_initialized():
        # torchrun sets these environment variables automatically
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank       = int(os.environ.get("RANK",       0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size > 1:
            dist.init_process_group(
                backend    = "nccl",
                init_method= "env://",
            )
        torch.cuda.set_device(local_rank)
    else:
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count()

    return rank, world_size, (rank == 0)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_ddp_loaders(
    config:    dict,
    hdf5_path: str,
    rank:      int,
    world_size:int,
) -> tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders with DistributedSampler so each GPU processes a
    different shard of the dataset.

    Train: shuffled, sharded across GPUs
    Val:   not sharded (each GPU evaluates the full val set and averages)
    """
    from torch.utils.data import random_split

    full_ds = GNMTrajectoryDataset(
        hdf5_path    = hdf5_path,
        obs_height   = config["obs_height"],
        obs_width    = config["obs_width"],
        obs_channels = config["obs_channels"],
    )

    # Validate action dimension
    if full_ds._action_dim is not None:
        expected = config.get("action_dim", 7)
        if full_ds._action_dim != expected:
            raise ValueError(
                f"action_dim mismatch: HDF5={full_ds._action_dim} "
                f"config={expected}"
            )

    n_total = len(full_ds)
    n_val   = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=generator
    )

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas = world_size,
        rank         = rank,
        shuffle      = True,
        drop_last    = True,
    )
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas = world_size,
        rank         = rank,
        shuffle      = False,
        drop_last    = False,
    )

    num_workers = config.get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size         = config["batch_size"],
        sampler            = train_sampler,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = (num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = config["batch_size"],
        sampler            = val_sampler,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = (num_workers > 0),
    )

    if rank == 0:
        print(
            f"[dataset] total={n_total}  train={n_train}  val={n_val}  "
            f"per-GPU train batches ≈ {len(train_loader)}"
        )
    return train_loader, val_loader


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all DDP ranks."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / dist.get_world_size()
    return tensor


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

        # Build system (NOT yet wrapped in DDP)
        self.system = PredictiveSemComSystem(config, agent=agent).to(device)

        # DataLoaders
        self.train_loader, self.val_loader = build_ddp_loaders(
            config, data_path, rank, world_size
        )

        # TensorBoard (rank 0 only)
        self.writer = None
        if self.is_main:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.out_dir, "tb_logs")
            )

        self.mse = nn.MSELoss()

    def _wrap_ddp(self, module: nn.Module) -> nn.Module:
        """Wrap an nn.Module in DDP for multi-GPU gradient synchronisation."""
        if self.world_size > 1:
            return DDP(
                module,
                device_ids        = [self.device.index],
                output_device     = self.device.index,
                find_unused_parameters = self.config.get("ddp_find_unused", False),
            )
        return module

    def _unwrap(self, module) -> nn.Module:
        """Unwrap DDP to access the underlying module."""
        return module.module if isinstance(module, DDP) else module

    def _task_loss(
        self, y_rec: torch.Tensor, actions_gt: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """
        Compute task loss using the real OpenVLA on rank 0 only.

        In multi-GPU mode, y_rec from all ranks is gathered to rank 0 where
        the frozen VLA computes dL/d(y_rec) for all samples.  The gradients
        are broadcast back so every rank gets the correct training signal
        through its local encoder / reshaper — no stub gradients are used.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return self.agent.task_loss_forward(
                y_rec, actions_gt.to(y_rec.device)
            )

        if not training:
            # Validation: just compute the scalar loss on rank 0, no backward needed.
            all_y_buf = [torch.empty_like(y_rec)      for _ in range(self.world_size)]
            all_a_buf = [torch.empty_like(actions_gt) for _ in range(self.world_size)]
            dist.all_gather(all_y_buf, y_rec.detach())
            dist.all_gather(all_a_buf, actions_gt.detach())
            if self.rank == 0:
                all_y = torch.cat(all_y_buf, dim=0)
                all_a = torch.cat(all_a_buf, dim=0)
                with torch.no_grad():
                    loss_tensor = (self.agent.task_loss_forward(all_y, all_a).detach() / self.world_size).reshape(1)
            else:
                loss_tensor = torch.zeros(1, device=self.device)
            dist.broadcast(loss_tensor, src=0)
            return loss_tensor.squeeze()

        B          = y_rec.size(0)
        actions_gt = actions_gt.to(self.device)

        # 1. Gather detached y_rec and actions from all ranks
        all_y_buf = [torch.empty_like(y_rec)      for _ in range(self.world_size)]
        all_a_buf = [torch.empty_like(actions_gt) for _ in range(self.world_size)]
        dist.all_gather(all_y_buf, y_rec.detach())
        dist.all_gather(all_a_buf, actions_gt.detach())

        # 2. Rank 0 runs the real VLA forward/backward over the full batch
        if self.rank == 0:
            all_y  = torch.cat(all_y_buf, dim=0).requires_grad_(True)
            all_a  = torch.cat(all_a_buf, dim=0)
            L_real = self.agent.task_loss_forward(all_y, all_a)
            L_real.backward()
            grad_all    = all_y.grad.contiguous()        # (world_size*B, C, H, W)
            loss_tensor = (L_real.detach() / self.world_size).reshape(1)
        else:
            grad_all    = torch.empty(
                self.world_size * B, *y_rec.shape[1:], device=self.device
            )
            loss_tensor = torch.zeros(1, device=self.device)

        # 3. Broadcast gradients and scalar loss from rank 0 to all ranks
        dist.broadcast(grad_all,    src=0)
        dist.broadcast(loss_tensor, src=0)

        # 4. Surrogate loss: d/d(y_rec)[(y_rec * g).sum()] == g exactly,
        #    giving each rank the correct OpenVLA gradient.
        #    Straight-through trick keeps the logged scalar meaningful.
        grad_local = grad_all[self.rank * B : (self.rank + 1) * B].detach()
        surrogate  = (y_rec * grad_local).sum()
        return surrogate - surrogate.detach() + loss_tensor.squeeze()

    def _task_loss_pair(
        self,
        y_rec:      torch.Tensor,   # (B, 3, H, W)
        y_align:    torch.Tensor,   # (B, 3, H, W)
        actions_gt: torch.Tensor,   # (B, 7)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute task loss for both y_rec and y_align in a SINGLE VLA forward+backward.

        Combines both tensors into one batch before the VLA pass, halving the
        number of backward passes through the 7B model compared to calling
        _task_loss twice.  Gradients are broadcast back to all ranks.
        """
        if not dist.is_initialized() or self.world_size == 1:
            L_task  = self.agent.task_loss_forward(y_rec,   actions_gt.to(y_rec.device))
            L_align = self.agent.task_loss_forward(y_align, actions_gt.to(y_align.device))
            return L_task, L_align

        need_grad  = torch.is_grad_enabled()
        B          = y_rec.size(0)
        actions_gt = actions_gt.to(self.device)
        WB         = self.world_size * B

        # 1. Gather detached y_rec, y_align, and actions from all ranks
        all_yrec_buf   = [torch.empty_like(y_rec)      for _ in range(self.world_size)]
        all_yalign_buf = [torch.empty_like(y_align)    for _ in range(self.world_size)]
        all_a_buf      = [torch.empty_like(actions_gt) for _ in range(self.world_size)]
        dist.all_gather(all_yrec_buf,   y_rec.detach())
        dist.all_gather(all_yalign_buf, y_align.detach())
        dist.all_gather(all_a_buf,      actions_gt.detach())

        # 2. Rank 0: single VLA forward+backward over combined batch (2*WB images)
        if self.rank == 0:
            all_y = torch.cat(all_yrec_buf + all_yalign_buf, dim=0)
            all_a = torch.cat(all_a_buf    + all_a_buf,      dim=0)
            if need_grad:
                # enable_grad ensures the computation graph is built even if
                # the caller context has grad disabled (e.g. during validation).
                with torch.enable_grad():
                    all_y.requires_grad_(True)
                    L_real = self.agent.task_loss_forward(all_y, all_a)
                    L_real.backward()
                # task_loss_forward normalises by total batch (2*WB); ×2 restores
                # the per-group gradient scale to match individual _task_loss calls.
                grad_all    = all_y.grad.contiguous() * 2   # (2*WB, C, H, W)
            else:
                # Validation: only need the scalar, skip expensive backward pass.
                with torch.no_grad():
                    L_real = self.agent.task_loss_forward(all_y, all_a)
                grad_all = torch.zeros(
                    2 * WB, *y_rec.shape[1:], device=self.device
                )
            loss_tensor = (L_real.detach() * 2 / self.world_size).reshape(1)
        else:
            grad_all    = torch.empty(
                2 * WB, *y_rec.shape[1:], device=self.device
            )
            loss_tensor = torch.zeros(1, device=self.device)

        # 3. Broadcast combined gradients and scalar from rank 0
        dist.broadcast(grad_all,    src=0)
        dist.broadcast(loss_tensor, src=0)

        half_loss = loss_tensor.squeeze() / 2

        if not need_grad:
            # Validation: return plain scalars, no surrogate needed.
            return half_loss, half_loss

        # 4. Split gradients: first WB rows → y_rec, last WB rows → y_align
        grad_rec   = grad_all[self.rank * B : (self.rank + 1) * B].detach()
        grad_align = grad_all[WB + self.rank * B : WB + (self.rank + 1) * B].detach()

        surr_rec    = (y_rec   * grad_rec).sum()
        surr_align  = (y_align * grad_align).sum()
        L_task  = surr_rec   - surr_rec.detach()   + half_loss
        L_align = surr_align - surr_align.detach() + half_loss
        return L_task, L_align

    def save_checkpoint(self, filename: str, extra: dict | None = None):
        """Save checkpoint from rank 0 only."""
        if not self.is_main:
            return
        path    = os.path.join(self.out_dir, filename)
        payload = {"system_state": self._unwrap(self.system).state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"  [ckpt] Saved → {path}")

    def load_checkpoint(self, path: str) -> dict:
        """Load checkpoint on all ranks (same weights needed everywhere)."""
        ckpt = torch.load(path, map_location=self.device)
        self._unwrap(self.system).load_state_dict(
            ckpt["system_state"], strict=False
        )
        if self.is_main:
            print(f"  [ckpt] Loaded ← {path}")
        return ckpt

    def _load_resume(self, path: str) -> tuple[int, float]:
        """Restore full training state for resuming. Returns (start_epoch, best)."""
        ckpt = self.load_checkpoint(path)
        if "optimizer_state" in ckpt and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best        = ckpt.get("best",  math.inf)
        if self.is_main:
            print(f"  [ckpt] Resuming from epoch {start_epoch}  best={best:.4f}")
        return start_epoch, best

    def _log(self, tag: str, values: dict, step: int):
        if self.is_main and self.writer is not None:
            for k, v in values.items():
                self.writer.add_scalar(f"{tag}/{k}", v, step)

    def _save_samples(self, tag: str, obs: torch.Tensor, y_rec: torch.Tensor,
                      epoch: int, n: int = 8, suffix: str = "",
                      tb_step: int | None = None):
        """Save a side-by-side grid of input obs vs reconstructed y_rec."""
        if not self.is_main:
            return
        obs   = obs[:n].detach().cpu().clamp(0, 1)
        y_rec = y_rec[:n].detach().cpu().clamp(0, 1)
        # interleave: obs_0, y_rec_0, obs_1, y_rec_1, ...
        pairs = torch.stack([obs, y_rec], dim=1).flatten(0, 1)  # (2n, C, H, W)
        grid  = make_grid(pairs, nrow=2, padding=2)

        sample_dir = os.path.join(self.out_dir, "samples", tag)
        os.makedirs(sample_dir, exist_ok=True)
        save_image(grid, os.path.join(sample_dir, f"ep{epoch:04d}{suffix}.png"))

        if self.writer is not None:
            step = tb_step if tb_step is not None else epoch
            self.writer.add_image(f"{tag}/obs_vs_yrec", grid, step)


# ========================================================================== #
#  Stage 1  —  VIB Encoder + Task-Oriented Reshaper                         #
# ========================================================================== #

class Stage1Trainer(BaseTrainer):
    """
    Multi-GPU training of JsccEncoder + Reshaper with frozen OpenVLA.

    Both Encoder and Reshaper are wrapped in DDP so their gradients are
    synchronised across the 4 T4 GPUs after each backward pass.

    Loss (VIB):
        L = L_task(OpenVLA(Reshaper(z_rec)), a_gt)   [distortion]
          + beta1 * KL(q(z|x) || N(0,I))             [rate]
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
        super().__init__(config, data_path, device, agent, rank, world_size)

        # Wrap only Encoder and Reshaper in DDP
        self.system.jscc_encoder = self._wrap_ddp(self.system.jscc_encoder)
        self.system.reshaper     = self._wrap_ddp(self.system.reshaper)

        params = (
            list(self._unwrap(self.system.jscc_encoder).parameters()) +
            list(self._unwrap(self.system.reshaper).parameters())
        )
        self.optimizer = optim.Adam(params, lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=1e-6
        )
        self.beta1              = config.get("vib_beta1",          1.0)
        self.l1_weight          = config.get("l1_weight",          10.0)
        self.l1_anneal_epochs   = config.get("l1_anneal_epochs",   5)
        self.task_weight        = config.get("task_weight",        1.0)
        self.task_anneal_epochs = config.get("task_anneal_epochs", 5)

        self.start_epoch = 1
        self.best        = math.inf
        if resume_ckpt:
            self.start_epoch, self.best = self._load_resume(resume_ckpt)

    def _run_epoch(self, loader, train: bool, epoch: int, l1_w: float = 0.0, task_w: float = 1.0) -> dict:
        self.system.jscc_encoder.train(train)
        self.system.reshaper.train(train)
        if train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        totals    = {"task": 0.0, "kl": 0.0, "l1": 0.0, "total": 0.0}
        n         = 0
        sample_obs   = None   # first-batch snapshots for visualisation
        sample_y_rec = None

        phase = "train" if train else "val"
        pbar = tqdm(
            loader,
            desc=f"  {phase} ep {epoch}",
            disable=not self.is_main,
            leave=False,
            dynamic_ncols=True,
        )
        with torch.set_grad_enabled(train):
            for obs_t, act_t, _ in pbar:
                obs_t = obs_t.to(self.device)
                act_t = act_t.to(self.device)
                B     = obs_t.size(0)

                z_prev = torch.zeros(B, self.config["latent_dim"], device=self.device)
                a_prev = torch.zeros(B, self.config["action_dim"],  device=self.device)

                out = self.system(
                    obs_t, z_prev, a_prev,
                    bypass_predictor=True,
                    sparsify=False,
                    sample_z=train,
                )

                L_kl   = JsccEncoder.kl_loss(out["mu"], out["log_var"])
                L_task = self._task_loss(out["y_rec"], act_t, training=train)
                L_l1   = F.l1_loss(out["y_rec"], obs_t)
                loss   = task_w * L_task + self.beta1 * L_kl + l1_w * L_l1

                if sample_obs is None:
                    sample_obs   = obs_t.detach()
                    sample_y_rec = out["y_rec"].detach()

                if train and n % 200 == 0:
                    global_step = (epoch - 1) * len(loader) + n
                    self._save_samples(
                        "Stage1_train",
                        obs_t.detach(), out["y_rec"].detach(),
                        epoch, suffix=f"_b{n:05d}", tb_step=global_step,
                    )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self._unwrap(self.system.jscc_encoder).parameters()) +
                        list(self._unwrap(self.system.reshaper).parameters()),
                        max_norm=5.0,
                    )
                    self.optimizer.step()

                totals["task"]  += L_task.item()
                totals["kl"]    += L_kl.item()
                totals["l1"]    += L_l1.item()
                totals["total"] += loss.item()
                n += 1

                pbar.set_postfix(
                    task=f"{L_task.item():.4f}",
                    kl=f"{L_kl.item():.4f}",
                    l1=f"{L_l1.item():.4f}",
                    total=f"{loss.item():.4f}",
                )

        # Average metrics across all GPUs
        avg = {}
        for k, v in totals.items():
            t = torch.tensor(v / max(n, 1), device=self.device)
            avg[k] = reduce_mean(t).item()
        return avg, sample_obs, sample_y_rec

    def train(self):
        best   = self.best
        epochs = self.config["epochs"]
        if self.is_main:
            print(f"\n[Stage1] VIB training for {epochs} epochs on "
                  f"{self.world_size} GPU(s) …")
            print(f"[Stage1] Effective batch = "
                  f"{self.config['batch_size']} × {self.world_size} = "
                  f"{self.config['batch_size'] * self.world_size}")

        epoch_bar = tqdm(
            range(self.start_epoch, epochs + 1),
            desc="[Stage1]",
            disable=not self.is_main,
            dynamic_ncols=True,
        )
        for ep in epoch_bar:
            # linear anneal: l1_weight → 0 over l1_anneal_epochs
            l1_w   = self.l1_weight   * max(0.0, 1.0 - (ep - 1) / self.l1_anneal_epochs)
            # linear warm-up: 0 → task_weight over task_anneal_epochs
            task_w = self.task_weight * min(1.0, 0.01 + (ep - 1) / self.task_anneal_epochs)

            tr, t_obs, t_yrec = self._run_epoch(self.train_loader, True,  ep, l1_w, task_w)
            vl, v_obs, v_yrec = self._run_epoch(self.val_loader,   False, ep, l1_w, task_w)
            self.scheduler.step()

            if self.is_main:
                self._log("Stage1", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage1", {"val_"   + k: v for k, v in vl.items()}, ep)
                self._log("Stage1", {"l1_weight": l1_w, "task_weight": task_w}, ep)
                self._save_samples("Stage1_train", t_obs, t_yrec, ep)
                self._save_samples("Stage1_val",   v_obs, v_yrec, ep)
                epoch_bar.set_postfix(
                    train=f"{tr['total']:.4f}",
                    val=f"{vl['total']:.4f}",
                    val_task=f"{vl['task']:.4f}",
                    val_kl=f"{vl['kl']:.4f}",
                    l1_w=f"{l1_w:.3f}",
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
            print(f"[Stage1] Done. Best val={best:.4f}")


# ========================================================================== #
#  Stage 2  —  Predictor (World Model / Digital Twin)                        #
# ========================================================================== #

class Stage2Trainer(BaseTrainer):
    """
    Multi-GPU training of Predictor with frozen Encoder + Reshaper.

    Only the Predictor is wrapped in DDP. Encoder and Reshaper stay frozen.

    Loss:
        L = MSE(ẑ, z_t.detach())
          + alpha * MSE(z_rec, z_t.detach())
          + beta  * L_task(OpenVLA(Reshaper(z_rec)), a_gt)
    """

    def __init__(
        self,
        config:       dict,
        data_path:    str,
        stage1_ckpt:  str,
        device:       torch.device,
        agent,
        rank:         int,
        world_size:   int,
        resume_ckpt:  str | None = None,
    ):
        super().__init__(config, data_path, device, agent, rank, world_size)
        self.load_checkpoint(stage1_ckpt)

        for p in self.system.jscc_encoder.parameters(): p.requires_grad_(False)
        for p in self.system.reshaper.parameters():     p.requires_grad_(False)
        if self.is_main:
            print("[Stage2] Encoder + Reshaper frozen.")

        self.system.predictor = self._wrap_ddp(self.system.predictor)

        self.optimizer = optim.Adam(
            self._unwrap(self.system.predictor).parameters(),
            lr=config["learning_rate"],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=1e-6
        )
        self.alpha = config.get("pred_alpha", 0.5)
        self.beta  = config.get("pred_beta",  0.1)

        self.start_epoch = 1
        self.best        = math.inf
        if resume_ckpt:
            self.start_epoch, self.best = self._load_resume(resume_ckpt)

    def _run_epoch(self, loader, train: bool, epoch: int) -> dict:
        self._unwrap(self.system.predictor).train(train)
        if train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        totals = {"pred": 0.0, "latent": 0.0, "task": 0.0, "total": 0.0}
        n      = 0
        steps  = self.config.get("rollout_steps", 5)

        with torch.set_grad_enabled(train):
            for bi, (obs_t, act_t, _) in enumerate(loader):
                obs_t = obs_t.to(self.device)
                act_t = act_t.to(self.device)
                B     = obs_t.size(0)

                with torch.no_grad():
                    _, _, z_prev = self.system.jscc_encoder(obs_t, sample=False)

                a_prev     = torch.zeros(B, self.config["action_dim"], device=self.device)
                h          = None
                batch_loss = torch.tensor(0.0, device=self.device)
                p_sum = l_sum = t_sum = 0.0

                for _ in range(steps):
                    with torch.no_grad():
                        _, _, z_t = self.system.jscc_encoder(obs_t, sample=False)

                    z_pred, h = self.system.predictor(z_prev, a_prev, h)
                    delta_z   = self.system.subtractor(z_t, z_pred)
                    received  = self.system.channel(delta_z)
                    z_rec     = self.system.adder(z_pred, received)

                    with torch.no_grad():
                        y_rec = self.system.reshaper(z_rec)

                    L_pred   = self.mse(z_pred, z_t.detach())
                    L_latent = self.mse(z_rec,  z_t.detach())
                    L_task   = self._task_loss(y_rec, act_t)

                    step_loss  = L_pred + self.alpha * L_latent + self.beta * L_task
                    batch_loss = batch_loss + step_loss

                    p_sum += L_pred.item()
                    l_sum += L_latent.item()
                    t_sum += L_task.item()
                    a_prev = act_t
                    z_prev = z_t.detach()

                batch_loss = batch_loss / steps
                if train:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self._unwrap(self.system.predictor).parameters(), 5.0
                    )
                    self.optimizer.step()

                totals["pred"]   += p_sum / steps
                totals["latent"] += l_sum / steps
                totals["task"]   += t_sum / steps
                totals["total"]  += batch_loss.item()
                n += 1

                if (train and self.is_main
                        and bi % self.config.get("log_interval", 10) == 0):
                    print(
                        f"    [{bi:4d}/{len(loader)}] "
                        f"pred={p_sum/steps:.4f}  "
                        f"task={t_sum/steps:.4f}"
                    )

        avg = {}
        for k, v in totals.items():
            t = torch.tensor(v / max(n, 1), device=self.device)
            avg[k] = reduce_mean(t).item()
        return avg

    def train(self):
        best   = self.best
        epochs = self.config["epochs"]
        if self.is_main:
            print(f"\n[Stage2] Predictor training for {epochs} epochs on "
                  f"{self.world_size} GPU(s) …")

        for ep in range(self.start_epoch, epochs + 1):
            tr = self._run_epoch(self.train_loader, True,  ep)
            vl = self._run_epoch(self.val_loader,   False, ep)
            self.scheduler.step()

            if self.is_main:
                self._log("Stage2", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage2", {"val_"   + k: v for k, v in vl.items()}, ep)
                print(
                    f"[Stage2] ep {ep:3d}/{epochs}  "
                    f"train={tr['total']:.4f}  val={vl['total']:.4f}  "
                    f"val_pred={vl['pred']:.5f}"
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
            print(f"[Stage2] Done. Best val={best:.4f}")


# ========================================================================== #
#  Stage 3  —  Learned tau  (JSAC extension, single GPU fine-tuning)        #
# ========================================================================== #

class Stage3Trainer(BaseTrainer):
    """
    Train Sparsifier.tau.  Not wrapped in DDP because tau is a tiny
    (latent_dim,) parameter — no inter-GPU synchronisation needed.

    Loss = MSE(z_rec, z_t) + lambda * rate_proxy
    """

    def __init__(
        self,
        config:      dict,
        data_path:   str,
        stage2_ckpt: str,
        device:      torch.device,
        agent,
        rank:        int,
        world_size:  int,
        resume_ckpt: str | None = None,
    ):
        super().__init__(config, data_path, device, agent, rank, world_size)

        if config.get("tau_mode", "fixed") != "learned":
            raise ValueError(
                "Stage3 requires tau_mode='learned'. "
                "For Globecom keep tau_mode='fixed' and sweep --tau."
            )

        self.load_checkpoint(stage2_ckpt)

        for name, p in self.system.named_parameters():
            p.requires_grad_(name == "sparsifier.tau")
        if self.is_main:
            print("[Stage3] All frozen except sparsifier.tau  "
                  f"(shape={list(self.system.sparsifier.tau.shape)})")

        self.optimizer = optim.Adam(
            [self.system.sparsifier.tau], lr=config["learning_rate"]
        )
        self.lam         = config.get("tau_lambda",      0.01)
        self.temperature = config.get("tau_temperature", 1.0)
        self.anneal_rate = config.get("tau_anneal_rate", 0.95)
        self.min_temp    = config.get("tau_min_temp",    0.05)

        self.start_epoch   = 1
        self.best          = math.inf
        self.start_temp    = self.temperature
        if resume_ckpt:
            ckpt = self.load_checkpoint(resume_ckpt)
            if "optimizer_state" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best        = ckpt.get("best",  math.inf)
            self.start_temp  = ckpt.get("temperature", self.temperature)
            if self.is_main:
                print(f"  [ckpt] Resuming from epoch {self.start_epoch}  "
                      f"best={self.best:.4f}  T={self.start_temp:.4f}")

    def _run_epoch(self, loader, train: bool, T: float) -> dict:
        self.system.train(train)
        if train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(0)  # tau training is short

        totals = {"recon": 0.0, "rate": 0.0, "total": 0.0, "nnz": 0.0}
        n = 0

        with torch.set_grad_enabled(train):
            for obs_t, act_t, _ in loader:
                obs_t = obs_t.to(self.device)
                B     = obs_t.size(0)

                z_prev = torch.zeros(B, self.config["latent_dim"], device=self.device)
                a_prev = torch.zeros(B, self.config["action_dim"],  device=self.device)

                out = self.system(
                    obs_t, z_prev, a_prev,
                    sparsify=True, sample_z=False,
                    temperature=T, hard=False,
                )
                with torch.no_grad():
                    _, _, z_t = self.system.jscc_encoder(obs_t, sample=False)

                L_recon = self.mse(out["z_rec"], z_t.detach())
                L_rate  = out["rate"]
                loss    = L_recon + self.lam * L_rate

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.system.sparsifier.tau.grad is not None:
                        self.system.sparsifier.tau.grad.data.clamp_(-0.05, 0.05)
                    self.optimizer.step()
                    with torch.no_grad():
                        self.system.sparsifier.tau.clamp_(min=0.0)

                totals["recon"] += L_recon.item()
                totals["rate"]  += L_rate.item()
                totals["total"] += loss.item()
                totals["nnz"]   += out["nnz"].item()
                n += 1

        avg = {}
        for k, v in totals.items():
            t = torch.tensor(v / max(n, 1), device=self.device)
            avg[k] = reduce_mean(t).item()
        return avg

    def train(self):
        best   = self.best
        epochs = self.config["epochs"]
        T      = self.start_temp
        if self.is_main:
            print(f"\n[Stage3] Learned tau for {epochs} epochs  "
                  f"lambda={self.lam}  T0={T}")

        for ep in range(self.start_epoch, epochs + 1):
            T  = max(T * self.anneal_rate, self.min_temp)
            tr = self._run_epoch(self.train_loader, True,  T)
            vl = self._run_epoch(self.val_loader,   False, T)

            if self.is_main:
                self._log("Stage3", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage3", {"val_"   + k: v for k, v in vl.items()}, ep)
                tau_vals = self.system.sparsifier.tau.detach()
                if self.writer:
                    self.writer.add_histogram("Stage3/tau", tau_vals, ep)
                print(
                    f"[Stage3] ep {ep:3d}/{epochs}  T={T:.4f}  "
                    f"val={vl['total']:.5f}  "
                    f"nnz={vl['nnz']:.1f}  "
                    f"tau_mean={tau_vals.mean():.4f}"
                )
                if vl["total"] < best:
                    best = vl["total"]
                    self.save_checkpoint("stage3_best.pt", {
                        "epoch": ep, "best": best, "temperature": T,
                        "optimizer_state": self.optimizer.state_dict(),
                        **vl,
                    })

        self.save_checkpoint("stage3_final.pt", {
            "epoch": epochs, "best": best, "temperature": T,
            "optimizer_state": self.optimizer.state_dict(),
        })
        if self.is_main and self.writer:
            self.writer.close()
        if self.is_main:
            print(f"[Stage3] Done. Best val={best:.5f}")