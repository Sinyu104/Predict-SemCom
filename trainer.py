"""
trainer.py  —  Multi-GPU training pipeline (4 × T4 on Linux server).

Launch with torchrun, NOT python:
    torchrun --nproc_per_node=4 main.py --train --stage 1 ...

DDP design
-----------
1. SemCom modules are DDP-wrapped per stage (only the trainable ones).
2. The OpenVLAAgent is NEVER DDP-wrapped — only rank 0 has the real VLA.
3. encode_image is called on rank 0 and the tokens are broadcast to all ranks
   before entering the forward pass (since only rank 0 has the real ViT).
4. task_loss_from_tokens (Stage 1) is computed on rank 0 over the gathered
   tokens_rec from all ranks; gradients are broadcast back.
5. Checkpoints are saved only from rank 0.

Stage overview
--------------
Stage 1  TokenEncoder + TokenDecoder   L_task (cross-entropy via OpenVLA LLM)
Stage 2  Predictor                     L_world = MSE(ẑ_{t+1}, sg(z_{t+1}))
Stage 3  JsccEncoder + JsccDecoder +   L = L_distortion + λ·L_rate
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

from models import SemComSystem
from dataset import GNMTrajectoryDataset


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
    config: dict, hdf5_path: str, rank: int, world_size: int
) -> tuple[DataLoader, DataLoader]:
    from torch.utils.data import random_split

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

        self.train_loader, self.val_loader = build_ddp_loaders(
            config, data_path, rank, world_size
        )
        self.writer = None
        if self.is_main:
            self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, "tb_logs"))

        self.mse = nn.MSELoss()

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
        Encode images to ViT tokens across DDP ranks.

        Only rank 0 has the real VLA.  All ranks gather their obs to rank 0,
        rank 0 encodes the full batch, then tokens are broadcast back.
        Returns detached tokens on self.device.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return self.agent.encode_image(obs).to(self.device)

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
        return all_tokens[self.rank * B : (self.rank + 1) * B].detach()

    def _task_loss_ddp(
        self,
        tokens_rec:  torch.Tensor,   # (B, N, D_vit) — output of TokenDecoder
        actions_gt:  torch.Tensor,   # (B, 7)
        instruction: str,
        training:    bool = True,
    ) -> torch.Tensor:
        """
        Task loss computed on rank 0, gradients broadcast back to all ranks.
        Gradient flows through tokens_rec → TokenDecoder → z_t → TokenEncoder.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return self.agent.task_loss_from_tokens(
                tokens_rec, instruction, actions_gt.to(tokens_rec.device)
            )

        B          = tokens_rec.size(0)
        actions_gt = actions_gt.to(self.device)

        # Gather detached tokens_rec and actions from all ranks
        all_tok_buf = [torch.empty_like(tokens_rec) for _ in range(self.world_size)]
        all_a_buf   = [torch.empty_like(actions_gt) for _ in range(self.world_size)]
        dist.all_gather(all_tok_buf, tokens_rec.detach())
        dist.all_gather(all_a_buf,   actions_gt.detach())

        loss_tensor = torch.zeros(1, device=self.device)
        N, D = tokens_rec.shape[1], tokens_rec.shape[2]

        if self.rank == 0:
            all_tok = torch.cat(all_tok_buf, dim=0).requires_grad_(True)
            all_a   = torch.cat(all_a_buf,   dim=0)
            if training:
                L = self.agent.task_loss_from_tokens(all_tok, instruction, all_a)
                L.backward()
                grad_all    = all_tok.grad.contiguous()
                loss_tensor = (L.detach() / self.world_size).reshape(1)
            else:
                with torch.no_grad():
                    L = self.agent.task_loss_from_tokens(all_tok, instruction, all_a)
                grad_all    = torch.zeros_like(all_tok)
                loss_tensor = (L.detach() / self.world_size).reshape(1)
        else:
            grad_all = torch.empty(
                self.world_size * B, N, D, device=self.device
            )

        dist.broadcast(grad_all,    src=0)
        dist.broadcast(loss_tensor, src=0)

        if not training:
            return loss_tensor.squeeze()

        grad_local = grad_all[self.rank * B : (self.rank + 1) * B].detach()
        surrogate  = (tokens_rec * grad_local).sum()
        return surrogate - surrogate.detach() + loss_tensor.squeeze()

    def save_checkpoint(self, filename: str, extra: dict | None = None):
        if not self.is_main:
            return
        path    = os.path.join(self.out_dir, filename)
        payload = {"system_state": self._unwrap(self.system).state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"  [ckpt] Saved → {path}")

    def load_checkpoint(self, path: str) -> dict:
        ckpt = torch.load(path, map_location="cpu")
        self._unwrap(self.system).load_state_dict(
            ckpt["system_state"], strict=False
        )
        if self.is_main:
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
            print(f"  [ckpt] Resuming from epoch {start_epoch}  best={best:.4f}")
        del ckpt
        torch.cuda.empty_cache()
        return start_epoch, best

    def _log(self, tag: str, values: dict, step: int):
        if self.is_main and self.writer is not None:
            for k, v in values.items():
                self.writer.add_scalar(f"{tag}/{k}", v, step)


# ========================================================================== #
#  Stage 1  —  TokenEncoder + TokenDecoder                                   #
# ========================================================================== #

class Stage1Trainer(BaseTrainer):
    """
    Train TokenEncoder and TokenDecoder end-to-end via the frozen OpenVLA
    task loss.  Gradient path:
        loss → logits → LLM → projector → tokens_rec
             → TokenDecoder → z_t → TokenEncoder

    Loss: L_task = CrossEntropy(LLM(Projector(tokens_rec)), a_gt)
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

        self.system.token_encoder = self._wrap_ddp(self.system.token_encoder)
        self.system.token_decoder = self._wrap_ddp(self.system.token_decoder)

        params = (
            list(self._unwrap(self.system.token_encoder).parameters()) +
            list(self._unwrap(self.system.token_decoder).parameters())
        )
        self.optimizer = optim.Adam(params, lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=1e-6
        )
        self.instruction = config.get("openvla_instruction", "")

        self.start_epoch = 1
        self.best        = math.inf
        if resume_ckpt:
            self.start_epoch, self.best = self._load_resume(resume_ckpt)

    def _run_epoch(self, loader, train: bool, epoch: int) -> dict:
        self.system.token_encoder.train(train)
        self.system.token_decoder.train(train)
        if train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        totals = {"task": 0.0, "total": 0.0}
        n      = 0
        phase  = "train" if train else "val"
        pbar   = tqdm(loader, desc=f"  {phase} ep {epoch}",
                      disable=not self.is_main, leave=False, dynamic_ncols=True)

        with torch.set_grad_enabled(train):
            for obs_t, act_t, _ in pbar:
                obs_t = obs_t.to(self.device)
                act_t = act_t.to(self.device)

                # Encode images on rank 0, broadcast tokens to all ranks
                vit_tokens = self._encode_all_ranks(obs_t)   # (B, N, D_vit), detached

                # Forward through trainable modules
                z_t        = self.system.token_encoder(vit_tokens)
                tokens_rec = self.system.token_decoder(z_t)

                # Task loss via frozen OpenVLA
                L_task = self._task_loss_ddp(tokens_rec, act_t, self.instruction, train)

                if train:
                    self.optimizer.zero_grad()
                    L_task.backward()
                    nn.utils.clip_grad_norm_(
                        list(self._unwrap(self.system.token_encoder).parameters()) +
                        list(self._unwrap(self.system.token_decoder).parameters()),
                        max_norm=5.0,
                    )
                    self.optimizer.step()

                totals["task"]  += L_task.item()
                totals["total"] += L_task.item()
                n += 1
                pbar.set_postfix(task=f"{L_task.item():.4f}")

        avg = {}
        for k, v in totals.items():
            t      = torch.tensor(v / max(n, 1), device=self.device)
            avg[k] = reduce_mean(t).item()
        return avg

    def train(self):
        best   = self.best
        epochs = self.config["epochs"]
        if self.is_main:
            print(f"\n[Stage1] TokenEncoder + TokenDecoder for {epochs} epochs "
                  f"on {self.world_size} GPU(s) …")

        for ep in range(self.start_epoch, epochs + 1):
            tr = self._run_epoch(self.train_loader, True,  ep)
            vl = self._run_epoch(self.val_loader,   False, ep)
            self.scheduler.step()

            if self.is_main:
                self._log("Stage1", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage1", {"val_"   + k: v for k, v in vl.items()}, ep)
                print(
                    f"[Stage1] ep {ep:3d}/{epochs}  "
                    f"train={tr['total']:.4f}  val={vl['total']:.4f}"
                )
                ckpt_payload = {
                    "epoch": ep, "best": best,
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    **vl,
                }
                self.save_checkpoint(f"stage1_ep{ep:04d}.pt", ckpt_payload)
                if vl["total"] < best:
                    best = vl["total"]
                    ckpt_payload["best"] = best
                    self.save_checkpoint("stage1_best.pt", ckpt_payload)

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
#  Stage 2  —  Predictor (action-conditioned V-JEPA 2)                       #
# ========================================================================== #

class Stage2Trainer(BaseTrainer):
    """
    Train the Predictor in JEPA style.

    Both context and target encoders (ViT + TokenEncoder) are frozen.
    The dataset triplet (obs_t, action_t, obs_tp1) is used as:
        obs_t   → z_{t-1}  (context / previous latent)
        action_t → a_{t-1} (action taken)
        obs_tp1 → z_t      (target / current latent, stop-gradient)

    Loss: L_world = ||ẑ_t^{pred} − sg(z_t)||²
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

        for p in self.system.token_encoder.parameters(): p.requires_grad_(False)
        for p in self.system.token_decoder.parameters(): p.requires_grad_(False)
        if self.is_main:
            print("[Stage2] TokenEncoder + TokenDecoder frozen.")

        self.system.predictor = self._wrap_ddp(self.system.predictor)
        self.optimizer = optim.Adam(
            self._unwrap(self.system.predictor).parameters(),
            lr=config["learning_rate"],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"], eta_min=1e-6
        )
        self.start_epoch = 1
        self.best        = math.inf
        if resume_ckpt:
            self.start_epoch, self.best = self._load_resume(resume_ckpt)

    def _run_epoch(self, loader, train: bool, epoch: int) -> dict:
        self._unwrap(self.system.predictor).train(train)
        if train and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        totals = {"world": 0.0, "total": 0.0}
        n      = 0
        phase  = "train" if train else "val"
        pbar   = tqdm(loader, desc=f"  {phase} ep {epoch}",
                      disable=not self.is_main, leave=False, dynamic_ncols=True)

        with torch.set_grad_enabled(train):
            for obs_t, act_t, obs_tp1 in pbar:
                obs_t   = obs_t.to(self.device)
                act_t   = act_t.to(self.device)
                obs_tp1 = obs_tp1.to(self.device)

                # Encode both frames; tokens are stop-gradient
                tok_prev = self._encode_all_ranks(obs_t)    # z_{t-1} context
                tok_curr = self._encode_all_ranks(obs_tp1)  # z_t target

                with torch.no_grad():
                    z_prev = self.system.token_encoder(tok_prev)   # (B, latent_dim)
                    z_t    = self.system.token_encoder(tok_curr)   # (B, latent_dim)

                # Predictor forward (only these parameters receive gradient)
                z_pred = self.system.predictor(z_prev, act_t)

                L_world = self.mse(z_pred, z_t.detach())

                if train:
                    self.optimizer.zero_grad()
                    L_world.backward()
                    nn.utils.clip_grad_norm_(
                        self._unwrap(self.system.predictor).parameters(), 5.0
                    )
                    self.optimizer.step()

                totals["world"] += L_world.item()
                totals["total"] += L_world.item()
                n += 1
                pbar.set_postfix(world=f"{L_world.item():.5f}")

        avg = {}
        for k, v in totals.items():
            t      = torch.tensor(v / max(n, 1), device=self.device)
            avg[k] = reduce_mean(t).item()
        return avg

    def train(self):
        best   = self.best
        epochs = self.config["epochs"]
        if self.is_main:
            print(f"\n[Stage2] Predictor training for {epochs} epochs "
                  f"on {self.world_size} GPU(s) …")

        for ep in range(self.start_epoch, epochs + 1):
            tr = self._run_epoch(self.train_loader, True,  ep)
            vl = self._run_epoch(self.val_loader,   False, ep)
            self.scheduler.step()

            if self.is_main:
                self._log("Stage2", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage2", {"val_"   + k: v for k, v in vl.items()}, ep)
                print(
                    f"[Stage2] ep {ep:3d}/{epochs}  "
                    f"train={tr['total']:.5f}  val={vl['total']:.5f}"
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


# ========================================================================== #
#  Stage 3  —  JSCC (Wyner-Ziv)                                              #
# ========================================================================== #

class Stage3Trainer(BaseTrainer):
    """
    Train JsccEncoder, JsccDecoder, and SideInfoEncoder.

    All Stage 1/2 modules are frozen.  The Predictor provides side information
    ẑ_t^{pred} — the receiver's prior estimate of z_t.

    Loss: L = ||z_t − ẑ_t||² + λ · KL(q(ŝ_t|z_t) || p(ŝ_t|ẑ_t^{pred}))
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
        self.load_checkpoint(stage2_ckpt)

        # Freeze everything from Stages 1 and 2
        for p in self.system.token_encoder.parameters():  p.requires_grad_(False)
        for p in self.system.token_decoder.parameters():  p.requires_grad_(False)
        for p in self.system.predictor.parameters():      p.requires_grad_(False)
        if self.is_main:
            print("[Stage3] TokenEncoder, TokenDecoder, Predictor frozen.")

        # Wrap trainable modules in DDP
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

                # Encode frames; all stop-gradient (frozen encoders)
                tok_prev = self._encode_all_ranks(obs_t)
                tok_curr = self._encode_all_ranks(obs_tp1)

                with torch.no_grad():
                    z_prev     = self.system.token_encoder(tok_prev)
                    z_t        = self.system.token_encoder(tok_curr)
                    # Predictor side information (receiver's prior estimate of z_t)
                    z_pred_si  = self.system.predictor(z_prev, act_t)

                # JSCC encoding (trainable, gradient flows here)
                mu_enc, log_var_enc, s_t = self.system.jscc_encoder(z_t, sample=train)

                # Wireless channel
                s_hat = self.system.channel(s_t)

                # Conditional prior (trainable, loss only)
                mu_prior, log_var_prior = self.system.side_info_encoder(z_pred_si)

                # JSCC decoding (trainable, gradient flows here)
                z_hat = self.system.jscc_decoder(s_hat, z_pred_si)

                # Losses
                L_distortion = self.mse(z_hat, z_t.detach())
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
                f"\n[Stage3] JSCC (Wyner-Ziv) training for {epochs} epochs "
                f"on {self.world_size} GPU(s) …  λ={self.lambda_rate}"
            )

        for ep in range(self.start_epoch, epochs + 1):
            tr = self._run_epoch(self.train_loader, True,  ep)
            vl = self._run_epoch(self.val_loader,   False, ep)
            self.scheduler.step()

            if self.is_main:
                self._log("Stage3", {"train_" + k: v for k, v in tr.items()}, ep)
                self._log("Stage3", {"val_"   + k: v for k, v in vl.items()}, ep)
                print(
                    f"[Stage3] ep {ep:3d}/{epochs}  "
                    f"val_dist={vl['distortion']:.5f}  "
                    f"val_rate={vl['rate']:.5f}  "
                    f"val_total={vl['total']:.5f}"
                )
                if vl["total"] < best:
                    best = vl["total"]
                    self.save_checkpoint("stage3_best.pt", {
                        "epoch": ep, "best": best,
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.scheduler.state_dict(),
                        **vl,
                    })

        self.save_checkpoint("stage3_final.pt", {
            "epoch": epochs, "best": best,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        })
        if self.is_main and self.writer:
            self.writer.close()
        if self.is_main:
            print(f"[Stage3] Done. Best val={best:.5f}")
