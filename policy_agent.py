"""
policy_agent.py  —  Policy wrappers for the VAE-latent SemCom system.

The policy lives entirely on the receiver.  Its input is the decoded image
x̃_t (rendered from the reconstructed VAE latent) — it has no knowledge of
the communication channel, VAE latents, or world model.

Two classes
-----------
Pi0FastPolicy
    Action-chunk policy inspired by π0-fast (Black et al., 2024).
    Wraps a frozen vision-language backbone and adds a lightweight transformer
    action-chunk head that outputs K consecutive actions in a single forward pass.

    Backbone options:
      • "openvla" : re-use the loaded OpenVLA vision backbone (DINOv2+SigLIP)
      • "stub"    : random features — CPU testing only

Pi0FastStub
    Lightweight CPU stub with the same interface.  No GPU or model download
    required.  Used for shape-checking and unit tests.

Action chunk (π0-fast style)
-----------------------------
Instead of predicting a single action at each step, the policy predicts a chunk
of K future actions:  â_{t:t+K} ∈ ℝ^{K×action_dim}.

At inference the robot executes the chunk and re-plans every K steps or after
a fixed receding-horizon re-plan interval.

Architecture of the action-chunk head
--------------------------------------
1. Vision encoder  →  visual features  (B, N_vis, D_vis)
2. Language embed  →  lang token       (B, 1, D_vis)   (fixed or learned)
3. Chunk query tokens: learned (K, D_vis), broadcast to (B, K, D_vis)
4. Transformer decoder: chunk queries attend to [lang | visual] via cross-attn
5. Linear head:  (B, K, D_vis) → (B, K, action_dim), tanh clamp to [-1, 1]

This is the "fast" variant (no diffusion / no flow-matching iterations):
the chunk is predicted in a single forward pass, making inference ~10× faster
than the full π0 diffusion model while retaining action-chunk benefits.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ========================================================================== #
#  Transformer decoder block (cross-attention)                                #
# ========================================================================== #

class _ChunkDecoderBlock(nn.Module):
    """Cross-attention + self-attention + FFN block for chunk queries."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_kv  = nn.LayerNorm(d_model)
        self.cross    = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.norm_q2  = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.norm_ff  = nn.LayerNorm(d_model)
        inner         = int(d_model * mlp_ratio)
        self.ffn      = nn.Sequential(
            nn.Linear(d_model, inner), nn.GELU(), nn.Linear(inner, d_model)
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Cross-attention: chunk tokens attend to vision+lang context
        q2, _ = self.cross(self.norm_q1(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + q2
        # Self-attention among chunk tokens
        q3, _ = self.self_attn(self.norm_q2(q), self.norm_q2(q), self.norm_q2(q))
        q = q + q3
        q = q + self.ffn(self.norm_ff(q))
        return q


# ========================================================================== #
#  Action-chunk head  (shared by both policy classes)                        #
# ========================================================================== #

class ActionChunkHead(nn.Module):
    """
    Predicts K consecutive actions given visual + language context.

    Inputs:
        context : (B, N_ctx, d_model)  — concatenated [lang_token | visual_tokens]
    Output:
        actions : (B, chunk_size, action_dim)  in [-1, 1]
    """

    def __init__(
        self,
        d_model:    int,
        chunk_size: int,
        action_dim: int,
        n_layers:   int = 4,
        n_heads:    int = 8,
    ):
        super().__init__()
        self.chunk_size  = chunk_size
        # Learnable query tokens: one per future step
        self.chunk_queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        self.blocks        = nn.ModuleList([
            _ChunkDecoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm          = nn.LayerNorm(d_model)
        self.action_head   = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        B = context.size(0)
        q = self.chunk_queries.expand(B, -1, -1)        # (B, K, d)
        for blk in self.blocks:
            q = blk(q, context)
        q = self.norm(q)
        return self.action_head(q)                       # (B, K, action_dim)


# ========================================================================== #
#  Pi0FastPolicy  (real model)                                               #
# ========================================================================== #

class Pi0FastPolicy(nn.Module):
    """
    π0-fast inspired policy: image + language → action chunk.

    backbone="openvla"  : reuse the OpenVLAAgent's vision backbone (DINOv2+SigLIP).
                          Pass the loaded agent as `vla_agent`.
    backbone="siglip"   : load a standalone SigLIP-B/16 (optional).
    backbone="stub"     : random features for CPU testing.

    Parameters
    ----------
    chunk_size  : number of future actions to predict (K)
    action_dim  : DoF per action (default 7 for Franka)
    d_model     : hidden dim for the action-chunk head transformer
    n_layers    : depth of action-chunk head transformer
    backbone    : "openvla" | "stub"
    vla_agent   : required when backbone="openvla"
    """

    def __init__(
        self,
        chunk_size:  int   = 16,
        action_dim:  int   = 7,
        d_model:     int   = 512,
        n_layers:    int   = 4,
        n_heads:     int   = 8,
        backbone:    str   = "stub",
        vla_agent          = None,
        instruction: str   = "",
    ):
        super().__init__()
        self.chunk_size  = chunk_size
        self.action_dim  = action_dim
        self.instruction = instruction
        self.backbone    = backbone
        self._vla_agent  = vla_agent

        # Visual feature projection → d_model
        # OpenVLA vision backbone produces D_vit=2176-dim tokens.
        # We project them down to d_model for the chunk head.
        D_vis = 2176 if backbone == "openvla" else 64
        self.vis_proj    = nn.Linear(D_vis, d_model)

        # Fixed sinusoidal language embedding (one vector per instruction char bucket).
        # For production: replace with a real text encoder (T5, CLIP text, etc.).
        self.lang_embed  = nn.Embedding(512, d_model)   # token-ID look-up for up to 512 chars
        self._instr_ids  = None                         # cached token IDs

        self.chunk_head  = ActionChunkHead(d_model, chunk_size, action_dim, n_layers, n_heads)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)  decoded image in [0,1]
        Returns (B, N_vis, D_vis)
        """
        if self.backbone == "openvla" and self._vla_agent is not None:
            with torch.no_grad():
                tokens = self._vla_agent.encode_image(x)   # (B, N, 2176)
            return tokens
        else:
            # Stub: uniform noise tokens of the right shape
            B = x.size(0)
            return torch.zeros(B, 256, 2176 if self.backbone == "openvla" else 64,
                               device=x.device, dtype=x.dtype)

    def _encode_instruction(self, B: int, device) -> torch.Tensor:
        """Returns language token (B, 1, d_model)."""
        # Simple hash the instruction string to a fixed-size ID sequence, mean-pool.
        if self._instr_ids is None:
            ids = [ord(c) % 512 for c in self.instruction[:32]] or [0]
            self._instr_ids = torch.tensor(ids, dtype=torch.long)
        ids = self._instr_ids.to(device)
        emb = self.lang_embed(ids).mean(0, keepdim=True).unsqueeze(0)  # (1, 1, d)
        return emb.expand(B, -1, -1)                                    # (B, 1, d)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training forward — returns action chunk predictions with gradient.

        x : (B, 3, H, W)  decoded image
        → actions : (B, chunk_size, action_dim) in [-1, 1]
        """
        B = x.size(0)
        vis_tokens  = self._get_visual_features(x)                 # (B, N, D_vis)
        vis_proj    = self.vis_proj(vis_tokens)                    # (B, N, d_model)
        lang_token  = self._encode_instruction(B, x.device)       # (B, 1, d_model)
        context     = torch.cat([lang_token, vis_proj], dim=1)    # (B, 1+N, d_model)
        return self.chunk_head(context)                            # (B, K, action_dim)

    @torch.inference_mode()
    def predict_action_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference: returns (B, chunk_size, action_dim) on CPU.
        """
        return self.forward(x).cpu()

    def set_instruction(self, s: str):
        self.instruction = s
        self._instr_ids  = None   # invalidate cache

    # LoRA stubs (for trainer compatibility)
    def add_lora(self, **kwargs):              pass
    def set_trainable_lora(self, **kwargs):    pass
    def lora_parameters(self):                 return []
    def lora_state_dict(self):                 return {}
    def load_lora_state_dict(self, sd):        pass


# ========================================================================== #
#  Pi0FastStub  (CPU placeholder for testing)                                #
# ========================================================================== #

class Pi0FastStub(nn.Module):
    """
    Lightweight CPU stub with the same interface as Pi0FastPolicy.
    Returns fixed zero actions — for shape-checking only.
    """

    def __init__(
        self,
        chunk_size:  int  = 16,
        action_dim:  int  = 7,
        instruction: str  = "",
        **kwargs,
    ):
        super().__init__()
        self.chunk_size  = chunk_size
        self.action_dim  = action_dim
        self.instruction = instruction
        self._loaded     = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return torch.zeros(B, self.chunk_size, self.action_dim,
                           device=x.device, dtype=x.dtype)

    @torch.inference_mode()
    def predict_action_chunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).cpu()

    def set_instruction(self, s: str):
        self.instruction = s

    # LoRA stubs
    def add_lora(self, **kwargs):              pass
    def set_trainable_lora(self, **kwargs):    pass
    def lora_parameters(self):                 return []
    def lora_state_dict(self):                 return {}
    def load_lora_state_dict(self, sd):        pass


# ========================================================================== #
#  LerobotPi0FastAgent  (real lerobot PI0Fast model)                         #
# ========================================================================== #

class LerobotPi0FastAgent(nn.Module):
    """
    Wrapper around lerobot's PI0FastPolicy for the VAE-latent SemCom pipeline.

    Takes decoded image x̃_t (B, 3, H, W) in [0, 1] and returns an action
    chunk (B, chunk_size, action_dim).  Handles batch preparation (language
    tokenisation, image key routing) internally.

    Robot state is not available from the image-only pipeline, so a zero
    state vector is used — acceptable for evaluation.

    Parameters
    ----------
    model_path    : HuggingFace hub ID or local directory of the pi0-fast
                    checkpoint (must contain config.json + model.safetensors)
    instruction   : natural-language task description
    image_key     : camera key expected by the loaded model config; if None,
                    the first key in config.image_features is used
    state_dim     : robot state dimension (zeros are passed; must be ≤ max_state_dim)
    """

    def __init__(
        self,
        model_path:  str,
        instruction: str = "do the task",
        cam1_key:    str | None = None,
        cam2_key:    str | None = None,
        state_dim:   int = 6,
        chunk_size:  int | None = None,
    ):
        super().__init__()
        self.model_path  = model_path
        self.instruction = instruction
        self._cam1_key   = cam1_key
        self._cam2_key   = cam2_key
        self.state_dim   = state_dim
        self._chunk_size = chunk_size   # None = use model default
        self._policy     = None
        self._tokenizer  = None
        self._lang_ids   = None
        self._lang_mask  = None

    # ------------------------------------------------------------------ #
    #  Lazy loader                                                         #
    # ------------------------------------------------------------------ #

    def _load(self):
        if self._policy is not None:
            return

        # lerobot.policies.__init__ imports the broken 'groot' module at
        # module level; mock it out before any lerobot.policies import.
        from unittest.mock import MagicMock
        for _m in [
            "lerobot.policies.groot",
            "lerobot.policies.groot.modeling_groot",
            "lerobot.policies.groot.configuration_groot",
            "lerobot.policies.groot.groot_n1",
        ]:
            sys.modules.setdefault(_m, MagicMock())

        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        from lerobot.configs.policies import PreTrainedConfig
        from transformers import AutoTokenizer

        print(f"[LerobotPi0FastAgent] Loading {self.model_path} …")
        if self._chunk_size is not None:
            cfg = PreTrainedConfig.from_pretrained(self.model_path)
            cfg.chunk_size     = self._chunk_size
            cfg.n_action_steps = self._chunk_size
            print(f"[LerobotPi0FastAgent] chunk_size overridden to {self._chunk_size}")
            self._policy = PI0FastPolicy.from_pretrained(self.model_path, config=cfg)
        else:
            self._policy = PI0FastPolicy.from_pretrained(self.model_path)
        self._policy.eval()

        tok_name = getattr(self._policy.config, "text_tokenizer_name",
                           "google/paligemma-3b-pt-224")
        print(f"[LerobotPi0FastAgent] Loading tokenizer ({tok_name}) …")
        self._tokenizer = AutoTokenizer.from_pretrained(tok_name)
        self._cache_lang_tokens()

        # Auto-detect camera keys from model config if not provided
        cfg_keys = list(getattr(self._policy.config, "image_features", {}).keys())
        if self._cam1_key is None:
            self._cam1_key = cfg_keys[0] if cfg_keys else "observation.images.base_0_rgb"
        if self._cam2_key is None:
            self._cam2_key = cfg_keys[1] if len(cfg_keys) >= 2 else None
        print(f"[LerobotPi0FastAgent] cam1_key='{self._cam1_key}'  cam2_key='{self._cam2_key}'")

        print("[LerobotPi0FastAgent] Ready.")

    def _cache_lang_tokens(self):
        """Pre-tokenise instruction with a zero (midpoint) state vector."""
        tok_max = getattr(self._policy.config, "tokenizer_max_length", 200)

        # Zeros → bin 128 (midpoint of 256 bins over [-1, 1])
        state_np   = np.zeros(self.state_dim)
        disc       = np.digitize(state_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_str  = " ".join(map(str, disc))
        cleaned    = self.instruction.strip().replace("_", " ").replace("\n", " ")
        prompt     = f"Task: {cleaned}, State: {state_str};\n"

        enc = self._tokenizer(
            prompt,
            return_tensors  = "pt",
            padding         = "max_length",
            max_length      = tok_max,
            truncation      = True,
        )
        self._lang_ids  = enc["input_ids"]       # (1, tok_max)
        self._lang_mask = enc["attention_mask"]  # (1, tok_max)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_instruction(self, s: str):
        self.instruction = s
        self._lang_ids   = None
        self._lang_mask  = None
        if self._tokenizer is not None:
            self._cache_lang_tokens()

    @property
    def chunk_size(self) -> int:
        self._load()
        return self._policy.config.chunk_size

    @property
    def action_dim(self) -> int:
        self._load()
        from lerobot.utils.constants import ACTION
        return self._policy.config.output_features[ACTION].shape[0]

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        x:  torch.Tensor,                    # (B, 3, H, W)  cam1
        x2: torch.Tensor | None = None,      # (B, 3, H, W)  cam2 (optional)
    ) -> torch.Tensor:
        """
        x  : (B, 3, H, W) decoded image from cam1 in [0, 1]
        x2 : (B, 3, H, W) decoded image from cam2 in [0, 1], or None
        Returns (B, chunk_size, action_dim) on CPU
        """
        self._load()
        device = next(self._policy.parameters()).device
        B = x.size(0)

        batch = {
            self._cam1_key: x.to(device),
            "observation.language.tokens":
                self._lang_ids.expand(B, -1).to(device),
            "observation.language.attention_mask":
                self._lang_mask.expand(B, -1).to(device),
        }
        if x2 is not None and self._cam2_key is not None:
            batch[self._cam2_key] = x2.to(device)

        return self._policy.predict_action_chunk(batch).cpu()

    # keep interface consistent with Pi0FastStub / Pi0FastPolicy
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_action_chunk(x)
