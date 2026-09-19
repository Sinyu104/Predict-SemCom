"""
vla_server.py  —  PI0Fast inference server (runs on the Linux server).

Runs on: Linux server with 4 × T4 GPUs
Transport: ROS2 pub/sub via Fast DDS Discovery Server

Message format
--------------
Client → Server:  /vla/request  (std_msgs/String)
    JSON: {
        "seq":            <int>,        sequence number for request/reply matching
        "jpeg_b64":       "<base64>",   JPEG-compressed RGB frame (base/front camera)
        "jpeg_b64_wrist": "<base64>",   optional wrist camera; falls back to jpeg_b64
        "instruction":    "pick up ...",
    }

Server → Client:  /vla/response (std_msgs/String)
    JSON: {"seq": <int>, "action": [[a0..a6], ...], "status": "ok"}
         action is a nested list of shape (chunk_size, 7)
    On error: {"seq": <int>, "action": [[0]*7], "status": "error", "message": "..."}

ROS2 environment (set on BOTH machines before launching)
---------------------------------------------------------
    export ROS_DOMAIN_ID=66
    export ROS_DISCOVERY_SERVER=10.32.33.49:11811

Usage
-----
On the Linux server (run BEFORE starting isaac_sim/vla_runner.py):

    python server/vla_server.py \\
        --model_dir physical-intelligence/pi0-fast \\
        --lora_path outputs/pi0fast_finetuned/lora_weights.pt

    # Or point to a specific checkpoint epoch:
    python server/vla_server.py \\
        --model_dir physical-intelligence/pi0-fast \\
        --lora_path outputs/pi0fast_finetuned/checkpoint_epoch10/lora_weights.pt

    # Full fine-tune checkpoint (no LoRA) — pass --full_weights instead:
    python server/vla_server.py \\
        --model_dir lerobot/pi0fast-base \\
        --full_weights outputs/pi0fast_so101_full_finetuned/full_weights.pt

Dependencies
------------
    pip install torch lerobot pillow numpy
    # ROS2: source /opt/ros/<distro>/setup.bash  (provides rclpy)
"""

import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image


# ── Silence lerobot's broken groot module before any lerobot import ────────── #

def _mock_groot():
    from unittest.mock import MagicMock
    for mod in [
        "lerobot.policies.groot",
        "lerobot.policies.groot.modeling_groot",
        "lerobot.policies.groot.configuration_groot",
        "lerobot.policies.groot.groot_n1",
    ]:
        sys.modules.setdefault(mod, MagicMock())


# ========================================================================== #
#  PI0Fast model wrapper                                                      #
# ========================================================================== #

class Pi0FastServer:
    """
    Loads PI0FastPolicy onto GPU and runs inference.  Two weight sources,
    mutually exclusive:

      • lora_path    — the base model plus LoRALinear adapters (same structure
                       as pi0fast_server.py's LoRA fine-tune), then lora_weights.pt.
      • full_weights — a full fine-tune checkpoint (full_weights.pt), which is a
                       plain PI0FastPolicy.state_dict() with NO adapters.  It is
                       loaded straight into the un-wrapped policy; the LoRA
                       insertion is skipped entirely, because full_weights.pt was
                       produced without LoRALinear and its keys are the vanilla
                       q_proj.weight / lm_head.weight, not q_proj.lora_A/B.

    Parameters
    ----------
    model_dir    : str          HuggingFace ID or local path to base PI0Fast weights
    lora_path    : str | None   Path to lora_weights.pt from a LoRA fine-tune
    full_weights : str | None   Path to full_weights.pt from a full fine-tune
                                 (takes precedence over lora_path)
    cam1_key     : str   Image key for base camera in policy batch
    cam2_key     : str   Image key for wrist camera in policy batch
    chunk_size   : int   Number of action steps to generate per request
    lora_rank    : int   LoRA rank (must match fine-tuning; ignored for full_weights)
    state_dim    : int   Robot state dim used in language prompt (zeros)
    action_dim   : int   Robot's REAL action DoF: 7 for Franka, 6 for SO-101
                         (5 joints + gripper).  NOT read from the model config —
                         pi0fast-base reports 32 there (the FAST tokeniser's padded
                         max_action_dim), so the returned chunk is sliced to this
                         many leading dims while detokenisation still uses the 32.
    device       : str   Torch device for inference
    """

    def __init__(
        self,
        model_dir:    str,
        lora_path:    str | None = None,
        full_weights: str | None = None,
        cam1_key:   str = "observation.images.base_0_rgb",
        cam2_key:   str = "observation.images.right_wrist_0_rgb",
        chunk_size: int = 10,
        lora_rank:  int = 16,
        state_dim:  int = 6,
        action_dim: int = 7,
        device:     str = "cuda:0",
    ):
        if (lora_path is None) == (full_weights is None):
            raise ValueError(
                "Pass exactly one of lora_path / full_weights "
                f"(got lora_path={lora_path!r}, full_weights={full_weights!r})."
            )
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _mock_groot()
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

        self.cam1_key   = cam1_key
        self.cam2_key   = cam2_key
        self.chunk_size = chunk_size
        self.state_dim  = state_dim
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        # Per-dim action de-normalisation stats, loaded from norm_stats.json next
        # to a full-finetune checkpoint. None → the model was trained on raw
        # actions and predict() returns them as-is.
        self.action_low = self.action_high = None

        import time as _time
        print(f"[Pi0FastServer] Loading base model from '{model_dir}' …", flush=True)
        print(f"[Pi0FastServer] Device: {self.device}", flush=True)

        t0 = _time.time()
        policy = PI0FastPolicy.from_pretrained(model_dir)
        print(f"[Pi0FastServer] from_pretrained done in {_time.time()-t0:.1f}s", flush=True)

        policy.config.chunk_size     = chunk_size
        policy.config.n_action_steps = chunk_size

        if full_weights is not None:
            # ── Full fine-tune: load a vanilla state_dict, no adapters ──── #
            # full_weights.pt is PI0FastPolicy.state_dict() gathered on rank 0
            # by pi0fast_server.py --full_finetune.  Its keys are the standard
            # q_proj.weight / lm_head.weight, so it must load into the UNMODIFIED
            # policy — inserting LoRALinear first would rename modules and leave
            # every trained tensor unmatched.
            t2 = _time.time()
            print(f"[Pi0FastServer] Loading full weights from '{full_weights}' …", flush=True)
            # Load to CPU, not the GPU: the base model already occupies ~13.8 GB
            # of GPU RAM, and mapping the 13.8 GB state_dict onto the same device
            # would need a second full copy and OOM a 15 GB card. load_state_dict
            # copies each CPU tensor into the existing GPU param in place.
            full_sd = torch.load(full_weights, map_location="cpu")
            n_tensors = len(full_sd)
            missing, unexpected = policy.load_state_dict(full_sd, strict=False)
            del full_sd   # free the 13.8 GB CPU copy once it's been copied into params
            print(f"[Pi0FastServer] Full weights loaded: {n_tensors} tensors in "
                  f"{_time.time()-t2:.1f}s  ({len(missing)} missing, {len(unexpected)} unexpected)",
                  flush=True)
            if unexpected:
                # A LoRA checkpoint fed here (lora_A/B keys) would show up as
                # unexpected and load nothing — fail loud rather than serve base.
                print(f"[Pi0FastServer] WARNING: unexpected keys (first 5): {unexpected[:5]}", flush=True)
            if missing:
                print(f"[Pi0FastServer] WARNING: missing keys (first 5): {missing[:5]}", flush=True)

            # ── Load normalisation stats persisted with the checkpoint ──── #
            # A full-finetune trained WITH normalisation emits actions in [-1,1];
            # predict() denormalises them with EXACTLY these per-dim stats. They
            # must come from the checkpoint, never be recomputed/hardcoded — a
            # mismatch reintroduces a silent per-dim offset.
            stats_path = os.path.join(
                os.path.dirname(os.path.abspath(full_weights)), "norm_stats.json")
            if os.path.isfile(stats_path):
                with open(stats_path) as sf:
                    st = json.load(sf)
                self.action_low  = np.asarray(st["action_low"],  dtype=np.float32)
                self.action_high = np.asarray(st["action_high"], dtype=np.float32)
                print(f"[Pi0FastServer] norm_stats.json loaded ({len(self.action_low)} dims) — "
                      f"actions will be denormalised from [-1,1].", flush=True)
            else:
                print(f"[Pi0FastServer] WARNING: no norm_stats.json next to the checkpoint — "
                      f"returning RAW model output. Correct ONLY for a checkpoint trained "
                      f"WITHOUT action normalisation; a normalised checkpoint MUST ship one.",
                      flush=True)
        else:
            # ── Apply LoRA structure — must exactly match pi0fast_server.py ── #
            # Create lora_A/B on the same device as the model weights to avoid
            # mixed CPU/GPU tensors which cause slow dtype conversion later.
            class LoRALinear(nn.Module):
                def __init__(self, linear, rank, alpha):
                    super().__init__()
                    dev  = linear.weight.device
                    dtype = linear.weight.dtype
                    in_f, out_f = linear.in_features, linear.out_features
                    self.weight = linear.weight
                    self.bias   = getattr(linear, "bias", None)
                    self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=dev, dtype=dtype) * 0.01)
                    self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dtype))
                    self.scale  = alpha / rank

                def forward(self, x):
                    out = F.linear(x, self.weight, self.bias)
                    out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
                    return out

            t1 = _time.time()
            lora_alpha = lora_rank * 2
            pali = policy.model.paligemma_with_expert.paligemma

            for layer in pali.model.language_model.layers:
                attn = layer.self_attn
                attn.q_proj = LoRALinear(attn.q_proj, lora_rank, lora_alpha)
                attn.k_proj = LoRALinear(attn.k_proj, lora_rank, lora_alpha)
                attn.v_proj = LoRALinear(attn.v_proj, lora_rank, lora_alpha)
                attn.o_proj = LoRALinear(attn.o_proj, lora_rank, lora_alpha)
            pali.lm_head = LoRALinear(pali.lm_head, lora_rank, lora_alpha)
            print(f"[Pi0FastServer] LoRA insertion done in {_time.time()-t1:.1f}s", flush=True)

            # ── Load fine-tuned LoRA weights ──────────────────────────────── #
            t2 = _time.time()
            print(f"[Pi0FastServer] Loading LoRA weights from '{lora_path}' …", flush=True)
            lora_sd = torch.load(lora_path, map_location=self.device)
            lora_params = {n: p for n, p in policy.named_parameters() if "lora_" in n}
            loaded, missing = 0, []
            for k, v in lora_sd.items():
                if k in lora_params:
                    lora_params[k].data.copy_(v)
                    loaded += 1
                else:
                    missing.append(k)
            print(f"[Pi0FastServer] LoRA tensors loaded: {loaded}/{len(lora_sd)} in {_time.time()-t2:.1f}s", flush=True)
            if missing:
                print(f"[Pi0FastServer] WARNING: missing LoRA keys: {missing[:5]}")

        t3 = _time.time()
        policy = policy.float()
        print(f"[Pi0FastServer] policy.float() done in {_time.time()-t3:.1f}s", flush=True)

        t4 = _time.time()
        policy.eval()
        policy.requires_grad_(False)
        print(f"[Pi0FastServer] eval+freeze done in {_time.time()-t4:.1f}s", flush=True)

        self.policy    = policy
        self._pali_tok = policy._paligemma_tokenizer
        self._tok_max  = getattr(policy.config, "tokenizer_max_length", 200)
        # Robot's real action DoF (from the --action_dim flag, NOT the config,
        # which reports the padded 32).  Used to slice the returned chunk and to
        # size the error fallback for whichever robot is being served.
        self.action_dim = action_dim
        cfg_dim = policy.config.output_features["action"].shape[0]

        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"[Pi0FastServer] {n_gpu} GPU(s) available. action_dim={self.action_dim} "
              f"(config/detok dim={cfg_dim}). Model ready.\n")
        self._debug_first_call = True

    def _build_language_tokens(self, instruction: str):
        import torch
        disc      = np.full(self.state_dim, 128, dtype=int)
        state_str = " ".join(map(str, disc))
        cleaned   = instruction.strip().replace("_", " ").replace("\n", " ")
        prompt    = f"Task: {cleaned}, State: {state_str};\n"
        enc = self._pali_tok(
            prompt, return_tensors="pt",
            padding="max_length", max_length=self._tok_max, truncation=True,
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @__import__("torch").inference_mode()
    def predict(
        self,
        obs_rgb:       np.ndarray,
        instruction:   str,
        obs_rgb_wrist: np.ndarray | None = None,
    ) -> list:
        """
        Run PI0Fast inference and return a full action chunk.

        Uses a fixed generation loop that matches the training layout:
        bos is passed as the first FAST token (causal segment), not appended
        to the language tokens (bidirectional segment) as lerobot's default
        sample_actions_fast does. The mismatch in the default path causes
        a degenerate hidden state that always predicts <bos> instead of Action.

        Returns
        -------
        list of lists: chunk_size × 7 floats  [[a0..a6], [a0..a6], ...]
        """
        import torch, math

        def to_tensor(arr):
            return (
                torch.from_numpy(arr)
                .permute(2, 0, 1).float().unsqueeze(0)
                .to(self.device) / 255.0
            )

        cam1_t = to_tensor(obs_rgb)
        cam2_t = to_tensor(obs_rgb_wrist if obs_rgb_wrist is not None else obs_rgb)
        lang_ids, lang_mask = self._build_language_tokens(instruction)

        batch = {
            self.cam1_key:                         cam1_t,
            self.cam2_key:                         cam2_t,
            "observation.language.tokens":         lang_ids,
            "observation.language.attention_mask": lang_mask.bool(),
        }

        model     = self.policy.model
        pali_tok  = self._pali_tok
        device    = self.device

        images, img_masks = self.policy._preprocess_images(batch)

        # Start with just [bos] in the FAST causal prefix.
        # With lm_head LoRA the model now predicts "Action" → ":" → " " naturally,
        # so we let it generate those text tokens freely, then force FAST tokens.
        bos_id            = pali_tok.bos_token_id
        action_prefix_ids = pali_tok.encode("Action: ", add_special_tokens=False)
        # Free text tokens to generate INSIDE the loop. The prefill step below
        # already samples the FIRST prefix token ("Action") before the loop, so
        # only the remaining len-1 prefix tokens (":", "▁") should be free.
        # Using the full len ran one extra free iteration, which consumed and
        # DISCARDED the first FAST codeword token — the one carrying dim0's
        # leading DCT coefficient — making dim0 decode to ~1190 (verified: a
        # dropped first token offsets dim0 alone, all other dims intact).
        n_text_prefix     = len(action_prefix_ids) - 1

        bos_tok     = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        bos_mask    = torch.ones((1, 1), dtype=torch.bool, device=device)

        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = model.embed_prefix_fast(
            images, img_masks, lang_ids, lang_mask.bool(),
            fast_action_tokens=bos_tok,
            fast_action_masks=bos_mask,
        )

        # Autoregressive generation with KV cache (O(N) instead of O(N²)).
        max_steps   = self.policy.config.max_decoding_steps
        temperature = self.policy.config.temperature
        lm_head     = model.paligemma_with_expert.paligemma.lm_head

        # FAST token range for forced sampling after the text prefix is generated.
        fast_skip  = self.policy.config.fast_skip_tokens
        vocab_size = pali_tok.vocab_size
        n_fast     = 2048
        fast_start = vocab_size - fast_skip - n_fast
        fast_end   = vocab_size - fast_skip
        fast_logit_mask = torch.full((vocab_size,), float("-inf"), device=device)
        fast_logit_mask[fast_start:fast_end] = 0.0  # additive mask: 0 = keep

        # Also permit the "|" end-of-action terminator during the FAST phase.
        # detokenize_actions() finds the action codeword by truncating at the first
        # "|"; training appended it after the FAST tokens. If generation never emits
        # it (the old forced-FAST-only loop couldn't — "|" is outside the FAST range),
        # the entire fixed-length token stream is DCT-decoded and junk coefficients
        # past the true codeword corrupt an action dimension (~20x out of range,
        # migrating between dims). Allowing "|" restores the boundary.
        pipe_id = pali_tok.convert_tokens_to_ids("|")
        fast_logit_mask[pipe_id] = 0.0

        generated = torch.zeros((1, max_steps), dtype=torch.long, device=device)

        # ── Prefill: run the full prefix once with KV cache enabled ──────── #
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d       = model._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

        (prefix_out, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        last_logits = lm_head(prefix_out[:, -1:, :])
        if self._debug_first_call:
            top5_raw = torch.topk(last_logits[0, -1], 5)
            tok_names = [pali_tok.decode([i.item()]) for i in top5_raw.indices]
            print(f"\n[DIAG] step=0 top-5 (raw): {list(zip(tok_names, top5_raw.values.tolist()))}", flush=True)
            print(f"[DIAG] prefix_embs shape: {prefix_embs.shape}, dtype: {prefix_embs.dtype}", flush=True)
            print(f"[DIAG] FAST range: [{fast_start}, {fast_end})  vocab={vocab_size}  skip={fast_skip}", flush=True)
            print(f"[DIAG] Generating {n_text_prefix} text tokens freely, then forcing FAST", flush=True)

        # Step 0: sample freely (expect "Action")
        if temperature > 0:
            next_token = torch.multinomial(torch.softmax(last_logits[:, -1] / temperature, dim=-1), 1)
        else:
            next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

        # current_pad_mask tracks which positions exist (for attention mask construction)
        current_pad_mask = prefix_pad_masks  # [1, prefix_len]

        # ── Decoding: generate text prefix freely, then force FAST tokens ── #
        # We collect only the FAST tokens (after the text prefix) in `generated`.
        fast_idx   = 0   # index into generated[]
        terminated = False   # True once the model emits "|" and closes the codeword
        text_generated = []

        for t in range(max_steps + n_text_prefix):
            # Feed previous token, get next logits
            next_emb = model.paligemma_with_expert.embed_language_tokens(next_token)
            next_emb = next_emb * math.sqrt(next_emb.shape[-1])
            if prefix_embs.dtype == torch.bfloat16:
                next_emb = next_emb.to(torch.bfloat16)

            current_pad_mask = torch.cat(
                [current_pad_mask, torch.ones((1, 1), dtype=torch.bool, device=device)], dim=1
            )
            current_pos   = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            step_att_mask = model._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1), dtype=next_emb.dtype
            )

            (step_out, _), past_key_values = model.paligemma_with_expert.forward(
                attention_mask=step_att_mask,
                position_ids=current_pos,
                past_key_values=past_key_values,
                inputs_embeds=[next_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )

            last_logits = lm_head(step_out[:, -1:, :])

            if t < n_text_prefix:
                # Still in text prefix phase — sample freely (model generates "Action", ":", " ")
                if temperature > 0:
                    next_token = torch.multinomial(torch.softmax(last_logits[:, -1] / temperature, dim=-1), 1)
                else:
                    next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)
                text_generated.append(next_token.item())
            else:
                # FAST phase — allow FAST tokens plus the "|" terminator.
                masked_logits = last_logits + fast_logit_mask
                if temperature > 0:
                    next_token = torch.multinomial(torch.softmax(masked_logits[:, -1] / temperature, dim=-1), 1)
                else:
                    next_token = torch.argmax(masked_logits[:, -1], dim=-1, keepdim=True)
                if fast_idx < max_steps:
                    generated[:, fast_idx] = next_token.squeeze(-1)
                    fast_idx += 1
                # Stop as soon as the model closes the action with "|". Everything up
                # to and including it is the real codeword (detokenize_actions truncates
                # at the "|"); this both removes the junk-coefficient corruption and
                # ends early instead of always running max_decoding_steps.
                if next_token.item() == pipe_id:
                    terminated = True
                    break

            if fast_idx >= max_steps:
                break

        # Prepend "Action: " so detokenize_actions sees the expected prefix.
        # detok_dim is the tokeniser's padded width (32 for pi0fast-base); the
        # real robot DoF (self.action_dim) is sliced out of the result below.
        detok_dim         = self.policy.config.output_features["action"].shape[0]
        prefix_tensor     = torch.tensor([action_prefix_ids], dtype=torch.long, device=device)
        # Only the tokens actually generated (including the closing "|"), NOT the
        # full max_steps buffer — trailing zero slots would decode to junk too.
        generated_full    = torch.cat([prefix_tensor, generated[:, :fast_idx]], dim=1)

        # Debug: show what the model generated
        gen_ids  = generated[0, :10].tolist()
        gen_strs = [pali_tok.decode([i]) for i in gen_ids]
        txt_strs = [pali_tok.decode([i]) for i in text_generated]
        print(f"[DIAG] text prefix generated: {txt_strs}", flush=True)
        print(f"[DIAG] generated[:10] ids  : {gen_ids}", flush=True)
        print(f"[DIAG] generated[:10] strs : {gen_strs}", flush=True)
        print(f"[DIAG] fast tokens used={fast_idx}/{max_steps}  terminated_by_'|'={terminated}", flush=True)
        print(f"[DIAG] vocab_size={pali_tok.vocab_size}  fast_skip={self.policy.config.fast_skip_tokens}", flush=True)

        continuous = self.policy.detokenize_actions(
            generated_full, action_horizon=self.chunk_size, action_dim=detok_dim)
        # continuous: (1, chunk_size, detok_dim=32) — the first self.action_dim
        # dims are the real robot channels (7 Franka / 6 SO-101), the rest are
        # tokeniser padding.  Slice to the robot's DoF.
        actions_t = continuous[0, :, :self.action_dim].cpu().float()
        # If the checkpoint was trained with per-dim normalisation, the model
        # emits actions in [-1,1]; map them back to physical units with the SAME
        # stats used at training (loaded from norm_stats.json).
        if self.action_low is not None:
            lo = torch.as_tensor(self.action_low[:self.action_dim], dtype=actions_t.dtype)
            hi = torch.as_tensor(self.action_high[:self.action_dim], dtype=actions_t.dtype)
            actions_t = (actions_t + 1.0) / 2.0 * (hi - lo) + lo
        actions = actions_t.tolist()

        if self._debug_first_call:
            self._debug_first_call = False
            print("\n[Pi0FastServer] ── FIRST INFERENCE DEBUG ──────────────────")
            print(f"  instruction : {instruction!r}")
            print(f"  cam1 shape  : {cam1_t.shape}  cam2: {cam2_t.shape}")
            print(f"  chunk_size  : {self.chunk_size}")
            print(f"  actions[0]  : {actions[0]}")
            print(f"  actions[-1] : {actions[-1]}")
            print("[Pi0FastServer] ─────────────────────────────────────────────\n")

        return actions   # list of chunk_size lists of 7 floats


# ========================================================================== #
#  TCP socket inference server                                                #
# ========================================================================== #

def run_server(args):
    """
    Start the TCP socket inference server.  Blocks until interrupted with Ctrl-C.

    Listens on --host:--port for length-prefixed JSON requests from ros2_bridge.py,
    runs PI0Fast inference, and returns length-prefixed JSON responses.
    """
    import socket

    def _resolve(path, filename):
        """Accept either the .pt file directly or a checkpoint dir containing it."""
        if os.path.isfile(path):
            return path
        candidate = os.path.join(path, filename)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"{filename} not found at '{path}'.")

    # Full weights take precedence; the two sources are mutually exclusive.
    lora_path = full_weights = None
    if args.full_weights is not None:
        full_weights = _resolve(args.full_weights, "full_weights.pt")
    else:
        lora_path = _resolve(args.lora_path, "lora_weights.pt")

    server = Pi0FastServer(
        model_dir    = args.model_dir,
        lora_path    = lora_path,
        full_weights = full_weights,
        cam1_key   = args.cam1_key,
        cam2_key   = args.cam2_key,
        chunk_size = args.chunk_size,
        lora_rank  = args.lora_rank,
        state_dim  = args.state_dim,
        action_dim = args.action_dim,
        device     = args.device,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)

    print(f"\n[Pi0FastServer] TCP socket server listening on {args.host}:{args.port}")
    print(f"[Pi0FastServer] chunk_size={args.chunk_size}  —  waiting for ros2_bridge …\n")

    n_requests = 0
    total_ms   = 0.0

    def handle_connection(conn, addr):
        """Serve one persistent connection until the client disconnects."""
        nonlocal n_requests, total_ms
        print(f"[Pi0FastServer] Bridge connected from {addr}")
        with conn:
            while True:
                # Read 4-byte length prefix
                raw_len = b""
                while len(raw_len) < 4:
                    chunk = conn.recv(4 - len(raw_len))
                    if not chunk:
                        print("[Pi0FastServer] Bridge disconnected.")
                        return
                    raw_len += chunk

                req_len = int.from_bytes(raw_len, "big")
                chunks, received = [], 0
                while received < req_len:
                    chunk = conn.recv(min(65536, req_len - received))
                    if not chunk:
                        print("[Pi0FastServer] Bridge disconnected mid-message.")
                        return
                    chunks.append(chunk)
                    received += len(chunk)

                data = json.loads(b"".join(chunks))
                seq  = data.get("seq", 0)

                try:
                    obs_rgb = np.array(
                        Image.open(io.BytesIO(base64.b64decode(data["jpeg_b64"]))).convert("RGB"),
                        dtype=np.uint8,
                    )
                    obs_rgb_wrist = None
                    if "jpeg_b64_wrist" in data:
                        obs_rgb_wrist = np.array(
                            Image.open(io.BytesIO(base64.b64decode(data["jpeg_b64_wrist"]))).convert("RGB"),
                            dtype=np.uint8,
                        )
                except Exception as e:
                    response = {"seq": seq, "action": [[0.0] * server.action_dim],
                                "status": "error", "message": f"Image decode error: {e}"}
                    payload  = json.dumps(response).encode()
                    conn.sendall(len(payload).to_bytes(4, "big") + payload)
                    continue

                instruction = data.get("instruction", args.instruction)

                t0           = time.perf_counter()
                action_chunk = server.predict(obs_rgb, instruction, obs_rgb_wrist)
                ms           = (time.perf_counter() - t0) * 1000

                n_requests += 1
                total_ms   += ms

                response = {"seq": seq, "action": action_chunk, "status": "ok"}
                payload  = json.dumps(response).encode()
                conn.sendall(len(payload).to_bytes(4, "big") + payload)

                if n_requests % 20 == 0:
                    print(
                        f"[Pi0FastServer] {n_requests} requests  "
                        f"avg_latency={total_ms/n_requests:.1f} ms"
                    )

    try:
        # Accept connections in a loop — reconnects if bridge restarts
        while True:
            conn, addr = sock.accept()
            handle_connection(conn, addr)

    except KeyboardInterrupt:
        print(f"\n[Pi0FastServer] Shutting down after {n_requests} requests.")
    finally:
        sock.close()


# ========================================================================== #
#  CLI                                                                        #
# ========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="PI0Fast TCP inference server — run alongside ros2_bridge.py"
    )
    p.add_argument("--model_dir",   type=str,
                   default="physical-intelligence/pi0-fast",
                   help="HuggingFace model ID or local path to base PI0Fast weights")
    p.add_argument("--lora_path",   type=str,
                   default="outputs/pi0fast_finetuned/lora_weights.pt",
                   help="Path to lora_weights.pt (or checkpoint directory containing it). "
                        "Used only when --full_weights is not given.")
    p.add_argument("--full_weights", type=str, default=None,
                   help="Path to full_weights.pt from a --full_finetune run (or a checkpoint "
                        "directory containing it). Loads a full state_dict with NO LoRA "
                        "adapters; takes precedence over --lora_path.")
    p.add_argument("--cam1_key",    type=str,
                   default="observation.images.base_0_rgb")
    p.add_argument("--cam2_key",    type=str,
                   default="observation.images.right_wrist_0_rgb")
    p.add_argument("--chunk_size",  type=int,   default=30,
                   help="Actions per request (must match fine-tuning chunk_size)")
    p.add_argument("--lora_rank",   type=int,   default=16,
                   help="LoRA rank (must match fine-tuning)")
    p.add_argument("--state_dim",   type=int,   default=6)
    p.add_argument("--action_dim",  type=int,   default=7,
                   help="Robot's real action DoF: 7 for Franka, 6 for SO-101. "
                        "The model emits 32 padded dims; the reply is sliced to this many. "
                        "NOT read from the model config (which reports the padded 32).")
    p.add_argument("--device",      type=str,   default="cuda:0")
    p.add_argument("--host",        type=str,   default="127.0.0.1",
                   help="TCP host to listen on (ros2_bridge.py must use the same)")
    p.add_argument("--port",        type=int,   default=5555,
                   help="TCP port to listen on")
    p.add_argument("--instruction", type=str,
                   default="pick up the red cube and place it on the tray",
                   help="Fallback instruction if not provided in the request")
    return p.parse_args()


if __name__ == "__main__":
    run_server(parse_args())
