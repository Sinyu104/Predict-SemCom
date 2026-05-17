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
    export ROS_DISCOVERY_SERVER=10.32.33.41:11811

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
import queue
import sys
import threading
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
    Loads PI0FastPolicy onto GPU, applies LoRALinear adapters (same structure
    as fine-tuning), loads LoRA weights from checkpoint, and runs inference.

    Parameters
    ----------
    model_dir   : str   HuggingFace ID or local path to base PI0Fast weights
    lora_path   : str   Path to lora_weights.pt from fine-tuning checkpoint
    cam1_key    : str   Image key for base camera in policy batch
    cam2_key    : str   Image key for wrist camera in policy batch
    chunk_size  : int   Number of action steps to generate per request
    lora_rank   : int   LoRA rank (must match fine-tuning)
    state_dim   : int   Robot state dim used in language prompt (zeros)
    device      : str   Torch device for inference
    """

    def __init__(
        self,
        model_dir:  str,
        lora_path:  str,
        cam1_key:   str = "observation.images.base_0_rgb",
        cam2_key:   str = "observation.images.right_wrist_0_rgb",
        chunk_size: int = 10,
        lora_rank:  int = 16,
        state_dim:  int = 6,
        device:     str = "cuda:0",
    ):
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

        print(f"[Pi0FastServer] Loading base model from '{model_dir}' …")
        print(f"[Pi0FastServer] Device: {self.device}")

        # Load on CPU first to avoid OOM, then move to GPU after LoRA insertion
        _orig_to = nn.Module.to
        def _cpu_only_to(self_m, *args, **kwargs):
            if args:
                try:
                    if torch.device(args[0]).type == "cuda":
                        return _orig_to(self_m, "cpu")
                except Exception:
                    pass
            return _orig_to(self_m, *args, **kwargs)
        nn.Module.to = _cpu_only_to
        try:
            policy = PI0FastPolicy.from_pretrained(model_dir)
        finally:
            nn.Module.to = _orig_to

        policy.config.chunk_size     = chunk_size
        policy.config.n_action_steps = chunk_size

        # ── Apply LoRA structure — must exactly match pi0fast_server.py ── #
        class LoRALinear(nn.Module):
            def __init__(self, linear, rank, alpha):
                super().__init__()
                in_f, out_f = linear.in_features, linear.out_features
                self.weight = linear.weight
                self.bias   = getattr(linear, "bias", None)
                self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
                self.scale  = alpha / rank

            def forward(self, x):
                out = F.linear(x, self.weight, self.bias)
                out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
                return out

        lora_alpha = lora_rank * 2
        pali = policy.model.paligemma_with_expert.paligemma

        pali.lm_head = LoRALinear(pali.lm_head, lora_rank, lora_alpha)
        for layer in pali.model.language_model.layers:
            attn = layer.self_attn
            attn.q_proj = LoRALinear(attn.q_proj, lora_rank, lora_alpha)
            attn.k_proj = LoRALinear(attn.k_proj, lora_rank, lora_alpha)
            attn.v_proj = LoRALinear(attn.v_proj, lora_rank, lora_alpha)

        # ── Load fine-tuned LoRA weights ──────────────────────────────── #
        print(f"[Pi0FastServer] Loading LoRA weights from '{lora_path}' …")
        lora_sd = torch.load(lora_path, map_location="cpu")
        missing, _ = policy.load_state_dict(lora_sd, strict=False)
        lora_missing = [k for k in missing if "lora_" in k]
        print(f"[Pi0FastServer] LoRA tensors loaded: {len(lora_sd) - len(lora_missing)}/{len(lora_sd)}")
        if lora_missing:
            print(f"[Pi0FastServer] WARNING: missing LoRA keys: {lora_missing[:5]}")

        policy = policy.float().to(self.device)
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)

        self.policy    = policy
        self._pali_tok = policy._paligemma_tokenizer
        self._tok_max  = getattr(policy.config, "tokenizer_max_length", 200)

        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"[Pi0FastServer] {n_gpu} GPU(s) available. Model ready.\n")
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

        Returns
        -------
        list of lists: chunk_size × 7 floats  [[a0..a6], [a0..a6], ...]
        """
        import torch

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

        # Reset internal action queue, then drain a fresh chunk
        self.policy.reset()
        actions = []
        for _ in range(self.chunk_size):
            act = self.policy.select_action(batch)   # (action_dim,) tensor
            actions.append(act[:7].cpu().float().tolist())

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
#  ROS2 inference server                                                      #
# ========================================================================== #

def run_server(args):
    """
    Start the ROS2 inference server.  Blocks until interrupted with Ctrl-C.
    """
    try:
        import rclpy
        from std_msgs.msg import String
    except ImportError:
        raise ImportError(
            "rclpy not found. Source your ROS2 workspace:\n"
            "  source /opt/ros/<distro>/setup.bash"
        )

    # Resolve lora_weights.pt (accept either a .pt file or a checkpoint directory)
    lora_path = args.lora_path
    if not os.path.isfile(lora_path):
        candidate = os.path.join(lora_path, "lora_weights.pt")
        if os.path.isfile(candidate):
            lora_path = candidate
        else:
            raise FileNotFoundError(
                f"lora_weights.pt not found at '{args.lora_path}'. "
                "Pass --lora_path outputs/pi0fast_finetuned/lora_weights.pt"
            )

    server = Pi0FastServer(
        model_dir  = args.model_dir,
        lora_path  = lora_path,
        cam1_key   = args.cam1_key,
        cam2_key   = args.cam2_key,
        chunk_size = args.chunk_size,
        lora_rank  = args.lora_rank,
        state_dim  = args.state_dim,
        device     = args.device,
    )

    rclpy.init()
    node = rclpy.create_node("vla_server")
    pub  = node.create_publisher(String, "/vla/response", 10)

    request_queue = queue.Queue()

    def on_request(msg):
        try:
            request_queue.put(json.loads(msg.data))
        except json.JSONDecodeError as e:
            node.get_logger().error(f"Malformed request JSON: {e}")

    node.create_subscription(String, "/vla/request", on_request, 10)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    print("\n[Pi0FastServer] ROS2 node started.")
    print("[Pi0FastServer] Subscribed to /vla/request")
    print("[Pi0FastServer] Publishing to  /vla/response")
    print(f"[Pi0FastServer] chunk_size={args.chunk_size}  —  waiting for client …\n")

    n_requests = 0
    total_ms   = 0.0

    try:
        while True:
            data = request_queue.get()
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
                reply      = String()
                reply.data = json.dumps({
                    "seq": seq, "action": [[0.0] * 7],
                    "status": "error", "message": f"Image decode error: {e}",
                })
                pub.publish(reply)
                continue

            instruction = data.get("instruction", args.instruction)

            t0           = time.perf_counter()
            action_chunk = server.predict(obs_rgb, instruction, obs_rgb_wrist)
            ms           = (time.perf_counter() - t0) * 1000

            n_requests += 1
            total_ms   += ms

            reply      = String()
            reply.data = json.dumps({"seq": seq, "action": action_chunk, "status": "ok"})
            pub.publish(reply)

            if n_requests % 20 == 0:
                print(
                    f"[Pi0FastServer] {n_requests} requests  "
                    f"avg_latency={total_ms/n_requests:.1f} ms"
                )

    except KeyboardInterrupt:
        print(f"\n[Pi0FastServer] Shutting down after {n_requests} requests.")
    finally:
        rclpy.shutdown()


# ========================================================================== #
#  CLI                                                                        #
# ========================================================================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="PI0Fast ROS2 inference server — runs on Linux server with 4× T4"
    )
    p.add_argument("--model_dir",   type=str,
                   default="physical-intelligence/pi0-fast",
                   help="HuggingFace model ID or local path to base PI0Fast weights")
    p.add_argument("--lora_path",   type=str,
                   default="outputs/pi0fast_finetuned/lora_weights.pt",
                   help="Path to lora_weights.pt (or checkpoint directory containing it)")
    p.add_argument("--cam1_key",    type=str,
                   default="observation.images.base_0_rgb")
    p.add_argument("--cam2_key",    type=str,
                   default="observation.images.right_wrist_0_rgb")
    p.add_argument("--chunk_size",  type=int,   default=10,
                   help="Actions per request (must match fine-tuning chunk_size)")
    p.add_argument("--lora_rank",   type=int,   default=16,
                   help="LoRA rank (must match fine-tuning)")
    p.add_argument("--state_dim",   type=int,   default=6)
    p.add_argument("--device",      type=str,   default="cuda:0")
    p.add_argument("--instruction", type=str,
                   default="pick up the red cube and place it on the tray",
                   help="Fallback instruction if not provided in the request")
    return p.parse_args()


if __name__ == "__main__":
    run_server(parse_args())
