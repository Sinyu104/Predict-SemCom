"""
semcom_server.py  —  Online closed-loop evaluation of the Wyner-Ziv semantic
communication pipeline, with a policy in the loop.

Same TCP/ROS front end as openvla_server.py, but the received observation is
pushed through the SemCom pipeline before it reaches the policy:

    o_t  ->  [VAE encode]      ->  z_t
             [JSCC encode]     ->  s_t
             [Fading channel]  ->  s~_t
    [z~_{t-6..t-1}], a_t -> [Ctrl-World DDIM] -> z_hat_t      (side information)
    (z_hat_t, s~_t)      -> [Refinement]      -> z~_t
                            [VAE decode]      -> x~_t
                            [Policy]          -> a_hat_t  -> back to the robot

Three processes, because the SemCom stack and OpenVLA cannot share a conda env:

    worldmodel   transformers 4.40.2 (OpenVLA pins it) + diffusers 0.35.2
                 -> diffusers needs EncoderDecoderCache (transformers >=4.42),
                    so ctrl_world_wrapper cannot import here
    predictor    transformers 4.57.6 + diffusers 0.38
                 -> Ctrl-World fine, OpenVLA will not load

    ros2_bridge.py   (py3.10, ROS2)      --TCP 5558-->
    semcom_server.py (predictor env)     --TCP 5557--> openvla_server.py (worldmodel)

Same split, same reason, as the existing rclpy/py3.12 bridge.  It also means the
policy is swappable (pi0-FAST, ACT, ...) without touching this file.

Unlike openvla_server.py this server is STATEFUL: Ctrl-World conditions on the
last `num_history` RECONSTRUCTED latents, so it keeps a per-episode buffer and
must be told when an episode starts.  The client sends an "episode" field in the
request JSON; when it changes, history is cleared.  A client that never sends
one gets a loud warning, because silently carrying history across a scene reset
conditions the predictor on frames from a different episode.

Modes (--mode), in increasing order of what is exercised:

    bypass      raw image straight to the policy.  Reproduces openvla_server.py
                exactly — the control for validating ROS plumbing, the episode
                state machine, and the reset protocol before any diffusion runs.
    vae         VAE encode -> decode only.  Isolates codec loss.
    nochannel   full pipeline with s~_t zeroed.  Shows what the Ctrl-World
                prediction alone supports (the "C" baseline in the Stage-2 eval).
    full        the complete pipeline.

Build up one mode at a time: each has a predictable expected result, so a
surprise localises immediately instead of being lost in the whole stack.

Usage
-----
  Terminal 1 — policy server (worldmodel env), unchanged:
    python server/openvla_server.py --serve --port 5557 \\
        --lora_path outputs/openvla_dagger_v2/checkpoint_epoch24

  Terminal 2 — this server (predictor env):
    # control run — should reproduce your OpenVLA-only success rate
    python server/semcom_server.py --mode bypass --port 5558

    # full pipeline at 10 dB
    python server/semcom_server.py --mode full --port 5558 \\
        --stage1_ckpt outputs/stage1_5cube_cam1_K8/stage1_best.pt \\
        --stage2_ckpt outputs/stage2_twophase_full/phase2.pt \\
        --snr_db 10 --channel rayleigh

  Terminal 3 — ROS bridge, pointed at THIS server instead of 5557:
    python3 server/ros2_bridge.py --host 127.0.0.1 --port 5558

Then run isaac_sim/vla_runner.py on the Windows side exactly as before.
"""

import argparse
import base64
import io
import json
import os
import pathlib
import socket
import sys
import time
from collections import deque

import numpy as np
import torch
from PIL import Image

REPO = str(pathlib.Path(__file__).resolve().parents[1])
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from config import CONFIG                      # noqa: E402
from models import SemComSystem                # noqa: E402


# ---------------------------------------------------------------------------
#  SemCom pipeline (stateful, one episode at a time)
# ---------------------------------------------------------------------------

class SemComPipeline:
    """
    Wraps VAE + Ctrl-World + SemComSystem and holds the per-episode history.

    step(obs_rgb, prev_action) -> (x_tilde_uint8, timings)

    `prev_action` is the action this server returned on the previous request —
    the one the robot has just executed to arrive at `obs_rgb`.  In the notation
    of CLAUDE.md that is a_t, the action conditioning for predicting z_hat_t, so
    there is no circular dependency on the action we are about to produce.
    """

    def __init__(self, config, vae, ctrl_world, system, device, mode):
        self.cfg        = config
        self.vae        = vae
        self.ctrl_world = ctrl_world
        self.system     = system
        self.device     = device
        self.mode       = mode

        self.num_history = config["num_history"]
        self.num_pred    = config["num_pred"]
        self.action_dim  = config["action_dim"]
        self.n_ddim      = config.get("ctrl_world_n_steps", 10)
        self.t_second    = config.get("noise_level_second", 0)

        self.episode = None
        self.reset()

    def reset(self):
        """Clear per-episode state.  Called on every episode boundary."""
        self.z_hist = deque(maxlen=self.num_history)
        self.a_hist = deque(maxlen=self.num_history + self.num_pred)
        # Actions are zero-padded until the robot has actually moved.
        for _ in range(self.num_history + self.num_pred):
            self.a_hist.append(np.zeros(self.action_dim, dtype=np.float32))

    # -- helpers ---------------------------------------------------------- #

    def _to_tensor(self, obs_rgb: np.ndarray) -> torch.Tensor:
        """(H,W,3) uint8 -> (1,3,H,W) float32 in [0,1], which VAEWrapper expects."""
        x = torch.from_numpy(obs_rgb).float().div_(255.0)
        return x.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _to_uint8(self, x: torch.Tensor) -> np.ndarray:
        """(1,3,H,W) float [0,1] -> (H,W,3) uint8."""
        x = x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (x * 255.0).round().astype(np.uint8)

    def _history_tensor(self) -> torch.Tensor:
        """
        (1, num_history, C, H, W).  Before the buffer fills, the oldest frame is
        repeated — at episode start the arm is at its home pose and the scene is
        static, so repeat-padding is a mild approximation.  It only affects the
        first `num_history` steps of a ~300-step episode.
        """
        frames = list(self.z_hist)
        while len(frames) < self.num_history:
            frames.insert(0, frames[0])
        return torch.stack(frames, dim=1)

    def _action_tensor(self) -> torch.Tensor:
        """(1, num_history + num_pred, action_dim), oldest first."""
        a = np.stack(list(self.a_hist), axis=0)
        return torch.from_numpy(a).unsqueeze(0).to(self.device)

    # -- main entry ------------------------------------------------------- #

    @torch.no_grad()
    def step(self, obs_rgb: np.ndarray, prev_action):
        t = {}
        if self.mode == "bypass":
            return obs_rgb, t

        self.a_hist.append(
            np.asarray(prev_action, dtype=np.float32)
            if prev_action is not None
            else np.zeros(self.action_dim, dtype=np.float32)
        )

        t0 = time.perf_counter()
        x     = self._to_tensor(obs_rgb)
        z_t   = self.vae.encode(x)                       # (1, C, H, W)
        t["vae_enc"] = time.perf_counter() - t0

        if self.mode == "vae":
            t0 = time.perf_counter()
            x_tilde = self.vae.decode(z_t)
            t["vae_dec"] = time.perf_counter() - t0
            self.z_hist.append(z_t[0])
            return self._to_uint8(x_tilde), t

        # -- Wyner-Ziv side information (receiver-only, no access to s~_t) -- #
        t0 = time.perf_counter()
        if len(self.z_hist) == 0:
            # First frame of an episode: nothing to condition on.  Fall back to
            # the observed latent so the refinement still has a sane prior.
            z_hat = z_t
        else:
            z_pred = self.ctrl_world.predict_next_latent(
                self._history_tensor(), self._action_tensor(), n_steps=self.n_ddim,
            )                                            # (1, num_pred, C, H, W)
            z_hat = z_pred[:, 0]
        t["ctrl_world"] = time.perf_counter() - t0

        # -- transmitter: JSCC over the channel ---------------------------- #
        t0 = time.perf_counter()
        _, _, s_t = self.system.jscc_encoder(z_t, sample=False)
        s_tilde   = self.system.channel(s_t)
        if self.mode == "nochannel":
            s_tilde = torch.zeros_like(s_tilde)
        t["jscc"] = time.perf_counter() - t0

        # -- receiver: refinement ------------------------------------------ #
        t0 = time.perf_counter()
        z_tilde = self.system.refinement_diffusion.sdedit_refine(
            z_hat, s_tilde, noise_level=self.t_second, n_steps=self.n_ddim,
        )
        t["refine"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        x_tilde = self.vae.decode(z_tilde)
        t["vae_dec"] = time.perf_counter() - t0

        # History is the RECONSTRUCTED latent — what the receiver would actually
        # have at the next step, not the ground-truth z_t.
        self.z_hist.append(z_tilde[0])
        return self._to_uint8(x_tilde), t


# ---------------------------------------------------------------------------
#  Model construction
# ---------------------------------------------------------------------------

def build_pipeline(args, config, device):
    if args.mode == "bypass":
        print("[SemCom] mode=bypass — no SemCom models loaded (policy only).")
        return SemComPipeline(config, None, None, None, device, "bypass")

    from vae_wrapper import VAEWrapper
    print(f"[SemCom] VAE: {config.get('vae_model_name')}")
    vae = VAEWrapper(config.get("vae_model_name", "stabilityai/sd-vae-ft-mse")).to(device)

    ctrl_world = None
    system     = None

    if args.mode in ("nochannel", "full"):
        from ctrl_world_wrapper import CtrlWorldWrapper
        svd_path = args.svd_path or config.get("svd_model_name")
        print(f"[SemCom] Ctrl-World: SVD={svd_path}  num_pred={config['num_pred']}")
        ctrl_world = CtrlWorldWrapper(
            action_dim          = config["action_dim"],
            num_history         = config["num_history"],
            num_pred            = config["num_pred"],
            svd_path            = svd_path,
            ctrl_ckpt           = None,
            freeze_unet         = True,
            finetune_cross_attn = config.get("finetune_unet_cross_attn", False),
            dtype               = torch.float16,
        ).to(device)

        # Stage-1 weights.  BOTH parts matter: loading only action_encoder_state
        # silently discards ~100M trained cross-attention parameters.
        ck = torch.load(args.stage1_ckpt, map_location="cpu")
        if "action_encoder_state" in ck:
            ctrl_world.action_encoder.load_state_dict(
                ck["action_encoder_state"], strict=False)
            print(f"[SemCom] Stage-1 action_encoder restored "
                  f"({len(ck['action_encoder_state'])} tensors)")
        else:
            print("[SemCom] WARNING: no action_encoder_state in Stage-1 checkpoint.")
        if "unet_cross_attn_state" in ck:
            ctrl_world.load_unet_cross_attn_state_dict(ck["unet_cross_attn_state"])
            print(f"[SemCom] Stage-1 unet_cross_attn restored "
                  f"({len(ck['unet_cross_attn_state'])} tensors)")
        else:
            print("[SemCom] WARNING: no unet_cross_attn_state — the UNet is stock SVD.")
        ctrl_world.eval()

        system = SemComSystem(config).to(device)
        sk = torch.load(args.stage2_ckpt, map_location="cpu")
        state = sk.get("system_state", sk)
        missing, unexpected = system.load_state_dict(state, strict=False)
        print(f"[SemCom] Stage-2 restored from {args.stage2_ckpt} "
              f"({len(state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected)")
        system.eval()

    return SemComPipeline(config, vae, ctrl_world, system, device, args.mode)


# ---------------------------------------------------------------------------
#  TCP server  (protocol identical to openvla_server.py, plus "episode")
# ---------------------------------------------------------------------------

def _send(conn, obj: dict):
    payload = json.dumps(obj).encode()
    conn.sendall(len(payload).to_bytes(4, "big") + payload)


def _recv_all(sock, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        c = sock.recv(min(65536, n - got))
        if not c:
            raise ConnectionError("policy server disconnected")
        chunks.append(c)
        got += len(c)
    return b"".join(chunks)


class PolicyClient:
    """
    Persistent TCP client to a policy server (openvla_server.py / vla_server.py).

    Speaks the same framed-JSON protocol the ROS bridge uses, so any existing
    policy server works unmodified — the only difference is that the image we
    send is the reconstructed x~_t rather than the raw observation.
    """

    def __init__(self, host: str, port: int, jpeg_quality: int = 95):
        self.jpeg_quality = jpeg_quality
        self._seq = 0
        print(f"[SemCom] Connecting to policy server at {host}:{port} …")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("[SemCom] Policy server connected.")

    def predict(self, img_rgb: np.ndarray, instruction: str) -> list:
        buf = io.BytesIO()
        Image.fromarray(img_rgb).save(buf, format="JPEG", quality=self.jpeg_quality)
        self._seq += 1
        payload = json.dumps({
            "seq":         self._seq,
            "jpeg_b64":    base64.b64encode(buf.getvalue()).decode("utf-8"),
            "instruction": instruction,
            "unnorm_key":  "franka_isaac",
        }).encode()
        self.sock.sendall(len(payload).to_bytes(4, "big") + payload)
        n = int.from_bytes(_recv_all(self.sock, 4), "big")
        reply = json.loads(_recv_all(self.sock, n))
        if reply.get("status") != "ok":
            print(f"[SemCom] policy error: {reply.get('message')}", flush=True)
            return [0.0] * 7
        return reply["action"]


def run_server(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config overrides
    for key, val in [("snr_db", args.snr_db), ("channel_type", args.channel),
                     ("D_jscc", args.D_jscc), ("num_history", args.num_history),
                     ("ctrl_world_n_steps", args.ctrl_world_n_steps),
                     ("noise_level_second", args.noise_level_second)]:
        if val is not None:
            CONFIG[key] = val
    # Closed loop knows one step at a time; K>1 needs future actions we do not have.
    CONFIG["num_pred"]    = 1
    CONFIG["clip_length"] = CONFIG["num_history"] + 1

    print(f"[SemCom] mode={args.mode}  snr={CONFIG['snr_db']}dB  "
          f"channel={CONFIG.get('channel_type')}  D_jscc={CONFIG['D_jscc']}")
    print(f"[SemCom] num_history={CONFIG['num_history']}  num_pred=1  "
          f"ddim_steps={CONFIG.get('ctrl_world_n_steps')}  "
          f"t''={CONFIG.get('noise_level_second')}")

    t_start  = time.perf_counter()
    pipeline = build_pipeline(args, CONFIG, device)

    policy = PolicyClient(args.policy_host, args.policy_port)
    print(f"[SemCom] Ready in {time.perf_counter() - t_start:.1f}s")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)
    print(f"[SemCom] TCP listening on {args.host}:{args.port}", flush=True)

    n_reqs, total_ms = 0, 0.0
    stage_ms  = {}
    prev_act  = None
    warned_no_episode = False

    def handle(conn, addr):
        nonlocal n_reqs, total_ms, prev_act, warned_no_episode
        print(f"[SemCom] Client from {addr}", flush=True)
        with conn:
            while True:
                raw = b""
                while len(raw) < 4:
                    c = conn.recv(4 - len(raw))
                    if not c:
                        return
                    raw += c
                req_len = int.from_bytes(raw, "big")
                chunks, got = [], 0
                while got < req_len:
                    c = conn.recv(min(65536, req_len - got))
                    if not c:
                        return
                    chunks.append(c)
                    got += len(c)

                data = json.loads(b"".join(chunks))
                seq  = data.get("seq", 0)

                # -- episode boundary -------------------------------------- #
                ep = data.get("episode")
                if ep is None:
                    if not warned_no_episode and args.mode != "bypass":
                        print("[SemCom] WARNING: client sent no 'episode' field. "
                              "History will NOT be reset between episodes, so the "
                              "predictor will condition on frames from the previous "
                              "scene. Update isaac_sim/vla_runner.py.", flush=True)
                        warned_no_episode = True
                elif ep != pipeline.episode:
                    pipeline.episode = ep
                    pipeline.reset()
                    prev_act = None
                    print(f"[SemCom] --- episode {ep}: history reset ---", flush=True)

                try:
                    obs_rgb = np.array(
                        Image.open(io.BytesIO(base64.b64decode(data["jpeg_b64"]))).convert("RGB"),
                        dtype=np.uint8,
                    )
                except Exception as e:
                    _send(conn, {"seq": seq, "action": [0.0] * 7,
                                 "status": "error", "message": str(e)})
                    continue

                instr = data.get("instruction", args.instruction)

                t0 = time.perf_counter()
                x_tilde, timings = pipeline.step(obs_rgb, prev_act)
                action = policy.predict(x_tilde, instr)
                prev_act = action
                ms = (time.perf_counter() - t0) * 1000

                for k, v in timings.items():
                    stage_ms[k] = stage_ms.get(k, 0.0) + v * 1000

                if n_reqs == 0:
                    print(f"[DEBUG] image {obs_rgb.shape} mean={obs_rgb.mean():.1f} "
                          f"-> x~ mean={np.asarray(x_tilde).mean():.1f}", flush=True)
                    print(f"[DEBUG] instruction: {instr!r}", flush=True)
                if n_reqs < 3:
                    print(f"[DEBUG] action = [{', '.join(f'{v:+.4f}' for v in action)}]",
                          flush=True)

                n_reqs   += 1
                total_ms += ms

                _send(conn, {"seq": seq, "action": action,
                             "exec_horizon": 1, "status": "ok"})

                if n_reqs % 20 == 0:
                    parts = "  ".join(f"{k}={v / n_reqs:.0f}"
                                      for k, v in sorted(stage_ms.items()))
                    print(f"[SemCom] {n_reqs} reqs  avg={total_ms / n_reqs:.1f} ms"
                          + (f"   [{parts}]" if parts else ""), flush=True)

    try:
        while True:
            conn, addr = sock.accept()
            handle(conn, addr)
    except KeyboardInterrupt:
        print(f"\n[SemCom] Done after {n_reqs} requests.")
    finally:
        sock.close()


def parse_args():
    p = argparse.ArgumentParser(
        description="Online closed-loop SemCom + policy inference server")
    p.add_argument("--mode", choices=["bypass", "vae", "nochannel", "full"],
                   default="bypass",
                   help="How much of the pipeline to exercise. Start with bypass "
                        "and confirm it reproduces the policy-only success rate.")

    # Policy — a separate process (different conda env); start it first.
    p.add_argument("--policy_host", type=str, default="127.0.0.1")
    p.add_argument("--policy_port", type=int, default=5557,
                   help="Where openvla_server.py --serve is listening.")
    p.add_argument("--instruction", type=str,
                   default="pick up the red cube and place it on the tray",
                   help="Fallback when the client sends none.")

    # SemCom checkpoints
    p.add_argument("--stage1_ckpt", type=str,
                   default="outputs/stage1_5cube_cam1_K8/stage1_best.pt",
                   help="Ctrl-World weights (action_encoder + unet_cross_attn).")
    p.add_argument("--stage2_ckpt", type=str,
                   default="outputs/stage2_twophase_full/phase2.pt",
                   help="JSCC + SideInfo + Refinement weights.")
    p.add_argument("--svd_path", type=str, default=None)

    # Channel / pipeline overrides
    p.add_argument("--snr_db",  type=float, default=None)
    p.add_argument("--channel", type=str, default=None,
                   choices=["awgn", "rayleigh", "cdl"])
    p.add_argument("--D_jscc",  type=int, default=None)
    p.add_argument("--num_history", type=int, default=None)
    p.add_argument("--ctrl_world_n_steps", type=int, default=None,
                   help="DDIM steps for Ctrl-World. Dominates per-step latency; "
                        "lower it if requests approach the client timeout.")
    p.add_argument("--noise_level_second", type=int, default=None,
                   help="SDEdit level t''. 0 = single forward pass (configured default).")

    # Network
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5558,
                   help="TCP port (5558 — avoids clashing with openvla_server on 5557)")
    return p.parse_args()


if __name__ == "__main__":
    run_server(parse_args())
