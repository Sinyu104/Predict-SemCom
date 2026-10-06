"""
semcom_server.py  —  Online closed-loop evaluation of the Wyner-Ziv semantic
communication pipeline, with a policy in the loop.

Same TCP/ROS front end as openvla_server.py, but the received observation is
pushed through the SemCom pipeline before it reaches the policy:

    o_t  ->  [VAE encode]                      ->  z_t
    m = 1[||z_t - z_{t-1}||^2 > thr] ; code = ConvEnc(z_t*m) ; route k channels
             [Fading channel over first k ch]  ->  s~_t
    [z~_{t-6..t-1}], a_t -> [Ctrl-World DDIM]   ->  z_hat_t   (prior / side information)
    dropped channels <- prior mean mu_p(z_hat_t) ; rec = ConvDec(s~_t)
    g = m * 1[||rec - z_hat_t||^2 > recv_thr] ; z~_t = g*rec + (1-g)*z_hat_t
                            [VAE decode]       ->  x~_t
                            [Policy]           ->  a_hat_t  -> back to the robot

    This is the CURRENT conv Wyner-Ziv codec (experiments/stage2_twophase/conv_codec.py,
    ConvWZCodec) + a receiver-side disagreement composite.  There is NO refinement
    diffusion.  A legacy SemComSystem (refinement) checkpoint is still accepted as a
    fallback, auto-detected from the checkpoint's cfg["arch"].

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
    predict     z_hat decoded straight to the policy.  Stage 1 ALONE, no Stage 2
                loaded at all.  Pair with --clean_history for the predictor's
                ceiling with drift removed.
    nochannel   conv codec with k=0 channels transmitted — every code channel is
                filled from the prior mu_p(z_hat), so the composite falls back to
                z_hat wherever it agrees.  The control for "did the channel help?".
    full        the complete pipeline (adaptive k routed by the change-cell count).

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
# The current (conv Wyner-Ziv) codec lives in the Stage-2 experiment package.
_S2 = os.path.join(REPO, "experiments", "stage2_twophase")
if _S2 not in sys.path:
    sys.path.insert(0, _S2)

from config import CONFIG                      # noqa: E402
from models import SemComSystem, FadingChannel # noqa: E402


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

    def __init__(self, config, vae, ctrl_world, system, device, mode,
                 codec=None, wz_channel=None, wz_params=None, sscc_cfg=None):
        self.cfg        = config
        self.vae        = vae
        self.ctrl_world = ctrl_world
        self.system     = system          # legacy SemComSystem (None when conv codec is used)
        self.device     = device
        self.mode       = mode

        # -- current conv Wyner-Ziv codec (ConvWZCodec) + receiver composite ---- #
        # When `codec` is set, the `full`/`nochannel` path uses it instead of the
        # legacy refinement diffusion.  Routing params come from the checkpoint cfg
        # (thr, recv_thr) and the CLI (route_thr, k_calm, k_dist).
        self.sscc_cfg   = sscc_cfg     # traditional streaming SSCC baseline (H.264/H.265 + LDPC + QAM)
        self.sscc_link  = None
        self.sscc_stats = {"frames": 0, "symbols": 0, "lost": 0, "outage": 0, "tx": 0}
        if sscc_cfg is not None:
            from baselines.traditional.nr_link import NRLink
            tb = int(round(sscc_cfg["kbps"] * 1000 / sscc_cfg["fps"] / 8))
            self.sscc_link = NRLink(sscc_cfg["channel"], sscc_cfg["snr_db"], tb_bytes=tb,
                                    re_budget=sscc_cfg["re_budget"], max_tx=sscc_cfg["max_tx"],
                                    bler_target=sscc_cfg["bler"], cdl_model=sscc_cfg["cdl_model"])
            print(f"[SemCom] SSCC NR link: {sscc_cfg['channel']} @ {sscc_cfg['snr_db']} dB -> MCS {self.sscc_link.mcs[0]} "
                  f"(Qm={self.sscc_link.mcs[1]}, R={self.sscc_link.mcs[2]:.2f}); TB {tb} B, budget "
                  f"{sscc_cfg['re_budget']} RE/frame, HARQ<= {sscc_cfg['max_tx']} tx", flush=True)
        self.codec      = codec
        self.wz_channel = wz_channel
        self.wz_thr = self.wz_recv_thr = self.route_thr = self.k_calm = self.k_dist = None
        if wz_params is not None:
            self.wz_thr      = wz_params["thr"]
            self.wz_recv_thr = wz_params["recv_thr"]
            self.route_thr   = wz_params["route_thr"]
            self.k_calm      = wz_params["k_calm"]
            self.k_dist      = wz_params["k_dist"]

        self.num_history = config["num_history"]
        self.num_pred    = config["num_pred"]
        self.action_dim  = config["action_dim"]
        self.n_ddim      = config.get("ctrl_world_n_steps", 10)
        self.t_second    = config.get("noise_level_second", 0)

        # Strided history + keyframe re-anchoring, matching experiments/rollout_collect.py.
        # Consecutive history was measured at 0.2759 latent error (copy-last-frame is
        # 0.287, i.e. no better than doing nothing); stride 8 drops it to 0.1639 with
        # no retraining, and N=16 anchors take it to 0.0838.
        self.stride   = int(config.get("history_stride", 1))
        self.reanchor = int(config.get("reanchor", 0))
        # Feed the TRUE latent into the history instead of the reconstruction.
        # Not deployable — the receiver never holds z_t for a non-anchor frame —
        # but it removes closed-loop drift entirely, so whatever damage survives
        # belongs to the channel and the refinement rather than to accumulation.
        self.clean_history = bool(config.get("clean_history", False))
        # Frames the strided window reaches back over.
        self.span = (self.num_history - 1) * self.stride + 1

        self.episode = None
        self.reset()

    def reset(self):
        """Clear per-episode state.  Called on every episode boundary."""
        # Buffers must span the whole strided window, not just num_history frames.
        self.z_hist = deque(maxlen=self.span)
        # One longer than the latent buffer: the newest action a_t belongs to the
        # future slots, so a full strided history still needs `span` entries behind it.
        self.a_hist = deque(maxlen=self.span + 1)
        # Actions are zero-padded until the robot has actually moved.
        for _ in range(self.span + 1):
            self.a_hist.append(np.zeros(self.action_dim, dtype=np.float32))
        self.frame_idx = 0        # index within the episode
        self.chunk     = None     # cached (1, num_pred, C, H, W) prediction
        self.chunk_pos = 0        # which frame of the cached chunk comes next
        self.last_zhat = None     # last Ctrl-World prediction latent (for the 3-panel dump)
        self._sscc_reset()
        self.z_prev_true = None   # previous TRUE latent z_{t-1} (transmitter, WZ-clean) for the change mask

    def _sscc_reset(self):
        """New episode -> fresh streaming encoder/decoder (first frame is a keyframe)."""
        c = getattr(self, "sscc_cfg", None)
        if c is None:
            return
        from baselines.traditional.stream_codec import StreamEncoder, StreamDecoder
        self.sscc_enc = StreamEncoder(c["codec"], gop=c["gop"], fps=c["fps"], kbps=c["kbps"])
        self.sscc_dec = StreamDecoder(c["codec"], conceal="drift")

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
        frames = list(self.z_hist)          # each (C, H, W), oldest first
        idx = self._hidx(len(frames))
        out = torch.stack([frames[j] for j in idx], dim=0).unsqueeze(0)
        assert out.ndim == 5 and out.shape[1] == self.num_history, \
            f"history tensor must be (1, {self.num_history}, C, H, W), got {tuple(out.shape)}"
        return out

    def _hidx(self, n):
        """Strided history indices into a buffer of length n, oldest first.

        Identical to rollout_collect.py / the offline rollouts: slots sit at
        n-1, n-1-stride, n-1-2*stride, ...  Clamped at 0, so early in an episode
        the oldest slots repeat the first frame rather than reaching off the end.
        """
        return [max(0, n - 1 - i * self.stride) for i in range(self.num_history)][::-1]

    def _action_tensor(self) -> torch.Tensor:
        """(1, num_history + num_pred, action_dim), oldest first.

        History slots use the SAME strided indices as the latents.  The num_pred
        future slots are the actions the world model is being asked to predict
        under — offline those come from the recorded demo, but online the policy
        has not produced them yet, so they repeat the most recent action.  With
        num_pred=1 there is no approximation: the single future slot is a_t, the
        action the robot just executed.  For num_pred>1 this is an approximation
        that only a chunking policy could remove.
        """
        a = np.stack(list(self.a_hist), axis=0)
        # a[-1] is a_t, the action that produced the frame being predicted, so the
        # history slots must come from a[:-1] (ending at a_{t-1}) and the future
        # slots start at a_t.  Offline this is cat([act[hidx], act[t:t+K]]), where
        # hidx tops out at t-1 — matching that split exactly matters, because an
        # off-by-one here shifts every history slot by one control step.
        past = a[:-1]
        idx = self._hidx(len(past))
        hist = past[idx]                                     # (num_history, action_dim)
        fut = np.repeat(a[-1][None, :], self.num_pred, axis=0)
        out = np.concatenate([hist, fut], axis=0)
        return torch.from_numpy(out).unsqueeze(0).to(self.device)

    # -- current conv Wyner-Ziv reconstruction --------------------------- #

    @torch.no_grad()
    def _wz_reconstruct(self, z_t, z_hat, z_prev):
        """Conv WZ codec + receiver disagreement composite (ConvWZCodec).

        Transmitter (WZ-blind to z_hat): change mask m from z_t vs the true z_{t-1},
        encode z_t*m, route k channels by the change-cell count, send the first k
        through the channel.  Receiver: fill the dropped channels with the prior mean
        mu_p(z_hat) (WZ-legal), decode, then composite z~ = g*rec + (1-g)*z_hat with
        g = m * 1[||rec - z_hat||^2 > recv_thr].  Mirrors decode_arms_alloc.py.
        """
        # change mask (WZ-clean).  No true previous latent yet -> send everything.
        if z_prev is None:
            m = torch.ones_like(z_t[:, :1])
        else:
            m = (((z_t - z_prev) ** 2).mean(1, keepdim=True) > self.wz_thr).float()

        if self.mode == "nochannel":
            k = 0                                               # transmit nothing -> all from prior
        else:
            k = self.k_dist if m.sum().item() > self.route_thr else self.k_calm

        mu_e, _ = self.codec.encode(z_t * m)                    # (1, Dch, H, W)
        mu_p, _ = self.codec.prior(z_hat)                       # prior mean from the prediction
        B, Dc, Hh, Ww = mu_e.shape
        s_t  = self.wz_channel(mu_e.flatten(1)).view(B, Dc, Hh, Ww)
        keep = (torch.arange(Dc, device=self.device) < k).float()[None, :, None, None]
        s_mix = keep * s_t + (1.0 - keep) * mu_p               # dropped channels <- prior mean
        rec  = self.codec.decode(s_mix)
        g = m * (((rec - z_hat) ** 2).mean(1, keepdim=True) > self.wz_recv_thr).float()
        return g * rec + (1.0 - g) * z_hat

    # -- main entry ------------------------------------------------------- #

    @torch.no_grad()
    def step(self, obs_rgb: np.ndarray, prev_action):
        t = {}
        if self.mode == "bypass":
            return obs_rgb, t

        if self.mode == "sscc":
            t0 = time.perf_counter()
            pkt = self.sscc_enc.encode(obs_rgb)
            rx, info = self.sscc_link.transmit(pkt)
            st = self.sscc_stats
            st["frames"] += 1; st["symbols"] += info["n_symbols"]; st["tx"] += info["n_tx"]
            st["lost"] += int(not info["ok"]); st["outage"] += int(info["outage"])
            t["sscc"] = time.perf_counter() - t0
            if rx is None:                       # frame lost: the edge has NO image, so no action can be computed
                self.sscc_dec.decode(None)       # keep decoder state consistent (references now stale)
                return None, t
            return self.sscc_dec.decode(rx), t

        self.a_hist.append(
            np.asarray(prev_action, dtype=np.float32)
            if prev_action is not None
            else np.zeros(self.action_dim, dtype=np.float32)
        )

        t0 = time.perf_counter()
        x     = self._to_tensor(obs_rgb)
        z_t   = self.vae.encode(x)                       # (1, C, H, W)
        t["vae_enc"] = time.perf_counter() - t0

        # Previous TRUE latent for the WZ-clean change mask (the transmitter holds its
        # own past observations).  Captured before overwriting, so this frame sees
        # z_{t-1} and the next frame will see this z_t.  Updated in EVERY mode so it is
        # always current, even across anchor / vae / predict early returns.
        z_prev = self.z_prev_true
        self.z_prev_true = z_t

        if self.mode == "vae_channel":
            # Baseline: send the RAW VAE latent (4*28*28 reals, no prediction / no WZ) over the channel.
            t0 = time.perf_counter()
            B, C, H, W = z_t.shape
            z_rx = self.wz_channel(z_t.flatten(1)).view(B, C, H, W)
            x_tilde = self.vae.decode(z_rx)
            t["vae_dec"] = time.perf_counter() - t0
            self.z_hist.append(z_t[0])
            return self._to_uint8(x_tilde), t

        if self.mode == "vae":
            t0 = time.perf_counter()
            x_tilde = self.vae.decode(z_t)
            t["vae_dec"] = time.perf_counter() - t0
            self.z_hist.append(z_t[0])
            return self._to_uint8(x_tilde), t

        # -- keyframe / warm-up anchoring ---------------------------------- #
        # The offline rollouts seed the loop with `span` ground-truth latents and
        # then re-anchor every `reanchor` frames.  An anchor is a frame the robot
        # sends uncoded, so the receiver holds the true latent; error at an anchor
        # is 0.0000 and the loop cannot accumulate past it.
        is_anchor = (
            self.frame_idx < self.span                                  # warm-up seed
            or (self.reanchor > 0 and self.frame_idx % self.reanchor == 0)
        )
        if is_anchor:
            x_tilde = self.vae.decode(z_t)
            self.z_hist.append(z_t[0])
            self.frame_idx += 1
            self.chunk = None                     # history jumped; drop stale chunk
            self.last_zhat = None                 # anchor: no prediction this frame
            t["anchor"] = True
            return self._to_uint8(x_tilde), t

        # -- Wyner-Ziv side information (receiver-only, no access to s~_t) -- #
        t0 = time.perf_counter()
        if len(self.z_hist) == 0:
            # First frame of an episode: nothing to condition on.  Fall back to
            # the observed latent so the refinement still has a sane prior.
            z_hat = z_t
        else:
            # One Ctrl-World call per num_pred frames, matching the offline loop.
            if self.chunk is None or self.chunk_pos >= self.num_pred:
                self.chunk = self.ctrl_world.predict_next_latent(
                    self._history_tensor(), self._action_tensor(), n_steps=self.n_ddim,
                )                                        # (1, num_pred, C, H, W)
                self.chunk_pos = 0
            # Ctrl-World runs in fp16; the refinement network is fp32.
            z_hat = self.chunk[:, self.chunk_pos].float()
            self.chunk_pos += 1
        self.last_zhat = z_hat        # stash the prediction (latent only, no decode cost)
        t["ctrl_world"] = time.perf_counter() - t0

        # -- raw prediction, no Stage 2 at all ----------------------------- #
        # `nochannel` is NOT this: it zeroes s~ but still runs the refinement, and
        # the refinement is residual (`z_hat + delta`), so it emits z_hat + delta(0).
        # Nothing else in this file renders z_hat itself, which is what separates a
        # Stage-1 failure from a Stage-2 one.  Pair with --clean_history to drop
        # drift as well and see the predictor's ceiling.
        if self.mode == "predict":
            x_tilde = self.vae.decode(z_hat)
            self.z_hist.append(z_t[0] if self.clean_history else z_hat[0])
            self.frame_idx += 1
            t["vae_dec"] = time.perf_counter() - t0
            return self._to_uint8(x_tilde), t

        # -- transmitter + receiver -------------------------------------- #
        t0 = time.perf_counter()
        if self.codec is not None:
            # CURRENT conv Wyner-Ziv path (mirrors decode_arms_alloc.py conv_wz).
            z_tilde = self._wz_reconstruct(z_t, z_hat, z_prev)
            t["wz"] = time.perf_counter() - t0
        else:
            # LEGACY refinement-diffusion path (SemComSystem checkpoint).
            _, _, s_t = self.system.jscc_encoder(z_t, sample=False)
            s_tilde   = self.system.channel(s_t)
            if self.mode == "nochannel":
                s_tilde = torch.zeros_like(s_tilde)
            t["jscc"] = time.perf_counter() - t0
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
        # The policy always sees the reconstruction; only what goes BACK into the
        # history changes, so --clean_history isolates the refinement from drift.
        self.z_hist.append(z_t[0] if self.clean_history else z_tilde[0])
        self.frame_idx += 1
        return self._to_uint8(x_tilde), t


# ---------------------------------------------------------------------------
#  Model construction
# ---------------------------------------------------------------------------

def build_pipeline(args, config, device):
    if args.mode == "bypass":
        print("[SemCom] mode=bypass — no SemCom models loaded (policy only).")
        return SemComPipeline(config, None, None, None, device, "bypass")

    if args.mode == "sscc":
        cfg = dict(codec=args.sscc_codec, gop=args.sscc_gop, fps=args.sscc_fps, kbps=args.sscc_kbps,
                   channel=config.get("channel_type", "awgn"), snr_db=config["snr_db"],
                   re_budget=args.sscc_re_budget, max_tx=args.sscc_max_tx, bler=args.sscc_bler,
                   cdl_model=args.cdl_model)
        print(f"[SemCom] mode=sscc: {cfg['codec']} CBR {cfg['kbps']} kbps @ {cfg['fps']} fps, GOP {cfg['gop']}, NR link")
        return SemComPipeline(config, None, None, None, device, "sscc", sscc_cfg=cfg)

    from vae_wrapper import VAEWrapper
    print(f"[SemCom] VAE: {config.get('vae_model_name')}")
    vae = VAEWrapper(config.get("vae_model_name", "stabilityai/sd-vae-ft-mse")).to(device)

    ctrl_world = None
    system     = None
    codec      = None
    wz_channel = None
    wz_params  = None
    cdl_kw = {"model": args.cdl_model, "delay_spread": args.cdl_delay_spread,
              "carrier_frequency": args.cdl_carrier_freq, "max_speed": args.cdl_max_speed}
    if args.mode == "vae_channel":
        wz_channel = FadingChannel(config["snr_db"], channel_type=config.get("channel_type", "awgn"),
                                   cdl_kwargs=cdl_kw)
        print(f"[SemCom] vae_channel: raw VAE latent over {config.get('channel_type','awgn')} "
              f"@ {config['snr_db']} dB" + (f" (CDL-{args.cdl_model})" if config.get('channel_type') == 'cdl' else ""))

    if args.mode in ("predict", "nochannel", "full"):
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

        # `predict` renders z_hat and never touches Stage 2, so it must not require
        # a Stage-2 checkpoint — the whole point is to score Stage 1 on its own.
        if args.mode != "predict":
            sk   = torch.load(args.stage2_ckpt, map_location="cpu")
            cfg2 = sk.get("cfg", {})
            arch = cfg2.get("arch", "legacy")
            if arch == "conv_wz":
                # CURRENT model: conv Wyner-Ziv codec (no refinement diffusion).
                from conv_codec import ConvWZCodec
                codec = ConvWZCodec(4, cfg2["conv_w"], cfg2["conv_dch"], cfg2["conv_nb"]).to(device)
                codec.load_state_dict(sk["net"])
                codec.eval()
                wz_channel = FadingChannel(config["snr_db"],
                                           channel_type=config.get("channel_type", "awgn"),
                                           cdl_kwargs=cdl_kw)
                wz_params = dict(thr=cfg2["thr"], recv_thr=cfg2["recv_thr"],
                                 route_thr=args.route_thr, k_calm=args.k_calm, k_dist=args.k_dist)
                print(f"[SemCom] conv_wz codec from {args.stage2_ckpt}: "
                      f"W{cfg2['conv_w']}/Dch{cfg2['conv_dch']}/nb{cfg2['conv_nb']} "
                      f"beta={cfg2.get('beta')} thr={cfg2['thr']} recv_thr={cfg2['recv_thr']} "
                      f"route: k={args.k_calm} if changed<= {args.route_thr} else k={args.k_dist}")
            elif arch in ("conv", "conv_nested"):
                raise NotImplementedError(
                    f"stage2_ckpt arch='{arch}' is not wired into the server yet; "
                    f"use a conv_wz checkpoint (train_masked_alloc_wz.py) or the legacy path.")
            else:
                # LEGACY: SemComSystem (refinement diffusion) checkpoint.
                system = SemComSystem(config).to(device)
                state = sk.get("system_state", sk)
                missing, unexpected = system.load_state_dict(state, strict=False)
                print(f"[SemCom] LEGACY Stage-2 restored from {args.stage2_ckpt} "
                      f"({len(state)} tensors, {len(missing)} missing, {len(unexpected)} unexpected)")
                system.eval()
        else:
            print("[SemCom] mode=predict — Stage 2 not loaded; the policy sees z_hat.")

    # Warm the channel up NOW: the CDL channel builds its Sionna model on the first call (~8 s),
    # which would otherwise land inside request 1 and time the client out.
    if wz_channel is not None:
        t0 = time.perf_counter()
        wz_channel(torch.randn(1, 4 * 28 * 28, device=device))
        print(f"[SemCom] channel warm-up {time.perf_counter() - t0:.1f}s", flush=True)

    return SemComPipeline(config, vae, ctrl_world, system, device, args.mode,
                          codec=codec, wz_channel=wz_channel, wz_params=wz_params)


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

    def predict(self, img_rgb: np.ndarray, instruction: str,
                passthrough_b64: str | None = None) -> list:
        """
        passthrough_b64 : the client's original JPEG, forwarded byte-for-byte.

        Only valid when the pipeline did NOT modify the image (bypass mode).
        Re-encoding an unchanged frame would put it through JPEG a second time
        — measured at 0.73/255 mean abs change — which would make the bypass
        control merely close to the old two-process path instead of identical
        to it.
        """
        if passthrough_b64 is not None:
            jpeg_b64 = passthrough_b64
        else:
            buf = io.BytesIO()
            Image.fromarray(img_rgb).save(buf, format="JPEG", quality=self.jpeg_quality)
            jpeg_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        self._seq += 1
        payload = json.dumps({
            "seq":         self._seq,
            "jpeg_b64":    jpeg_b64,
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
    # K>1 needs future actions the receiver does not have: to predict z_t..z_{t+K-1}
    # the world model is conditioned on a_t..a_{t+K-1}, but the policy emits one
    # action at a time from images this loop has not rendered yet.  The offline
    # rollouts read those actions from the recorded demo, which deployment cannot.
    # K=1 is therefore the deployable default; K>1 is allowed for comparison
    # against the offline numbers, with the future slots repeating the last action.
    CONFIG["num_pred"]       = args.num_pred
    CONFIG["clip_length"]    = CONFIG["num_history"] + CONFIG["num_pred"]
    CONFIG["history_stride"] = args.history_stride
    CONFIG["reanchor"]       = args.reanchor
    CONFIG["clean_history"]  = args.clean_history
    CONFIG["refine_disagreement_gate"] = args.disagree_gate
    if args.clean_history:
        print("[SemCom] clean_history=ON — Ctrl-World sees TRUE latents. DIAGNOSTIC ONLY, "
              "not a deployable configuration; numbers from this run are an upper bound.")

    print(f"[SemCom] mode={args.mode}  snr={CONFIG['snr_db']}dB  "
          f"channel={CONFIG.get('channel_type')}  D_jscc={CONFIG['D_jscc']}")
    print(f"[SemCom] num_history={CONFIG['num_history']}  num_pred={CONFIG['num_pred']}  "
          f"stride={args.history_stride}  reanchor={args.reanchor}  "
          f"ddim_steps={CONFIG.get('ctrl_world_n_steps')}  "
          f"t''={CONFIG.get('noise_level_second')}")
    if args.num_pred > 1:
        print(f"[SemCom] WARNING: num_pred={args.num_pred} > 1 — the {args.num_pred} future "
              f"action slots repeat the last action, since the policy has not produced "
              f"them yet.  Offline rollouts used the demo's true future actions, so this "
              f"is NOT the same conditioning those numbers were measured under.")
    if args.reanchor > 0:
        span = (CONFIG["num_history"] - 1) * args.history_stride + 1
        print(f"[SemCom] anchors: first {span} frames seeded uncoded, then every "
              f"{args.reanchor} frames.  Anchor frames bypass the channel entirely.")

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
                    # Warn in EVERY mode, including bypass. Bypass keeps no
                    # history so it is harmless here — but bypass is precisely
                    # the run used to validate the protocol, so staying quiet
                    # would let a missing field reach `full` undetected.
                    if not warned_no_episode:
                        harmless = " (harmless in bypass, FATAL in other modes)" \
                                   if args.mode == "bypass" else ""
                        print(f"[SemCom] WARNING: client sent no 'episode' field"
                              f"{harmless}. History will NOT be reset between "
                              f"episodes, so the predictor would condition on "
                              f"frames from the previous scene. The Windows "
                              f"checkout needs commit 196730e "
                              f"(VLAClient.new_episode()).", flush=True)
                        warned_no_episode = True
                elif ep != pipeline.episode:
                    pipeline.episode = ep
                    pipeline.reset()
                    prev_act = None
                    print(f"[SemCom] --- episode {ep}: history reset ---", flush=True)

                # -- disturbance marker ------------------------------------ #
                # Windows tags the frames where it injected a pose/action
                # disturbance. Read it exactly like seq/episode. It changes no
                # data flow — it only labels the frame, so we can (a) log the
                # moment and (b) tag its dump for later analysis of how the
                # channel/refinement behaved on the frames that actually need it.
                disturbed = bool(data.get("disturbed", False))
                if disturbed:
                    print(f"[SemCom] disturbance @ ep{ep} seq{seq}", flush=True)

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
                if x_tilde is None:              # SSCC frame lost -> reply with an error; the client applies zero action
                    _send(conn, {"seq": seq, "action": [0.0] * 7, "status": "error", "message": "frame lost on the channel"})
                    prev_act = None; n_reqs += 1
                    if n_reqs % 20 == 0:
                        st = pipeline.sscc_stats
                        print(f"[SemCom] SSCC: {st['symbols'] / max(st['frames'], 1):.0f} REs/frame, "
                              f"{100 * st['lost'] / max(st['frames'], 1):.1f}% lost "
                              f"({100 * st['outage'] / max(st['frames'], 1):.1f}% outage), "
                              f"{st['tx'] / max(st['frames'] - st['outage'], 1):.2f} tx/frame", flush=True)
                    continue
                # In bypass the frame is untouched, so forward the client's own
                # bytes and keep the control bit-exact with the 2-process path.
                action = policy.predict(
                    x_tilde, instr,
                    passthrough_b64=data["jpeg_b64"] if args.mode == "bypass" else None,
                )
                prev_act = action
                ms = (time.perf_counter() - t0) * 1000

                if args.dump_dir and n_reqs % args.dump_every == 0:
                    try:
                        os.makedirs(args.dump_dir, exist_ok=True)
                        obs = np.asarray(obs_rgb)
                        rec = np.asarray(x_tilde)
                        if rec.shape != obs.shape:      # bypass forwards the original
                            rec = obs
                        # 3-panel [received | predicted z_hat | reconstructed] when a
                        # prediction exists (full/nochannel modes, non-anchor frames).
                        # At a disturbance z_hat shows the pre-event scene and rec should
                        # show the channel's correction, so the three side by side are the
                        # direct read on whether the Wyner-Ziv channel did its job.
                        panels = [obs]
                        with torch.no_grad():
                            if getattr(pipeline, "last_zhat", None) is not None:
                                pred = pipeline._to_uint8(pipeline.vae.decode(pipeline.last_zhat))
                                if pred.shape == obs.shape:
                                    panels.append(pred)
                        panels.append(rec)
                        pair = np.concatenate(panels, axis=1)
                        # "_D" marks a disturbed frame, so the frames that stress
                        # the channel are findable with a glob after the run.
                        tag = "_D" if disturbed else ""
                        Image.fromarray(pair.astype(np.uint8)).save(
                            os.path.join(args.dump_dir,
                                         f"{str(ep) if ep is not None else 'noep'}"
                                         f"_{n_reqs:05d}{tag}.png"))
                    except Exception as e:                        # never kill the loop
                        print(f"[SemCom] dump failed: {e}", flush=True)

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

                if args.mode == "sscc" and n_reqs % 20 == 0:
                    st = pipeline.sscc_stats
                    print(f"[SemCom] SSCC: {st['symbols'] / max(st['frames'], 1):.0f} REs/frame, "
                          f"{100 * st['lost'] / max(st['frames'], 1):.1f}% lost "
                          f"({100 * st['outage'] / max(st['frames'], 1):.1f}% outage), "
                          f"{st['tx'] / max(st['frames'] - st['outage'], 1):.2f} tx/frame", flush=True)

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
    p.add_argument("--mode",
                   choices=["bypass", "vae", "vae_channel", "sscc", "predict", "nochannel", "full"],
                   default="bypass",
                   help="How much of the pipeline to exercise. Start with bypass "
                        "and confirm it reproduces the policy-only success rate. "
                        "predict renders z_hat with NO Stage 2 — the only mode that "
                        "scores Stage 1 alone (nochannel still runs the refinement).")

    # Policy — a separate process (different conda env); start it first.
    p.add_argument("--policy_host", type=str, default="127.0.0.1")
    p.add_argument("--policy_port", type=int, default=5557,
                   help="Where openvla_server.py --serve is listening.")
    p.add_argument("--instruction", type=str,
                   default="pick up the red cube and place it on the tray",
                   help="Fallback when the client sends none.")
    p.add_argument("--clean_history", action="store_true",
                   help="Condition Ctrl-World on TRUE latents instead of reconstructions. "
                        "NOT deployable (the receiver never holds z_t for a coded frame) — "
                        "a diagnostic that removes closed-loop drift, so what remains is "
                        "the channel and refinement alone. Pair with --reanchor 0 so every "
                        "frame actually goes through the channel.")
    p.add_argument("--dump_dir", type=str, default=None,
                   help="Save observation|reconstruction pairs here as PNGs. The policy "
                        "acts on the RIGHT half; if the cube is the wrong colour or absent "
                        "there, the policy never had the information. Off by default.")
    p.add_argument("--dump_every", type=int, default=10,
                   help="Save one pair every N frames when --dump_dir is set.")

    # SemCom checkpoints
    p.add_argument("--stage1_ckpt", type=str,
                   default="outputs/stage1_rollout_ft/stage1_rollout_ep1.pt",
                   help="Ctrl-World weights (action_encoder + unet_cross_attn). Default pairs with "
                        "the conv_wz codec (both use CLEAN history at stride 1; keep --history_stride 1). "
                        "Mismatching the predictor lineage silently breaks the reconstruction.")
    p.add_argument("--stage2_ckpt", type=str,
                   default="outputs/masked_alloc_wz_b1e-4/masked_alloc.pt",
                   help="Stage-2 weights. Auto-detected from cfg['arch']: 'conv_wz' -> current "
                        "ConvWZCodec (conv enc + prior + dec, no refinement); otherwise the legacy "
                        "SemComSystem (refinement diffusion).")
    p.add_argument("--svd_path", type=str, default=None)

    # conv Wyner-Ziv adaptive-rate router (conv_wz checkpoints only; thr/recv_thr come from the ckpt)
    p.add_argument("--route_thr", type=float, default=110.0,
                   help="conv_wz: send k_dist code channels if the changed-cell count exceeds this, "
                        "else k_calm. WZ-clean (uses only z_t, z_{t-1}).")
    p.add_argument("--k_calm", type=int, default=2,
                   help="conv_wz: code channels transmitted on calm frames (of Dch).")
    p.add_argument("--k_dist", type=int, default=4,
                   help="conv_wz: code channels transmitted on disturbed frames (of Dch).")

    # Channel / pipeline overrides
    p.add_argument("--snr_db",  type=float, default=None)
    p.add_argument("--channel", type=str, default=None,
                   choices=["awgn", "rayleigh", "cdl"])
    p.add_argument("--D_jscc",  type=int, default=None)
    p.add_argument("--sscc_codec", default="h264", choices=["h264", "h265"])
    p.add_argument("--sscc_kbps", type=float, default=100.0, help="CBR video bitrate (typical low-res streaming)")
    p.add_argument("--sscc_fps", type=int, default=10, help="control / frame rate")
    p.add_argument("--sscc_gop", type=int, default=10, help="keyframe interval in frames")
    p.add_argument("--sscc_re_budget", type=int, default=8000, help="NR resource elements allocated per frame period")
    p.add_argument("--sscc_max_tx", type=int, default=4, help="HARQ: max transmissions per frame (within the budget)")
    p.add_argument("--sscc_bler", type=float, default=0.1, help="TS 38.214 CQI rule: BLER target for MCS selection")
    p.add_argument("--num_history", type=int, default=None)
    p.add_argument("--history_stride", type=int, default=1,
                   help="m: spacing between history slots. Offline, stride 8 cut closed-loop "
                        "latent error from 0.2759 to 0.1639 with no retraining (copy-last-frame "
                        "is 0.287). Must match the stride the Stage-1 checkpoint was fine-tuned "
                        "on — stride_k8 was trained at 8.")
    p.add_argument("--reanchor", type=int, default=0,
                   help="N: send an uncoded frame every N steps, so the receiver holds the true "
                        "latent and drift cannot accumulate past it. 0 = off. Offline, N=16 took "
                        "latent error from 0.1639 to 0.0838. Costs one uncoded frame per N.")
    p.add_argument("--num_pred", type=int, default=1,
                   help="K: frames per Ctrl-World call. 1 is the deployable default. K>1 needs "
                        "future actions the receiver does not have (see run_server) and is only "
                        "meaningful for comparison against the offline rollouts.")
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
