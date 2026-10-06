"""
Causal, frame-by-frame H.264 / H.265 for closed-loop control.

video_codec.py encodes whole clips offline.  A robot control loop cannot do that:
frame t must be encoded, sent, decoded and acted on before frame t+1 exists.  This
module keeps encoder and decoder STATE across frames, with low-delay settings
(no B-frames, no lookahead, one packet in / one frame out):

    enc = StreamEncoder("h264", qp=28, gop=10)
    pkt = enc.encode(frame_rgb)              # bytes for THIS frame (I at gop boundaries, P otherwise)
    dec = StreamDecoder("h264")
    img = dec.decode(pkt)                    # RGB frame, or None if pkt is None (lost)

Loss handling (receiver knows a frame is missing, e.g. from a CRC failure):
    conceal="freeze" : repeat the last good frame and ignore P-frames until the next
                       keyframe (the decoder's references are stale).  Default; what a
                       packet-loss-aware receiver does.
    conceal="drift"  : keep decoding P-frames against the stale reference (smearing).
"""
from __future__ import annotations

import numpy as np
import av

try:
    av.logging.set_level(av.logging.PANIC)
except Exception:
    pass

_ENC = {"h264": "libx264", "avc": "libx264", "h265": "libx265", "hevc": "libx265"}
_DEC = {"h264": "h264", "avc": "h264", "h265": "hevc", "hevc": "hevc"}


def _is_keyframe(codec: str, data: bytes) -> bool:
    """True if this Annex-B access unit contains an IDR (h264) / IRAP (hevc) NAL."""
    b = data
    i = 0
    n = len(b)
    while i + 3 < n:
        if b[i] == 0 and b[i + 1] == 0 and (b[i + 2] == 1 or (b[i + 2] == 0 and i + 3 < n and b[i + 3] == 1)):
            j = i + (3 if b[i + 2] == 1 else 4)
            if j >= n:
                break
            if _DEC[codec] == "h264":
                if (b[j] & 0x1F) == 5:
                    return True
            else:
                t = (b[j] >> 1) & 0x3F
                if 16 <= t <= 21:
                    return True
            i = j
        else:
            i += 1
    return False


class StreamEncoder:
    def __init__(self, codec: str = "h264", qp: int = 28, gop: int = 10,
                 width: int = 224, height: int = 224, fps: int = 10,
                 kbps: float | None = None, vbv_frames: float = 1.0):
        """kbps given => constant-bitrate rate control (typical for streaming); the encoder picks QP per frame.
        vbv_frames = rate-buffer size in frame periods (1.0 => every frame, keyframes included, is ~kbps/fps).
        kbps None => fixed QP."""
        self.codec = codec.lower()
        self.width, self.height = width, height
        enc = _ENC[self.codec]
        ctx = av.CodecContext.create(enc, "w")
        ctx.width, ctx.height = width, height
        ctx.pix_fmt = "yuv420p"
        ctx.framerate = fps
        ctx.time_base = __import__("fractions").Fraction(1, fps)
        ctx.gop_size = int(gop)
        ctx.max_b_frames = 0
        if kbps is not None:
            br = int(kbps * 1000); buf = int(br / fps * vbv_frames)
            if enc == "libx264":
                ctx.options = {"b": str(br), "maxrate": str(br), "bufsize": str(buf), "tune": "zerolatency",
                               "bf": "0", "nal-hrd": "cbr", "sc_threshold": "0",
                               "g": str(int(gop)), "keyint_min": str(int(gop))}
            else:
                ctx.options = {"x265-params":
                               f"bitrate={int(kbps)}:vbv-maxrate={int(kbps)}:vbv-bufsize={max(buf // 1000, 1)}:"
                               f"bframes=0:rc-lookahead=0:frame-threads=1:keyint={int(gop)}:min-keyint={int(gop)}:"
                               f"scenecut=0:repeat-headers=1:log-level=none"}
        elif enc == "libx264":
            ctx.options = {"qp": str(int(qp)), "tune": "zerolatency", "bf": "0",
                           "sc_threshold": "0", "g": str(int(gop)), "keyint_min": str(int(gop))}
        else:
            ctx.options = {"x265-params":
                           f"qp={int(qp)}:bframes=0:rc-lookahead=0:frame-threads=1:lookahead-slices=0:"
                           f"keyint={int(gop)}:min-keyint={int(gop)}:scenecut=0:repeat-headers=1:"
                           f"no-open-gop=1:log-level=none"}
        ctx.open()
        self.ctx = ctx
        self._pts = 0

    def encode(self, frame_rgb: np.ndarray) -> bytes:
        f = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_rgb, dtype=np.uint8), format="rgb24")
        f = f.reformat(format="yuv420p")
        f.pts = self._pts
        self._pts += 1
        out = b"".join(bytes(p) for p in self.ctx.encode(f))
        return out


class StreamDecoder:
    def __init__(self, codec: str = "h264", width: int = 224, height: int = 224,
                 conceal: str = "drift"):
        self.codec = codec.lower()
        self.width, self.height = width, height
        self.conceal = conceal
        ctx = av.CodecContext.create(_DEC[self.codec], "r")
        ctx.thread_count = 1
        ctx.options = {"flags": "low_delay"}
        ctx.open()
        self.ctx = ctx
        self.last = np.full((height, width, 3), 128, np.uint8)
        self.stale = False          # references are stale after a loss until the next keyframe
        self.n_lost = 0

    def decode(self, data: bytes | None) -> np.ndarray:
        """Return the displayed RGB frame.  data=None => this frame's packet was lost."""
        if data is None or len(data) == 0:
            self.n_lost += 1
            self.stale = True
            return self.last.copy()
        key = _is_keyframe(self.codec, data)
        if self.stale and self.conceal == "freeze":
            if not key:
                return self.last.copy()          # wait for the next keyframe
            self.stale = False
        elif key:
            self.stale = False
        try:
            frames = self.ctx.decode(av.Packet(data))
        except Exception:
            frames = []
        if frames:
            img = frames[-1].to_ndarray(format="rgb24")
            if img.shape == (self.height, self.width, 3):
                self.last = img
        return self.last.copy()
