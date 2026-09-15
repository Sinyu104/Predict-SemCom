"""
H.264 / H.265 source coding for the traditional SSCC baseline.

Frames are encoded as a short raw Annex-B elementary stream (no container
overhead) via PyAV / libx264 / libx265, producing a compressed byte string.
A single image is just a clip of length 1 (intra-only).

Decoding is made robust to bit errors: a corrupted bitstream makes the
video decoder throw or emit fewer frames, so we catch failures and pad the
output to the expected length.  This is the realistic, intended behaviour of
a separate-coding baseline — it is what produces the cliff effect.
"""

from __future__ import annotations

import io
import numpy as np

import av

# libx264 / libx265 emit copious warnings on corrupted input — silence them.
try:
    av.logging.set_level(av.logging.PANIC)
except Exception:
    pass


# ── codec name → (encoder, raw-stream format) ───────────────────────────── #
_CODECS = {
    "h264": ("libx264", "h264"),
    "h265": ("libx265", "hevc"),
    "hevc": ("libx265", "hevc"),
    "avc":  ("libx264", "h264"),
}


def _resolve_codec(codec: str) -> tuple[str, str]:
    key = codec.lower()
    if key not in _CODECS:
        raise ValueError(
            f"Unknown codec '{codec}'. Choose from {sorted(_CODECS)}."
        )
    return _CODECS[key]


def encode_frames(
    frames: np.ndarray,
    codec: str = "h265",
    crf: int | None = 28,
    qp: int | None = None,
    gop: int = 12,
    fps: int = 10,
) -> tuple[bytes, str]:
    """
    Encode an RGB clip to a compressed elementary stream.

    Rate control (pick one):
        qp  : constant quantization parameter (CQP). Fixed quantizer every
              frame -> clean, monotonic rate-vs-quality points, ideal for
              rate-distortion curves. Range 0 (lossless) .. 51 (worst).
        crf : constant-rate-factor (perceptual VBR). Used only when qp is None.
              Range ~18 (high quality) .. 35 (low); 28 is a good default.

    Args:
        frames: (T, H, W, 3) uint8 RGB.
        codec:  "h264" / "h265".
        gop:    group-of-pictures size (keyframe interval).
        fps:    nominal frame rate stored in the stream.

    Returns:
        (bitstream_bytes, stream_format)  — pass stream_format to decode_frames.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3) RGB, got {frames.shape}")
    frames = np.ascontiguousarray(frames.astype(np.uint8))
    encoder, stream_fmt = _resolve_codec(codec)

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=stream_fmt)
    stream = container.add_stream(encoder, rate=fps)
    stream.width   = int(frames.shape[2])
    stream.height  = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.gop_size = int(gop)

    # Rate control. QP (constant quantizer) takes precedence over CRF.
    # x265 takes all knobs via one ':'-joined x265-params string; x264 uses
    # individual private options.
    if qp is not None:
        if encoder == "libx265":
            stream.options = {"x265-params": f"qp={int(qp)}:log-level=none"}
        else:
            stream.options = {"qp": str(int(qp))}
    else:
        if crf is None:
            raise ValueError("Provide either crf or qp for rate control.")
        if encoder == "libx265":
            stream.options = {"crf": str(crf), "x265-params": "log-level=none"}
        else:
            stream.options = {"crf": str(crf)}

    for f in frames:
        vframe = av.VideoFrame.from_ndarray(f, format="rgb24")
        vframe = vframe.reformat(format="yuv420p")
        for pkt in stream.encode(vframe):
            container.mux(pkt)
    for pkt in stream.encode():          # flush
        container.mux(pkt)
    container.close()

    return buf.getvalue(), stream_fmt


def decode_frames(
    data: bytes,
    stream_fmt: str,
    expected_T: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, int]:
    """
    Decode a (possibly bit-corrupted) elementary stream back to RGB frames.

    Missing / undecodable frames are replaced with mid-gray (128) so the
    output always has shape (expected_T, height, width, 3).

    Returns:
        (frames_uint8, n_decoded)  where n_decoded is how many frames the
        decoder actually produced before failing (n_decoded < expected_T
        signals channel-induced bitstream corruption).
    """
    out: list[np.ndarray] = []
    try:
        container = av.open(io.BytesIO(data), format=stream_fmt)
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            # A corrupted stream can make the decoder resync to a garbage
            # header and emit a frame with the wrong geometry — drop those.
            if img.shape != (height, width, 3):
                continue
            out.append(img)
            if len(out) >= expected_T:
                break
        container.close()
    except Exception:
        # Corrupted bitstream — keep whatever decoded cleanly before the throw.
        pass

    n_decoded = len(out)
    gray = np.full((height, width, 3), 128, dtype=np.uint8)
    while len(out) < expected_T:
        out.append(gray.copy())
    frames = np.stack(out[:expected_T], axis=0).astype(np.uint8)
    return frames, n_decoded
