"""
Traditional separate source--channel coding (SSCC) baseline.

Pipeline:   image(s)  ->  H.264 / H.265  ->  bits
                       ->  5G-LDPC encode  ->  QAM map
                       ->  AWGN / Rayleigh / 3GPP-CDL channel  (NVIDIA Sionna 2.0)
                       ->  QAM demap (LLR)  ->  LDPC decode  ->  bits
                       ->  H.264 / H.265 decode  ->  reconstructed image(s)

This is the classical foil to the JSCC semantic system: above a threshold
SNR the LDPC code corrects every bit and reconstruction is near-perfect;
below it the video bitstream is corrupted and the decoder collapses
(the "cliff effect").  The channel + SNR convention matches
``models.RayleighChannel`` (N0 = 1 / snr_lin, coherent ZF equalization).
Rate control is by constant QP (for rate-distortion sweeps) or CRF.
"""

from .video_codec import encode_frames, decode_frames
from .bit_utils import bytes_to_bits, bits_to_bytes

__all__ = [
    "encode_frames",
    "decode_frames",
    "SionnaCodedChannel",
    "bytes_to_bits",
    "bits_to_bytes",
]


def __getattr__(name):
    # Import Sionna (and TensorFlow) only when the channel is actually used,
    # so the codec / metrics path works without them installed.
    if name == "SionnaCodedChannel":
        from .sionna_channel import SionnaCodedChannel
        return SionnaCodedChannel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
