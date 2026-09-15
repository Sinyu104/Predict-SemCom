"""Bit <-> byte packing helpers (MSB-first, matches numpy.unpackbits).

Kept separate from the Sionna channel so the codec / metrics path can run
without TensorFlow + Sionna installed.
"""

from __future__ import annotations

import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    """bytes -> uint8 array of 0/1, length 8*len(data)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """0/1 array -> bytes (trailing bits beyond a byte boundary are dropped)."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    usable = (bits.size // 8) * 8
    return np.packbits(bits[:usable]).tobytes()
