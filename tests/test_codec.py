"""Tests for the codec math (pure functions in codec.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kronecker_embeddings import codec_output_dim, encode_single, kronecker_codec


# ---------------- helpers ----------------

def hand_encode(byte_seq: bytes, char_dim: int = 256, pos_dim: int = 32,
                length_normalize: bool = True, z_normalize: bool = True,
                eps: float = 1e-6) -> np.ndarray:
    """
    Reference implementation in numpy, written from the math definition.
    Uses Bessel's correction (ddof=1) to match torch.std()'s default — the
    production codec uses torch's default.
    """
    L = min(len(byte_seq), pos_dim)
    M = np.zeros((char_dim, pos_dim), dtype=np.float32)
    for i, b in enumerate(byte_seq[:L]):
        M[b, i] = 1.0
    if length_normalize and L > 0:
        M *= 1.0 / math.sqrt(L)
    v = M.flatten()
    if z_normalize:
        v = (v - v.mean()) / (v.std(ddof=1) + eps)
    return v


# ---------------- kappa correctness ----------------

@pytest.mark.parametrize("text", ["a", "ab", "abc", "hello", "hi!", "X"])
def test_kappa_matches_hand_computed(text):
    """codec output must match the by-hand numpy implementation exactly."""
    ours = encode_single(text.encode("utf-8")).numpy()
    ref = hand_encode(text.encode("utf-8"))
    np.testing.assert_allclose(ours, ref, atol=1e-4, rtol=1e-4)


def test_codec_output_dim_helper():
    assert codec_output_dim() == 256 * 32
    assert codec_output_dim(256, 16) == 256 * 16
    assert codec_output_dim(128, 64) == 128 * 64


# ---------------- determinism ----------------

def test_determinism():
    """Same input -> identical output across calls (no random ops)."""
    b = "hello".encode("utf-8")
    out1 = encode_single(b)
    out2 = encode_single(b)
    assert torch.equal(out1, out2)


# ---------------- single-byte-substitution property ----------------

def test_one_byte_difference_cosine():
    """
    For two strings of identical length L that differ in exactly one byte
    position, the codec output cosine similarity should be (L - 1) / L
    (after length-normalization, before z-norm — z-norm preserves this
    geometry up to a constant offset that washes out under cosine on
    centered vectors).

    We use sufficiently distinct bytes (one ASCII letter and a different
    ASCII letter at the same index) so the resulting vectors have no
    shared one-hot positions at the differing index.
    """
    a = b"abcd"
    b = b"xbcd"  # differs at index 0
    # Compute with length-normalize ON, z-norm OFF for the clean prediction.
    out_a = encode_single(a, z_normalize=False)
    out_b = encode_single(b, z_normalize=False)
    cos = (out_a @ out_b) / (out_a.norm() * out_b.norm() + 1e-12)
    expected = (len(a) - 1) / len(a)
    assert abs(cos.item() - expected) < 1e-4


# ---------------- case sensitivity ----------------

def test_case_sensitivity_orthogonal():
    """Upper and lower case of distinct ASCII letters should be (nearly) orthogonal."""
    out_lower = encode_single(b"run", z_normalize=False)
    out_upper = encode_single(b"RUN", z_normalize=False)
    cos = (out_lower @ out_upper) / (out_lower.norm() * out_upper.norm() + 1e-12)
    # ASCII lower- and upper-case letters live in different byte rows;
    # zero shared one-hot positions => cosine == 0.
    assert abs(cos.item()) < 1e-6


# ---------------- batched call ----------------

def test_batched_matches_loop():
    texts = ["a", "ab", "abc", "abcd"]
    byte_seqs = [t.encode("utf-8") for t in texts]
    pos_dim = 32

    # Build batched inputs.
    bb = torch.zeros((len(byte_seqs), pos_dim), dtype=torch.uint8)
    lens = torch.zeros(len(byte_seqs), dtype=torch.long)
    for i, bs in enumerate(byte_seqs):
        L = min(len(bs), pos_dim)
        bb[i, :L] = torch.frombuffer(bytearray(bs[:L]), dtype=torch.uint8)
        lens[i] = L

    batched = kronecker_codec(bb, lens)
    looped = torch.stack([encode_single(bs) for bs in byte_seqs])
    torch.testing.assert_close(batched, looped, atol=1e-5, rtol=1e-5)


# ---------------- empty-token safety ----------------

def test_empty_token_safe():
    out = encode_single(b"")
    assert out.shape == (256 * 32,)
    # All-zero codec input -> z-norm gives the std=eps fallback; just check finite.
    assert torch.isfinite(out).all().item()


# ---------------- length_normalize off ----------------

def test_no_length_normalize():
    out = encode_single(b"abc", length_normalize=False, z_normalize=False)
    # Sum of one-hot entries equals L=3 when length_normalize off.
    assert abs(out.sum().item() - 3.0) < 1e-5
