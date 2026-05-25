"""Tests for tokenizer_utils.py."""

from __future__ import annotations

import pytest

from kronecker_embeddings import (
    build_byte_buffer,
    coverage_stats,
    token_id_to_bytes,
    utf8_safe_truncate,
)

try:
    from transformers import AutoTokenizer  # noqa: F401
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# ---------------- utf8_safe_truncate ----------------

def test_truncate_short_passthrough():
    assert utf8_safe_truncate(b"hello", 32) == b"hello"


def test_truncate_long_ascii():
    assert utf8_safe_truncate(b"a" * 50, 10) == b"a" * 10


def test_truncate_does_not_split_codepoint():
    """The Devanagari 'क' is 3 bytes (E0 A4 95). Truncating at 1 or 2 must yield empty/valid."""
    word = "क" * 5  # 15 bytes total, all UTF-8
    raw = word.encode("utf-8")
    # Truncate to 4 bytes: only first character (3 bytes) fits; bytes 4 is the start of next codepoint.
    out = utf8_safe_truncate(raw, 4)
    # Must decode cleanly:
    assert out.decode("utf-8") == "क"
    # Truncate to 7 bytes: 2 characters (6 bytes) fit; byte 7 is partial.
    out = utf8_safe_truncate(raw, 7)
    assert out.decode("utf-8") == "कक"


def test_truncate_zero_max():
    assert utf8_safe_truncate(b"abc", 0) == b""


# ---------------- token_id_to_bytes / build_byte_buffer (HF needed) ----------------

@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers required")
def test_byte_fallback_token_returns_single_byte():
    """A llama-2-style tokenizer has byte-fallback tokens like <0x41>. They should decode to a single byte."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    # Find a byte-fallback token id.
    vocab = tok.get_vocab()
    byte_id = None
    for piece, tid in vocab.items():
        if piece == "<0x41>":  # ASCII 'A' = 0x41
            byte_id = tid
            break
    if byte_id is None:
        pytest.skip("no <0x41> byte-fallback token in this tokenizer")
    raw = token_id_to_bytes(tok, byte_id)
    assert raw == b"\x41"


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers required")
def test_build_byte_buffer_gpt2():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    bb, lb = build_byte_buffer(tok, pos_dim=16)
    assert bb.shape[0] == tok.vocab_size
    assert bb.shape[1] == 16
    assert lb.shape == (tok.vocab_size,)
    # GPT-2 has byte-level vocab; most tokens have small bytes.
    assert (lb <= 16).all().item()


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers required")
def test_coverage_stats():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    _, lb = build_byte_buffer(tok, pos_dim=16)
    stats = coverage_stats(lb, pos_dim=16)
    assert stats["vocab_size"] == tok.vocab_size
    assert 0 <= stats["pct_within_pos_dim"] <= 100
    assert stats["max_observed_length"] <= 16


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers required")
def test_special_tokens_get_literal_bytes():
    """Special tokens like <|endoftext|> should encode their literal piece string."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    eot_id = tok.convert_tokens_to_ids("<|endoftext|>")
    raw = token_id_to_bytes(tok, eot_id)
    # Should produce a deterministic non-empty byte sequence.
    assert len(raw) > 0
    # And contain '<' and '>' bytes for the literal string fallback.
    assert b"<" in raw and b">" in raw
