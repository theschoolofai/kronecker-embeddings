"""Tests for the KroneckerEmbedding nn.Module wrapper."""

from __future__ import annotations

import pytest
import torch

from kronecker_embeddings import KroneckerEmbedding

try:
    from transformers import AutoTokenizer  # noqa: F401
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


pytestmark = pytest.mark.skipif(
    not _HAS_TRANSFORMERS, reason="transformers is required for embedding tests"
)


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


# ---------------- construction + forward shape ----------------

def test_construct_from_tokenizer_and_forward(gpt2_tokenizer):
    emb = KroneckerEmbedding(
        vocab_size=gpt2_tokenizer.vocab_size,
        d_model=64,
        tokenizer=gpt2_tokenizer,
        pos_dim=16,
    )
    ids = torch.tensor([[100, 200, 300, 400]], dtype=torch.long)
    out = emb(ids)
    assert out.shape == (1, 4, 64)
    assert torch.isfinite(out).all().item()
    # nn.Embedding compatibility aliases:
    assert emb.num_embeddings == gpt2_tokenizer.vocab_size
    assert emb.embedding_dim == 64


# ---------------- dynamic == cached ----------------

def test_dynamic_and_cached_match(gpt2_tokenizer):
    """dynamic and cached modes must produce identical codec outputs."""
    common = dict(
        vocab_size=gpt2_tokenizer.vocab_size,
        d_model=32,
        tokenizer=gpt2_tokenizer,
        pos_dim=16,
    )
    emb_dyn = KroneckerEmbedding(**common, mode="dynamic")
    emb_cache = KroneckerEmbedding(**common, mode="cached")
    # Force identical projection weights to compare full forward output.
    emb_cache.projection.load_state_dict(emb_dyn.projection.state_dict())

    ids = torch.tensor([[0, 50, 1000, 5000, 30000]], dtype=torch.long)
    out_dyn = emb_dyn(ids)
    out_cache = emb_cache(ids)
    torch.testing.assert_close(out_dyn, out_cache, atol=1e-5, rtol=1e-5)


# ---------------- state_dict round-trip ----------------

def test_state_dict_only_projection(gpt2_tokenizer):
    emb = KroneckerEmbedding(
        vocab_size=gpt2_tokenizer.vocab_size,
        d_model=32,
        tokenizer=gpt2_tokenizer,
        pos_dim=16,
    )
    sd = emb.state_dict()
    keys = list(sd.keys())
    # The only persistent parameter is the projection.
    assert keys == ["projection.weight"]
    assert sd["projection.weight"].shape == (32, emb.D)


def test_save_load_projection(gpt2_tokenizer, tmp_path):
    emb1 = KroneckerEmbedding(
        vocab_size=gpt2_tokenizer.vocab_size,
        d_model=32,
        tokenizer=gpt2_tokenizer,
        pos_dim=16,
    )
    p = tmp_path / "ke.pt"
    torch.save(emb1.state_dict(), p)

    emb2 = KroneckerEmbedding(
        vocab_size=gpt2_tokenizer.vocab_size,
        d_model=32,
        tokenizer=gpt2_tokenizer,
        pos_dim=16,
    )
    emb2.load_state_dict(torch.load(p, weights_only=True))

    ids = torch.tensor([[100, 200, 300]], dtype=torch.long)
    out1 = emb1(ids)
    out2 = emb2(ids)
    torch.testing.assert_close(out1, out2)


# ---------------- byte override ----------------

def test_byte_override_changes_position(gpt2_tokenizer):
    emb = KroneckerEmbedding(
        vocab_size=gpt2_tokenizer.vocab_size,
        d_model=32,
        tokenizer=gpt2_tokenizer,
        pos_dim=16,
    )
    ids = torch.tensor([100, 200, 300], dtype=torch.long)
    base = emb.forward_with_byte_override(ids)
    overridden = emb.forward_with_byte_override(ids, {1: b"hello"})
    # Position 0 and 2 unchanged.
    torch.testing.assert_close(base[0], overridden[0])
    torch.testing.assert_close(base[2], overridden[2])
    # Position 1 changed.
    assert not torch.allclose(base[1], overridden[1])


def test_byte_override_rejects_batched(gpt2_tokenizer):
    emb = KroneckerEmbedding(
        vocab_size=gpt2_tokenizer.vocab_size,
        d_model=32,
        tokenizer=gpt2_tokenizer,
        pos_dim=16,
    )
    ids = torch.tensor([[100, 200, 300], [400, 500, 600]], dtype=torch.long)
    with pytest.raises(ValueError, match="single-row"):
        emb.forward_with_byte_override(ids, {1: b"x"})


# ---------------- construct without tokenizer (pre-built buffers) ----------------

def test_construct_from_prebuilt_buffers():
    V, pos_dim = 50, 8
    bb = torch.zeros((V, pos_dim), dtype=torch.uint8)
    lb = torch.zeros((V,), dtype=torch.int16)
    bb[5, :3] = torch.tensor([0x61, 0x62, 0x63], dtype=torch.uint8)  # "abc"
    lb[5] = 3
    emb = KroneckerEmbedding(
        vocab_size=V, d_model=16, char_dim=256, pos_dim=pos_dim,
        byte_buffer=bb, length_buffer=lb,
    )
    out = emb(torch.tensor([5], dtype=torch.long))
    assert out.shape == (1, 16)
    assert torch.isfinite(out).all().item()
