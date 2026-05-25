"""
Example 1: minimal `nn.Embedding` -> `KroneckerEmbedding` swap.

Defines a tiny encoder-style model, then shows the two ways to construct
its input embedding:
  (a) a stock nn.Embedding (vocab_size * d_model trainable parameters).
  (b) a KroneckerEmbedding (only D * d_model trainable parameters, where
      D = char_dim * pos_dim, independent of vocab_size).

This example uses ``pos_dim=16`` (D = 256 * 16 = 4096) to match the
paper's §6.9 124M parameter accounting. The reduction factor is
``vocab_size / D = 50,257 / 4096 ≈ 12.27×``, independent of ``d_model``.
At the paper's ``d_model=768`` setting this is 38.6M -> 3.1M
(~91% input-side reduction, paper §6.9 Table 11); the same ratio
applies at the ``d_model=128`` setting used here.

Run::

    python examples/01_minimal_swap.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from kronecker_embeddings import KroneckerEmbedding


class TinyEncoder(nn.Module):
    """A toy 1-layer transformer encoder for demonstration."""

    def __init__(self, embedding: nn.Module, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.embedding = embedding
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embedding(input_ids)
        h_attn, _ = self.attn(h, h, h)
        return self.ln(h_attn + h)


def main() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # GPT-2 has no pad token by default
    text = ["hello world", "Kronecker embeddings."]
    enc = tok(text, return_tensors="pt", padding=True)
    ids = enc["input_ids"]

    d_model = 128

    # ---- (a) stock nn.Embedding ----
    stock_emb = nn.Embedding(tok.vocab_size, d_model)
    stock_params = sum(p.numel() for p in stock_emb.parameters())
    print(f"nn.Embedding trainable params:        {stock_params:>14,}")

    # ---- (b) KroneckerEmbedding ----
    # pos_dim=16 -> D=4096, matching paper §6.9's 124M setting.
    k_emb = KroneckerEmbedding(
        vocab_size=tok.vocab_size,
        d_model=d_model,
        tokenizer=tok,
        pos_dim=16,
    )
    k_params = sum(p.numel() for p in k_emb.parameters())
    print(f"KroneckerEmbedding trainable params:  {k_params:>14,}")
    print(f"reduction factor:                     {stock_params / max(k_params, 1):>14.2f}x")

    # ---- both produce (B, L, d_model) shape ----
    stock_out = stock_emb(ids)
    k_out = k_emb(ids)
    assert stock_out.shape == k_out.shape, (stock_out.shape, k_out.shape)
    print(f"\nBoth produce output of shape {tuple(k_out.shape)}.")
    print("KroneckerEmbedding is a drop-in replacement for nn.Embedding.")

    # ---- use it inside a tiny model ----
    model_stock = TinyEncoder(stock_emb, d_model=d_model)
    model_k = TinyEncoder(k_emb, d_model=d_model)
    out_a = model_stock(ids)
    out_b = model_k(ids)
    assert out_a.shape == out_b.shape
    print(f"\nTinyEncoder forward output shape: {tuple(out_b.shape)}.")
    print("\nDone.")


if __name__ == "__main__":
    main()
