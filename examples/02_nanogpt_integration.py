"""
Example 2: patch nanoGPT's GPT class to use KroneckerEmbedding.

nanoGPT (https://github.com/karpathy/nanoGPT) constructs its input embedding
as `self.transformer.wte = nn.Embedding(vocab_size, n_embd)`. This example
shows the surgical replacement:

    self.transformer.wte = KroneckerEmbedding(vocab_size, n_embd, tokenizer=tok)

The rest of the model (positional embeddings, attention blocks, lm_head)
stays exactly as in upstream nanoGPT.

Note: the lm_head is a separate Linear in nanoGPT and remains untouched —
KroneckerEmbedding deliberately does not include the output head. Weight
tying with the lm_head is NOT possible because the codec output dimension
D = char_dim * pos_dim does not match d_model in general.

A full forked training repo (kronecker-nanogpt) is forthcoming. This file
is a worked code snippet, not a runnable end-to-end trainer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from kronecker_embeddings import KroneckerEmbedding


# --- nanoGPT-style skeleton (simplified for the demo) ---

class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPTSkeleton(nn.Module):
    """
    Skeleton mirroring nanoGPT.GPT.__init__. The body (`self.transformer.h`)
    is omitted for brevity — point is to show the surgical wte swap.
    """

    def __init__(self, config: GPTConfig, tokenizer=None, use_kronecker: bool = False):
        super().__init__()
        self.config = config

        if use_kronecker:
            assert tokenizer is not None, "tokenizer required for use_kronecker=True"
            # pos_dim=16 (D=4096) matches paper §6.9 124M parameter accounting:
            # nn.Embedding(50257, 768) = 38.6M  vs  Linear(4096, 768) = 3.1M (~12.5x).
            wte = KroneckerEmbedding(
                vocab_size=config.vocab_size,
                d_model=config.n_embd,
                tokenizer=tokenizer,
                pos_dim=16,
                mode="dynamic",
            )
        else:
            wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte=wte,
            wpe=nn.Embedding(config.block_size, config.n_embd),
            # h=nn.ModuleList(...),    # transformer blocks would go here
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Note: NO weight tying with wte when use_kronecker=True (D != n_embd).

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.ln_f(tok_emb + pos_emb)
        return self.lm_head(x)


def main() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    cfg = GPTConfig()

    stock = GPTSkeleton(cfg, use_kronecker=False)
    kron = GPTSkeleton(cfg, tokenizer=tok, use_kronecker=True)

    n_stock = sum(p.numel() for p in stock.parameters() if p.requires_grad)
    n_kron = sum(p.numel() for p in kron.parameters() if p.requires_grad)

    print("nanoGPT-style skeleton, n_embd=768, vocab=50257")
    print(f"  Stock nn.Embedding params:        {n_stock:>14,}")
    print(f"  Kronecker model params:           {n_kron:>14,}")
    print(f"  Embedding savings (stock - kron): {n_stock - n_kron:>14,}")

    ids = tok("Hello, world", return_tensors="pt").input_ids
    logits_stock = stock(ids)
    logits_kron = kron(ids)
    print(f"\nForward output shape (both): {tuple(logits_kron.shape)}")
    print(f"\nThe full nanoGPT training fork lives at the kronecker-nanogpt repo (forthcoming).")


if __name__ == "__main__":
    main()
