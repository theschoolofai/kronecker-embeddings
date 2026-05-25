"""
Example 3: forced-OOV (out-of-vocabulary) inference at a chosen position.

The codec is deterministic — any byte sequence produces an embedding,
including byte sequences that have no corresponding single token in the
vocab. This makes the input embedding effectively character-extensible
at test time: you can put a word the model has never seen as one logical
position in the prompt.

This example reproduces the §6.12 forced-OOV inference pattern:
construct a prompt of token ids, then override one position's underlying
bytes with an arbitrary string.

Run::

    python examples/03_inference_with_oov.py
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from kronecker_embeddings import KroneckerEmbedding


def main() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")

    emb = KroneckerEmbedding(
        vocab_size=tok.vocab_size,
        d_model=128,
        tokenizer=tok,
        pos_dim=32,
    )

    # A normal tokenized prompt.
    prompt = "the word is"
    ids = tok(prompt, return_tensors="pt").input_ids  # shape (1, L)
    print(f"prompt tokens: {tok.convert_ids_to_tokens(ids[0].tolist())}")

    # The standard forward pass treats every position as its tokenized form.
    standard_out = emb(ids)  # (1, L, 128)
    print(f"standard forward output shape: {tuple(standard_out.shape)}")

    # Now override position L-1 (the last token, " is") with a made-up word.
    # The bytes are passed through the codec directly — no tokenization required.
    L = ids.size(1)
    overridden = emb.forward_with_byte_override(
        ids[0],                          # single-row mode: pass 1D ids
        position_byte_overrides={L - 1: "frobnication".encode("utf-8")},
    )
    print(f"\noverride output shape: {tuple(overridden.shape)}")

    # Verify positions 0..L-2 unchanged.
    diff_other = (standard_out[0, :L - 1] - overridden[:L - 1]).abs().max()
    diff_last = (standard_out[0, L - 1] - overridden[L - 1]).abs().max()
    print(f"max diff at unchanged positions: {diff_other.item():.6e}")
    print(f"max diff at overridden position: {diff_last.item():.6e}")
    print()
    print("Forced-OOV pattern: the embedding for an arbitrary byte string ")
    print("is produced deterministically — no vocab lookup required.")


if __name__ == "__main__":
    main()
