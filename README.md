# Kronecker Embeddings

**Byte-level structured token representations for parameter-efficient language models. Reference implementation.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python ≥3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/arXiv-IN__PROCESS-b31b1b.svg)](https://arxiv.org/)

> **Status:** Accompanying paper submitted to arXiv; ID in process. The badge and citation will be updated once the ID is assigned.

A deterministic byte-level factorization that replaces `nn.Embedding` with a
fixed codec + a single trainable projection. Parameters scale with the
codec output dimension `D = char_dim × pos_dim`, **not** with vocabulary
size. At the paper's 124M GPT-2 setting (V=50,257, d_model=768, `D=4096`
via `pos_dim=16`), the stock `nn.Embedding(50,257, 768)` has 38.6 M
trainable params; the Kronecker projection `Linear(4096, 768)` has 3.1 M
— a **~12.3× input-side reduction (~91% saving)**, matching paper §6.9
Table 11. The saving grows with vocab and flattens with `D`, so it grows
as you scale up.

---

## Quickstart

```bash
pip install kronecker-embeddings
```

```python
import torch
from transformers import AutoTokenizer
from kronecker_embeddings import KroneckerEmbedding

tok = AutoTokenizer.from_pretrained("gpt2")
emb = KroneckerEmbedding(
    vocab_size=tok.vocab_size,
    d_model=512,
    tokenizer=tok,        # buffers built once at construction
    pos_dim=16,           # D = char_dim * pos_dim = 256 * 16 = 4096 (paper §6.9 setting)
)

ids = tok("hello world", return_tensors="pt").input_ids
out = emb(ids)            # (1, L, 512) — matches nn.Embedding.forward
```

That's it. Use `emb` anywhere you'd use `nn.Embedding`.

## What this is

Each token has a UTF-8 byte sequence — `"hello"` is 5 bytes, `"नमस्ते"` is 18.
The Kronecker codec maps that byte sequence to a fixed `D = char_dim × pos_dim`
dimensional vector deterministically (no learned parameters):

```
kappa(b) = (1/√L) · vec( Σ_{p=1..L} c_{b_p} ⊗ p_p )
```

where `c_v` is the v-th basis vector in R^256 (one-hot for byte value v) and
`p_p` is the p-th basis vector in R^`pos_dim` (one-hot for byte position p).
The result is z-normalized per-token, then passed through a single learned
linear projection `Linear(D, d_model)`.

**Architecture.** Fixed codec + **one** `nn.Linear(D, d_model, bias=False)`,
exactly as described in paper §3. No GELU, no hidden layer, no residual,
no norm beyond the codec's z-norm.

*Note:* paper §3.4 mentions an optional self-entropy regularizer on the
projection's row distribution, used in the 138M companion run (paper §6.7).
It is a production add-on, not part of the canonical method, and is not
shipped in this reference package.

The codec is **content-aware by byte**: tokens that share UTF-8 byte
prefixes share codec output positions, giving the embedding a built-in
orthographic structure (the §6.1–6.4 result). Tokenizers that fragment
language at the BPE level (Hindi, code, rare languages) get the same byte
locality as English at no extra vocab cost.

## Why use this

- **Parameter efficiency.** Embedding params scale with `D` (`pos_dim=16`
  → D=4096 in the paper's 124M experiments; `pos_dim=32` → D=8192 in
  production), not vocab size. The bigger your tokenizer, the bigger the win.
- **Orthographic structure built in.** Spelling variants, plural/singular
  pairs, and case variants land near each other in input-embedding space
  by construction. See `probes/cross_model_probe.py` for the full §6.1–6.4
  numbers across 6 public LMs.
- **Forced-OOV at inference.** Any byte sequence — including strings that
  don't have a single vocab token — produces a valid embedding. See
  `examples/03_inference_with_oov.py`.
- **Drop-in compatibility.** Same `forward(input_ids) -> (..., d_model)`
  contract as `nn.Embedding`. State dict round-trips with a single
  `projection.weight` tensor; the byte buffers are reconstructed from the
  tokenizer at load time.
- **Deployment.** No vocab-resize migration when adding/removing tokens —
  the codec handles arbitrary byte sequences. The byte buffer is ~0.9 MB
  for a 50K-vocab tokenizer at `pos_dim=16`, ~4.5 MB for a 131K-vocab
  tokenizer at `pos_dim=32`.

## Operational variants — `mode="dynamic"` vs `mode="cached"`

| Mode | Memory | Forward compute | Use when |
|---|---|---|---|
| `"dynamic"` (default) | `V × (pos_dim + 2)` bytes (~0.9 MB at V=50K, `pos_dim=16`; ~4.5 MB at V=131K production, `pos_dim=32`) | ~1-4 ms / micro-batch (scatter_add) | Frontier-scale training or memory-tight inference. The default. |
| `"cached"` | `V × D × 4` bytes (gigabytes at frontier scale) | ~0 (just `index_select`) | Small vocab (V ≲ 50K) or you have RAM to burn. |

Both produce **bit-identical** outputs (verified by `tests/test_embedding.py`).

Switch via the `mode=` kwarg:
```python
emb = KroneckerEmbedding(..., mode="cached")
```

## Examples

[`examples/01_minimal_swap.py`](examples/01_minimal_swap.py) — drop-in
replacement for `nn.Embedding` inside a toy encoder.

[`examples/02_nanogpt_integration.py`](examples/02_nanogpt_integration.py) —
patching nanoGPT's `transformer.wte` to use Kronecker. (A full training
fork lives at the forthcoming `kronecker-nanogpt` repo.)

[`examples/03_inference_with_oov.py`](examples/03_inference_with_oov.py) —
forced-OOV inference: override a position's bytes with a string that has
no single-token equivalent.

## Probes

The four probe scripts from the paper:

- [`probes/cross_model_probe.py`](probes/cross_model_probe.py) — §6.1–6.4
  embedding-geometry probe over arbitrary HF models. No checkpoint needed.
- [`probes/layered_probe.py`](probes/layered_probe.py) — §6.8 layered probe.
  Requires trained checkpoints (see [`probes/README.md`](probes/README.md)).
- [`probes/robustness_probe.py`](probes/robustness_probe.py) — §6.11 110-prompt
  spelling-robustness probe.
- [`probes/generation_probe.py`](probes/generation_probe.py) — §6.12 20-prompt
  generation probe with forced-OOV substitution.

Reference outputs from the paper's runs are in
[`probes/data/`](probes/data/) — reviewers can inspect them directly without
re-running.

## Citation

```bibtex
@article{shravan2026kronecker,
  title={Kronecker Embeddings: Byte-Level Structured Token Representations
         for Parameter-Efficient Language Models},
  author={Shravan, Rohan},
  journal={arXiv preprint arXiv:ARXIV_ID_PLACEHOLDER},
  year={2026},
  url={https://github.com/theschoolofai/kronecker-embeddings}
}
```

(The arXiv id will be replaced upon acceptance.)

## See also

- **`kronecker-nanogpt`** *(forthcoming)* — the full Kronecker-vs-stock
  training fork of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT),
  including matched 124M checkpoints that drop into the probes here.

## License

Apache 2.0 — see [LICENSE](LICENSE). Copyright 2026 The School of AI.

## Contact

- Issues / PRs: [github.com/theschoolofai/kronecker-embeddings/issues](https://github.com/theschoolofai/kronecker-embeddings/issues)
- Email: `rshravan@theschoolofai.in`
