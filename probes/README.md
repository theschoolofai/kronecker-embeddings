# Probes

Reviewer-runnable probe scripts from the paper. Each script ships with the
exact prompt sets used in the paper and produces JSON output in the same
schema as the reference runs in `probes/data/`.

| Probe | Paper § | Needs trained checkpoint? | Reference output |
|---|---|---|---|
| `cross_model_probe.py` | §6.1–6.4 | No (uses public HF model embeddings) | [`data/raw/cross_model/`](data/raw/cross_model/) — per-model `.txt` outputs |
| `layered_probe.py` | §6.8 | **Yes** (BPE + Kron checkpoints) | [`data/layered_probe_results.json`](data/layered_probe_results.json) |
| `robustness_probe.py` | §6.11 | **Yes** (BPE + Kron checkpoints) | [`data/robustness_results.json`](data/robustness_results.json) |
| `generation_probe.py` | §6.12 | **Yes** (BPE + Kron checkpoints) | [`data/generation_probe_results.json`](data/generation_probe_results.json) |

## Quickstart

### Cross-model probe (no checkpoints required)
```bash
python probes/cross_model_probe.py \
    --models meta-llama/Llama-3.2-1B Qwen/Qwen3-8B google/gemma-3-1b-pt \
             mistralai/Mistral-Nemo-Base-2407 deepseek-ai/DeepSeek-R1 \
             sarvamai/sarvam-30b \
    --output cross_model_results.json
```
Operates on each model's public input-embedding table directly. The default
`--models` set in the paper was the 6 LMs listed above.

### Checkpoint-dependent probes

The 124M-parameter BPE + Kronecker checkpoints used in the paper were
trained on a one-off run and are not redistributable. To reproduce these
probes:

1. Train two matched 124M nanoGPT models (or any pair of transformer LMs)
   for the same number of steps on the same data:
   - Arm A: stock `nn.Embedding(V, d_model)` for the input embedding.
   - Arm B: `KroneckerEmbedding(vocab_size=V, d_model=d_model, tokenizer=tok)`.
2. Pass their checkpoint paths via `--bpe-checkpoint` and `--kron-checkpoint`.
3. Wire the small `run_pair` / `string_to_embedding` stubs in each probe
   script to your model wrapper (model classes vary; ~10 lines of glue
   each).

A reference training fork — `kronecker-nanogpt` — is forthcoming and will
include working checkpoints that drop into these probes without modification.

## Reference output

`probes/data/` contains the actual JSON outputs from the paper's runs.
Reviewers can inspect these directly:

- **`layered_probe_results.json`** — paper §6.8 Table 10. 6 morphological
  families × 8 probes = 48 probes, 4 representations (E_raw, E_post, L0, L1),
  3 metrics (loose / root-substring / strict-family morph@10) for both
  BPE and Kron 124M checkpoints (seed 1337, step 4623).
- **`robustness_results.json`** — paper §6.11 Tables 6 & 7. Full per-prompt
  output for 110 (clean, typo) pairs across 11 categories. Per pair, both
  arms' KL(clean‖typo), final-hidden cosine, log-prob drop, top-1 stability.
- **`generation_probe_results.json`** — paper §6.12. 20 prompts × {normal,
  misspelled} + 4 forced-OOV prompts × 2 arms = 88 records with greedy +
  sampled completions, sampler settings (`T=0.7, top_p=0.9, max=30, seed=42`),
  and per-record input token counts.
- **`raw/cross_model/`** — paper §6.2–6.5 raw `.txt` / `.csv` outputs from
  the cross-model probe on 6 public LMs (DeepSeek-V3-Base, GPT-OSS-120B,
  Llama-3.2-1B, Qwen3-32B, Gemma-3-1B-pt, SmolLM2-135M). Contains the
  per-family morph@5 rows, anisotropy norms, edit-distance tables, and
  cross-tokenizer Jaccard stability values referenced in the paper.
- **`raw/layered_probe/`** — human-readable per-arm SUMMARY markdown for
  the §6.8 layered probe.

## Methodology notes

- **Tokenization fairness.** All probes use each arm's *own* tokenizer for
  encoding. For the Kron arm the same vocab + a Kronecker codec replaces
  the input embedding; the lm_head is a separate trainable Linear.
- **Free OOV slot.** The generation probe uses token id 50300 (a free slot
  in the GPT-2 alignment-padding zone [50257..50303] that the tokenizer
  never produces) as a placeholder for byte-encoded overrides. The Kron arm
  rewrites its byte buffer at id 50300 with the override's UTF-8 bytes per
  prompt; the BPE arm has no equivalent and runs the prompt untouched.
- **Sampler.** Default `top_p=0.9, T=0.7, max_new_tokens=30, seed=42`. The
  generation probe applies a greedy backoff if any single token id repeats
  5+ consecutive times in the output (avoids degenerate loops).
- **Determinism.** With identical seed and identical checkpoints the probes
  should reproduce the published JSONs exactly. With different checkpoints
  (e.g. your own training run) the absolute numbers will differ but the
  Kron-vs-BPE relative ranking is stable in our experience.
