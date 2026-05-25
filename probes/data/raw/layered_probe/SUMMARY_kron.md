# Morphological probe — KRON (step 4623, kronecker_s1337_step0004623_final.pt)

## Architectural notes
- Vocab: 50304 | d_model: 768 | layers: 12 | embed_type: kronecker
- Architecture: Vanilla nanoGPT GPT-2 124M, pre-norm Blocks (ln_1 before attn, ln_2 before mlp), single-stream
- E_raw: Kron: frozen byte-level PF buffer (V×4096) → Linear(4096→768) → (D,)
- Preprocessing before layer 0: none (vanilla nanoGPT: x = drop(tok_emb + wpe(pos)), drop is identity in eval; first transformer block applies ln_1 internally inside h[0])
- E_post == E_raw after centering? **True** (vanilla GPT-2: wpe(0) is a constant offset across vocab)
- Stream collapse: n/a (single-stream)

- Centered ‖·‖₂ mean per layer: E_raw=20.192, E_post=20.192, L0=18.675, L1=17.807

## Sanity checks
- Self-retrieval: PASS
- Centered random-pair cosines (should be near zero):
  - E_raw: -0.0033
  - E_post: -0.0033
  - L0:    -0.0025
  - L1:    -0.0030
- Probes in vocab: PASS

## Overall morphological@10 (mean across 6 families)

| Layer | L (loose) | M (root-substring) | S (strict-family) |
|---|---:|---:|---:|
| E_raw | 0.967 | 0.244 | 0.221 |
| E_post | 0.967 | 0.244 | 0.221 |
| L0 | 0.967 | 0.265 | 0.237 |
| L1 | 0.967 | 0.258 | 0.231 |
| Δ (L0 − E_raw) | +0.000 | +0.021 | +0.017 |
| Δ (L1 − L0) | +0.000 | -0.006 | -0.006 |

## Per-family

| Family | Layer | L | M | S |
|---|---|---:|---:|---:|
| run | E_raw | 0.963 | 0.263 | 0.263 |
| run | L0 | 0.963 | 0.287 | 0.287 |
| run | L1 | 0.963 | 0.287 | 0.287 |
| compute | E_raw | 0.988 | 0.013 | 0.013 |
| compute | L0 | 0.988 | 0.013 | 0.013 |
| compute | L1 | 0.988 | 0.013 | 0.013 |
| nation | E_raw | 0.963 | 0.237 | 0.200 |
| nation | L0 | 0.963 | 0.263 | 0.200 |
| nation | L1 | 0.963 | 0.263 | 0.200 |
| code | E_raw | 0.988 | 0.163 | 0.113 |
| code | L0 | 0.988 | 0.163 | 0.113 |
| code | L1 | 0.988 | 0.163 | 0.113 |
| act | E_raw | 0.950 | 0.350 | 0.175 |
| act | L0 | 0.950 | 0.425 | 0.237 |
| act | L1 | 0.950 | 0.388 | 0.212 |
| build | E_raw | 0.950 | 0.438 | 0.562 |
| build | L0 | 0.950 | 0.438 | 0.575 |
| build | L1 | 0.950 | 0.438 | 0.562 |

## Notable findings

- Strict-family signal is small across layers (max Δ=0.017); morphological clustering is mostly already in E_raw or doesn't develop in 2 layers.
- E_raw L=0.967: top-10 are largely different surface forms, so M/S measure morphological vs semantic spread.

## Example top-10 retrievals (one probe per family)

| Family | Probe | E_raw top-10 | L1 top-10 |
|---|---|---|---|
| run | 'run' | 'runs' · 'Run' · 'ru' · 'tun' · 'fun' · 'pun' · 'mun' · 'runner' · 'oun' · 'sun' | 'runs' · 'Run' · 'ru' · 'tun' · 'sun' · 'pun' · 'running' · 'gun' · 'runner' · 'fun' |
| compute | 'compute' | 'Comp' · 'compl' · 'com' · 'camp' · 'compan' · 'Compl' · 'comm' · 'Com' · 'comb' · 'rompt' | 'Comp' · 'compl' · 'camp' · 'com' · 'compan' · 'Compar' · 'Compl' · 'comm' · 'Com' · 'omp' |
| nation | 'nation' | 'Nation' · 'national' · 'cation' · 'iation' · 'uation' · 'vation' · 'ration' · 'lation' · 'iations' · 'potion' | 'Nation' · 'cation' · 'national' · 'lation' · 'iation' · 'vation' · 'uation' · 'ration' · 'potion' · 'ination' |
| code | 'code' | 'Code' · 'codes' · 'coded' · 'cod' · 'cade' · 'mode' · 'node' · 'Cod' · 'core' · 'Mode' | 'Code' · 'codes' · 'coded' · 'cod' · 'cade' · 'mode' · 'Cod' · 'node' · 'core' · 'cone' |
| act | 'act' | 'acts' · 'Act' · 'oct' · 'ac' · 'acted' · 'ect' · 'actus' · 'actic' · 'ict' · 'activ' | 'acts' · 'Act' · 'acted' · 'oct' · 'actus' · 'ect' · 'actic' · 'ac' · 'ict' · 'apt' |
| build | 'build' | 'Build' · 'building' · 'built' · 'builder' · 'builders' · 'Building' · 'Built' · 'Builder' · 'bull' · 'bu' | 'Build' · 'building' · 'built' · 'builder' · 'Building' · 'Built' · 'builders' · 'Builder' · 'bu' · 'buy' |

