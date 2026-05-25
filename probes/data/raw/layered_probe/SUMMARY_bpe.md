# Morphological probe — BPE (step 4623, bpe_s1337_step0004623_final.pt)

## Architectural notes
- Vocab: 50304 | d_model: 768 | layers: 12 | embed_type: bpe
- Architecture: Vanilla nanoGPT GPT-2 124M, pre-norm Blocks (ln_1 before attn, ln_2 before mlp), single-stream
- E_raw: BPE: wte(id) is plain nn.Embedding lookup → (D,)
- Preprocessing before layer 0: none (vanilla nanoGPT: x = drop(tok_emb + wpe(pos)), drop is identity in eval; first transformer block applies ln_1 internally inside h[0])
- E_post == E_raw after centering? **True** (vanilla GPT-2: wpe(0) is a constant offset across vocab)
- Stream collapse: n/a (single-stream)

- Centered ‖·‖₂ mean per layer: E_raw=0.839, E_post=0.839, L0=6.899, L1=8.344

## Sanity checks
- Self-retrieval: PASS
- Centered random-pair cosines (should be near zero):
  - E_raw: +0.0016
  - E_post: +0.0016
  - L0:    -0.0004
  - L1:    +0.0012
- Probes in vocab: PASS

## Overall morphological@10 (mean across 6 families)

| Layer | L (loose) | M (root-substring) | S (strict-family) |
|---|---:|---:|---:|
| E_raw | 0.898 | 0.302 | 0.283 |
| E_post | 0.898 | 0.302 | 0.283 |
| L0 | 0.906 | 0.285 | 0.267 |
| L1 | 0.942 | 0.169 | 0.156 |
| Δ (L0 − E_raw) | +0.008 | -0.017 | -0.017 |
| Δ (L1 − L0) | +0.035 | -0.117 | -0.111 |

## Per-family

| Family | Layer | L | M | S |
|---|---|---:|---:|---:|
| run | E_raw | 0.887 | 0.362 | 0.425 |
| run | L0 | 0.887 | 0.362 | 0.375 |
| run | L1 | 0.938 | 0.150 | 0.150 |
| compute | E_raw | 0.963 | 0.050 | 0.050 |
| compute | L0 | 0.963 | 0.062 | 0.062 |
| compute | L1 | 0.963 | 0.062 | 0.062 |
| nation | E_raw | 0.900 | 0.287 | 0.250 |
| nation | L0 | 0.912 | 0.275 | 0.237 |
| nation | L1 | 0.950 | 0.225 | 0.175 |
| code | E_raw | 0.925 | 0.175 | 0.175 |
| code | L0 | 0.925 | 0.150 | 0.150 |
| code | L1 | 0.963 | 0.075 | 0.075 |
| act | E_raw | 0.863 | 0.487 | 0.275 |
| act | L0 | 0.900 | 0.362 | 0.200 |
| act | L1 | 0.963 | 0.200 | 0.100 |
| build | E_raw | 0.850 | 0.450 | 0.525 |
| build | L0 | 0.850 | 0.500 | 0.575 |
| build | L1 | 0.875 | 0.300 | 0.375 |

## Notable findings

- Strict-family signal is small across layers (max Δ=0.111); morphological clustering is mostly already in E_raw or doesn't develop in 2 layers.

## Example top-10 retrievals (one probe per family)

| Family | Probe | E_raw top-10 | L1 top-10 |
|---|---|---|---|
| run | 'run' | ' Run' · 'Run' · ' run' · ' runs' · 'runs' · 'running' · ' running' · 'haul' · ' ran' · 'winter' | 'end' · 'runs' · 'Ire' · ' Sakuya' · 'uilt' · 'dyl' · ' \u200e' · '}"' · 'javascript' · '}}}' |
| compute | 'compute' | 'Comp' · ' Comp' · ' uncomp' · ' comp' · ' recomp' · 'omp' · ' decomp' · ' incomp' · 'Mod' · 'whe' | 'Comp' · ' Comp' · ' uncomp' · ' comp' · ' decomp' · ' recomp' · 'omp' · ' incomp' · 'exp' · ' disp' |
| nation | 'nation' | 'Nation' · ' nation' · 'world' · 'population' · 'itution' · 'ation' · 'avage' · ' regime' · 'establishment' · 'child' | 'gamer' · ' aggro' · 'iseum' · ' fullback' · ' NASL' · ' Luthor' · 'senal' · ' Blackhawks' · 'ventus' · 'エル' |
| code | 'code' | ' code' · 'Code' · 'codes' · ' Code' · ' codes' · ' coding' · 'ntax' · 'source' · 'words' · 'File' | 'codes' · 'Code' · 'query' · 'browser' · ' code' · 'file' · 'xml' · 'package' · 'map' · 'PIN' |
| act | 'act' | 'acts' · ' lact' · 'fact' · ' impact' · ' effect' · 'ost' · 'ect' · ' interact' · 'issue' · ' Impact' | 'Redditor' · ' warr' · 'ktop' · '\x00' · '*=-' · '�' · '' · 'EStream' · ' gobl' · ' lvl' |
| build | 'build' | 'Build' · ' build' · ' Build' · ' builds' · ' built' · ' rebuild' · ' building' · 'Building' · ' rebuilding' · 'built' | 'Build' · ' Build' · ' build' · ' rebuild' · ' building' · ' built' · 'Building' · 'Create' · 'Develop' · 'make' |

