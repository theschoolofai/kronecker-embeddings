# Cross-tokenizer Kronecker stability

For each probe, the top-5 Kronecker-nearest in-vocab tokens were retrieved from each of the 6 models' vocabularies (in pure byte-level Kron space). We compute Jaccard similarity between every pair of models' top-5 canonical surface form sets (15 pairs).

## Per-probe Jaccard (over 15 model pairs)

| Probe | Mean Jaccard | Min | Max | n_pairs |
|---|---:|---:|---:|---:|
| `run` | 0.378 | 0.111 | 0.667 | 15 |
| `running` | 0.262 | 0.143 | 0.600 | 15 |
| `compute` | 0.773 | 0.600 | 1.000 | 15 |
| `computer` | 0.707 | 0.400 | 1.000 | 15 |
| `magnet` | 0.392 | 0.000 | 0.667 | 15 |
| `nation` | 0.512 | 0.250 | 0.667 | 15 |
| `creation` | 0.302 | 0.143 | 1.000 | 15 |
| `import` | 0.512 | 0.143 | 1.000 | 15 |

## Top-5 canonical surface forms per probe per model

| Probe | Llama-3.2-1B | Qwen3-32B | Gemma-3 | DeepSeek-V3 | GPT-OSS-120B | SmolLM2 |
|---|---|---|---|---|---|---|
| `run` | `run`, `rung`, `runs`, `ru`, `runner` | `run`, `runs`, `ru`, `runner`, `ron` | `run`, `rune`, `rund`, `runk`, `rung` | `run`, `rund`, `ru`, `runner`, `ron` | `run`, `runs`, `ru`, `runde`, `runner` | `run`, `rund`, `runk`, `runc`, `runs` |
| `running` | `running`, `running`, `ranking`, `inning`, `rending` | `running`, `running`, `rending`, `ounding`, `rowning` | `running`, `running`, `panning`, `funding`, `winning` | `running`, `running`, `rinnings`, `turning`, `ranking` | `running`, `running`, `ranking`, `funding`, `winning` | `running`, `running`, `winning`, `turning`, `aunting` |
| `compute` | `compute`, `computed`, `computer`, `comput`, `compute` | `compute`, `computed`, `computer`, `comput`, `compute` | `compute`, `computed`, `computer`, `comput`, `computers` | `compute`, `computer`, `comput`, `compute`, `computer` | `compute`, `computed`, `computer`, `comput`, `compute` | `compute`, `computer`, `comput`, `computers`, `compute` |
| `computer` | `computer`, `compute`, `computer`, `computed`, `comput` | `computer`, `compute`, `computer`, `computed`, `comput` | `computer`, `computers`, `compute`, `computer`, `computed` | `computer`, `compute`, `computer`, `comput`, `compute` | `computer`, `compute`, `computed`, `computer`, `comput` | `computer`, `computers`, `compute`, `computer`, `comput` |
| `magnet` | `mag`, `magnitude`, `market`, `markets`, `magn` | `mag`, `magnitude`, `market`, `markets`, `mage` | `magnet`, `magnetic`, `magnet`, `magn`, `maine` | `magn`, `mag`, `market`, `markets`, `mage` | `magn`, `maine`, `mag`, `market`, `mannen` | `magnetic`, `magn`, `mag`, `magnitude`, `market` |
| `nation` | `nation`, `national`, `lation`, `vation`, `ration` | `nation`, `national`, `vation`, `lation`, `cation` | `nation`, `national`, `mation`, `ration`, `vation` | `nation`, `national`, `lation`, `ration`, `iation` | `nation`, `national`, `ration`, `uation`, `iation` | `nation`, `national`, `ration`, `iation`, `cation` |
| `creation` | `creation`, `creation`, `creat`, `citation`, `ertation` | `creation`, `creation`, `creat`, `irection`, `cription` | `creation`, `creational`, `creation`, `creatic`, `creat` | `creation`, `creational`, `creation`, `culation`, `creative` | `creation`, `creation`, `ervation`, `cryption`, `cription` | `creation`, `creation`, `creat`, `cription`, `irection` |
| `import` | `import`, `importe`, `imports`, `import`, `important` | `import`, `imports`, `importe`, `import`, `important` | `import`, `importe`, `imports`, `imported`, `import` | `import`, `import`, `important`, `importance`, `empor` | `import`, `importe`, `imports`, `import`, `important` | `import`, `import`, `important`, `empor`, `imp` |

## One-liner

Averaged across **8** probe strings and **15** cross-tokenizer pairs, Kronecker neighbor retrievals from different tokenizers have mean Jaccard **0.480** on canonical surface forms. Highest agreement: `compute` (Jaccard 0.773). Lowest: `running` (Jaccard 0.262).