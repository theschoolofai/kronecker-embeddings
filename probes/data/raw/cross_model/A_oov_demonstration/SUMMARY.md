# OOV-as-single-token: cross-model comparison

Top-3 Kronecker-nearest in-vocab tokens for each probe, across 6 models.
Retrieval is in pure byte-level Kronecker space (no trained projection).
Tokens shown with their decoded surface form.

**Sanity:** at least one of top-3 retrievals contains the probe as a byte-substring (or vice versa) in **78/120** (probe × model) cases.

## Per-probe top-3 retrievals across 6 models

| Probe | Llama-3.2-1B | Qwen3-32B | Gemma-3-1B-pt | DeepSeek-V3 | GPT-OSS-120B | SmolLM2 |
|---|---|---|---|---|---|---|
| `kubernetes` | `kube`, `Hibernate`, ` usernames` | `kube`, `Hibernate`, ` usernames` | `kubernetes`, `kube`, `hibernate` | ` uber`, `Internet`, ` username` | `Hibernate`, `hibernate`, ` usernames` | `internet`, `Internet`, ` username` |
| `tensorflow` | `tensorflow`, `tensor`, `sensor` | `tensorflow`, `tensor`, `sensor` | `tensorflow`, `tensor`, `sensor` | `Sensor`, `Tensor`, `ten` | `tensorflow`, `tensor`, `sensor` | `tensor`, `sensor`, `Tensor` |
| `asynchronously` | ` synchronous`, ` synchron`, ` synchronize` | ` synchronous`, ` synchron`, ` synchronize` | ` synchronously`, ` synchronous`, ` synchron` | ` synchronous`, ` synchron`, ` Synchron` | ` synchronous`, ` synchron`, ` Synchron` | ` synchronous`, ` synchron`, `async` |
| `deserialization` | `deserialize`, ` specialization`, `Deserialize` | `deserialize`, ` specialization`, `Deserialize` | `deserialize`, ` specialization`, `Deserialize` | ` specialization`, ` initialization`, ` reorganization` | `deserialize`, ` specialization`, `Deserialize` | ` specialization`, ` initialization`, ` generalization` |
| `vibecoding` | ` preceding`, ` including`, ` intending` | ` preceding`, `.isLoading`, ` impending` | ` preceding`, `vib`, `piperidin` | ` preceding`, ` including`, ` impending` | ` preceding`, ` intending`, ` including` | ` preceding`, ` including`, ` intending` |
| `shoggoth` | ` Forgot`, ` forgot`, ` hodnot` | ` Forgot`, ` forgot`, `thought` | `sho`, ` Forgot`, ` forgot` | ` forgot`, ` hodnot`, `Thought` | `sho`, ` forgot`, ` Forgot` | ` forgot`, `thought`, ` hog` |
| `tiramisu` | `tir`, `thritis`, `.frames` | `thritis`, `_frames`, `(frames` | `riram`, `tir`, ` iremos` | ` cramps`, ` frames`, ` dramas` | `tir`, ` iremos`, ` promis` | `thritis`, ` cramps`, ` dramas` |
| `rizzler` | `riz`, `ripple`, `ricular` | `riz`, `ripple`, `ricular` | `rizzle`, `rizz`, `riz` | `rizzle`, `rizzly`, `rizz` | `riz`, `Puzzle`, ` izgled` | `riz`, `ricular`, `ritz` |
| `getUserById` | `getUser`, `getAs`, `getQuery` | `getUser`, `getAs`, `getQuery` | `getUser`, `GetUser`, `setUser` | `get`, `letcher`, `gett` | `get`, `getitem`, `Fetcher` | `get`, `getitem`, `letcher` |
| `fetchAsync` | `fetch`, `fetchAll`, `retch` | `fetch`, `fetchAll`, `retch` | `fetch`, `fetchAll`, `fetched` | `fetch`, `retch`, ` etch` | `fetch`, `retch`, `Fetch` | `fetch`, `fetched`, `retch` |
| `isPrimeNumber` | `iscrim`, ` stringBuffer`, ` randomNumber` | `iscrim`, ` stringBuffer`, ` randomNumber` | `isPrime`, ` secondNumber`, ` secretNumber` | ` shrimp`, ` shrine`, ` strive` | `istri`, `istrik`, `istoire` | `iscrim`, ` strive`, `istries` |
| `parseJSON` | `parse`, `parser`, `parsed` | `parse`, `parser`, `parsed` | `parse`, `parser`, `parsed` | `parse`, `parser`, `pause` | `parse`, `parser`, `parsed` | `parse`, `parser`, `parsed` |
| `unhappiness` | ` Shipping`, ` Shopping`, `_shipping` | ` Shipping`, ` Shopping`, ` snapping` | ` Chaplin`, `enhancing`, ` Wrapping` | ` Shipping`, ` Shopping`, ` trapping` | ` Shopping`, ` Shipping`, `-shopping` | ` sleepiness`, ` Shipping`, ` Shopping` |
| `magnetization` | `.Organization`, ` organization`, ` localization` | `.localization`, ` polarization`, ` organization` | ` monetization`, ` amortization`, `magnetic` | ` constipation`, ` quantization`, ` minimization` | ` constipation`, `magn`, `.Organization` | `magnetic`, ` constipation`, ` annihilation` |
| `antidisestablishmentarianism` | `anti`, `andidates`, `antidad` | `anti`, `andidates`, `antidad` | `antiti`, `anti`, `andidates` | `anti`, `andidates`, `andid` | `anti`, `andidates`, ` utilisent` | `anti`, `anted`, `antis` |
| `supercalifragilisticexpialidocious` | `super`, `Super`, `sumer` | `super`, `sumer`, `Super` | `super`, `supers`, `superclass` | `super`, `superscript`, `Super` | `super`, `Super`, `sup` | `super`, `Super`, `surgical` |
| `namaste` | `nama`, `names`, ` amassed` | `nama`, `names`, ` amassed` | `nama`, ` abaste`, ` amante` | `nama`, `names`, `nam` | `nama`, ` amante`, ` abaste` | `names`, `nam`, `nament` |
| `धन्यवाद` | `ानसभ`, `सन`, `़न` | `วันนี้`, `วันที่`, `บัญชี` | `वर्तमान`, `कन्या`, `वण्यात` | `िन्द`, `संख्या`, `ाम्` | `कर्ताओं`, `ाध्यक्ष`, `ाच्या` | `\xe0\xa4`, `大`, `म` |
| `感謝` | `感`, `感觉`, `\xe6\x84` | `感謝`, `感`, `感觉` | `感謝`, `感`, `感謝石頭` | `感謝`, `感`, `感触` | `感`, `感觉`, `感谢` | `\xe6`, `ل`, `\xe2\x84` |
| `Schadenfreude` | `Sch`, `Scheduler`, `Scheme` | `Sch`, `Scheduler`, `Scheme` | `Scha`, ` chad`, `Sched` | `Sch`, `Scheme`, ` shade` | `Sched`, `Sch`, `Scheduler` | `Sched`, `Sch`, ` clade` |

## Byte-overlap rate per category (top-3, any of 6 models)

Fraction of (probe × model) cells where at least one top-3 retrieval is a byte-substring of the probe (or vice versa). This is a coarse sanity check: high values mean Kronecker is picking up byte structure.

| Category | Cells | Cells with byte overlap | Rate |
|---|---:|---:|---:|
| technical_terms | 24 | 16 | 67% |
| neologisms_brands | 24 | 13 | 54% |
| code_identifiers | 24 | 19 | 79% |
| compound_words | 24 | 13 | 54% |
| multilingual | 24 | 17 | 71% |
