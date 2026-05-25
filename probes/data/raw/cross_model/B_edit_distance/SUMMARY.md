# Edit-distance falloff

Cosine similarity in Trained-BPE space (mean-centered, mean-of-subtokens query) vs Kronecker space (whole-string byte encoding), averaged across 6 models. Variants are grouped into 4 buckets:

- **self**: identical to anchor (cosine = 1.0 by construction)
- **same_canonical**: typographic variant — same `canonical_form()` as anchor
- **morphological**: morphological relative — different canonical form
- **unrelated**: control

## Aggregate table (mean across 6 models)

| anchor | variant | ED | description | bucket | trained_cos | kron_cos |
|---|---|---:|---|---|---:|---:|
| run | run | 0 | self | self | 1.000 | 1.000 |
| run | Run | 1 | case_swap | same_canonical | 0.657 | 0.667 |
| run | RUN | 3 | all_caps | same_canonical | 0.494 | -0.000 |
| run | runs | 1 | morph_+s | morphological | 0.506 | 0.866 |
| run | ran | 2 | morph_irregular | morphological | 0.324 | 0.667 |
| run | runner | 3 | morph_+er | morphological | 0.349 | 0.707 |
| run | running | 4 | morph_+ing | morphological | 0.469 | 0.654 |
| run | rune | 1 | near_unrelated | unrelated | 0.271 | 0.866 |
| run | road | 3 | unrelated_short | unrelated | 0.129 | 0.288 |
| run | kitchen | 6 | unrelated_long | unrelated | 0.070 | -0.001 |
| compute | compute | 0 | self | self | 1.000 | 1.000 |
| compute | Compute | 1 | case_swap | same_canonical | 0.590 | 0.857 |
| compute | computes | 1 | morph_+s | morphological | 0.281 | 0.935 |
| compute | computed | 1 | morph_+d | morphological | 0.383 | 0.935 |
| compute | computer | 1 | morph_+r | morphological | 0.277 | 0.935 |
| compute | computing | 2 | morph_+ing | morphological | 0.326 | 0.756 |
| compute | computation | 3 | morph_+ation | morphological | 0.308 | 0.683 |
| compute | commute | 1 | near_unrelated | unrelated | 0.124 | 0.857 |
| compute | compose | 2 | unrelated | unrelated | 0.192 | 0.714 |
| compute | kitchen | 7 | unrelated_long | unrelated | 0.041 | -0.001 |
| nation | nation | 0 | self | self | 1.000 | 1.000 |
| nation | Nation | 1 | case_swap | same_canonical | 0.455 | 0.833 |
| nation | nations | 1 | morph_+s | morphological | 0.232 | 0.926 |
| nation | national | 2 | morph_+al | morphological | 0.344 | 0.866 |
| nation | nationally | 4 | morph_+ally | morphological | 0.300 | 0.774 |
| nation | station | 1 | near_unrelated | unrelated | 0.185 | -0.001 |
| nation | notion | 2 | near_unrelated_2 | unrelated | 0.136 | 0.833 |
| nation | ration | 1 | near_unrelated_3 | unrelated | 0.206 | 0.833 |
| nation | kitchen | 6 | unrelated | unrelated | 0.033 | 0.154 |
| import | import | 0 | self | self | 1.000 | 1.000 |
| import | Import | 1 | case_swap | same_canonical | 0.443 | 0.833 |
| import | imports | 1 | morph_+s | morphological | 0.453 | 0.926 |
| import | imported | 2 | morph_+ed | morphological | 0.693 | 0.866 |
| import | importing | 3 | morph_+ing | morphological | 0.735 | 0.816 |
| import | important | 3 | near_unrelated | unrelated | 0.192 | 0.816 |
| import | export | 2 | antonym | unrelated | 0.401 | 0.666 |
| import | kitchen | 7 | unrelated | unrelated | 0.016 | -0.001 |
| code | code | 0 | self | self | 1.000 | 1.000 |
| code | Code | 1 | case_swap | same_canonical | 0.677 | 0.750 |
| code | codes | 1 | morph_+s | morphological | 0.529 | 0.894 |
| code | coded | 1 | morph_+d | morphological | 0.369 | 0.894 |
| code | coding | 2 | morph_+ing | morphological | 0.366 | 0.612 |
| code | codec | 1 | morph_+c | morphological | 0.330 | 0.894 |
| code | decode | 2 | morph_prefix | morphological | 0.193 | -0.001 |
| code | encode | 2 | morph_prefix_2 | morphological | 0.253 | -0.001 |
| code | kitchen | 6 | unrelated | unrelated | 0.138 | -0.001 |

## Bucket means (across all 6 models)

| Bucket | Pairs | trained_cos (mean) | kron_cos (mean) |
|---|---:|---:|---:|
| same_canonical (typographic) | 36 | 0.553 | 0.657 |
| morphological | 126 | 0.382 | 0.743 |
| unrelated (control) | 84 | 0.152 | 0.430 |

## Spread (morphological − unrelated)

- **Trained-BPE** spread = +0.230 (morphological 0.382 − unrelated 0.152)
- **Kronecker** spread   = +0.313 (morphological 0.743 − unrelated 0.430)

Larger spread = method is more discriminating between meaningful morphological relatives and random unrelated strings.

## Same-canonical-form pairs (typographic variants)

| anchor | variant | trained_cos | kron_cos |
|---|---|---:|---:|
| run | Run | 0.657 | 0.667 |
| run | RUN | 0.494 | -0.000 |
| compute | Compute | 0.590 | 0.857 |
| nation | Nation | 0.455 | 0.833 |
| import | Import | 0.443 | 0.833 |
| code | Code | 0.677 | 0.750 |

**Trained-BPE mean cosine on same-canonical-form pairs: 0.553**
**Kronecker mean cosine on same-canonical-form pairs: 0.657**

## Morphological pairs

| anchor | variant | trained_cos | kron_cos |
|---|---|---:|---:|
| run | runs | 0.506 | 0.866 |
| run | ran | 0.324 | 0.667 |
| run | runner | 0.349 | 0.707 |
| run | running | 0.469 | 0.654 |
| compute | computes | 0.281 | 0.935 |
| compute | computed | 0.383 | 0.935 |
| compute | computer | 0.277 | 0.935 |
| compute | computing | 0.326 | 0.756 |
| compute | computation | 0.308 | 0.683 |
| nation | nations | 0.232 | 0.926 |
| nation | national | 0.344 | 0.866 |
| nation | nationally | 0.300 | 0.774 |
| import | imports | 0.453 | 0.926 |
| import | imported | 0.693 | 0.866 |
| import | importing | 0.735 | 0.816 |
| code | codes | 0.529 | 0.894 |
| code | coded | 0.369 | 0.894 |
| code | coding | 0.366 | 0.612 |
| code | codec | 0.330 | 0.894 |
| code | decode | 0.193 | -0.001 |
| code | encode | 0.253 | -0.001 |

**Trained-BPE mean cosine on morphological pairs: 0.382**
**Kronecker mean cosine on morphological pairs: 0.743**

## Unrelated pairs (control)

| anchor | variant | trained_cos | kron_cos |
|---|---|---:|---:|
| run | rune | 0.271 | 0.866 |
| run | road | 0.129 | 0.288 |
| run | kitchen | 0.070 | -0.001 |
| compute | commute | 0.124 | 0.857 |
| compute | compose | 0.192 | 0.714 |
| compute | kitchen | 0.041 | -0.001 |
| nation | station | 0.185 | -0.001 |
| nation | notion | 0.136 | 0.833 |
| nation | ration | 0.206 | 0.833 |
| nation | kitchen | 0.033 | 0.154 |
| import | important | 0.192 | 0.816 |
| import | export | 0.401 | 0.666 |
| import | kitchen | 0.016 | -0.001 |
| code | kitchen | 0.138 | -0.001 |

**Trained-BPE mean cosine on unrelated pairs: 0.152**
**Kronecker mean cosine on unrelated pairs: 0.430**
