# Release runbook

**Current state (as of initial publication):** GitHub repo `github.com/theschoolofai/kronecker-embeddings` is public with the full reference implementation. PyPI package `kronecker-embeddings==0.1.0` is uploaded and `pip install kronecker-embeddings` works. The arXiv ID is not yet assigned; the README badge displays `IN_PROCESS`, and the BibTeX citation + `pyproject.toml` `Paper` URL carry the literal token `ARXIV_ID_PLACEHOLDER`.

**Remaining release-day actions** are limited to wiring the real arXiv ID through every public surface and setting the GitHub website field.

---

## Step 1 — Find-replace `ARXIV_ID_PLACEHOLDER` in the GitHub repo

The token appears in exactly **2 production locations** in the public repo:

```
README.md:<line>          journal={arXiv preprint arXiv:ARXIV_ID_PLACEHOLDER},
pyproject.toml:44         Paper = "https://arxiv.org/abs/ARXIV_ID_PLACEHOLDER"
```

(Verify with `grep -rn ARXIV_ID_PLACEHOLDER --exclude="RELEASE_RUNBOOK.md" .` — the runbook itself contains the token as documentation; exclude it from the find-replace.)

Run, with `REAL_ID` set to the assigned arXiv ID (e.g. `2606.12345`):

```bash
git clone git@github.com:theschoolofai/kronecker-embeddings.git
cd kronecker-embeddings

REAL_ID="2606.12345"   # ← replace with assigned ID
sed -i.bak "s/ARXIV_ID_PLACEHOLDER/${REAL_ID}/g" README.md pyproject.toml
rm README.md.bak pyproject.toml.bak

# Verify zero placeholders remain in production files:
grep -rn ARXIV_ID_PLACEHOLDER --exclude=RELEASE_RUNBOOK.md . 
# Expected: no output.
```

**Note on `pyproject.toml`:** the placeholder substitution here matters only for **future PyPI version bumps** (0.2.0, 0.1.1, etc.). PyPI 0.1.0 is already published with `ARXIV_ID_PLACEHOLDER` in its metadata; PyPI releases are **immutable per version** — there is no way to edit a published wheel's metadata. The published 0.1.0's `Paper` URL will permanently point to `arxiv.org/abs/ARXIV_ID_PLACEHOLDER`. This is an accepted known state. The live citation surface is the GitHub README (which we are fixing here); a future 0.1.1 release will carry the corrected URL.

---

## Step 2 — Update the README badge separately

The badge encodes its text in the URL, not as the literal token `ARXIV_ID_PLACEHOLDER`, so step 1's find-replace **does not catch it**. Update it manually.

The badge URL uses `IN__PROCESS` (double underscore) to render `IN_PROCESS` per shields.io's URL-encoding rules. Replace both pieces:

```bash
# In the cloned repo from step 1:
sed -i.bak \
  -e "s|arXiv-IN__PROCESS-b31b1b|arXiv-${REAL_ID}-b31b1b|g" \
  -e "s|](https://arxiv.org/)|](https://arxiv.org/abs/${REAL_ID})|g" \
  README.md
rm README.md.bak

# Also remove the "> **Status:** ..." line below the badges (the ID is now assigned):
sed -i.bak '/> \*\*Status:\*\* Accompanying paper submitted to arXiv; ID in process/d' README.md
rm README.md.bak

# Final verification:
grep -n "IN_PROCESS\|IN__PROCESS" README.md           # expect: no output
grep -n "ARXIV_ID_PLACEHOLDER" README.md pyproject.toml  # expect: no output
```

---

## Step 3 — Find-replace `ARXIV_ID_PLACEHOLDER` on the 4 HF model cards

The 4 Hugging Face model card repos that depend on this package use the **same `ARXIV_ID_PLACEHOLDER` token** by convention. A single find-replace per card:

```bash
# For each of the 4 dependent HF model card repos under theschoolofai/:
git clone git@hf.co:theschoolofai/<model>.git
cd <model>
sed -i.bak "s/ARXIV_ID_PLACEHOLDER/${REAL_ID}/g" README.md
rm README.md.bak
git add README.md
git commit -m "Set arXiv ID to ${REAL_ID}"
git push
cd ..
```

---

## Step 4 — Commit + push the GitHub edits

```bash
# In the kronecker-embeddings clone from step 1+2:
git add README.md pyproject.toml
git commit -m "Set arXiv ID to ${REAL_ID}; remove pre-assignment status line"
git push origin main
```

---

## Step 5 — Set the GitHub website field to the arXiv abstract URL

```bash
gh repo edit theschoolofai/kronecker-embeddings \
  --homepage "https://arxiv.org/abs/${REAL_ID}"
```

---

## Step 6 — Final verification

```bash
# From a clean directory, fresh-clone and verify:
git clone git@github.com:theschoolofai/kronecker-embeddings.git /tmp/verify_release
cd /tmp/verify_release

# Both must return zero output:
grep -rn ARXIV_ID_PLACEHOLDER --exclude-dir=.git .
grep -rn "IN_PROCESS\|IN__PROCESS" --exclude-dir=.git .

# Verify the real ID appears where expected:
grep -n "${REAL_ID}" README.md pyproject.toml

# Verify GitHub website field points to arXiv:
gh repo view theschoolofai/kronecker-embeddings --json homepageUrl
```

And visually:
- Visit `https://github.com/theschoolofai/kronecker-embeddings` — badge should read `arXiv: ${REAL_ID}`, click should land on the arXiv abstract page.
- Visit `https://pypi.org/project/kronecker-embeddings/` — note that 0.1.0's metadata still shows `arxiv.org/abs/ARXIV_ID_PLACEHOLDER` (immutable); plan a 0.1.1 release when convenient to update.

---

## Post-release checklist

- [ ] GitHub README badge reads `arXiv: <REAL_ID>` and links to the real abstract.
- [ ] GitHub README BibTeX uses the real ID.
- [ ] `pyproject.toml` `Paper` URL uses the real ID (for future PyPI version bumps).
- [ ] Status line about "ID in process" is removed from the README.
- [ ] 4 HF model cards have the real ID baked in.
- [ ] GitHub repo's website field links to the arXiv abstract.
- [ ] `grep ARXIV_ID_PLACEHOLDER` across the GitHub repo returns zero (excluding `RELEASE_RUNBOOK.md`).
- [ ] `grep IN_PROCESS` across the GitHub README returns zero.
- [ ] (Optional, when convenient) Publish PyPI 0.1.1 to update the `Paper` URL in the package metadata — current 0.1.0 retains the placeholder permanently.
