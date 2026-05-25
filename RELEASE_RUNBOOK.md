# Release runbook

Literal commands to publish this staging tree on submission day. Run from the
**staging repo root** (`kronecker-embeddings-staging/`) unless noted otherwise.

The release process has four steps:

1. Fill in the arXiv ID across all placeholder locations.
2. Push the staging tree to the public GitHub repo.
3. Set the public repo's website field to the arXiv abstract URL.
4. *(Optional)* Publish the package to PyPI.

---

## 0. Prerequisites (one-time)

```bash
# Ensure gh CLI is authenticated to the org account that owns
# github.com/theschoolofai/kronecker-embeddings:
gh auth status

# Confirm you are on the right working tree:
pwd          # → .../kronecker-embeddings-staging
git status   # → clean (no uncommitted changes)
```

---

## 1. Fill in the arXiv ID

Replace the literal token `ARXIV_ID_PLACEHOLDER` with the real ID
(e.g. `2606.12345`) across the **exact 4 locations** below. The
release-day find-replace must catch every one.

```bash
# Verify the placeholder is present at exactly the expected locations:
grep -rn "ARXIV_ID_PLACEHOLDER" .
# Expected output (4 hits):
#   README.md:7:[![Paper](https://img.shields.io/badge/arXiv-ARXIV_ID_PLACEHOLDER-b31b1b.svg)](https://arxiv.org/abs/ARXIV_ID_PLACEHOLDER)
#   README.md:133:  journal={arXiv preprint arXiv:ARXIV_ID_PLACEHOLDER},
#   pyproject.toml:44:Paper = "https://arxiv.org/abs/ARXIV_ID_PLACEHOLDER"
#   RELEASE_RUNBOOK.md   (these instructions — ignore)

# Apply the replacement (set REAL_ID first):
REAL_ID="2606.12345"   # ← REPLACE WITH ACTUAL arXiv ID
sed -i.bak "s/ARXIV_ID_PLACEHOLDER/${REAL_ID}/g" README.md pyproject.toml
rm README.md.bak pyproject.toml.bak

# Verify zero placeholders remain (except in this runbook):
grep -rn "ARXIV_ID_PLACEHOLDER" --exclude="RELEASE_RUNBOOK.md" .
# Expected: no output.

# Commit:
git add README.md pyproject.toml
git commit -m "Set arXiv ID to ${REAL_ID}"
```

---

## 2. Push to the public repo

The public stub repo `github.com/theschoolofai/kronecker-embeddings` should
exist already with a placeholder README. We force the staged tree on top of
it (the placeholder README has no history worth preserving).

```bash
# Add the public repo as a remote:
git remote add origin git@github.com:theschoolofai/kronecker-embeddings.git

# Fetch + verify what's on the public side currently:
git fetch origin
git log --oneline origin/main | head -5      # → just the placeholder commit(s)

# Push staging history as the new main. The placeholder will be replaced.
# (--force is intentional and safe here: the public repo holds only a stub.)
git push --force origin main:main
```

If the public repo doesn't exist yet, create it first:

```bash
gh repo create theschoolofai/kronecker-embeddings \
    --public \
    --description "Byte-level structured token representations for parameter-efficient language models. Reference implementation." \
    --license apache-2.0 \
    --homepage "https://arxiv.org/abs/${REAL_ID}"
git remote add origin git@github.com:theschoolofai/kronecker-embeddings.git
git push -u origin main
```

---

## 3. Set the public repo's website / topics

```bash
gh repo edit theschoolofai/kronecker-embeddings \
    --homepage "https://arxiv.org/abs/${REAL_ID}" \
    --add-topic transformers \
    --add-topic embeddings \
    --add-topic byte-level \
    --add-topic language-models \
    --add-topic pytorch
```

---

## 4. *(Optional)* Publish to PyPI

Skip this step if you are not publishing the package on PyPI. If you do
publish, the HF model cards' `pip install kronecker-embeddings` snippet
becomes live.

```bash
# Build the sdist + wheel:
python -m pip install --upgrade build twine
python -m build
ls dist/   # → kronecker_embeddings-0.1.0.tar.gz, kronecker_embeddings-0.1.0-py3-none-any.whl

# Smoke-check on TestPyPI first (recommended):
twine upload --repository testpypi dist/*

# Real release:
twine upload dist/*
```

After PyPI publish, verify in a fresh venv:

```bash
python -m venv /tmp/pypi_check && /tmp/pypi_check/bin/pip install kronecker-embeddings
/tmp/pypi_check/bin/python -c "from kronecker_embeddings import KroneckerEmbedding; print('ok')"
```

---

## Post-release checklist

- [ ] Public repo's main branch contains the staged tree (4 placeholders → real ID).
- [ ] Public repo website field links to the arXiv abstract.
- [ ] HF model cards (separately maintained, depends on this package) have
      their own `ARXIV_ID_PLACEHOLDER` find-replace done. The same token
      `ARXIV_ID_PLACEHOLDER` is used by convention so a single grep across
      both surfaces catches every occurrence.
- [ ] `pip install kronecker-embeddings` works in a fresh venv *(if PyPI step ran)*.
