# Bioconda recipe for MAGICC (WS7.10)

Addresses **R2-o5** ("please distribute via Bioconda").

**Status: prepared and validated locally. NOT SUBMITTED.**
Submitting to bioconda-recipes is the author's decision and has deliberately not
been done. Nothing here has been pushed to any remote.

## Layout

```
conda-recipe/
  magicc/
    meta.yaml        # the recipe, ready to drop into bioconda-recipes/recipes/magicc/
  README.md          # this file
```

## Source and checksum — verified, not copied from a template

The recipe builds the **PyPI sdist**.

| | |
|---|---|
| PyPI package name | **`magicc`** |
| Version | 0.3.1 |
| URL | `https://pypi.io/packages/source/m/magicc/magicc-0.3.1.tar.gz` |
| SHA256 | `aa36d2091487af864f264c1caed21ab67aa885cd6a062291629b7b3607e2926c` |

> **The PyPI distribution is named `magicc`.** There is no `magicc-genome`
> package — `https://pypi.org/pypi/magicc-genome/json` returns **404**. The
> install command is **`pip install magicc`**, and that is the name any
> citation or availability statement must use.

## Design decisions

* **`noarch: python`** — the package is pure Python; the only compiled code is
  Numba JIT, generated at run time. One build serves every platform.
* **Run dependencies mirror `pyproject.toml`** with lower bounds only. Exact
  pins belong in the container and the conda-lock file
  (`docker/requirements-lock.txt`, `conda/conda-lock.yml`), not in a
  distribution recipe that must co-solve with a user's other packages.
* **`run_exports` with `max_pin="x.x"`** so downstream recipes pinning against
  MAGICC get a sane compatibility range.
* **Tests do not run a prediction.** Bioconda's build workers have no network
  access, and the ~170 MB ONNX model is not in the sdist, so a prediction test
  would fail for reasons unrelated to packaging. The tests instead verify the
  entry point, the module imports, and that both shipped data resources are
  present and correct (the k-mer list must have exactly 9,249 lines).

## The model-download caveat, stated plainly

The sdist ships `selected_kmers.txt` and `normalization_params.json` but **not**
`magicc_v5.onnx` (169.7 MB), which exceeds what belongs in a conda package. On
first prediction the CLI downloads it to `~/.magicc/`.

For reproducible or offline use this is a real limitation, and it is why the
container images exist: `docker/Dockerfile` bakes the model in, verifies its
SHA256 at build time, and the build **fails** if a run-time download is
attempted. Reviewers wanting a hermetic install should prefer the container over
the conda package.

## Local validation performed

```bash
# 1. checksum verification against the published sdist
curl -sSL -o /tmp/magicc-0.3.1.tar.gz \
    https://pypi.io/packages/source/m/magicc/magicc-0.3.1.tar.gz
sha256sum /tmp/magicc-0.3.1.tar.gz   # must equal the SHA256 above

# 2. recipe render + lint
conda render conda-recipe/magicc
conda build --check conda-recipe/magicc
```

The 0.3.0 recipe was additionally cross-checked by `cmp`-ing the downloaded
sdist against the locally built one; the two were byte-identical.

## If and when the author decides to submit

1. Fork `bioconda/bioconda-recipes`.
2. Copy `conda-recipe/magicc/` to `recipes/magicc/`.
3. Replace the placeholder DOI in `extra.identifiers` with the published DOI.
4. Open a pull request; Bioconda CI builds and tests the recipe.

None of these steps have been performed.
