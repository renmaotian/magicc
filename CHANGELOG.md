# Changelog

All notable changes to MAGICC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] — 2026-08-27

**Repairs a regression that 0.3.2 introduced, found by 0.3.2's own acceptance
test within the hour and fixed before anything else depended on it.**
0.3.3 is otherwise identical to 0.3.2; use it in preference to 0.3.2.

### Fixed

- **0.3.2 refused to start in an install whose `magicc` package was shadowed
  by a leftover directory.** To build the release-asset URL, 0.3.2 introduced
  `from magicc import __version__` at the top of `magicc/cli.py`. If an empty
  `site-packages/magicc/` directory survives an uninstall — which it does
  whenever Numba has written its `__pycache__/*.nbc` cache there, i.e. after
  any prediction, and is then followed by `pip install -e .` — Python treats
  `magicc` as a *namespace* package that shadows the real one (defect D5).
  In that state the import raises and **0.3.2 exits with
  `ImportError: cannot import name '__version__' from 'magicc' (unknown
  location)`** before doing anything. 0.3.1 and earlier kept working, so this
  was a regression, and the message told a user nothing actionable.

  0.3.3 resolves the version defensively: the normal import first, then by
  reading `__version__` out of the `__init__.py` sitting **beside `cli.py`**
  (always the real one, whatever `import magicc` resolves to), then a literal
  that `tests/test_model_integrity.py` pins to `magicc.__version__` so it
  cannot drift. When a fallback is used, MAGICC prints a warning naming the
  likely cause and the repair (`pip uninstall magicc`, delete the leftover
  `site-packages/magicc/`, reinstall) and **carries on** rather than dying.

  The model URL is still built from the resolved version, so it still points at
  the immutable release asset for the running version.

- 4 further regression tests (suite: **96**).

### Notes

- `0.3.2` remains on PyPI and as a GitHub release; it is correct for every
  normal install and its release asset is the same file. It is superseded
  rather than yanked, and 0.3.3 supersedes it for all purposes.

## [0.3.2] — 2026-08-27

**A reproducibility defect in how the model was obtained, stated plainly.**

The model is unchanged. **V5 is still the released model**, SHA256
`b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`, and every
prediction this release makes is byte-identical to one made by 0.3.0 or 0.3.1
on the same input — verified on a 25-genome panel across plain FASTA, gzip,
`--input-list` and `--extension auto` at 1 and 4 threads. Only the way the
model is *fetched and validated* has changed.

### Fixed

- **The model was downloaded from a mutable branch URL, without checksum
  verification.** 0.3.0 and 0.3.1 do not ship the 162 MB ONNX model in the
  PyPI wheel or sdist (it exceeds PyPI's per-file limit), so on first run the
  CLI fetched it from
  `https://github.com/renmaotian/magicc/raw/main/models/magicc_v5.onnx`.
  That reference is the **`main` branch**, not a tag: any later commit touching
  the model would silently change what an *already released* version of MAGICC
  downloads. The only validation performed was `size > 1 MB`, so a wrong,
  truncated or substituted model would have been used without complaint. An
  install advertised as reproducible therefore depended on a mutable remote
  artefact.

  **0.3.2 fetches the immutable release asset of its own version**,
  `https://github.com/renmaotian/magicc/releases/download/v0.3.2/magicc_v5.onnx`
  — a GitHub release asset, frozen once published — and **verifies SHA256
  against `b84346…b3096` before the model is used**, failing with an error that
  names the expected digest, the observed digest and the offending file. The
  expected hash lives in exactly one place, `magicc.cli.MODEL_SHA256`.

  This was the packaged-CLI half of the defect fixed inside the Docker image in
  0.3.1: the image was made self-contained, but `pip install magicc` was not.

### Added

- **The checksum is verified on every run, not only on first download.** A
  model found in the installed package, in the project layout, or already in
  `~/.magicc` is hashed before it is loaded, so a cache that is corrupted or
  altered after it was written is caught on the next invocation rather than
  never. Cost: ~0.4 s per run on the reference host.
- **A download announcement.** The first fetch prints one line naming the file,
  its size, the URL it comes from, the cache path it goes to, and the SHA256 it
  will be checked against — the download is no longer silent.
- **An actionable offline error.** If the download fails, MAGICC reports the
  URL, the cache path (`~/.magicc/magicc_v5.onnx`), the reason, the expected
  size and hash, and the two ways to supply the file by hand (copy it to the
  cache path, or pass `--model`). It exits 1 with that message instead of an
  unhandled `URLError` traceback. `README.md` documents the same route.
- **Atomic installation of a downloaded model.** The transfer lands on a
  temporary file beside the cache path and is renamed into place only after the
  checksum matches, so an interrupted or rejected download cannot leave a file
  that a later run would treat as a valid cache.
- **15 regression tests** (`tests/test_model_integrity.py`) covering the URL
  form (asserting it is a release asset and not a branch ref), the digest
  helpers, cached-model verification, corrupted-cache rejection, download and
  verification over `file://`, rejection of a wrong artefact with nothing left
  behind, and the offline error's contents. The suite grows to **92 tests**.

### Changed

- `--model` still accepts an explicit path and is deliberately **not**
  checksum-verified, so alternative models (e.g. the taxonomic-holdout models)
  can be evaluated on purpose. Its `--help` text now says so.
- Container image tags and labels move from `0.3.1` to `0.3.2`. The images were
  already immune to this defect — they bundle the model — and their build-time
  smoke test still asserts that no download occurs.

## [0.3.1] — 2026-08-25

The model is unchanged. **V5 is still the released model**, SHA256
`b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`, and every
prediction this release makes is bit-identical to one made by 0.3.0 on the same
input. This release adds input handling, tests, and the reproducibility and
distribution machinery that the accompanying manuscript describes — none of
which was present in the 0.3.0 distribution.

### Added

- **Gzip-compressed FASTA input**, accepted everywhere a plain FASTA is:
  `.fasta`, `.fa`, `.fna`, `.fas`, `.ffn` and their `.gz` forms. Compression is
  detected from file contents rather than the file name, so `X.fasta` and
  `X.fasta.gz` produce identical predictions **and the same `genome_name`** in
  the output. Verified byte-identical across 100 real assemblies.
- **`--input-list` / `-I`**: score an arbitrary set of genomes listed one path
  per line. Blank lines and `#` comments are ignored; absolute and relative
  paths may be mixed (relative paths resolve against the working directory,
  then against the directory holding the list); `~` is expanded; duplicates are
  dropped; every missing path is reported with its line number *before* the run
  starts; output row order follows the list. Mutually exclusive with `--input`.
- **`--extension auto`**: match any recognised FASTA extension, compressed or
  not, in a directory that mixes them. An explicit `--extension` now also
  matches the corresponding `.gz` form (`--extension .fasta` picks up both
  `*.fasta` and `*.fasta.gz`).
- **Test suite** (`tests/`, 77 tests) covering FASTA I/O, gzip equivalence,
  input discovery and CLI equivalence across input modes, run through the real
  installed console script.
- **Snakemake reproduction workflow** (`workflow/`). One command,
  `bash workflow/run_reproduction.sh`, verifies the frozen artefacts by
  checksum, predicts on the five leakage-free benchmark sets, computes per-set
  and pooled statistics with a cluster bootstrap, and compares them against the
  recorded values — exiting non-zero if any reproduced value falls outside the
  configured tolerance. Every parameter that affects a result, including seeds
  and the tolerance, is in `workflow/config/config.yaml`. `--smoke` runs a fast
  structural check on 25 genomes per set, labelled as such and explicitly not a
  reproduction of the headline numbers.
- **Docker image definition** (`docker/`): base image pinned by immutable
  content digest, all 13 Python dependencies pinned to exact versions *and*
  verified against recorded SHA256 hashes (`--require-hashes`), the V5 model
  baked in and checksummed twice at build time (against a literal hash and
  against `results/revision/model_card.json`), and a build-time smoke test that
  asserts `.gz == plain`, that `--input-list` works, and that **no model
  download occurred** — so the image is provably self-contained and runs with
  `--network none`.
- **Apptainer/Singularity definition** (`containers/magicc.def`) built from the
  same Docker image, plus `containers/build_containers.sh`.
- **Conda specification** (`conda/`): `environment.yml` plus a `conda-lock.yml`
  and fully resolved explicit locks for `linux-64`, `osx-64` and `osx-arm64`.
- **Bioconda recipe** (`conda-recipe/magicc/meta.yaml`), `noarch: python`,
  built and tested locally. Submission to bioconda-recipes remains an author
  decision and has not been made.
- **`CHANGELOG.md`** (this file).
- Small artefacts required for a clean clone to work unaided: the 9,249-k-mer
  list and its annotation, the normalisation parameters, the model card, the
  recorded metrics table the workflow checks against, and the determinism
  statement.

### Fixed

- The CLI silently downloaded the ONNX model from GitHub when no local copy was
  found, including inside containers that were supposed to be hermetic. The
  container images now bake the model into the installed package so the
  download path is never reached, and the image build fails if it is.
- An empty `site-packages/magicc/` namespace directory could shadow the
  installed package after an editable install.
- `conda/environment.yml` documented the install as
  `pip install magicc-genome`. **No such package exists on PyPI**; the
  distribution is and always has been `magicc`.

### Changed

- Container image tags and labels move from `0.3.0` to `0.3.1`.
- `README.md` documents compressed input, `--input-list`, `--extension auto`,
  the reproduction workflow, the container and conda routes, and states plainly
  which artefacts live in this repository and which are distributed elsewhere.

### Notes and limitations

- **Training is not seeded.** The released V5 weights cannot be re-derived
  bit-exactly; they are pinned by content hash instead. *Inference* is
  deterministic — identical across thread counts (1/2/4), batch sizes
  (1/64/512), plain vs gzip input, directory vs list input, repeated runs, and
  independently built environments (host, Docker, conda-lock). See
  `results/revision/reproducibility/DETERMINISM.md`.
- **Apptainer: build validated, execution not.** The SIF builds rootless from
  the Docker image and inspects correctly, but could not be executed on the
  development host, which forbids unprivileged user namespaces and has no
  setuid `starter-suid`. This is stated rather than glossed over.
- The reproduction workflow covers the **MAGICC arm only**. Reproducing the
  comparator tools needs separate environments and >20 GB of reference
  databases; their per-genome predictions are released as result files instead.

## [0.3.0] — 2026-03-19

- Released model **V5** (SHA256
  `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`),
  replacing V4.

## [0.2.1] — 2026-03-14

- Packaging fixes.

## [0.2.0] — 2026-03-14

- Model V4: assembly statistics removed; seven k-mer summary features retained.

## [0.1.0] — 2026-02-19

- First public release (model V3).

[0.3.3]: https://github.com/renmaotian/magicc/releases/tag/v0.3.3
[0.3.2]: https://github.com/renmaotian/magicc/releases/tag/v0.3.2
[0.3.1]: https://github.com/renmaotian/magicc/releases/tag/v0.3.1
[0.3.0]: https://pypi.org/project/magicc/0.3.0/
[0.2.1]: https://pypi.org/project/magicc/0.2.1/
[0.2.0]: https://pypi.org/project/magicc/0.2.0/
[0.1.0]: https://pypi.org/project/magicc/0.1.0/
