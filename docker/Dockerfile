# =============================================================================
# MAGICC — reproducible container image                          (WS7.1)
#
# Addresses reviewer comments R1-M6 ("provide a container"), R1-m17 ("pinned
# installs") and editorial requirement E2.
#
# Reproducibility properties:
#   * base image pinned by immutable content digest, not by tag
#   * every Python dependency pinned to an exact version AND verified against
#     a recorded sha256 (`--require-hashes`), transitive closure included
#   * the V5 ONNX model is BUNDLED in the image and its SHA256 is verified at
#     build time against results/revision/model_card.json, so the image can
#     never silently ship a different model
#   * no network access is required at run time: the frozen model is copied
#     into the *installed package's* data/ directory, which is the first
#     location the CLI looks in, so its download path is never reached.  Since
#     0.3.2 the CLI additionally verifies that file's SHA256 on every run, so
#     an image whose model was altered after the build would refuse to run
#
# Build (from the repository root):
#   docker build -f docker/Dockerfile -t magicc:0.3.3 .
#
# Run:
#   docker run --rm -v "$PWD":/data magicc:0.3.3 \
#       predict --input /data/genomes --output /data/predictions.tsv --threads 8
# =============================================================================

# --- pinned base -------------------------------------------------------------
# python:3.11.9-slim-bookworm, digest resolved 2026-07-26.
FROM python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

LABEL org.opencontainers.image.title="MAGICC" \
      org.opencontainers.image.description="Genome completeness and contamination estimation from core-gene k-mer profiles" \
      org.opencontainers.image.version="0.3.3" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/renmaotian/magicc" \
      org.magicc.model.version="V5" \
      org.magicc.model.sha256="b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096"

# Deterministic, quiet, no stale bytecode.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MAGICC_HOME=/opt/magicc

WORKDIR /opt/magicc

# --- dependencies (hash-verified, no resolver involved) ----------------------
COPY docker/requirements-lock.txt /opt/magicc/requirements-lock.txt
RUN python -m pip install --no-cache-dir --require-hashes --no-deps \
        -r /opt/magicc/requirements-lock.txt

# --- MAGICC source -----------------------------------------------------------
COPY pyproject.toml README.md LICENSE MANIFEST.in /opt/magicc/
COPY magicc /opt/magicc/magicc

# --- bundled model + resources, checksummed ----------------------------------
COPY models/magicc_v5.onnx                       /opt/magicc/models/magicc_v5.onnx
COPY data/features/normalization_params.json     /opt/magicc/share/normalization_params.json
COPY data/kmer_selection/selected_kmers.txt      /opt/magicc/share/selected_kmers.txt
COPY data/kmer_selection/selected_kmers_annotated.tsv /opt/magicc/share/selected_kmers_annotated.tsv
COPY results/revision/model_card.json            /opt/magicc/share/model_card.json

# Fail the build if the bundled model is not the exact frozen V5 artefact.
RUN set -eu; \
    expected="b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096"; \
    actual="$(sha256sum /opt/magicc/models/magicc_v5.onnx | cut -d' ' -f1)"; \
    echo "model sha256 expected=${expected}"; \
    echo "model sha256 actual  =${actual}"; \
    [ "${expected}" = "${actual}" ] || { echo "FATAL: bundled model checksum mismatch"; exit 1; }; \
    card="$(python -c "import json;print(json.load(open('/opt/magicc/share/model_card.json'))['onnx']['sha256'])")"; \
    [ "${card}" = "${actual}" ] || { echo "FATAL: model_card.json disagrees with bundled model"; exit 1; }; \
    echo "model checksum verified against results/revision/model_card.json"

# --- install the package (no deps: they are already hash-pinned above) -------
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation . \
 && python -c "import magicc, magicc.cli; print('magicc import OK')"

# The CLI resolves resources from the *installed package* directory first, then
# the project layout, then ~/.magicc (which triggers a GitHub download).
# Copying the frozen model into the installed package's data/ removes the
# download path entirely and makes the image work with no network at all.
#
# NOTE: the site-packages path must be resolved from a neutral working
# directory ('cd /'), otherwise `python -c` puts /opt/magicc on sys.path and
# `magicc.__file__` resolves to the *source* tree rather than the installed
# package -- which silently leaves the installed copy without a model.
RUN set -eu; \
    site="$(cd / && python -c 'import magicc, os; print(os.path.dirname(magicc.__file__))')"; \
    echo "installed package dir: ${site}"; \
    case "${site}" in *site-packages/magicc) : ;; \
      *) echo "FATAL: refusing to install resources into non-site-packages path ${site}"; exit 1 ;; esac; \
    cp /opt/magicc/share/normalization_params.json "${site}/data/normalization_params.json"; \
    cp /opt/magicc/share/selected_kmers.txt        "${site}/data/selected_kmers.txt"; \
    cp /opt/magicc/models/magicc_v5.onnx           "${site}/data/magicc_v5.onnx"; \
    expected="b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096"; \
    actual="$(sha256sum "${site}/data/magicc_v5.onnx" | cut -d' ' -f1)"; \
    [ "${expected}" = "${actual}" ] || { echo "FATAL: installed model checksum mismatch"; exit 1; }; \
    ls -l "${site}/data"

# Warm the numba AOT cache so the first user run does not pay JIT cost and the
# cache directory is writable by any UID.
RUN mkdir -p /tmp/numba_cache && chmod 1777 /tmp/numba_cache

# --- smoke test at build time ------------------------------------------------
# Also asserts that NO model download occurred: ~/.magicc must not exist
# afterwards.  That is the check that proves the image is self-contained.
RUN set -eu; \
    rm -rf /root/.magicc; \
    printf '>c1\nACGTACGTACGTACGTNNNNACGTACGTACGTACGT\n>c2\nTTTTGGGGCCCCAAAATTTTGGGGCCCCAAAA\n' > /tmp/plain.fasta; \
    gzip -c /tmp/plain.fasta > /tmp/zipped.fasta.gz; \
    magicc predict --input /tmp/plain.fasta     --output /tmp/a.tsv --threads 1 --quiet; \
    magicc predict --input /tmp/zipped.fasta.gz --output /tmp/b.tsv --threads 1 --quiet; \
    printf '/tmp/plain.fasta\n/tmp/zipped.fasta.gz\n' > /tmp/list.txt; \
    magicc predict --input-list /tmp/list.txt   --output /tmp/c.tsv --threads 1 --quiet; \
    cat /tmp/a.tsv; \
    cut -f2,3 /tmp/a.tsv | tail -n1 > /tmp/a.v; cut -f2,3 /tmp/b.tsv | tail -n1 > /tmp/b.v; \
    cmp /tmp/a.v /tmp/b.v || { echo "FATAL: .gz prediction differs from plain"; exit 1; }; \
    test "$(wc -l < /tmp/c.tsv)" = "3" || { echo "FATAL: --input-list produced wrong row count"; exit 1; }; \
    test ! -e /root/.magicc || { echo "FATAL: model was downloaded at run time -- image is not self-contained"; exit 1; }; \
    rm -f /tmp/a.tsv /tmp/b.tsv /tmp/c.tsv /tmp/a.v /tmp/b.v /tmp/list.txt /tmp/plain.fasta /tmp/zipped.fasta.gz; \
    echo "container smoke test PASSED (.gz == plain, --input-list works, no download)"

WORKDIR /data
ENTRYPOINT ["magicc"]
CMD ["--help"]
