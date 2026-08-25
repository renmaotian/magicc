#!/usr/bin/env bash
# =============================================================================
# Build both MAGICC container images (WS7.1 + WS7.2) from the repository root.
#
#   bash containers/build_containers.sh
#
# Produces:
#   docker image  magicc:0.3.1
#   containers/magicc_0.3.1.sif
#
# Both embed the frozen V5 ONNX model and fail to build if its SHA256 does not
# match results/revision/model_card.json.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="magicc:0.3.1"
SIF="${REPO}/containers/magicc_0.3.1.sif"
MODEL_SHA="b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096"

cd "${REPO}"

echo "== 1/3  verifying the frozen model on disk =="
actual="$(sha256sum models/magicc_v5.onnx | cut -d' ' -f1)"
[ "${actual}" = "${MODEL_SHA}" ] || { echo "FATAL: models/magicc_v5.onnx checksum mismatch"; exit 1; }
echo "   models/magicc_v5.onnx OK (${actual})"

echo "== 2/3  docker build =="
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t "${TAG}" .

echo "== 3/3  apptainer build (from the Docker image, no root required) =="
if command -v apptainer >/dev/null 2>&1; then
    apptainer build --force "${SIF}" "docker-daemon://${TAG}"
    apptainer inspect "${SIF}" | grep -i magicc || true
    echo "   SIF: ${SIF}"
    sha256sum "${SIF}"
elif command -v singularity >/dev/null 2>&1; then
    singularity build --force "${SIF}" "docker-daemon://${TAG}"
else
    cat >&2 <<'EOF'
   apptainer/singularity not found -- skipping the SIF.
   Install it (conda-forge provides an unprivileged build:
       mamba create -n apptainer -c conda-forge apptainer=1.4.5
   ) and re-run, or run the documented command by hand:
       apptainer build containers/magicc_0.3.1.sif docker-daemon://magicc:0.3.1
EOF
fi

echo "done."
