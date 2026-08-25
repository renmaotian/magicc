#!/usr/bin/env bash
# =============================================================================
# ONE-COMMAND END-TO-END REPRODUCTION of MAGICC's headline benchmark claims.
#                                                          (WS7.11 / R1-m18)
#
#   bash workflow/run_reproduction.sh              # full: 5,000 genomes
#   bash workflow/run_reproduction.sh --smoke      # fast structural test
#   bash workflow/run_reproduction.sh -j 2 --config threads_per_job=8
#
# Any additional arguments are passed straight through to snakemake.
#
# Requires: snakemake (>=7) on PATH, and a conda environment containing the
# `magicc` console script -- named by `magicc_env` in
# workflow/config/config.yaml (default: magicc2). Create it with:
#     conda-lock install --name magicc conda/conda-lock.yml
#     pip install --no-deps .
#
# Output: results/revision/reproducibility/workflow/HEADLINE_REPRODUCTION.md
# Exit status is non-zero if a reproduced value falls outside tolerance.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

EXTRA=()
SMOKE=0
for arg in "$@"; do
    if [[ "${arg}" == "--smoke" ]]; then SMOKE=1; else EXTRA+=("${arg}"); fi
done

if ! command -v snakemake >/dev/null 2>&1; then
    cat >&2 <<'EOF'
FATAL: snakemake is not on PATH.

    mamba create -n magicc_workflow -c conda-forge -c bioconda snakemake-minimal
    conda activate magicc_workflow

EOF
    exit 127
fi

echo "== MAGICC headline reproduction =="
echo "   repo      : ${REPO}"
echo "   snakemake : $(snakemake --version)"
[[ ${SMOKE} -eq 1 ]] && echo "   mode      : SMOKE (subsampled; NOT a reproduction of the headline numbers)"

CMD=(snakemake
     --snakefile "${REPO}/workflow/Snakefile"
     --configfile "${REPO}/workflow/config/config.yaml"
     --directory "${REPO}"
     --rerun-incomplete
     --printshellcmds
     -j 1)
[[ ${SMOKE} -eq 1 ]] && CMD+=(--config subsample=25)
[[ ${#EXTRA[@]} -gt 0 ]] && CMD+=("${EXTRA[@]}")

"${CMD[@]}"

REPORT="${REPO}/results/revision/reproducibility/workflow/HEADLINE_REPRODUCTION.md"
echo
echo "== report: ${REPORT}"
[[ -f "${REPORT}" ]] && sed -n '1,40p' "${REPORT}"
