# MAGICC reproduction workflow (WS7.4 / WS7.11)

Addresses **R1-M6** (workflow manager), **R1-M8** (reproducibility artifacts) and
**R1-m18** (one-command end-to-end reproduction).

## Input data (not in this repository)

The workflow scores **real benchmark genomes**, which are far too large for git.
Before running it you need, laid out under the repository root:

| Path | What | Where to get it |
|---|---|---|
| `data/benchmarks/{set_A_v2,set_B_v2,set_C_clean,set_D_clean,set_E}/fasta/` | 5,000 benchmark assemblies | figshare deposition (see the paper's Data availability) |
| `data/benchmarks/<set>/metadata.tsv` | per-sample ground truth | figshare, and mirrored in the [data repository](https://github.com/renmaotian/magicc-data) |

Everything else the workflow needs — the frozen model, the 9,249-k-mer list, the
normalisation parameters, the model card and the recorded metrics table it
checks against — **is in this repository** and is verified by checksum before
any prediction runs.

## One command

```bash
bash workflow/run_reproduction.sh
```

This regenerates MAGICC's **pooled leakage-free headline numbers** from raw
FASTA — model checksum verification, prediction on 5,000 benchmark genomes,
per-set and pooled statistics with cluster-bootstrap CIs, and an automatic
comparison against the recorded values. It exits non-zero if anything falls
outside tolerance.

Fast structural check (25 genomes per set, ~1 min):

```bash
bash workflow/run_reproduction.sh --smoke
```

A subsampled run is labelled `SUBSAMPLED_STRUCTURAL_ONLY` in the report and is
explicitly *not* a reproduction of the headline numbers.

## What is covered

| | |
|---|---|
| **Covered** | The MAGICC V5 arm of the headline benchmark: artefact verification → prediction on `set_A_v2`, `set_B_v2`, `set_C_clean`, `set_D_clean`, `set_E` → per-set MAE/bias/R²/RMSE → pooled statistics with a cluster bootstrap over reference genomes → comparison with `results/revision/metrics/ws5.5_table_S2_rebuilt.tsv`. |
| **Not covered** | Competitor tools (CheckM2, CoCoPyE, DeepCheck, GUNC), which need separate conda environments and reference databases totalling >20 GB; the GPU retraining; and the real-data workstreams. Their predictions are released as result files (figshare) and the metrics framework that recomputes every published statistic from them ships with the analysis code, not with this package. |
| **Deliberately excluded** | The submitted manuscript's Sets C and D. Their dominant genomes came from the training split (100 % of Set C, 90.3 % of Set D); those results are withdrawn, not reproduced. |

## Layout

```
workflow/
  Snakefile                     the DAG: verify -> predict -> evaluate -> pool -> report
  run_reproduction.sh           the one-command entry point
  config/config.yaml            every parameter that affects a result, incl. seeds
  scripts/verify_inputs.py      checksums the model, k-mer list and normalisation
  scripts/make_input_list.py    deterministic, natural-sorted genome list
  scripts/evaluate.py           MAE / bias / R² / RMSE + cluster bootstrap
  scripts/compare_and_report.py comparison with recorded values; writes the report
  envs/magicc.yaml              conda env for `--use-conda` runs
```

## Rules

| Rule | Jobs | Threads | Purpose |
|---|---|---|---|
| `verify_inputs` | 1 | 1 | fails immediately unless `models/magicc_v5.onnx`, the 9,249-k-mer list and the normalisation parameters all match `results/revision/model_card.json` |
| `predict` | 5 | `threads_per_job` (default 4) | `magicc predict --input-list …` per set |
| `evaluate` | 5 | 1 | per-set statistics |
| `pool` | 1 | 1 | pooled five-set statistics |
| `report` | 1 | 1 | comparison + `HEADLINE_REPRODUCTION.md` |

Per-set jobs mean a failure is localised and a rerun resumes rather than
restarting.

## Statistical conventions (not defaults — decisions)

* **R² is the coefficient of determination**, `1 − SS_res/SS_tot`. Squared
  Pearson r is computed separately and labelled `r_pearson_sq`; it is never
  called R². (Protocol §4.4d — the submitted manuscript mixed the two.)
* **Cluster bootstrap** over reference genomes, 2,000 iterations, seed 20260726.
  Clusters are `(set, reference)` pairs, because benchmark sets reuse each
  reference across simulations — `set_C_clean` and `set_D_clean` have only 100
  references for 1,000 genomes. A naive i.i.d. bootstrap achieves 54 % coverage
  where the cluster bootstrap achieves 95 % (WS5.4).
* **R² is omitted** where the true value has zero variance (R1-m19).

## Resources and runtime

Default `-j 1` with `threads_per_job: 4` — deliberately conservative. Raise with:

```bash
bash workflow/run_reproduction.sh -j 2 --config threads_per_job=16
```

Full run: 5,000 genomes.

## Environments

The workflow calls `magicc` inside the conda env named by `magicc_env` in
`config/config.yaml` (default `magicc2`). For a hermetic run:

```bash
# option A -- locked conda environment
conda-lock install --name magicc conda/conda-lock.yml
pip install --no-deps .
snakemake -s workflow/Snakefile --configfile workflow/config/config.yaml \
          -j 1 --config magicc_env=magicc

# option B -- snakemake manages the environment
snakemake -s workflow/Snakefile --configfile workflow/config/config.yaml \
          -j 1 --use-conda --config magicc_env=""

# option C -- container (most hermetic; model baked in and checksummed)
docker run --rm --network none -v "$PWD":/data magicc:0.3.1 \
    predict --input /data/data/benchmarks/set_E/fasta --output /data/pred.tsv
```

## Outputs

```
results/revision/reproducibility/workflow/
  checks/inputs_verified.json      artefact checksum results
  predictions/{set}.tsv            predictions, one file per set
  predictions/{set}.inputs.txt     the exact genome list scored
  metrics/{set}.json               per-set statistics
  metrics/pooled_leakage_free.json pooled statistics
  logs/predict_{set}.log           full CLI logs
  HEADLINE_REPRODUCTION.md         the report
  reproduction_summary.json        machine-readable summary
```
