<p align="center">
  <img src="magicc_logo.png" alt="MAGICC logo" width="300">
</p>

# MAGICC

**Metagenome-Assembled Genome Inference of Completeness and Contamination**

Accurate and ultra-fast genome quality assessment using core gene k-mer profiles and deep learning.

## Installation

```bash
pip install magicc
```

Or from source:

```bash
git clone https://github.com/renmaotian/magicc.git
cd magicc
pip install -e .
```

**Note**: Git LFS is required to clone the repository (the ONNX model is ~180 MB).

### Container (recommended for reproducible or offline use)

The container bundles the frozen V5 model, so nothing is downloaded at run time
and the exact model is guaranteed by checksum.

```bash
# Docker
docker build -f docker/Dockerfile -t magicc:0.3.1 .
docker run --rm -v "$PWD":/data magicc:0.3.1 \
    predict --input /data/genomes --output /data/predictions.tsv --threads 8

# Apptainer / Singularity (HPC)
apptainer build magicc_0.3.1.sif docker-daemon://magicc:0.3.1
apptainer run --containall --bind "$PWD":/data magicc_0.3.1.sif \
    predict --input /data/genomes --output /data/predictions.tsv --threads 8
```

The base image is pinned by digest and every Python dependency is pinned to an
exact version and verified against a recorded SHA256
(`docker/requirements-lock.txt`). The build fails if the bundled model does not
match `results/revision/model_card.json`.

### Pinned conda environment

```bash
conda-lock install --name magicc conda/conda-lock.yml   # exact, solver-free
conda activate magicc && pip install --no-deps .
```

`conda/environment.yml` is the human-readable source;
`conda/conda-lock.yml` and `conda/conda-{linux-64,osx-64,osx-arm64}.lock` are the
fully resolved locks.

### Dependencies

- Python >= 3.8
- numpy >= 1.20
- numba >= 0.53
- scipy >= 1.7
- h5py >= 3.0
- onnxruntime >= 1.10

Exact versions used for the published results: `numpy 1.26.4`, `numba 0.63.1`,
`scipy 1.17.0`, `h5py 3.15.1`, `onnxruntime 1.23.2` on Python 3.11.

## Usage

```bash
# Predict quality for all FASTA files in a directory (uses all CPUs by default)
magicc predict --input /path/to/genomes/ --output predictions.tsv

# Single genome
magicc predict --input genome.fasta --output predictions.tsv

# Gzip-compressed input works everywhere a plain FASTA does
magicc predict --input genome.fasta.gz --output predictions.tsv

# Specify threads and file extension
magicc predict --input /path/to/genomes/ --output predictions.tsv --threads 8 --extension .fa

# Accept any recognised FASTA extension, compressed or not, in one directory
magicc predict --input /path/to/genomes/ --extension auto --output predictions.tsv

# Submit an arbitrary set of genomes listed in a text file
magicc predict --input-list genome_paths.txt --output predictions.tsv
```

### Compressed input

Plain-text and **gzip-compressed** FASTA are both accepted: `.fasta`, `.fa`,
`.fna`, `.fas`, `.ffn` and their `.gz` forms. Compression is detected from the
file contents, so `X.fasta` and `X.fasta.gz` yield **identical** predictions and
the same `genome_name` in the output.

For directory input, `--extension` also matches the corresponding gzip form —
the default `--extension .fasta` picks up both `*.fasta` and `*.fasta.gz`. Use
`--extension auto` when a directory mixes extensions.

### Genome list files

`--input-list` takes a text file with one genome path per line:

```
# Human gut MAGs, batch 3
/data/mags/MAG_0001.fa.gz
/data/mags/MAG_0002.fa.gz

relative/path/MAG_0003.fasta
~/genomes/MAG_0004.fna
```

- blank lines and lines starting with `#` are ignored
- compressed and uncompressed files may be mixed freely
- paths may be absolute or relative (resolved against the current directory,
  then against the directory containing the list file); `~` is expanded
- every missing path is reported with its line number before the run starts
- output rows follow the order of the list file (directory input is sorted by
  file name); duplicate paths are dropped

`--input` and `--input-list` are mutually exclusive.

### Options

```
magicc predict [OPTIONS]

Required (exactly one input):
  --input, -i       Path to a genome FASTA file (plain or .gz) or a directory
  --input-list, -I  Text file listing genome paths, one per line
Required:
  --output, -o      Output TSV file path

Optional:
  --threads, -t     Number of threads (default: 0 = all CPUs)
  --batch-size      Batch size for ONNX inference (default: 64)
  --extension, -x   Extension filter for directory input (default: .fasta).
                    Also matches the .gz form; use "auto" for any FASTA extension
  --model           Path to ONNX model file (auto-downloads if not found)
  --quiet, -q       Suppress progress output
  --verbose, -v     Verbose debug output
```

### Output

Tab-separated file with three columns:

| genome_name | pred_completeness | pred_contamination |
|-------------|-------------------|--------------------|
| genome_001  | 95.2341           | 2.1567             |
| genome_002  | 78.4521           | 15.3421            |

- **pred_completeness**: Predicted completeness (%), range [50, 100]
- **pred_contamination**: Predicted contamination (%), range [0, 100]

## Reproducibility

```bash
# Reproduce the headline benchmark numbers end to end, in one command
bash workflow/run_reproduction.sh

# Fast structural check of the workflow (25 genomes per set)
bash workflow/run_reproduction.sh --smoke

# Test suite (77 tests)
python -m pytest tests/ -q
```

`workflow/run_reproduction.sh` verifies the frozen artefacts by checksum,
predicts on the five leakage-free benchmark sets, computes per-set and pooled
statistics with a cluster bootstrap, and compares them against the recorded
values. **It exits non-zero if any reproduced value falls outside the tolerance
declared in `workflow/config/config.yaml`.** It needs the benchmark assemblies,
which are too large for git — see `workflow/README.md` and the paper's Data
availability statement.

| Artifact | Location |
|---|---|
| Determinism statement + machine-readable report | `results/revision/reproducibility/DETERMINISM.md`, `determinism_report.json` |
| Model card (architecture, training, limitations, checksums) | `results/revision/model_card.json` |
| Selected 9,249 9-mers, and the annotated list | `magicc/data/selected_kmers.txt`, `data/kmer_selection/selected_kmers_annotated.tsv` |
| Recorded metrics the reproduction is checked against | `results/revision/metrics/ws5.5_table_S2_rebuilt.tsv` |
| Snakemake workflow | `workflow/` |
| Version history | [`CHANGELOG.md`](CHANGELOG.md) |

**Note on the model artifact.** The released V5 weights
(SHA256 `b84346650ce21a66acd488e9f2eab1ca72333ba4dd50fed79070ec182b2b3096`) are
pinned by checksum rather than by seed: training was not seeded, so the weights
cannot be re-derived bit-exactly. Inference *is* deterministic — identical
across thread counts, batch sizes, input encodings and independently built
environments.

## What is in this repository, and what is not

This repository is the **software**: the installable package, the frozen V5
model, the tests, the reproduction workflow, and the container and conda
specifications. It is deliberately not the research workspace.

| | Where |
|---|---|
| Package, model, tests, workflow, containers, conda | here |
| Benchmark accession lists, split files, per-sample metadata, SHA256 manifests, the leakage audit, CAMI II identifiers and truth tables | [`renmaotian/magicc-data`](https://github.com/renmaotian/magicc-data) |
| Benchmark genome assemblies and the full per-genome result files | figshare (see the paper's Data availability statement) |
| Analysis and training scripts, figure code | released with the paper (see its Code availability statement) |

A few small files under `data/` and `results/` **are** tracked here, because a
clean clone cannot build the container or run the workflow without them: the
k-mer list and its annotation, the normalisation parameters, the model card, the
recorded metrics table, and the determinism statement. Nothing else under those
directories is published from this repository.

### Benchmark sets C and D

The benchmark sets named *C* and *D* in the originally submitted manuscript were
built from reference genomes that overlapped the training split (100 % of set C,
90.3 % of set D). **Those results are withdrawn.** They are replaced by
`set_C_clean` and `set_D_clean`, rebuilt from held-out test-split references,
and it is the clean sets that the reproduction workflow scores. See the data
repository for the full inventory and the provenance audit.

## Citation

If you use MAGICC in your research, please cite:

> Tian, R. (2026). MAGICC: Accurate and ultra-fast genome quality assessment using core gene k-mer profiles and deep learning. *In preparation*.

## License

MIT License. See [LICENSE](LICENSE) for details.
