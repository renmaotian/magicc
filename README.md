# MAGICC

**Metagenome-Assembled Genome Inference of Completeness and Contamination**

Ultra-fast genome quality assessment using core gene k-mer profiles and deep learning.

## Overview

MAGICC predicts **completeness** and **contamination** of metagenome-assembled genomes (MAGs) using a multi-modal deep neural network trained on 1,000,000 synthetic genomes. It combines:

- **9,249 core gene k-mer features** (canonical 9-mers from bacterial and archaeal core genes)
- **26 assembly statistics** (contig metrics, GC composition, k-mer summary statistics)

MAGICC achieves **~1,700x faster** processing per thread compared to CheckM2, while providing competitive accuracy and superior contamination detection -- particularly for cross-phylum contamination, chimeric assemblies, and underrepresented taxa.

## Key Results

| Tool | Comp MAE | Cont MAE | Speed (genomes/min/thread) |
|------|----------|----------|---------------------------|
| **MAGICC** | **2.79%** | **4.04%** | **~1,700** |
| CheckM2 | 6.08% | 27.71% | ~1.0 |
| CoCoPyE | 7.98% | 22.02% | ~0.8 |
| DeepCheck | 10.36% | 31.12% | ~0.8* |

*Evaluated on 3,200 benchmark genomes across controlled completeness/contamination gradients, Patescibacteria, and Archaea.*

## Installation

### From PyPI

```bash
pip install magicc-genome
```

### From source

```bash
git clone https://github.com/renmaotian/magicc.git
cd magicc
pip install -e .
```

**Note**: Git LFS is required to clone the repository (the ONNX model is ~180 MB). Install Git LFS first:

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Then initialize
git lfs install
```

### Dependencies

- Python >= 3.8
- numpy >= 1.20
- numba >= 0.53
- scipy >= 1.7
- h5py >= 3.0
- onnxruntime >= 1.10

## Quick Start

### Command Line

```bash
# Predict quality for all FASTA files in a directory
magicc predict --input /path/to/genomes/ --output predictions.tsv

# Single genome
magicc predict --input genome.fasta --output predictions.tsv

# Multi-threaded feature extraction
magicc predict --input /path/to/genomes/ --output predictions.tsv --threads 8

# Specify file extension
magicc predict --input /path/to/genomes/ --output predictions.tsv --extension .fa
```

### Python Module

```bash
python -m magicc predict --input /path/to/genomes/ --output predictions.tsv
```

### Output Format

The output is a tab-separated file with three columns:

| genome_name | pred_completeness | pred_contamination |
|-------------|-------------------|-------------------|
| genome_001  | 95.2341           | 2.1567            |
| genome_002  | 78.4521           | 15.3421           |

- **pred_completeness**: Predicted completeness (%) -- range [50, 100]
- **pred_contamination**: Predicted contamination (%) -- range [0, 100]

## CLI Options

```
magicc predict [OPTIONS]

Required:
  --input, -i       Path to genome FASTA file(s) or directory
  --output, -o      Output TSV file path

Optional:
  --threads, -t     Number of threads for feature extraction (default: 1)
  --batch-size      Batch size for ONNX inference (default: 64)
  --extension, -x   Genome file extension filter (default: .fasta)
  --model           Path to ONNX model file
  --normalization   Path to normalization parameters JSON
  --kmers           Path to selected k-mers file
  --quiet, -q       Suppress progress output
  --verbose, -v     Verbose debug output
```

## How It Works

### Feature Extraction

1. **K-mer counting**: For each genome, MAGICC counts occurrences of 9,249 pre-selected canonical 9-mers derived from bacterial (85 BCG) and archaeal (128 UACG) core genes. Raw counts (not frequencies) are used because counts reflect gene copy number and completeness.

2. **Assembly statistics**: 26 features computed in a single pass -- contig length metrics (N50, L50, etc.), GC composition statistics, distributional features (GC bimodality, outlier fraction), and k-mer summary statistics.

3. **Normalization**: K-mer features undergo log(count+1) transformation followed by Z-score standardization. Assembly statistics use feature-appropriate normalization (log10, min-max, or robust scaling).

### Neural Network Architecture

MAGICC uses a multi-branch fusion network with Squeeze-and-Excitation (SE) attention:

- **K-mer branch**: 9,249 -> 4,096 -> 1,024 -> 256 (with SE attention blocks)
- **Assembly branch**: 26 -> 128 -> 64
- **Cross-attention fusion**: Assembly features attend to k-mer embeddings
- **Output**: Completeness [50-100%] and contamination [0-100%]

The model was trained on 1,000,000 synthetic genomes (800K train / 100K validation / 100K test) derived from 100,000 high-quality GTDB reference genomes, with realistic fragmentation patterns and contamination scenarios spanning 0-100%.

### Speed

| Step | Time per genome | 100K genomes (1 thread) |
|------|----------------|------------------------|
| K-mer counting | ~18 ms | ~30 min |
| Assembly stats | ~20 ms | ~33 min |
| Normalization | ~0.25 ms | ~25 sec |
| ONNX inference | ~0.18 ms | ~18 sec |
| **Total** | **~38 ms** | **~63 min** |

## Benchmark Data

This repository includes benchmark datasets and predictions for reproducibility:

- **Motivating analysis** (`data/benchmarks/motivating_v2/`): Sets A, B, C demonstrating limitations of existing tools
- **Benchmark sets** (`data/benchmarks/set_{A_v2,B_v2,C,D,E}/`): Controlled completeness/contamination gradients, Patescibacteria, Archaea, and mixed genomes
- **100K test evaluation** (`data/benchmarks/test_100k/`): Large-scale test set results

Each set includes metadata (true labels) and prediction TSV files from MAGICC, CheckM2, CoCoPyE, and DeepCheck.

## Training Data

The model was trained on synthetic genomes with the following composition:
- 15% pure genomes (50-100% completeness, 0% contamination)
- 15% complete genomes (100% completeness, 0-100% contamination)
- 30% within-phylum contamination (1-3 contaminant genomes from same phylum)
- 30% cross-phylum contamination (1-5 contaminant genomes from different phyla)
- 5% reduced genome organisms (Patescibacteria, DPANN, symbionts)
- 5% archaeal genomes

## Repository Structure

```
magicc/                     # Python package
  cli.py                    # Command-line interface
  kmer_counter.py           # Numba-accelerated k-mer counting
  assembly_stats.py         # Assembly statistics computation
  normalization.py          # Feature normalization
  model.py                  # Neural network architecture (PyTorch)
  pipeline.py               # Integrated pipeline
  data/                     # Bundled model and data files
    magicc_v3.onnx          # ONNX model (179.5 MB)
    selected_kmers.txt      # 9,249 selected k-mers
    normalization_params.json
models/
  magicc_v3.onnx            # ONNX model (same as above)
data/
  kmer_selection/
    selected_kmers.txt      # 9,249 canonical 9-mers
  features/
    normalization_params.json
  benchmarks/               # Benchmark datasets (metadata + predictions)
scripts/                    # Analysis and benchmarking scripts
results/                    # Accuracy metrics and figures
manuscript/                 # Manuscript figures
```

## Citation

If you use MAGICC in your research, please cite:

> Ren, M. (2026). MAGICC: Ultra-fast genome quality assessment using core gene k-mer profiles and deep learning. *In preparation*.

## License

MIT License. See [LICENSE](LICENSE) for details.
