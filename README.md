# MAGICC

**Metagenome-Assembled Genome Inference of Completeness and Contamination**

Ultra-fast genome quality assessment using core gene k-mer profiles and deep learning.

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

### Dependencies

- Python >= 3.8
- numpy >= 1.20
- numba >= 0.53
- scipy >= 1.7
- h5py >= 3.0
- onnxruntime >= 1.10

## Usage

```bash
# Predict quality for all FASTA files in a directory (uses all CPUs by default)
magicc predict --input /path/to/genomes/ --output predictions.tsv

# Single genome
magicc predict --input genome.fasta --output predictions.tsv

# Specify threads and file extension
magicc predict --input /path/to/genomes/ --output predictions.tsv --threads 8 --extension .fa
```

### Options

```
magicc predict [OPTIONS]

Required:
  --input, -i       Path to genome FASTA file(s) or directory
  --output, -o      Output TSV file path

Optional:
  --threads, -t     Number of threads (default: 0 = all CPUs)
  --batch-size      Batch size for ONNX inference (default: 64)
  --extension, -x   Genome file extension filter (default: .fasta)
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

## Citation

If you use MAGICC in your research, please cite:

> Tian, R. (2026). MAGICC: Ultra-fast genome quality assessment using core gene k-mer profiles and deep learning. *In preparation*.

## License

MIT License. See [LICENSE](LICENSE) for details.
