"""
Shared pytest fixtures for the MAGICC test suite.

Fixtures build a temporary sandbox from a small number of *real* benchmark
genomes (``data/benchmarks/set_E/fasta/``).  The originals are never modified:
every fixture copies (or gzip-copies) into a pytest ``tmp_path_factory``
directory.

If the benchmark genomes are not present (e.g. a fresh clone without the data
repository), the genome-backed fixtures are skipped and synthetic FASTA files
are used instead where possible.
"""

from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
SET_E_FASTA_DIR = PROJECT_DIR / 'data' / 'benchmarks' / 'set_E' / 'fasta'
MODEL_PATH = PROJECT_DIR / 'models' / 'magicc_v5.onnx'
NORM_PATH = PROJECT_DIR / 'data' / 'features' / 'normalization_params.json'
KMER_PATH = PROJECT_DIR / 'data' / 'kmer_selection' / 'selected_kmers.txt'

# Small, deliberately heterogeneous set: few contigs, many contigs, mid-size.
FIXTURE_GENOMES = ['genome_0', 'genome_1', 'genome_2', 'genome_4']


def _require(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not available at {path}")


@pytest.fixture(scope='session')
def real_genome_paths() -> list[Path]:
    """Paths to the untouched original benchmark FASTA files."""
    _require(SET_E_FASTA_DIR, 'set_E benchmark FASTA directory')
    paths = [SET_E_FASTA_DIR / f'{g}.fasta' for g in FIXTURE_GENOMES]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f"missing benchmark genomes: {missing}")
    return paths


@pytest.fixture(scope='session')
def plain_dir(tmp_path_factory, real_genome_paths) -> Path:
    """Directory of uncompressed ``*.fasta`` copies of the fixture genomes."""
    d = tmp_path_factory.mktemp('plain_fasta')
    for src in real_genome_paths:
        shutil.copy2(src, d / src.name)
    return d


@pytest.fixture(scope='session')
def gz_dir(tmp_path_factory, real_genome_paths) -> Path:
    """Directory of gzip-compressed ``*.fasta.gz`` copies of the fixture genomes."""
    d = tmp_path_factory.mktemp('gz_fasta')
    for src in real_genome_paths:
        dest = d / (src.name + '.gz')
        with open(src, 'rb') as fin, gzip.open(dest, 'wb', compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout)
    return d


@pytest.fixture(scope='session')
def mixed_dir(tmp_path_factory, real_genome_paths) -> Path:
    """
    Directory mixing compressed and uncompressed files and several extensions:
    ``.fasta``, ``.fa.gz``, ``.fna``, ``.fasta.gz``.
    """
    d = tmp_path_factory.mktemp('mixed_fasta')
    layout = ['.fasta', '.fa.gz', '.fna', '.fasta.gz']
    for src, suffix in zip(real_genome_paths, layout):
        stem = src.stem
        dest = d / f'{stem}{suffix}'
        if suffix.endswith('.gz'):
            with open(src, 'rb') as fin, gzip.open(dest, 'wb', compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout)
        else:
            shutil.copy2(src, dest)
    return d


@pytest.fixture(scope='session')
def model_resources():
    """Paths to model / normalization / k-mer resources, skipping if unavailable."""
    for path, what in [(MODEL_PATH, 'ONNX model'),
                       (NORM_PATH, 'normalization params'),
                       (KMER_PATH, 'selected k-mer list')]:
        _require(path, what)
    return {
        'model_path': str(MODEL_PATH),
        'norm_path': str(NORM_PATH),
        'kmer_path': str(KMER_PATH),
    }


@pytest.fixture
def tiny_fasta(tmp_path) -> Path:
    """A minimal 2-contig FASTA with multi-line, mixed-case, and blank-line quirks."""
    p = tmp_path / 'tiny.fasta'
    p.write_text(
        '>c1 description here\n'
        'ACGTacgtACGT\n'
        'NNNNacgt\n'
        '\n'
        '>c2\n'
        'ttttGGGGcccc\n'
    )
    return p


@pytest.fixture
def tiny_fasta_gz(tiny_fasta) -> Path:
    """gzip copy of :func:`tiny_fasta`."""
    dest = Path(str(tiny_fasta) + '.gz')
    with open(tiny_fasta, 'rb') as fin, gzip.open(dest, 'wb') as fout:
        shutil.copyfileobj(fin, fout)
    return dest


def read_tsv(path) -> dict[str, tuple[str, str]]:
    """Read a MAGICC prediction TSV into {genome_name: (comp_str, cont_str)}."""
    rows = {}
    with open(path) as f:
        header = f.readline().rstrip('\n').split('\t')
        assert header == ['genome_name', 'pred_completeness', 'pred_contamination'], header
        for line in f:
            if not line.strip():
                continue
            name, comp, cont = line.rstrip('\n').split('\t')
            rows[name] = (comp, cont)
    return rows


def tsv_order(path) -> list[str]:
    """Genome names in file order."""
    with open(path) as f:
        f.readline()
        return [ln.split('\t')[0] for ln in f if ln.strip()]
