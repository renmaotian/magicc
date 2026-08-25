"""
WS7.8 -- unit tests for compressed/uncompressed FASTA reading in magicc.cli.

Covers:
  * gzip and plain reading return byte-identical contig lists
  * FASTA validation on gzip input
  * empty files, header-only files, single-contig genomes
  * genome-name derivation (``X.fasta``, ``X.fasta.gz``, ``X.fna.gz`` -> ``X``)
  * magicc.fragmentation.read_fasta / load_original_contigs gz round-trip
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from magicc import cli
from magicc.fragmentation import load_original_contigs, read_fasta


# ---------------------------------------------------------------------------
# read_fasta_contigs: gz vs plain
# ---------------------------------------------------------------------------
def test_read_contigs_plain(tiny_fasta):
    contigs = cli.read_fasta_contigs(str(tiny_fasta))
    assert contigs == ['ACGTACGTACGTNNNNACGT', 'TTTTGGGGCCCC']


def test_read_contigs_gz_matches_plain(tiny_fasta, tiny_fasta_gz):
    assert cli.read_fasta_contigs(str(tiny_fasta_gz)) == cli.read_fasta_contigs(str(tiny_fasta))


def test_read_contigs_real_genome_gz_matches_plain(plain_dir, gz_dir):
    """Full-size real genome: gz and plain must yield identical contigs."""
    for plain in sorted(plain_dir.glob('*.fasta')):
        gz = gz_dir / (plain.name + '.gz')
        assert gz.is_file()
        a = cli.read_fasta_contigs(str(plain))
        b = cli.read_fasta_contigs(str(gz))
        assert a == b, f'contig mismatch for {plain.name}'
        assert len(a) > 0


def test_read_contigs_gz_without_gz_extension(tmp_path, tiny_fasta):
    """A gzip stream named ``*.fasta`` must still be read correctly (magic-byte sniff)."""
    disguised = tmp_path / 'disguised.fasta'
    with gzip.open(disguised, 'wb') as fout:
        fout.write(tiny_fasta.read_bytes())
    assert cli.read_fasta_contigs(str(disguised)) == cli.read_fasta_contigs(str(tiny_fasta))


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------
def test_read_contigs_empty_file(tmp_path):
    p = tmp_path / 'empty.fasta'
    p.write_text('')
    assert cli.read_fasta_contigs(str(p)) == []


def test_read_contigs_empty_gz(tmp_path):
    p = tmp_path / 'empty.fasta.gz'
    with gzip.open(p, 'wt') as f:
        f.write('')
    assert cli.read_fasta_contigs(str(p)) == []


def test_read_contigs_header_only(tmp_path):
    p = tmp_path / 'headers.fasta'
    p.write_text('>a\n>b\n')
    assert cli.read_fasta_contigs(str(p)) == []


def test_read_contigs_single_contig(tmp_path):
    p = tmp_path / 'one.fasta'
    p.write_text('>only\nACGTACGT\nACGT\n')
    assert cli.read_fasta_contigs(str(p)) == ['ACGTACGTACGT']


def test_read_contigs_no_trailing_newline(tmp_path):
    p = tmp_path / 'nonl.fasta'
    p.write_text('>x\nACGT')
    assert cli.read_fasta_contigs(str(p)) == ['ACGT']


def test_read_contigs_crlf(tmp_path):
    p = tmp_path / 'crlf.fasta'
    p.write_bytes(b'>x\r\nACGT\r\nGGTT\r\n')
    assert cli.read_fasta_contigs(str(p)) == ['ACGTGGTT']


def test_read_contigs_truncated_gz_raises(tmp_path, tiny_fasta):
    """A corrupt/truncated gzip file must not silently look like an empty genome."""
    good = tmp_path / 'good.fasta.gz'
    with gzip.open(good, 'wb') as fout:
        fout.write(tiny_fasta.read_bytes() * 200)
    truncated = tmp_path / 'trunc.fasta.gz'
    truncated.write_bytes(good.read_bytes()[:-40])
    with pytest.raises(cli.FastaReadError):
        cli.read_fasta_contigs(str(truncated))


def test_read_contigs_corrupt_gz_midstream_raises(tmp_path, tiny_fasta):
    """Bytes flipped in the middle of the deflate stream must raise, not truncate."""
    good = tmp_path / 'good.fasta.gz'
    with gzip.open(good, 'wb') as fout:
        fout.write(tiny_fasta.read_bytes() * 500)
    data = bytearray(good.read_bytes())
    mid = len(data) // 2
    for i in range(mid, mid + 32):
        data[i] ^= 0xFF
    corrupt = tmp_path / 'corrupt.fasta.gz'
    corrupt.write_bytes(bytes(data))
    with pytest.raises(cli.FastaReadError):
        cli.read_fasta_contigs(str(corrupt))


def test_read_contigs_missing_file_raises(tmp_path):
    with pytest.raises(cli.FastaReadError):
        cli.read_fasta_contigs(str(tmp_path / 'does_not_exist.fasta'))


def test_read_contigs_binary_input_raises(tmp_path):
    """Non-UTF8 binary input must be rejected, not decoded into a bogus contig."""
    p = tmp_path / 'binary.fasta'
    p.write_bytes(b'>hdr\n' + bytes(range(200, 256)) * 20 + b'\n')
    with pytest.raises(cli.FastaReadError):
        cli.read_fasta_contigs(str(p))


# ---------------------------------------------------------------------------
# validate_fasta
# ---------------------------------------------------------------------------
def test_validate_fasta_plain_and_gz(tiny_fasta, tiny_fasta_gz):
    assert cli.validate_fasta(str(tiny_fasta)) is True
    assert cli.validate_fasta(str(tiny_fasta_gz)) is True


def test_validate_fasta_rejects_non_fasta(tmp_path):
    p = tmp_path / 'notfasta.fasta'
    p.write_text('ACGT\nACGT\n')
    assert cli.validate_fasta(str(p)) is False


def test_validate_fasta_rejects_missing(tmp_path):
    assert cli.validate_fasta(str(tmp_path / 'nope.fasta')) is False


# ---------------------------------------------------------------------------
# Genome-name derivation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('filename,expected', [
    ('genome_0.fasta', 'genome_0'),
    ('genome_0.fasta.gz', 'genome_0'),
    ('genome_0.fa', 'genome_0'),
    ('genome_0.fa.gz', 'genome_0'),
    ('genome_0.fna', 'genome_0'),
    ('genome_0.fna.gz', 'genome_0'),
    ('GCA_000009265.1.fna', 'GCA_000009265.1'),
    ('GCA_000009265.1.fna.gz', 'GCA_000009265.1'),
    ('GCA_000009265.1.fasta.gz', 'GCA_000009265.1'),
    ('sample.FASTA.GZ', 'sample'),
    # No recognised FASTA extension: fall back to stripping the last suffix,
    # matching pre-existing (pathlib.Path.stem) behaviour.
    ('weird.txt', 'weird'),
    ('GCA_000009265.1_genomic.fna', 'GCA_000009265.1_genomic'),
])
def test_genome_name_from_path(filename, expected):
    assert cli.genome_name_from_path(filename) == expected
    assert cli.genome_name_from_path('/some/dir/' + filename) == expected


# ---------------------------------------------------------------------------
# magicc.fragmentation helpers (already gz-aware; assert it stays that way)
# ---------------------------------------------------------------------------
def test_fragmentation_read_fasta_gz_round_trip(tiny_fasta, tiny_fasta_gz):
    assert read_fasta(str(tiny_fasta_gz)) == read_fasta(str(tiny_fasta))
    assert load_original_contigs(str(tiny_fasta_gz)) == load_original_contigs(str(tiny_fasta))


def test_fragmentation_read_fasta_real_genome_gz(plain_dir, gz_dir):
    plain = sorted(plain_dir.glob('*.fasta'))[0]
    gz = gz_dir / (plain.name + '.gz')
    assert read_fasta(str(gz)) == read_fasta(str(plain))
    assert load_original_contigs(str(gz)) == load_original_contigs(str(plain))
