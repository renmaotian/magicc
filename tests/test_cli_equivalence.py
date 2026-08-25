"""
WS7.8 / WS7.9 / WS7.12 -- end-to-end equivalence and determinism tests.

These run the real MAGICC pipeline (k-mer counting + ONNX inference) on a
handful of real benchmark genomes, so they are the slowest tests in the suite
(~1-2 min total).  They assert:

  * gz round trip: predictions from ``X.fasta`` and ``X.fasta.gz`` are
    *bit-identical* in the output TSV
  * ``--input-list`` gives the same predictions as directory input
  * a mixed compressed/uncompressed list gives the same predictions
  * determinism: repeated runs at the same thread count are byte-identical
  * determinism: 1 thread vs 8 threads give identical predictions
  * empty / invalid genomes are skipped rather than crashing the run
  * a single-contig genome is handled

Run only the fast tests with:  pytest tests -m "not slow"
"""

from __future__ import annotations

import filecmp
import gzip
import hashlib
import shutil
from pathlib import Path

import pytest

from magicc import cli
from conftest import read_tsv, tsv_order

pytestmark = pytest.mark.slow


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope='module')
def run_predict(model_resources, tmp_path_factory):
    """Factory returning a function that runs cli.predict and returns the TSV path."""
    outdir = tmp_path_factory.mktemp('preds')
    counter = {'n': 0}

    def _run(tag: str, **kwargs) -> Path:
        counter['n'] += 1
        out = outdir / f'{counter["n"]:02d}_{tag}.tsv'
        kwargs.setdefault('threads', 4)
        cli.predict(output_path=str(out), **model_resources, **kwargs)
        return out

    return _run


# ---------------------------------------------------------------------------
# gz round-trip equivalence
# ---------------------------------------------------------------------------
def test_gz_roundtrip_identical_predictions(run_predict, plain_dir, gz_dir):
    plain = run_predict('plain_dir', input_path=str(plain_dir))
    gz = run_predict('gz_dir', input_path=str(gz_dir))
    assert tsv_order(plain) == tsv_order(gz)
    assert filecmp.cmp(plain, gz, shallow=False), (
        f'gz predictions differ from plain:\n{plain.read_text()}\n---\n{gz.read_text()}'
    )


def test_gz_roundtrip_single_file(run_predict, plain_dir, gz_dir):
    plain = run_predict('plain_one', input_path=str(plain_dir / 'genome_0.fasta'), threads=1)
    gz = run_predict('gz_one', input_path=str(gz_dir / 'genome_0.fasta.gz'), threads=1)
    assert filecmp.cmp(plain, gz, shallow=False)


def test_mixed_extension_dir_matches_plain(run_predict, plain_dir, mixed_dir):
    plain = run_predict('plain_for_mixed', input_path=str(plain_dir))
    mixed = run_predict('mixed_auto', input_path=str(mixed_dir), extension='auto')
    assert read_tsv(plain) == read_tsv(mixed)


# ---------------------------------------------------------------------------
# --input-list equivalence
# ---------------------------------------------------------------------------
def test_input_list_matches_directory(run_predict, plain_dir, tmp_path):
    lst = tmp_path / 'plain_list.txt'
    lst.write_text(
        '# genomes for the equivalence test\n\n'
        + '\n'.join(str(p) for p in sorted(plain_dir.glob('*.fasta'))) + '\n'
    )
    from_dir = run_predict('list_dir_ref', input_path=str(plain_dir))
    from_list = run_predict('from_list', input_list=str(lst))
    assert filecmp.cmp(from_dir, from_list, shallow=False)


def test_input_list_mixed_compression_matches_directory(run_predict, plain_dir, gz_dir, tmp_path):
    lst = tmp_path / 'mixed_list.txt'
    lst.write_text(
        f'{plain_dir / "genome_0.fasta"}\n'
        f'{gz_dir / "genome_1.fasta.gz"}\n'
        '# a comment in the middle\n'
        f'{plain_dir / "genome_2.fasta"}\n'
        '\n'
        f'{gz_dir / "genome_4.fasta.gz"}\n'
    )
    from_dir = run_predict('mixedlist_dir_ref', input_path=str(plain_dir))
    from_list = run_predict('mixed_list', input_list=str(lst))
    assert read_tsv(from_dir) == read_tsv(from_list)


def test_input_list_relative_paths(run_predict, plain_dir, monkeypatch, tmp_path):
    lst = tmp_path / 'rel_list.txt'
    lst.write_text('genome_0.fasta\ngenome_1.fasta\n')
    monkeypatch.chdir(plain_dir)
    out = run_predict('rel_list', input_list=str(lst), threads=1)
    assert tsv_order(out) == ['genome_0', 'genome_1']


# ---------------------------------------------------------------------------
# Determinism (WS7.12)
# ---------------------------------------------------------------------------
def test_determinism_repeat_same_threads(run_predict, plain_dir):
    a = run_predict('det_a', input_path=str(plain_dir), threads=4)
    b = run_predict('det_b', input_path=str(plain_dir), threads=4)
    assert sha256(a) == sha256(b)


def test_determinism_one_vs_many_threads(run_predict, plain_dir):
    t1 = run_predict('det_t1', input_path=str(plain_dir), threads=1)
    t8 = run_predict('det_t8', input_path=str(plain_dir), threads=8)
    assert sha256(t1) == sha256(t8), (
        f'thread count changed predictions:\n{t1.read_text()}\n---\n{t8.read_text()}'
    )


def test_determinism_batch_size_invariant(run_predict, plain_dir):
    b1 = run_predict('bs1', input_path=str(plain_dir), threads=1, batch_size=1)
    b64 = run_predict('bs64', input_path=str(plain_dir), threads=1, batch_size=64)
    assert read_tsv(b1) == read_tsv(b64)


# ---------------------------------------------------------------------------
# Degenerate genomes in a real run
# ---------------------------------------------------------------------------
def test_empty_and_single_contig_genomes(run_predict, plain_dir, tmp_path):
    d = tmp_path / 'degenerate'
    d.mkdir()
    shutil.copy2(plain_dir / 'genome_0.fasta', d / 'good.fasta')
    (d / 'empty.fasta').write_text('')
    with gzip.open(d / 'empty_gz.fasta.gz', 'wt') as f:
        f.write('')
    (d / 'single_contig.fasta').write_text('>c\n' + 'ACGTTGCA' * 5000 + '\n')
    out = run_predict('degenerate', input_path=str(d), threads=2)
    rows = read_tsv(out)
    assert set(rows) == {'good', 'single_contig'}
    for comp, cont in rows.values():
        assert 50.0 <= float(comp) <= 100.0
        assert 0.0 <= float(cont) <= 100.0


def test_all_genomes_invalid_raises(run_predict, tmp_path, model_resources):
    d = tmp_path / 'all_bad'
    d.mkdir()
    (d / 'a.fasta').write_text('')
    (d / 'b.fasta').write_text('>only_header\n')
    with pytest.raises(RuntimeError):
        cli.predict(input_path=str(d), output_path=str(tmp_path / 'x.tsv'),
                    threads=1, **model_resources)


def test_corrupt_gz_is_skipped_not_fatal(run_predict, plain_dir, tmp_path):
    d = tmp_path / 'with_corrupt'
    d.mkdir()
    shutil.copy2(plain_dir / 'genome_0.fasta', d / 'good.fasta')
    good_gz = d / 'broken.fasta.gz'
    with gzip.open(good_gz, 'wb') as f:
        f.write((plain_dir / 'genome_1.fasta').read_bytes())
    good_gz.write_bytes(good_gz.read_bytes()[:-2000])
    out = run_predict('corrupt', input_path=str(d), threads=2)
    assert set(read_tsv(out)) == {'good'}


def test_no_genomes_found_raises(tmp_path, model_resources):
    d = tmp_path / 'empty_dir'
    d.mkdir()
    with pytest.raises(RuntimeError) as exc:
        cli.predict(input_path=str(d), output_path=str(tmp_path / 'x.tsv'),
                    threads=1, **model_resources)
    assert 'No genome files found' in str(exc.value)


# ---------------------------------------------------------------------------
# CLI end-to-end via main()
# ---------------------------------------------------------------------------
def test_main_end_to_end_gz_and_list(tmp_path, gz_dir, model_resources):
    lst = tmp_path / 'l.txt'
    lst.write_text('\n'.join(str(p) for p in sorted(gz_dir.glob('*.gz'))) + '\n')
    out = tmp_path / 'main_out.tsv'
    cli.main([
        'predict', '--input-list', str(lst), '--output', str(out),
        '--threads', '2', '--quiet',
        '--model', model_resources['model_path'],
        '--normalization', model_resources['norm_path'],
        '--kmers', model_resources['kmer_path'],
    ])
    assert out.is_file()
    assert tsv_order(out) == ['genome_0', 'genome_1', 'genome_2', 'genome_4']
