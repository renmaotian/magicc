"""
WS7.8 / WS7.9 -- tests for genome discovery, ``--extension`` semantics and
``--input-list`` parsing.

Covers:
  * backward-compatible ``--extension`` behaviour (default ``.fasta``)
  * automatic acceptance of ``<extension>.gz``
  * ``--extension auto`` for mixed extensions
  * list-file parsing: comments, blank lines, whitespace, relative/absolute
    paths, mixed compressed/uncompressed
  * clear errors for missing files, empty lists, directories in a list
  * ``--input`` / ``--input-list`` mutual exclusion
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from magicc import cli


def names(pairs):
    return [n for n, _ in pairs]


# ---------------------------------------------------------------------------
# discover_genomes -- directory input
# ---------------------------------------------------------------------------
def test_discover_default_extension_plain(plain_dir):
    found = cli.discover_genomes(str(plain_dir), '.fasta')
    assert names(found) == ['genome_0', 'genome_1', 'genome_2', 'genome_4']


def test_discover_default_extension_finds_gz(gz_dir):
    """``--extension .fasta`` also matches ``*.fasta.gz`` (new in WS7.8)."""
    found = cli.discover_genomes(str(gz_dir), '.fasta')
    assert names(found) == ['genome_0', 'genome_1', 'genome_2', 'genome_4']
    assert all(p.endswith('.fasta.gz') for _, p in found)


def test_discover_explicit_gz_extension(gz_dir):
    found = cli.discover_genomes(str(gz_dir), '.fasta.gz')
    assert names(found) == ['genome_0', 'genome_1', 'genome_2', 'genome_4']


def test_discover_extension_without_leading_dot(plain_dir):
    assert cli.discover_genomes(str(plain_dir), 'fasta') == \
           cli.discover_genomes(str(plain_dir), '.fasta')


def test_discover_ncbi_compound_suffix(tmp_path):
    """NCBI-style '_genomic.fna' suffixes must match, plain and gzipped."""
    (tmp_path / 'GCA_000009265.1_ASM926v1_genomic.fna').write_text('>c\nACGT\n')
    with gzip.open(tmp_path / 'GCA_000009525.1_ASM952v1_genomic.fna.gz', 'wt') as f:
        f.write('>c\nACGT\n')
    (tmp_path / 'GCA_000010925.1_other.fna').write_text('>c\nACGT\n')
    found = cli.discover_genomes(str(tmp_path), '_genomic.fna')
    assert names(found) == ['GCA_000009265.1_ASM926v1_genomic',
                           'GCA_000009525.1_ASM952v1_genomic']


def test_discover_extension_is_still_a_filter(mixed_dir):
    """Backward compatibility: an explicit extension must not pick up other suffixes."""
    found = cli.discover_genomes(str(mixed_dir), '.fna')
    assert names(found) == ['genome_2']


def test_discover_auto_extension_mixed(mixed_dir):
    """``--extension auto`` accepts every known FASTA extension, gz or not."""
    found = cli.discover_genomes(str(mixed_dir), 'auto')
    assert names(found) == ['genome_0', 'genome_1', 'genome_2', 'genome_4']
    suffixes = sorted(Path(p).name.replace(n, '') for n, p in found)
    assert suffixes == ['.fa.gz', '.fasta', '.fasta.gz', '.fna']


def test_discover_auto_ignores_non_fasta(tmp_path):
    (tmp_path / 'a.fasta').write_text('>x\nACGT\n')
    (tmp_path / 'notes.txt').write_text('hello')
    (tmp_path / 'table.tsv').write_text('a\tb\n')
    (tmp_path / 'archive.tar.gz').write_bytes(b'\x1f\x8b\x00')
    assert names(cli.discover_genomes(str(tmp_path), 'auto')) == ['a']


def test_discover_single_file_plain_and_gz(plain_dir, gz_dir):
    p = plain_dir / 'genome_0.fasta'
    assert cli.discover_genomes(str(p), '.fasta') == [('genome_0', str(p))]
    g = gz_dir / 'genome_0.fasta.gz'
    assert cli.discover_genomes(str(g), '.fasta') == [('genome_0', str(g))]
    # A single explicit file is used regardless of the extension filter.
    assert cli.discover_genomes(str(g), '.fna') == [('genome_0', str(g))]


def test_discover_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        cli.discover_genomes(str(tmp_path / 'no_such_dir'), '.fasta')
    assert 'does not exist' in str(exc.value)


def test_discover_empty_dir_returns_empty(tmp_path):
    assert cli.discover_genomes(str(tmp_path), '.fasta') == []


def test_discover_skips_subdirectories(tmp_path):
    (tmp_path / 'a.fasta').write_text('>x\nACGT\n')
    (tmp_path / 'sub.fasta').mkdir()
    assert names(cli.discover_genomes(str(tmp_path), '.fasta')) == ['a']


def test_discover_duplicate_names_across_extensions(tmp_path):
    """``x.fasta`` and ``x.fasta.gz`` in one directory must both be reported."""
    (tmp_path / 'x.fasta').write_text('>c\nACGT\n')
    with gzip.open(tmp_path / 'x.fasta.gz', 'wt') as f:
        f.write('>c\nACGT\n')
    found = cli.discover_genomes(str(tmp_path), '.fasta')
    assert len(found) == 2
    assert names(found) == ['x', 'x']


# ---------------------------------------------------------------------------
# read_genome_list / discover from list file
# ---------------------------------------------------------------------------
def test_list_basic_absolute(tmp_path, plain_dir):
    lst = tmp_path / 'paths.txt'
    lst.write_text('\n'.join(str(p) for p in sorted(plain_dir.glob('*.fasta'))) + '\n')
    found = cli.discover_genomes_from_list(str(lst))
    assert names(found) == ['genome_0', 'genome_1', 'genome_2', 'genome_4']


def test_list_comments_blank_lines_and_whitespace(tmp_path, plain_dir):
    p0 = plain_dir / 'genome_0.fasta'
    p1 = plain_dir / 'genome_1.fasta'
    lst = tmp_path / 'paths.txt'
    lst.write_text(
        '# MAGICC genome list\n'
        '\n'
        f'  {p0}  \n'
        '\t\n'
        '   # indented comment\n'
        f'{p1}\n'
        '\n'
    )
    found = cli.discover_genomes_from_list(str(lst))
    assert found == [('genome_0', str(p0)), ('genome_1', str(p1))]


def test_list_preserves_file_order(tmp_path, plain_dir):
    order = ['genome_4', 'genome_0', 'genome_2', 'genome_1']
    lst = tmp_path / 'paths.txt'
    lst.write_text('\n'.join(str(plain_dir / f'{g}.fasta') for g in order) + '\n')
    assert names(cli.discover_genomes_from_list(str(lst))) == order


def test_list_mixed_compressed_and_plain(tmp_path, plain_dir, gz_dir):
    lst = tmp_path / 'paths.txt'
    lst.write_text(
        f'{plain_dir / "genome_0.fasta"}\n'
        f'{gz_dir / "genome_1.fasta.gz"}\n'
        f'{plain_dir / "genome_2.fasta"}\n'
        f'{gz_dir / "genome_4.fasta.gz"}\n'
    )
    found = cli.discover_genomes_from_list(str(lst))
    assert names(found) == ['genome_0', 'genome_1', 'genome_2', 'genome_4']
    assert [Path(p).name.endswith('.gz') for _, p in found] == [False, True, False, True]


def test_list_relative_to_cwd(tmp_path, plain_dir, monkeypatch):
    lst = tmp_path / 'paths.txt'
    lst.write_text('genome_0.fasta\ngenome_1.fasta\n')
    monkeypatch.chdir(plain_dir)
    found = cli.discover_genomes_from_list(str(lst))
    assert names(found) == ['genome_0', 'genome_1']
    assert all(Path(p).is_absolute() for _, p in found)


def test_list_relative_to_list_file_dir(tmp_path, plain_dir, monkeypatch):
    """Paths relative to the list file resolve even when CWD is elsewhere."""
    lst = plain_dir / 'paths.txt'
    lst.write_text('genome_0.fasta\ngenome_2.fasta\n')
    monkeypatch.chdir(tmp_path)
    found = cli.discover_genomes_from_list(str(lst))
    assert names(found) == ['genome_0', 'genome_2']


def test_list_tilde_expansion(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'g.fasta').write_text('>c\nACGT\n')
    monkeypatch.setenv('HOME', str(home))
    lst = tmp_path / 'paths.txt'
    lst.write_text('~/g.fasta\n')
    found = cli.discover_genomes_from_list(str(lst))
    assert found == [('g', str(home / 'g.fasta'))]


def test_list_missing_file_reports_line_numbers(tmp_path, plain_dir):
    lst = tmp_path / 'paths.txt'
    lst.write_text(
        f'{plain_dir / "genome_0.fasta"}\n'
        '/definitely/not/here.fasta\n'
        f'{plain_dir / "genome_1.fasta"}\n'
        'also_missing.fna\n'
    )
    with pytest.raises(FileNotFoundError) as exc:
        cli.discover_genomes_from_list(str(lst))
    msg = str(exc.value)
    # Every missing entry, with its line number, must be reported.
    assert 'line 2' in msg and '/definitely/not/here.fasta' in msg
    assert 'line 4' in msg and 'also_missing.fna' in msg
    assert '2 path(s)' in msg


def test_list_file_itself_missing(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        cli.discover_genomes_from_list(str(tmp_path / 'nope.txt'))
    assert 'Genome list file not found' in str(exc.value)


def test_list_empty_raises(tmp_path):
    lst = tmp_path / 'paths.txt'
    lst.write_text('# only a comment\n\n\n')
    with pytest.raises(ValueError) as exc:
        cli.discover_genomes_from_list(str(lst))
    assert 'no genome paths' in str(exc.value).lower()


def test_list_directory_entry_rejected(tmp_path, plain_dir):
    lst = tmp_path / 'paths.txt'
    lst.write_text(f'{plain_dir}\n')
    with pytest.raises(FileNotFoundError) as exc:
        cli.discover_genomes_from_list(str(lst))
    assert 'not a file' in str(exc.value).lower()


def test_list_duplicate_paths_deduplicated(tmp_path, plain_dir):
    p0 = plain_dir / 'genome_0.fasta'
    lst = tmp_path / 'paths.txt'
    lst.write_text(f'{p0}\n{p0}\n')
    found = cli.discover_genomes_from_list(str(lst))
    assert found == [('genome_0', str(p0))]


def test_list_is_directory_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.discover_genomes_from_list(str(tmp_path))


# ---------------------------------------------------------------------------
# CLI argument wiring
# ---------------------------------------------------------------------------
def test_parser_accepts_input_list(tmp_path):
    parser = cli.build_parser()
    args = parser.parse_args(['predict', '--input-list', 'p.txt', '--output', 'o.tsv'])
    assert args.input_list == 'p.txt'
    assert args.input is None


def test_parser_input_and_input_list_mutually_exclusive(capsys):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['predict', '--input', 'd', '--input-list', 'p.txt', '-o', 'o.tsv'])
    err = capsys.readouterr().err
    assert 'not allowed with' in err or 'mutually exclusive' in err


def test_parser_requires_one_of_input_or_list(capsys):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['predict', '--output', 'o.tsv'])
    err = capsys.readouterr().err
    assert '--input' in err


def test_predict_rejects_both_inputs(tmp_path, plain_dir, model_resources):
    with pytest.raises(ValueError) as exc:
        cli.predict(
            input_path=str(plain_dir),
            input_list=str(tmp_path / 'p.txt'),
            output_path=str(tmp_path / 'out.tsv'),
            **model_resources,
        )
    assert 'mutually exclusive' in str(exc.value)


def test_predict_requires_an_input(tmp_path, model_resources):
    with pytest.raises(ValueError) as exc:
        cli.predict(input_path=None, input_list=None,
                    output_path=str(tmp_path / 'out.tsv'), **model_resources)
    assert '--input' in str(exc.value)


def test_cli_main_missing_list_exits_1(tmp_path, capsys):
    rc = None
    try:
        cli.main(['predict', '--input-list', str(tmp_path / 'nope.txt'),
                  '-o', str(tmp_path / 'o.tsv'), '-q'])
    except SystemExit as e:
        rc = e.code
    assert rc == 1
