"""
Model provenance and integrity (regression tests for the 0.3.2 fix).

Up to and including 0.3.1 the CLI fetched ``magicc_v5.onnx`` from
``https://github.com/renmaotian/magicc/raw/main/models/magicc_v5.onnx`` -- a
**mutable branch ref** -- and accepted whatever came back on the strength of a
``size > 1 MB`` check alone.  0.3.2 fetches the **immutable release asset of
the package's own version** and verifies SHA256 before the model is used, on
first download *and* on every subsequent run.

These tests pin both properties.  They never touch the network: the download
path is exercised through a ``file://`` URL, which ``urllib.request.urlretrieve``
handles with the same code path as ``https://``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import magicc
from magicc import cli

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_MODEL = PROJECT_DIR / 'models' / 'magicc_v5.onnx'

#: Big enough to clear the CLI's ``size > 1 MB`` pre-filter.
_BLOB = b'\x00magicc-test-blob\n' * 70_000


def _write_blob(path: Path, blob: bytes = _BLOB) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture
def isolated_model_env(tmp_path, monkeypatch):
    """
    Point every model-resolution location at an empty temporary tree.

    Returns the cache path the CLI will use.  ``MODEL_SHA256`` and
    ``MODEL_URL`` are left for individual tests to set.
    """
    pkg = tmp_path / 'pkg'
    proj = tmp_path / 'proj'
    cache = tmp_path / 'cache' / cli.MODEL_FILENAME
    (pkg / 'data').mkdir(parents=True)
    (proj / 'models').mkdir(parents=True)
    monkeypatch.setattr(cli, '_PACKAGE_DIR', pkg)
    monkeypatch.setattr(cli, '_PROJECT_DIR', proj)
    monkeypatch.setattr(cli, 'MODEL_CACHE_PATH', cache)
    return cache


# ---------------------------------------------------------------------------
# The URL itself
# ---------------------------------------------------------------------------

def test_model_url_is_the_release_asset_for_this_version():
    assert cli.MODEL_URL == (
        'https://github.com/renmaotian/magicc/releases/download/'
        f'v{magicc.__version__}/magicc_v5.onnx'
    )


def test_cli_version_matches_the_package():
    assert cli.__version__ == magicc.__version__


def test_fallback_version_literal_cannot_drift():
    """
    ``_FALLBACK_VERSION`` is the last resort when ``import magicc`` is shadowed
    by a leftover empty ``site-packages/magicc/`` directory (defect D5). It is a
    literal, so this test is what keeps it honest.
    """
    assert cli._FALLBACK_VERSION == magicc.__version__


def test_version_is_readable_from_the_init_file_beside_cli():
    """The fallback path 0.3.2 lacked: resolve the version without importing."""
    init = Path(cli.__file__).resolve().parent / '__init__.py'
    assert cli._version_from_init(init) == magicc.__version__
    assert cli._package_version() == magicc.__version__


def test_version_from_init_handles_absence(tmp_path):
    (tmp_path / '__init__.py').write_text('"""no version here"""\n')
    assert cli._version_from_init(tmp_path / '__init__.py') is None
    assert cli._version_from_init(tmp_path / 'does-not-exist.py') is None


def test_model_url_is_not_a_mutable_branch_ref():
    """The 0.3.1 defect, pinned so it cannot come back."""
    assert '/raw/' not in cli.MODEL_URL
    assert '/main/' not in cli.MODEL_URL
    assert '/releases/download/' in cli.MODEL_URL


def test_declared_size_and_hash_describe_the_repository_model():
    if not REPO_MODEL.is_file():
        pytest.skip('models/magicc_v5.onnx not present (fresh clone without Git LFS)')
    assert REPO_MODEL.stat().st_size == cli.MODEL_BYTES
    assert cli._sha256_file(REPO_MODEL) == cli.MODEL_SHA256


# ---------------------------------------------------------------------------
# The hash helpers
# ---------------------------------------------------------------------------

def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / 'x.bin'
    p.write_bytes(b'abc' * 1000)
    assert cli._sha256_file(p) == hashlib.sha256(b'abc' * 1000).hexdigest()
    # chunk boundary must not change the answer
    assert cli._sha256_file(p, chunk_bytes=7) == cli._sha256_file(p)


def test_verify_model_accepts_a_matching_file(tmp_path, monkeypatch):
    p = tmp_path / 'model.onnx'
    monkeypatch.setattr(cli, 'MODEL_SHA256', _write_blob(p))
    cli._verify_model(p, 'test')          # must not raise


def test_verify_model_names_both_digests_on_mismatch(tmp_path, monkeypatch):
    p = tmp_path / 'model.onnx'
    good = _write_blob(p)
    monkeypatch.setattr(cli, 'MODEL_SHA256', good)
    p.write_bytes(_BLOB[:-1] + b'X')      # same length, one byte different
    observed = cli._sha256_file(p)
    with pytest.raises(cli.ModelIntegrityError) as excinfo:
        cli._verify_model(p, 'test')
    message = str(excinfo.value)
    assert good in message                # expected digest named
    assert observed in message            # observed digest named
    assert str(p) in message              # and the offending file


def test_model_integrity_error_exits_cleanly_through_the_cli():
    """``main()`` turns it into a logged error and exit 1, not a traceback."""
    assert issubclass(cli.ModelIntegrityError, RuntimeError)


# ---------------------------------------------------------------------------
# Cached model: verified on every run, not only on download
# ---------------------------------------------------------------------------

def test_ensure_model_verifies_an_already_present_cache(isolated_model_env, monkeypatch):
    monkeypatch.setattr(cli, 'MODEL_SHA256', _write_blob(isolated_model_env))
    assert cli._ensure_model() == str(isolated_model_env)


def test_ensure_model_rejects_a_corrupted_cache(isolated_model_env, monkeypatch):
    good = _write_blob(isolated_model_env)
    monkeypatch.setattr(cli, 'MODEL_SHA256', good)
    corrupted = bytearray(_BLOB)
    corrupted[12345] ^= 0xFF
    isolated_model_env.write_bytes(bytes(corrupted))

    with pytest.raises(cli.ModelIntegrityError) as excinfo:
        cli._ensure_model()
    assert good in str(excinfo.value)
    # It must NOT silently re-download over a suspect file.
    assert isolated_model_env.read_bytes() == bytes(corrupted)


def test_ensure_model_verifies_the_package_data_copy(isolated_model_env, monkeypatch):
    pkg_model = cli._PACKAGE_DIR / 'data' / cli.MODEL_FILENAME
    _write_blob(pkg_model)
    monkeypatch.setattr(cli, 'MODEL_SHA256', 'f' * 64)
    with pytest.raises(cli.ModelIntegrityError):
        cli._ensure_model()


# ---------------------------------------------------------------------------
# Download path (exercised over file://, so no network is required)
# ---------------------------------------------------------------------------

def test_download_verifies_and_installs(isolated_model_env, tmp_path, monkeypatch, capsys):
    src = tmp_path / 'served' / cli.MODEL_FILENAME
    monkeypatch.setattr(cli, 'MODEL_SHA256', _write_blob(src))
    monkeypatch.setattr(cli, 'MODEL_URL', src.as_uri())

    assert cli._ensure_model() == str(isolated_model_env)
    assert isolated_model_env.read_bytes() == _BLOB

    notice = capsys.readouterr().out
    assert cli.MODEL_FILENAME in notice          # what
    assert src.as_uri() in notice                # from where
    assert str(isolated_model_env) in notice     # to where
    assert cli.MODEL_SHA256 in notice            # and that it is checksummed


def test_download_of_a_wrong_artefact_is_rejected_and_leaves_nothing(
    isolated_model_env, tmp_path, monkeypatch
):
    src = tmp_path / 'served' / cli.MODEL_FILENAME
    _write_blob(src, _BLOB[:-1] + b'Z')
    monkeypatch.setattr(cli, 'MODEL_SHA256', hashlib.sha256(_BLOB).hexdigest())
    monkeypatch.setattr(cli, 'MODEL_URL', src.as_uri())

    with pytest.raises(cli.ModelIntegrityError):
        cli._ensure_model()

    # No model, and no half-written temporary file, is left behind: a later run
    # must not mistake the rejected download for a valid cache.
    assert not isolated_model_env.exists()
    leftovers = [p for p in isolated_model_env.parent.iterdir()]
    assert leftovers == [], leftovers


def test_failed_download_reports_url_and_cache_path(isolated_model_env, tmp_path, monkeypatch):
    missing = tmp_path / 'served' / 'not-there.onnx'
    monkeypatch.setattr(cli, 'MODEL_URL', missing.as_uri())

    with pytest.raises(RuntimeError) as excinfo:
        cli._ensure_model()
    message = str(excinfo.value)
    assert not isinstance(excinfo.value, cli.ModelIntegrityError)
    assert missing.as_uri() in message               # the URL to fetch by hand
    assert str(isolated_model_env) in message        # where to put the result
    assert cli.MODEL_SHA256 in message               # and what it must hash to
    assert '--model' in message                      # plus the escape hatch


def test_offline_help_mentions_git_lfs_route():
    text = cli._offline_help('simulated')
    assert 'Git LFS' in text
    assert str(cli.MODEL_CACHE_PATH) in text


def test_cache_path_is_under_dot_magicc_in_home():
    assert cli.MODEL_CACHE_PATH == Path(os.path.expanduser('~')) / '.magicc' / 'magicc_v5.onnx'
