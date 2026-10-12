"""Deployment synchronization checks must be deterministic and fail closed."""

from pathlib import Path

import pytest

from src.deployment_source_fingerprint import runtime_source_fingerprint


def _source_tree(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "app.py").write_text("print('app')\n")
    (tmp_path / "requirements.txt").write_text("streamlit==1.65.0\n")
    (tmp_path / "src" / "engine.py").write_text("MODEL_VERSION = 1\n")
    (tmp_path / "src" / "stadiums.json").write_text('{"name":"Buffalo"}\n')
    return tmp_path


@pytest.mark.parametrize("changed", ["app.py", "requirements.txt", "src/engine.py", "src/stadiums.json"])
def test_fingerprint_changes_with_deployed_source(tmp_path, changed):
    root = _source_tree(tmp_path)
    first = runtime_source_fingerprint(root)
    assert first == runtime_source_fingerprint(root)
    target = root / changed
    target.write_text(target.read_text() + " updated")
    assert runtime_source_fingerprint(root) != first


def test_fingerprint_ignores_unrelated_docs_and_python_cache(tmp_path):
    root = _source_tree(tmp_path)
    first = runtime_source_fingerprint(root)
    (root / "README.md").write_text("not a runtime dependency")
    cache = root / "src" / "__pycache__"
    cache.mkdir()
    (cache / "engine.cpython-312.pyc").write_bytes(b"ignored")
    assert runtime_source_fingerprint(root) == first


@pytest.mark.parametrize("missing", ["app.py", "requirements.txt", "src"])
def test_fingerprint_refuses_missing_runtime_inputs(tmp_path, missing):
    root = _source_tree(tmp_path)
    target = root / missing
    if target.is_dir():
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
    else:
        target.unlink()
    with pytest.raises(FileNotFoundError):
        runtime_source_fingerprint(root)
