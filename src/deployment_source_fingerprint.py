"""Hash public-demo source files to detect stale Streamlit Cloud deployments.

This is a synchronization check, not authenticated remote code attestation.
It intentionally excludes model/data artifacts and protected validation inputs.
"""

from hashlib import sha256
from pathlib import Path

_REQUIRED = ("app.py", "requirements.txt")
_SOURCE_SUFFIXES = frozenset({".py", ".json"})


def runtime_source_fingerprint(root: Path) -> str:
    """Return a path-aware SHA-256 over the app's source/dependency files."""
    root = Path(root).resolve()
    source_dir = root / "src"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing deployed source directory: {source_dir}")

    files = [root / name for name in _REQUIRED]
    for file in source_dir.rglob("*"):
        if file.is_file() and file.suffix in _SOURCE_SUFFIXES:
            files.append(file)

    digest = sha256()
    for file in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = file.relative_to(root).as_posix().encode("utf-8")
        data = file.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


if __name__ == "__main__":
    print(runtime_source_fingerprint(Path(__file__).resolve().parents[1]))
