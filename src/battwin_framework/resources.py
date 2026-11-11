"""Read-only bundled resources and writable inference-cache locations."""

from pathlib import Path
import hashlib
import json
import tempfile

ASSETS = Path(__file__).resolve().parent / "assets"


def default_cache_directory():
    """Keep compatibility copies outside both the installed package and project."""
    return Path(tempfile.gettempdir()) / "battwin-framework-cache"


def bundle_directory():
    """Directory containing the unchanged bundled workbooks and trained models."""
    return ASSETS


def verify_assets():
    """Verify the 20 required model artifacts and four original data workbooks."""
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = ASSETS / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Bundled artifact changed or missing: {relative}")
    return manifest
