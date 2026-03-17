"""
File I/O tools — read and write local files.

Restricted to a configurable base directory to prevent the model from
reading/writing arbitrary system files.
"""

import os
from pathlib import Path

import config

# Default sandbox directory for file operations
_SANDBOX_DIR = config.DATA_DIR / "files"
_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

_MAX_READ_CHARS = 8000


def _safe_path(filepath: str) -> Path:
    """Resolve a path and ensure it's within the sandbox."""
    p = Path(filepath)
    # If relative, resolve against sandbox
    if not p.is_absolute():
        p = _SANDBOX_DIR / p
    p = p.resolve()
    # Must be inside sandbox
    if not str(p).startswith(str(_SANDBOX_DIR.resolve())):
        raise ValueError(f"Access denied: path must be inside {_SANDBOX_DIR}")
    return p


def read_file(filepath: str) -> dict:
    """
    Read a file's contents.

    Returns:
        {"path": str, "content": str | None, "truncated": bool, "error": str | None}
    """
    try:
        p = _safe_path(filepath)
    except ValueError as e:
        return {"path": filepath, "content": None, "truncated": False, "error": str(e)}

    if not p.exists():
        return {"path": str(p), "content": None, "truncated": False,
                "error": "File not found"}

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"path": str(p), "content": None, "truncated": False,
                "error": f"Read failed: {e}"}

    truncated = len(text) > _MAX_READ_CHARS
    if truncated:
        text = text[:_MAX_READ_CHARS] + "\n[Content truncated]"

    return {"path": str(p), "content": text, "truncated": truncated, "error": None}


def write_file(filepath: str, content: str) -> dict:
    """
    Write content to a file (creates parent directories as needed).

    Returns:
        {"path": str, "bytes_written": int, "error": str | None}
    """
    try:
        p = _safe_path(filepath)
    except ValueError as e:
        return {"path": filepath, "bytes_written": 0, "error": str(e)}

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as e:
        return {"path": str(p), "bytes_written": 0, "error": f"Write failed: {e}"}

    return {"path": str(p), "bytes_written": len(content.encode("utf-8")), "error": None}


def list_files(directory: str = ".") -> dict:
    """
    List files in a directory within the sandbox.

    Returns:
        {"path": str, "files": list[str], "error": str | None}
    """
    try:
        p = _safe_path(directory)
    except ValueError as e:
        return {"path": directory, "files": [], "error": str(e)}

    if not p.is_dir():
        return {"path": str(p), "files": [], "error": "Not a directory"}

    try:
        entries = []
        for item in sorted(p.iterdir()):
            prefix = "d " if item.is_dir() else "f "
            entries.append(prefix + item.name)
        return {"path": str(p), "files": entries, "error": None}
    except Exception as e:
        return {"path": str(p), "files": [], "error": f"List failed: {e}"}
