from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from .. import config


def get_temp_dir() -> Path:
    """Return the shared temporary directory for the app."""
    return config.ensure_directory(config.TEMP_DIR)


def save_uploaded_file(file_name: str, data: bytes, suffix: Optional[str] = None) -> Path:
    """
    Persist an uploaded file into the temporary directory.

    Args:
        file_name: The original file name supplied by the browser.
        data: Raw byte contents of the file.
        suffix: Optional extension override (e.g. ".pdf")

    Returns:
        Path to the saved file.
    """
    temp_dir = get_temp_dir()
    ext = suffix if suffix else Path(file_name).suffix or ".bin"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    target = temp_dir / unique_name
    target.write_bytes(data)
    return target


def cleanup_temp_dir() -> None:
    """Remove temporary artefacts."""
    temp_dir = get_temp_dir()
    for path in temp_dir.iterdir():
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except PermissionError:
            # On Windows a file may still be locked; ignore silently.
            continue

