import sys as _sys

# Provide backward-compatible module name so imports like `from app.models` work
# even when the package is loaded outside of its original PYTHONPATH context.
if "app" not in _sys.modules:
    _sys.modules["app"] = _sys.modules[__name__]

from .main import get_fastapi_app  # noqa: E402,F401

