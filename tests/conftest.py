"""Test bootstrap: make src/ importable for any test selection.

Previously individual test files patched sys.path at import time, so running
a single test file in isolation (pytest tests/test_x.py) failed to import
assistant_app unless a path-patching file happened to load first. This
conftest removes that ordering dependence.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
