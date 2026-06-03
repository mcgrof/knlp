"""Ensure the repo root is on sys.path so `import trellis_lm` resolves during
collection regardless of where pytest is invoked from."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
