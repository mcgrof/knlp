# SPDX-License-Identifier: MIT
"""Dataset loaders for the content-bearing trace replay stage.

Package named ``content_datasets`` (NOT ``datasets``) so it never shadows the
pip ``datasets`` module that the loaders themselves import.

Each loader returns a NORMALIZED list of records:

  * LMSYS  -> ``{"conversation_id": str, "turns": [{"role", "content"}, ...]}``
  * LongBench -> ``{"doc_id": str, "document": str, "questions": [str, ...]}``

The pure normalizers (``normalize_lmsys``, ``normalize_longbench``) are
importable and unit-testable without any download; the ``load_*`` entrypoints
add the HF-datasets fetch + idempotent run-dir JSON cache and graceful skip.
"""

from .lmsys import load_lmsys, normalize_lmsys
from .longbench import load_longbench, normalize_longbench

__all__ = [
    "load_lmsys",
    "normalize_lmsys",
    "load_longbench",
    "normalize_longbench",
]
