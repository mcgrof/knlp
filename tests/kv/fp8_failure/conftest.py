"""Make the tests importable regardless of where pytest is invoked from: repo root on sys.path for
`import tools.kv.fp8_failure...` (namespace package), this dir for `import _fixtures`, and tools/kv
for the tier-1 `import k_bias_common`. CPU-only, offline: no network, no GPU."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_REPO, _HERE, os.path.join(_REPO, "tools", "kv")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
