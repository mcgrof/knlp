#!/usr/bin/env python3
"""Apply CAS code fixes on the cartridges checkout.

Two independent patches; each is idempotent and reports its own
outcome, so a skip in one never suppresses the other:

1. attention.py: compiled FlexAttention on capable CUDA.  The pinned
   upstream tree already compiles flex; the raw-kernel form only
   appears in an RDNA3-modified tree.  Applying to upstream is
   therefore a no-op and is reported as already-compiled rather than
   failing.  Env CARTRIDGES_COMPILE_FLEX toggles the patched form at
   runtime.

2. clients/base.py: the target-flattening edge case that collapses a
   teacher row to top-1 when cumulative mass never reaches the
   threshold.  Env CARTRIDGES_FLATTEN_FIX (default "1") applies it.
   Set it to "0" to keep the legacy flatten semantics — required when
   reproducing a historical objective whose stored rows were consumed
   under the unpatched flattener, where the fix would silently change
   the target support out from under the reproduction control.

Prints one status line per patch plus a machine-readable summary.
"""

import json
import os
import pathlib
import sys

_root = os.environ.get("CART_ROOT", "/root/cartridges")
if "--cart-root" in sys.argv:
    _root = sys.argv[sys.argv.index("--cart-root") + 1]
CART = pathlib.Path(_root) / "cartridges"

status = {}

# --- (1) attention.py: compiled flex on capable CUDA -----------------------
ap = CART / "models" / "attention.py"
s = ap.read_text()
if "_CR_COMPILE" in s:
    status["attention"] = "already_patched"
elif "flex_attention_train = flex_attention" in s:
    s = s.replace(
        "flex_attention_train = flex_attention  # no compile on RDNA3",
        "import os as _os\n"
        "_CR_COMPILE = (_os.environ.get('CARTRIDGES_COMPILE_FLEX', '1') == '1'\n"
        "               and torch.cuda.is_available()\n"
        "               and torch.cuda.get_device_capability()[0] >= 8)\n"
        "flex_attention_train = (torch.compile(flex_attention, dynamic=False, mode='max-autotune-no-cudagraphs')\n"
        "                        if _CR_COMPILE else flex_attention)  # compiled on CUDA >=sm80",
    )
    s = s.replace(
        "flex_attention_generate = flex_attention  # no compile on RDNA3 (64KB shared mem) ",
        "flex_attention_generate = (torch.compile(flex_attention, dynamic=True)\n"
        "                           if _CR_COMPILE else flex_attention)  # compiled on CUDA >=sm80",
    )
    ap.write_text(s)
    status["attention"] = "patched"
elif "torch.compile(flex_attention" in s:
    # pinned upstream form: already compiled, nothing to restore
    status["attention"] = "upstream_already_compiled"
else:
    status["attention"] = "unrecognized_shape"

# --- (2) clients/base.py: flatten edge-case fix ----------------------------
FLATTEN_FIX = os.environ.get("CARTRIDGES_FLATTEN_FIX", "1") == "1"
bp = CART / "clients" / "base.py"
b = bp.read_text()
marker = "# EDGE FIX: rows whose top-K never reach `threshold`"
old = "        cut_idx    = (cum_mass >= threshold).argmax(axis=1)   # [T]"
if marker in b:
    status["flatten"] = "already_patched"
elif not FLATTEN_FIX:
    status["flatten"] = "skipped_legacy_flatten_requested"
elif old in b:
    new = (
        old + "\n        " + marker + "\n"
        "        # get argmax==0 (keeps only top-1). Keep all K for those rows\n"
        "        # so the teacher distribution isn't silently collapsed.\n"
        "        _reached = cum_mass[:, -1] >= threshold\n"
        "        cut_idx  = np.where(_reached, cut_idx, K - 1)"
    )
    bp.write_text(b.replace(old, new))
    status["flatten"] = "patched"
else:
    status["flatten"] = "flatten_line_not_found"

for k, v in status.items():
    print(f"[patch] {k}: {v}")
print("PATCH_STATUS " + json.dumps(status))
