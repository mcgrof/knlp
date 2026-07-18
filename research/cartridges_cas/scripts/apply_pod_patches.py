#!/usr/bin/env python3
"""Apply CAS code fixes on the cartridges checkout: (1) restore compiled FlexAttention on
CUDA/Hopper (upstream intent; the raw kernel was an RDNA3 workaround); (2) fix the
target-flattening edge case that collapses the top-k teacher distribution to top-1
when cumulative mass never reaches the threshold."""
import os, sys, pathlib

_root = os.environ.get("CART_ROOT", "/root/cartridges")
if "--cart-root" in sys.argv:
    _root = sys.argv[sys.argv.index("--cart-root") + 1]
CART = pathlib.Path(_root) / "cartridges"

# --- (1) attention.py: compiled flex on capable CUDA ---
ap = CART / "models" / "attention.py"
s = ap.read_text()
assert "flex_attention_train = flex_attention" in s, "attention.py already patched or shape changed"
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
print("attention.py patched (compiled flex on CUDA>=sm80, env CARTRIDGES_COMPILE_FLEX to toggle)")

# --- (2) clients/base.py: flatten edge-case fix ---
bp = CART / "clients" / "base.py"
b = bp.read_text()
old = "        cut_idx    = (cum_mass >= threshold).argmax(axis=1)   # [T]"
assert old in b, "base.py flatten line not found (shape changed?)"
new = (old +
       "\n        # EDGE FIX: rows whose top-K never reach `threshold` get argmax==0\n"
       "        # (keeps only top-1). Keep all K for those rows so the teacher\n"
       "        # distribution isn't silently collapsed to a hard label.\n"
       "        _reached = cum_mass[:, -1] >= threshold\n"
       "        cut_idx  = np.where(_reached, cut_idx, K - 1)")
b = b.replace(old, new)
bp.write_text(b)
print("base.py flatten edge-case patched (keep all K when threshold unreached)")
print("PATCHES_APPLIED")
