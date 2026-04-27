#!/usr/bin/env python3
"""Asymmetric K16/V8 vLLM patches — scouting only, NOT a final landing.

After a design review (see docs/lmcache_asym_kv_review_brief.md and the
ChatGPT Pro response folded into docs/lmcache_asym_kv_status_20260427.md),
several of the patches we originally discovered are now known to be
dangerous or wrong-headed.  This file ONLY applies the patches that are
either correct on their own or safe scouting.  The dangerous ones are
documented and intentionally not applied.

Patches applied (safe / scouting):
  [1] Tuple-handling helper at every site that does
      kv_cache_dtype.startswith() / == on a string.  Lets the engine
      boot far enough to expose downstream gaps; a final design must
      replace these with explicit asymmetric-aware checks (V-dtype
      tells you "is V quantized?", not "is K quantized?").
  [3] get_kv_cache_torch_dtype: accept tuple for the engine-init path.
      This one is a config-plumbing fix and stays.
  [4a] get_fp8_dtype_for_flashattn: accept tuple.  Same rationale.
  [7] lmcache vllm_service_factory: _ensure_engine() before
      LookupClientFactory.create_lookup_client().  Orthogonal,
      enables bypass-lookup in single-process LLM() runs.

Patches NOT applied (known dangerous or wrong-headed; left here as
documentation of what to remove):

  [2] reshape_and_cache_flash tuple-to-V collapse — DANGEROUS.
      Quantizes K through the FP8 path and defeats the entire research
      goal.  The right fix is a new reshape_and_cache_flash_asym op
      that stores K losslessly and V as FP8.  Will be added in
      Milestone 3 of the landing plan.
  [4b] flash_attn schedule cache_dtype tuple unwrap — wrong backend.
      FA3 does not support asymmetric K/V for head_dim=128.  Selector
      must instead route asymmetric to FlashInfer and fail closed if
      anything else was requested.
  [5] flashinfer init tuple unwrap — premature; the proper version
      lives inside the FlashInfer backend's __init__ where it already
      resolves k_cache_dtype and v_cache_dtype separately, but the
      forward path then passes only kv_data_type.  The fix is in the
      forward path (pass k_data_type/v_data_type to every plan() call),
      not at init.  Captured in Milestone 4.
  [6] Selector reduction asym->V-dtype — wrong intent.  Picks an
      FP8-aware backend but erases that K is NOT FP8.  The selector
      should see asymmetric and pick FlashInfer specifically, with a
      hard fail-closed for any other backend.
  [8] env-var force-FlashInfer (_LMCACHE_ASYM_FORCE_FLASHINFER) —
      wrong mechanism.  Process-global, implicit, and can poison
      unrelated model loads.  Replace with explicit selector logic
      (see Milestone 5 in the status doc).
  [9] FlashInfer _check_cached_qkv_data_type relaxation — DANGEROUS.
      Plans the kernel as homogeneous FP8 KV but feeds it BF16 K.
      Silent-corruption surface.  The fix is to extend the FlashInfer
      prefill plan() signature with separate k_data_type/v_data_type
      and validate them exactly (Milestone 1).

Run from the pod root after cloning vllm-asymmetric-kv-plumbing and
lmcache-asymmetric-kv-codec:

    python3 lmcache_asym_vllm_patches.py /root/vllm-src /root/lmcache
"""
import re
import sys
from pathlib import Path


def patch_kv_cache_dtype_string_methods(vllm_root: Path):
    """Wrap every .startswith()/== call on kv_cache_dtype with a helper
    that handles the asym tuple form.  SCOUTING ONLY: lets the engine
    boot far enough to expose downstream gaps.  A final landing must
    replace these with explicit asymmetric-aware checks because
    `cache_dtype_v(spec)` answers "is V quantized?" but several of the
    rewritten call sites are really asking "is the whole KV cache
    quantized?", which is not the same question."""
    targets = [
        vllm_root / "v1" / "attention",
        vllm_root / "model_executor" / "layers" / "attention",
    ]
    helper_import = "from vllm.config.cache import cache_dtype_v as _cdv\n"
    patterns = [
        (
            re.compile(r"\bself\.kv_cache_dtype\.startswith\("),
            "_cdv(self.kv_cache_dtype).startswith(",
        ),
        (
            re.compile(r"(?<![\w.])kv_cache_dtype\.startswith\("),
            "_cdv(kv_cache_dtype).startswith(",
        ),
        (
            re.compile(r"\bself\.kv_cache_dtype\s*==\s*"),
            "_cdv(self.kv_cache_dtype) == ",
        ),
        (re.compile(r"(?<![\w.])kv_cache_dtype\s*==\s*"), "_cdv(kv_cache_dtype) == "),
        (
            re.compile(r"assert\s+self\.kv_cache_dtype\s+in\s+\{"),
            "assert _cdv(self.kv_cache_dtype) in {",
        ),
    ]
    n_files, n_subs = 0, 0
    for tgt in targets:
        for f in tgt.rglob("*.py"):
            text = f.read_text()
            before = text
            for pat, repl in patterns:
                text = pat.sub(repl, text)
            if text == before:
                continue
            if helper_import.strip() not in text:
                lines = text.split("\n")
                # Find safe insertion point — after the last contiguous
                # `from vllm....` import block, never inside `from X import (`.
                insert_at = 0
                paren_depth = 0
                for i, ln in enumerate(lines):
                    body = ln if "#" not in ln else ln[: ln.find("#")]
                    paren_depth += body.count("(") - body.count(")")
                    if paren_depth == 0 and (
                        ln.startswith("from vllm.") or ln.startswith("import vllm")
                    ):
                        insert_at = i + 1
                lines.insert(insert_at, helper_import.rstrip())
                text = "\n".join(lines)
            f.write_text(text)
            n_subs += sum(len(p.findall(before)) for p, _ in patterns)
            n_files += 1
    print(f"  [1] tuple-handling helper (scouting): {n_files} files, {n_subs} subs")


def patch_torch_utils_get_kv_cache_torch_dtype(vllm_root: Path):
    """get_kv_cache_torch_dtype() must accept tuple — collapse to V-half.
    This one is a config-plumbing fix and is correct as-is."""
    p = vllm_root / "utils" / "torch_utils.py"
    t = p.read_text()
    old = (
        "def get_kv_cache_torch_dtype(\n"
        "    cache_dtype: str | torch.dtype | None,\n"
        "    model_dtype: str | torch.dtype | None = None,\n"
        ") -> torch.dtype:\n"
        "    if isinstance(cache_dtype, str):"
    )
    new = (
        "def get_kv_cache_torch_dtype(\n"
        "    cache_dtype,\n"
        "    model_dtype: str | torch.dtype | None = None,\n"
        ") -> torch.dtype:\n"
        "    if isinstance(cache_dtype, tuple):\n"
        "        cache_dtype = cache_dtype[1]\n"
        "    if isinstance(cache_dtype, str):"
    )
    if old in t:
        p.write_text(t.replace(old, new))
        print("  [3] get_kv_cache_torch_dtype tuple-aware: patched")


def patch_get_fp8_dtype_for_flashattn(vllm_root: Path):
    """FA boundary helper: accept tuple.  Config-plumbing only."""
    p = vllm_root / "v1" / "attention" / "backends" / "flash_attn.py"
    t = p.read_text()
    old = (
        "    @staticmethod\n"
        "    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str)"
        " -> torch.dtype:\n"
        '        if kv_cache_dtype in ("fp8", "fp8_e4m3"):'
    )
    new = (
        "    @staticmethod\n"
        "    def get_fp8_dtype_for_flashattn(kv_cache_dtype) -> torch.dtype:\n"
        "        if isinstance(kv_cache_dtype, tuple):\n"
        "            kv_cache_dtype = kv_cache_dtype[1]\n"
        '        if kv_cache_dtype in ("fp8", "fp8_e4m3"):'
    )
    if old in t:
        p.write_text(t.replace(old, new))
        print("  [4a] get_fp8_dtype_for_flashattn tuple-aware: patched")


def patch_lmcache_ensure_engine_before_lookup(lmcache_root: Path):
    """vllm_service_factory.maybe_create_lookup_client() must call
    _ensure_engine() before LookupClientFactory.create_lookup_client()
    so that bypass lookup has a real engine (single-process LLM())."""
    p = lmcache_root / "integration" / "vllm" / "vllm_service_factory.py"
    t = p.read_text()
    old = (
        "        self._ensure_metadata()\n"
        "        assert self.metadata is not None\n"
        "        return LookupClientFactory.create_lookup_client(\n"
        "            self.lmcache_config,\n"
        "            self.metadata,\n"
        "            self.lmcache_engine,\n"
        "        )"
    )
    new = (
        "        self._ensure_metadata()\n"
        "        self._ensure_engine()\n"
        "        assert self.metadata is not None\n"
        "        return LookupClientFactory.create_lookup_client(\n"
        "            self.lmcache_config,\n"
        "            self.metadata,\n"
        "            self.lmcache_engine,\n"
        "        )"
    )
    if old in t:
        p.write_text(t.replace(old, new))
        print("  [7] lmcache _ensure_engine before lookup_client: patched")


def fix_paren_block_imports(vllm_root: Path):
    """Move any `from vllm.config.cache import cache_dtype_v as _cdv` line
    that landed inside an open `from X import (` block to after the close
    paren."""
    needle = "from vllm.config.cache import cache_dtype_v as _cdv"
    for f in vllm_root.rglob("*.py"):
        txt = f.read_text()
        if needle not in txt:
            continue
        out = []
        deferred = []
        open_parens = 0
        for ln in txt.split("\n"):
            if ln.strip() == needle and open_parens > 0:
                deferred.append(ln)
                continue
            out.append(ln)
            body = ln if "#" not in ln else ln[: ln.find("#")]
            open_parens += body.count("(") - body.count(")")
            if open_parens == 0 and deferred:
                out.extend(deferred)
                deferred = []
        new = "\n".join(out)
        if new != txt:
            f.write_text(new)


def main():
    if len(sys.argv) != 3:
        print("usage: lmcache_asym_vllm_patches.py " "<vllm-src-root> <lmcache-root>")
        sys.exit(1)
    vllm_root = Path(sys.argv[1]) / "vllm"
    lmcache_root = Path(sys.argv[2])
    if not vllm_root.exists():
        # Maybe they passed the actual vllm pkg dir
        vllm_root = Path(sys.argv[1])
    print(f"Applying SAFE/SCOUTING patches to {vllm_root} and {lmcache_root}")
    print(
        "(See docstring for the dangerous patches that are intentionally"
        " NOT applied.)"
    )
    patch_kv_cache_dtype_string_methods(vllm_root)
    fix_paren_block_imports(vllm_root)
    patch_torch_utils_get_kv_cache_torch_dtype(vllm_root)
    patch_get_fp8_dtype_for_flashattn(vllm_root)
    patch_lmcache_ensure_engine_before_lookup(lmcache_root)
    print("\nApplied.  This is scouting-only — do NOT call asym forward")
    print("until Milestone 1 (FlashInfer prefill plan asym extension +")
    print("standalone proof) lands.")


if __name__ == "__main__":
    main()
