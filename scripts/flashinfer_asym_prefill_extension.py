#!/usr/bin/env python3
"""Milestone 1: extend FlashInfer prefill plan() with k_data_type/v_data_type.

The fork at mcgrof/flashinfer:asymmetric-kv-dtype only added the asym
kwargs to BatchDecodeWithPagedKVCacheWrapper.plan().  This script adds
the same to BatchPrefillWithPagedKVCacheWrapper.plan() so the prefill
path can plan an honestly-asymmetric K/V kernel.

Per the design review:
  - Defaults preserve backward-compat: if k_data_type or v_data_type
    is None, use kv_data_type.
  - If kv_data_type is None and k/v_data_type are not, derive
    kv_data_type from v_data_type (V is the cache element format).
  - Cache _cached_k_data_type and _cached_v_data_type separately.
  - Validate q/k/v dtypes at run() against the cached values; do NOT
    use the relaxed BF16-K-vs-FP8-kv hack from the prior pod.

Usage:
    python3 flashinfer_asym_prefill_extension.py /root/flashinfer-src
"""
import re
import sys
from pathlib import Path


def patch_prefill_plan_signature(fi_root: Path):
    """Add k_data_type and v_data_type kwargs to
    BatchPrefillWithPagedKVCacheWrapper.plan() signature, mirroring the
    decode wrapper."""
    p = fi_root / "flashinfer" / "prefill.py"
    t = p.read_text()
    # Find BatchPrefillWithPagedKVCacheWrapper.plan signature.
    # The signature has q_data_type, kv_data_type, o_data_type already.
    # We add k_data_type and v_data_type immediately after kv_data_type.
    pat = re.compile(
        r"(\b(?:def plan\(\s*self,\s*[^)]+?"
        r"kv_data_type:\s*Optional\[Union\[str,\s*torch\.dtype\]\]\s*=\s*None,\n))",
        re.DOTALL,
    )
    insertion = (
        "        k_data_type: Optional[Union[str, torch.dtype]] = None,\n"
        "        v_data_type: Optional[Union[str, torch.dtype]] = None,\n"
    )
    n = len(pat.findall(t))
    if n == 0:
        print("  [signature] no plan() with kv_data_type found")
        return False
    # Add insertion after every kv_data_type in plan() signatures.
    new = pat.sub(r"\1" + insertion, t)
    p.write_text(new)
    print(f"  [signature] added k_data_type/v_data_type to {n} plan() signature(s)")
    return True


def patch_prefill_plan_body(fi_root: Path):
    """Inside plan() body, default unset k_data_type/v_data_type from
    kv_data_type, canonicalize, cache them as _cached_k_data_type and
    _cached_v_data_type, and include them in the JIT/dispatch key.

    Looks for the existing kv_data_type canonicalization block:
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
    and inserts k/v defaulting + canonicalization right after.
    """
    p = fi_root / "flashinfer" / "prefill.py"
    t = p.read_text()
    needle = (
        "        if kv_data_type is None:\n"
        "            kv_data_type = q_data_type\n"
        "        kv_data_type = canonicalize_torch_dtype(kv_data_type)\n"
    )
    addition = (
        "        # Asymmetric K/V: if k_data_type/v_data_type omitted,\n"
        "        # both default to kv_data_type (back-compat).  When set,\n"
        "        # they may differ (e.g., k=bf16, v=fp8_e4m3).\n"
        "        if k_data_type is None:\n"
        "            k_data_type = kv_data_type\n"
        "        k_data_type = canonicalize_torch_dtype(k_data_type)\n"
        "        if v_data_type is None:\n"
        "            v_data_type = kv_data_type\n"
        "        v_data_type = canonicalize_torch_dtype(v_data_type)\n"
    )
    n = t.count(needle)
    if n == 0:
        print("  [body] kv_data_type canonicalize block not found")
        return False
    t = t.replace(needle, needle + addition)

    # Cache k/v dtypes alongside _cached_kv_data_type wherever it's set.
    cache_needle = "        self._cached_kv_data_type = kv_data_type\n"
    cache_add = (
        "        self._cached_kv_data_type = kv_data_type\n"
        "        self._cached_k_data_type = k_data_type\n"
        "        self._cached_v_data_type = v_data_type\n"
    )
    if cache_needle in t:
        # Replace exactly once per occurrence.
        t = t.replace(cache_needle, cache_add)
    p.write_text(t)
    print(f"  [body] added k/v defaults + caching ({n} occurrence(s))")
    return True


def patch_prefill_run_validation(fi_root: Path):
    """In BatchPrefillWithPagedKVCacheWrapper.run(), validate that the
    paged K and V caches match _cached_k_data_type and _cached_v_data_type
    respectively — no relaxation."""
    p = fi_root / "flashinfer" / "prefill.py"
    t = p.read_text()
    # Find any _check_cached_qkv_data_type call inside prefill run() that
    # was using the symmetric helper, and replace with explicit q/k/v checks.
    pat = re.compile(
        r"(_check_cached_qkv_data_type\(\s*"
        r"q,\s*([a-zA-Z_][\w\.]*),\s*"
        r"self\._cached_q_data_type,\s*"
        r"self\._cached_kv_data_type\s*\))"
    )
    matches = pat.findall(t)
    if not matches:
        print("  [run] no _check_cached_qkv_data_type call to replace")
        return False
    # For each unique k_cache var name, replace with the explicit form.
    # Build the replacement as a plain multi-line block; pre-resolve
    # expected dtypes into local names so we don't need nested f-strings
    # (Python 3.12 doesn't allow nested quotes the way earlier versions did).
    seen = set()
    for full, k_var in matches:
        if (full, k_var) in seen:
            continue
        seen.add((full, k_var))
        # In prefill.run() the caller has already done
        # `k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, ...)`
        # so we can reference both variables directly.  The first arg of
        # _check_cached_qkv_data_type IS the k_cache tensor; v_cache is
        # the sibling variable in the same scope.
        v_var = "v_cache" if k_var == "k_cache" else k_var
        validation_block = """# Asymmetric-aware paged KV validation. No dtype relaxation.
        # Replaces the legacy symmetric _check_cached_qkv_data_type helper
        # which conflated K and V into a single kv_data_type and would
        # reject BF16 K when V is FP8 (the entire point of asymmetric).
        _exp_q = self._cached_q_data_type
        _exp_k = getattr(self, "_cached_k_data_type", self._cached_kv_data_type)
        _exp_v = getattr(self, "_cached_v_data_type", self._cached_kv_data_type)
        if q.dtype != _exp_q:
            raise ValueError(
                "q dtype " + str(q.dtype) +
                " != cached q_data_type " + str(_exp_q)
            )
        if {k_var}.dtype != _exp_k:
            raise ValueError(
                "k_cache dtype " + str({k_var}.dtype) +
                " != cached k_data_type " + str(_exp_k)
            )
        if {v_var}.dtype != _exp_v:
            raise ValueError(
                "v_cache dtype " + str({v_var}.dtype) +
                " != cached v_data_type " + str(_exp_v)
            )
        """.format(
            k_var=k_var, v_var=v_var
        )
        # Drop the legacy call entirely.  Our explicit checks above are
        # strictly stronger.
        replacement = validation_block
        t = t.replace(full, replacement, 1)
    p.write_text(t)
    print(f"  [run] replaced {len(seen)} validation site(s)")
    return True


def main():
    if len(sys.argv) != 2:
        print("usage: flashinfer_asym_prefill_extension.py <flashinfer-src-root>")
        sys.exit(1)
    fi_root = Path(sys.argv[1])
    print(f"Extending FlashInfer prefill plan() asymmetric K/V at {fi_root}")
    ok1 = patch_prefill_plan_signature(fi_root)
    ok2 = patch_prefill_plan_body(fi_root)
    ok3 = patch_prefill_run_validation(fi_root)
    if all([ok1, ok2, ok3]):
        print("\nMilestone 1 prefill extension applied.")
        print("Run scripts/test_flashinfer_asym_kv.py to verify before")
        print("attempting any vLLM integration.")
    else:
        print("\nWARN: some patches did not apply; review fork state.")


if __name__ == "__main__":
    main()
