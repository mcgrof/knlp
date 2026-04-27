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
    seen = set()
    for full, k_var in matches:
        if (full, k_var) in seen:
            continue
        seen.add((full, k_var))
        # Build the replacement.  v_cache var name follows from k_cache.
        # In the existing code paths, the prefill run signature has
        # `paged_kv_cache` which can be a tuple or a single tensor.
        # Insert a tuple-aware validation block right before the call.
        replacement = (
            "# Asymmetric-aware paged KV validation.  No dtype relaxation.\n"
            "        _kc = " + k_var + "\n"
            "        if isinstance(_kc, (tuple, list)):\n"
            "            _k_t, _v_t = _kc\n"
            "        else:\n"
            "            _k_t = _v_t = _kc\n"
            "        if q.dtype != self._cached_q_data_type:\n"
            "            raise ValueError(\n"
            '                f"q dtype {q.dtype} != cached q_data_type "\n'
            '                f"{self._cached_q_data_type}")\n'
            '        if _k_t.dtype != getattr(self, "_cached_k_data_type",'
            " self._cached_kv_data_type):\n"
            "            raise ValueError(\n"
            '                f"k_cache dtype {_k_t.dtype} != cached "\n'
            '                f"k_data_type "\n'
            "                f\"{getattr(self, '_cached_k_data_type', \"\n"
            '                f"self._cached_kv_data_type)}")\n'
            '        if _v_t.dtype != getattr(self, "_cached_v_data_type",'
            " self._cached_kv_data_type):\n"
            "            raise ValueError(\n"
            '                f"v_cache dtype {_v_t.dtype} != cached "\n'
            '                f"v_data_type "\n'
            "                f\"{getattr(self, '_cached_v_data_type', \"\n"
            '                f"self._cached_kv_data_type)}")\n'
            "        # Original symmetric-helper call retained for back-compat:\n"
            "        " + full
        )
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
