"""fp8_failure: shared core + the canonical artifact contract (the result schema everything keys
off). Reuses the validated tools/kv harnesses (FlexKVHarness, quantizers, attention discovery)
rather than rebuilding them -- the brief says inspect first. The one thing that is genuinely new
here is the RESULT SCHEMA / run manifest: every result row must carry enough provenance to be
reproducible and to be safely aggregated WITHOUT silently overwriting a per-model CSV.
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys

# reuse the tier-1 core (FlexKVHarness, parse_spec, _quant_lastdims, discover_attention, etc.)
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # tools/kv
import k_bias_common as kbc  # noqa: E402

# measurement levels -- label EVERY result as exactly one; never compare throughput across them
MEAS = (
    "activation_audit",  # hooks + distribution stats, no cache simulation
    "fake_quant_teacher_forced",  # quant/dequant during a teacher-forced forward
    "hf_dynamic_cache",  # incremental DynamicCache writes/reads
    "full_serving",  # vLLM + FlashInfer / deployed backend
    "standalone_kernel",  # isolated attention/cache kernel
)

# the canonical failure classes (a result is exactly one, or "unknown")
FAILURE_CLASSES = (
    "tolerant",
    "bias_induced",
    "mixed_subspace",
    "scale_granularity",
    "attention_score_scale",
    "value_sensitive",
    "layer_local",
    "backend_artifact",
    "unknown",
)

# claim-ladder verdict labels
CLAIM_VERDICTS = (
    "supported",
    "partially_supported",
    "unsupported",
    "refuted",
    "not_tested",
)


# --------------------------------------------------------------------- run manifest / provenance
def git_state(repo=None):
    repo = repo or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    try:
        commit = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", repo, "status", "--porcelain"], text=True
            ).strip()
        )
        return commit, dirty
    except Exception:
        return "unknown", None


def versions():
    out = {}
    for mod in ("torch", "transformers", "vllm", "flashinfer", "triton"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def config_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


@dataclasses.dataclass
class RunManifest:
    """Provenance recorded once per run (the brief's required manifest fields)."""

    run_id: str
    model_id: str
    model_revision: str = "main"
    tokenizer_revision: str = "main"
    cmdline: str = ""
    config_hash: str = ""
    device: str = ""
    driver: str = ""
    dtype: str = "bfloat16"
    fp8_variant: str = "e4m3fn"  # e4m3fn / e4m3fnuz / e5m2
    fp8_rounding: str = "rne"
    saturation: str = "clamp_pm448"
    scale_axis: str = "per_tensor"
    scale_dtype: str = "float32"
    quant_point: str = "cache_write"  # cache_write | cache_read
    calib_dataset: str = ""
    calib_indices: tuple = ()
    eval_dataset: str = ""
    eval_indices: tuple = ()
    seed: int = 0
    measurement_level: str = "fake_quant_teacher_forced"

    def finalize(self):
        c, d = git_state()
        return dict(
            **dataclasses.asdict(self),
            git_commit=c,
            git_dirty=d,
            versions=versions(),
            argv=" ".join(sys.argv),
        )


def write_manifest(out_dir, manifest: RunManifest):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest.finalize(), f, indent=2, default=str)


# ------------------------------------------------------- artifact safety: NEVER silent-overwrite
def model_csv_path(out_dir, short_name, name):
    """Per-MODEL path (the brief: never silently overwrite a shared per-model CSV; aggregate
    explicitly). Returns a model-namespaced path."""
    d = os.path.join(out_dir, short_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def append_row(path, row, fields=None):
    """Append one row, header-if-new. Per-model incremental write so a mid-loop crash preserves
    completed work (the failure mode we hit in tier-2/phi). Returns the path."""
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(row.keys()))
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in (fields or row.keys())})
    return path


def aggregate(out_dir, name, fields=None):
    """Explicit aggregation step: gather per-model <short>/<name> CSVs into out_dir/<name>.
    Does NOT overwrite the per-model files; the aggregate is rebuildable."""
    import csv
    import glob

    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, "*", name))):
        rows += list(csv.DictReader(open(p)))
    if not rows:
        return None
    fields = fields or list(rows[0].keys())
    with open(os.path.join(out_dir, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return os.path.join(out_dir, name)


# ---------------------------------------------------------------------------- recovery fraction
def recovery_fraction(error_normal_fp8, error_repair, error_native=0.0):
    """(err_normal - err_repair) / (err_normal - err_native). 1.0 = full recovery, 0 = none.
    Retain raw values; clamp only for presentation."""
    denom = error_normal_fp8 - error_native
    if abs(denom) < 1e-12:
        return 0.0
    return (error_normal_fp8 - error_repair) / denom
