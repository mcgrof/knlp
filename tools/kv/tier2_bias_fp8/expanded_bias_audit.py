"""Phase 1 (expanded audit): bias magnitude + activation stress with 64 prompts. Thin wrapper that
runs the validated tier-1 bias_audit (weights) + k_bias_activation_audit (pre/post-bias/post-RoPE K
stress) into one output dir. Kept as a named entry per the Tier-2 spec; the science lives in the
tier-1 scripts + k_bias_common."""

import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tools/kv
ap = argparse.ArgumentParser()
ap.add_argument("--model")
ap.add_argument("--short-name")
ap.add_argument("--models-file")
ap.add_argument("--only", nargs="*")
ap.add_argument("--seq-len", type=int, default=2048)
ap.add_argument("--num-prompts", type=int, default=64)
ap.add_argument("--dataset", default="wikitext")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--output-dir", required=True)
a = ap.parse_args()
import _t2common as t2

specs = t2.load_models_spec(a.models_file, [a.model] if a.model else None, a.only)
if a.model and a.short_name:
    specs = [dict(short_name=a.short_name, model_id=a.model)]
PY = sys.executable
os.makedirs(a.output_dir, exist_ok=True)
for m in specs:
    mid, sn = m["model_id"], m["short_name"]
    subprocess.run(
        [
            PY,
            os.path.join(HERE, "bias_audit.py"),
            "--model",
            mid,
            "--short-name",
            sn,
            "--device",
            "cpu",
            "--trust-remote-code",
            "--output-dir",
            a.output_dir,
        ]
    )
    subprocess.run(
        [
            PY,
            os.path.join(HERE, "k_bias_activation_audit.py"),
            "--model",
            mid,
            "--short-name",
            sn,
            "--num-prompts",
            str(a.num_prompts),
            "--seq-len",
            str(a.seq_len),
            "--device",
            a.device,
            "--trust-remote-code",
            "--output-dir",
            a.output_dir,
        ]
    )
print(f"[expanded-audit] {len(specs)} models -> {a.output_dir}")
