"""Phase 5 (alpha causal replication): delegates to the validated tier-1 k_bias_alpha_sweep with the
Tier-2 alpha grid (0,0.25,0.5,1,1.5,2). Named per the Tier-2 spec; the causal logic lives in
k_bias_alpha_sweep + k_bias_common (AlphaKBiasPatch)."""

import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*")
ap.add_argument("--models-file")
ap.add_argument("--only", nargs="*")
ap.add_argument("--seq-len", type=int, default=2048)
ap.add_argument("--num-prompts", type=int, default=32)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--output-dir", required=True)
a = ap.parse_args()
import _t2common as t2

specs = t2.load_models_spec(a.models_file, a.models, a.only)
PY = sys.executable
os.makedirs(a.output_dir, exist_ok=True)
for m in specs:
    subprocess.run(
        [
            PY,
            os.path.join(HERE, "k_bias_alpha_sweep.py"),
            "--model",
            m["model_id"],
            "--short-name",
            m["short_name"],
            "--alphas",
            "0.0,0.25,0.5,1.0,1.5,2.0",
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
print(f"[alpha-replication] {len(specs)} models -> {a.output_dir}")
