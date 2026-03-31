#!/usr/bin/env python3
"""Quick eval wrapper for powerlifting-coef-calc on the existing pass@k harness."""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TASK_DIR = Path("/workspace/skillsbench/tasks/powerlifting-coef-calc")
MODEL_PATH = "/workspace/models/Qwen2.5-7B-Instruct"

def load_skills():
    skills_dir = TASK_DIR / "environment" / "skills"
    parts = []
    for skill_dir in sorted(skills_dir.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            parts.append(f"### Skill: {skill_dir.name}\n\n{skill_md.read_text()}")
    return "\n\n".join(parts)

def build_prompt(with_skills=True):
    instruction = (TASK_DIR / "instruction.md").read_text()
    data_readme = (TASK_DIR / "environment" / "data-readme.md").read_text()

    parts = []
    if with_skills:
        parts.append(f"# Reference Skills\n\n{load_skills()}\n")
    parts.append(f"# Task\n\n{instruction}\n")
    parts.append(f"\n# Data Documentation\n\n{data_readme}\n")
    parts.append(
        "\n# Available Files\n"
        "- /root/data/openipf.xlsx (Excel workbook with 'Data' sheet and empty 'Dots' sheet)\n"
        "- /root/data/data-readme.md (column documentation)\n"
    )
    parts.append(
        "\n# Instructions\n\n"
        "Write a complete Python script that:\n"
        "1. Reads /root/data/openipf.xlsx\n"
        "2. Creates/populates the 'Dots' sheet with the required columns and Excel formulas\n"
        "3. Saves the result back to /root/data/openipf.xlsx\n\n"
        "Use openpyxl for reading/writing. Output ONLY a Python script in ```python ... ``` block."
    )
    return "\n".join(parts)

def generate(model, tokenizer, prompt, temperature=0.8):
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. Write clean, correct code."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=False).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=4096, temperature=max(temperature, 0.01),
            do_sample=True, top_p=0.95, pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), gen_time, len(new_tokens)

def extract_python(text):
    m = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()

def run_tests(out_dir):
    test_dir = TASK_DIR / "tests"
    os.makedirs("/tests", exist_ok=True)
    for f in test_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, f"/tests/{f.name}")
    result = subprocess.run(
        ["python3", "-m", "pytest", "/tests/test_outputs.py", "-rA", "-v", "--tb=short"],
        capture_output=True, text=True, cwd="/root"
    )
    Path(out_dir) .joinpath("test_output.log").write_text(result.stdout + result.stderr)
    lines = result.stdout.split("\n")
    passed = sum(1 for l in lines if " PASSED" in l and "::" in l)
    failed = sum(1 for l in lines if " FAILED" in l and "::" in l)
    errors = sum(1 for l in lines if " ERROR" in l and "::" in l)
    total = passed + failed + errors
    return {"passed": passed, "failed": failed + errors, "total": total,
            "pass_rate": round(passed / max(total, 1), 3)}

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

def main():
    arm = sys.argv[1]  # "full-skill" or "no-skill"
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    prompt = build_prompt(with_skills=(arm == "full-skill"))
    out_base = Path(f"/workspace/tier1-results/powerlifting-coef-calc/{arm}-pass{n_samples}")
    out_base.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"Task: powerlifting | Arm: {arm} | Samples: {n_samples} | Prompt tokens: {prompt_tokens}")

    # Backup original xlsx
    orig_xlsx = Path("/root/data/openipf.xlsx")
    backup = Path("/root/data/openipf_backup.xlsx")
    shutil.copy2(orig_xlsx, backup)

    n_correct = 0
    samples = []

    for i in range(n_samples):
        sample_dir = out_base / f"sample_{i}"
        sample_dir.mkdir(exist_ok=True)

        # Restore original
        shutil.copy2(backup, orig_xlsx)

        print(f"\n--- Sample {i+1}/{n_samples} ---")
        response, gen_time, n_tok = generate(model, tokenizer, prompt)
        (sample_dir / "response.txt").write_text(response)

        script = extract_python(response)
        (sample_dir / "script.py").write_text(script)
        print(f"Gen: {gen_time:.1f}s, {n_tok} tok, script: {len(script)} chars")

        # Execute
        script_path = Path("/root/solve_generated.py")
        script_path.write_text(script)
        try:
            result = subprocess.run(
                ["python3", str(script_path)], capture_output=True, text=True,
                timeout=120, cwd="/root"
            )
            success = result.returncode == 0
            (sample_dir / "exec_stdout.log").write_text(result.stdout)
            (sample_dir / "exec_stderr.log").write_text(result.stderr)
            if not success:
                print(f"Exec FAIL: {result.stderr[-200:]}")
            else:
                print(f"Exec OK")
        except subprocess.TimeoutExpired:
            success = False
            print("Exec TIMEOUT")

        # Run tests
        test_result = run_tests(str(sample_dir))
        print(f"Tests: {test_result['passed']}/{test_result['total']}")

        sample_info = {"sample_id": i, "gen_time_sec": round(gen_time, 3),
                       "gen_tokens": n_tok, "exec_success": success, **test_result}
        samples.append(sample_info)
        if test_result["passed"] == test_result["total"] and test_result["total"] > 0:
            n_correct += 1

    n = len(samples)
    summary = {
        "task": "powerlifting-coef-calc", "arm": arm, "num_samples": n,
        "num_correct": n_correct, "prompt_tokens": prompt_tokens,
        "pass_at_1": round(pass_at_k(n, n_correct, 1), 4),
        "pass_at_3": round(pass_at_k(n, n_correct, 3), 4) if n >= 3 else None,
        "samples": samples,
    }
    with open(out_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY: powerlifting / {arm}")
    print(f"Correct: {n_correct}/{n}")
    for s in samples:
        print(f"  Sample {s['sample_id']}: exec={'OK' if s['exec_success'] else 'FAIL'}, tests={s['passed']}/{s['total']}")
    print(f"pass@1={summary['pass_at_1']:.4f}")
    if summary['pass_at_3']: print(f"pass@3={summary['pass_at_3']:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
