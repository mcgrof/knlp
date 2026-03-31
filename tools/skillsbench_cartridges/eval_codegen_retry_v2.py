#!/usr/bin/env python3
"""Evaluate cartridge vs full-skill on code-gen tasks with retry loop.
v2: Manual generation loop for cartridge to avoid generate() cache_position issues.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache


def manual_generate(model, tokenizer, input_ids, attention_mask=None,
                    past_key_values=None, position_ids=None,
                    max_new_tokens=4096, temperature=0.1):
    """Generate tokens manually, one at a time. Works with pre-populated KV cache."""
    device = model.device
    if attention_mask is None:
        attention_mask = torch.ones(1, input_ids.shape[1], dtype=torch.long, device=device)

    t0 = time.time()
    with torch.no_grad():
        # First forward pass: process the prompt
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        kv = out.past_key_values
        seq_len = kv[0][0].shape[2]  # Total seq len in KV cache

        # Sample first token
        logits = out.logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = [next_token.item()]

        for i in range(max_new_tokens - 1):
            pos = torch.tensor([[seq_len + i]], device=device)
            attn = torch.ones(1, seq_len + i + 1, dtype=torch.long, device=device)
            out = model(
                input_ids=next_token,
                attention_mask=attn,
                position_ids=pos,
                past_key_values=kv,
                use_cache=True,
            )
            kv = out.past_key_values
            logits = out.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break

    gen_time = time.time() - t0
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response, gen_time, len(generated)


class ConversationManager:
    def __init__(self, model, tokenizer, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.messages = [{"role": "system", "content": system_prompt}]
        self.total_gen_time = 0
        self.total_gen_tokens = 0

    def add_user(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def generate(self, max_new_tokens=4096, temperature=0.1):
        text = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False).to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_time = time.time() - t0
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.total_gen_time += gen_time
        self.total_gen_tokens += len(new_tokens)
        self.add_assistant(response)
        return response, gen_time, len(new_tokens)


class CartridgeConversationManager:
    def __init__(self, model, tokenizer, system_prompt, cartridge_path):
        self.model = model
        self.tokenizer = tokenizer
        self.messages = [{"role": "system", "content": system_prompt}]
        self.cartridge_kv_raw = torch.load(cartridge_path, map_location="cpu", weights_only=False)
        self.cart_seq_len = self.cartridge_kv_raw[0][0].shape[2]
        self.total_gen_time = 0
        self.total_gen_tokens = 0

    def add_user(self, content):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def generate(self, max_new_tokens=4096, temperature=0.1):
        text = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Rebuild cartridge KV each time (multi-turn needs fresh copy)
        past_kv = DynamicCache()
        for layer_idx, (k, v) in enumerate(self.cartridge_kv_raw):
            past_kv.update(k.to(self.model.device), v.to(self.model.device), layer_idx)

        position_ids = torch.arange(
            self.cart_seq_len, self.cart_seq_len + prompt_len,
            dtype=torch.long, device=self.model.device
        ).unsqueeze(0)
        attention_mask = torch.ones(
            1, self.cart_seq_len + prompt_len,
            dtype=torch.long, device=self.model.device
        )

        response, gen_time, n_tokens = manual_generate(
            self.model, self.tokenizer,
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            past_key_values=past_kv,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        self.total_gen_time += gen_time
        self.total_gen_tokens += n_tokens
        self.add_assistant(response)
        return response, gen_time, n_tokens


def extract_python_code(text):
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    stripped = text.strip()
    if stripped.startswith(("import ", "from ", "#!/")):
        return stripped
    return None


def run_code(code, timeout=120):
    code_path = Path("/tmp/generated_solution.py")
    code_path.write_text(code)
    try:
        result = subprocess.run(
            ["python3", str(code_path)],
            capture_output=True, text=True, timeout=timeout, cwd="/root"
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT after 120s", -1


def run_tests(task_dir, result_dir):
    test_dir = Path(task_dir) / "tests"
    os.makedirs("/tests", exist_ok=True)
    for f in test_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, f"/tests/{f.name}")
    result = subprocess.run(
        ["python3", "-m", "pytest", "/tests/test_outputs.py", "-rA", "-v", "--tb=short"],
        capture_output=True, text=True, cwd="/root", timeout=120
    )
    log_path = Path(result_dir) / "logs" / "test_output.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr)
    lines = result.stdout.split("\n")
    passed = sum(1 for l in lines if " PASSED" in l and "::" in l)
    failed = sum(1 for l in lines if " FAILED" in l and "::" in l)
    errors = sum(1 for l in lines if " ERROR" in l and "::" in l)
    total = passed + failed + errors
    return {"passed": passed, "failed": failed + errors, "total": total,
            "pass_rate": round(passed / max(total, 1), 3),
            "reward": 1.0 if passed == total and total > 0 else 0.0}


def build_initial_prompt(instruction, skill_text=None, data_files=None):
    parts = []
    if skill_text:
        parts.append(f"<skills>\n{skill_text}\n</skills>\n")
    parts.append(f"<task>\n{instruction}\n</task>\n")
    if data_files:
        for name, preview in data_files.items():
            parts.append(f'<file name="{name}">\n{preview}\n</file>\n')
    parts.append(
        "Write a complete Python script that solves this task. "
        "The script should be self-contained and executable. "
        "Use numpy.loadtxt for data loading. "
        "Output ONLY the Python code in a single ```python code block."
    )
    return "\n".join(parts)


def build_retry_prompt(error_msg, attempt):
    return (
        f"The code failed with this error:\n```\n{error_msg[-1500:]}\n```\n\n"
        f"Fix the code and output the complete corrected Python script "
        f"in a ```python code block. (Attempt {attempt}/3)"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--arm", choices=["full-skill", "no-skill", "cartridge"], required=True)
    parser.add_argument("--cartridge-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    instruction = (task_dir / "instruction.md").read_text()
    skills_dir = task_dir / "environment" / "skills"
    skill_text = ""
    if skills_dir.exists():
        for sd in sorted(skills_dir.iterdir()):
            skill_md = sd / "SKILL.md"
            if skill_md.exists():
                skill_text += skill_md.read_text() + "\n\n"

    env_dir = task_dir / "environment"
    data_files = {}
    for f in env_dir.rglob("*"):
        if f.is_file() and f.suffix in (".txt", ".csv", ".dat") and f.name != "Dockerfile":
            content = f.read_text()
            lines = content.split("\n")[:20]
            rel = f.relative_to(env_dir)
            data_files[str(rel)] = "\n".join(lines) + ("\n..." if len(content.split("\n")) > 20 else "")

    data_dir = env_dir / "data"
    if data_dir.exists():
        os.makedirs("/root/data", exist_ok=True)
        for f in data_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, f"/root/data/{f.name}")

    print(f"Task: {task_dir.name}")
    print(f"Arm: {args.arm}")
    print(f"Skill: {len(skill_text)} chars | Data: {list(data_files.keys())}")

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    sys_prompt = ("You are a precise coding assistant. Write complete, executable Python scripts. "
                  "Output only the code in a ```python block. When fixing errors, output the COMPLETE "
                  "corrected script, not just the changed lines.")

    if args.arm == "cartridge":
        assert args.cartridge_dir
        cart_path = str(Path(args.cartridge_dir) / "cartridge.pt")
        conv = CartridgeConversationManager(model, tokenizer, sys_prompt, cart_path)
        initial_prompt = build_initial_prompt(instruction, skill_text=None, data_files=data_files)
    elif args.arm == "full-skill":
        conv = ConversationManager(model, tokenizer, sys_prompt)
        initial_prompt = build_initial_prompt(instruction, skill_text=skill_text, data_files=data_files)
    else:
        conv = ConversationManager(model, tokenizer, sys_prompt)
        initial_prompt = build_initial_prompt(instruction, skill_text=None, data_files=data_files)

    attempts = []
    conv.add_user(initial_prompt)

    for attempt in range(1, args.max_retries + 1):
        print(f"\n--- Attempt {attempt}/{args.max_retries} ---")
        for cleanup in ["/root/period.txt"]:
            if os.path.exists(cleanup):
                os.remove(cleanup)

        response, gen_time, n_tokens = conv.generate()
        code = extract_python_code(response)
        attempt_info = {"attempt": attempt, "gen_time": round(gen_time, 3), "gen_tokens": n_tokens}

        if not code:
            print("No code extracted")
            print(f"Response preview: {response[:300]}")
            attempt_info["code_extracted"] = False
            attempt_info["success"] = False
            attempts.append(attempt_info)
            # For cartridge, might just be generating explanation instead of code
            if attempt < args.max_retries:
                conv.add_user("Please output ONLY the complete Python code in a ```python code block. No explanation needed.")
            continue

        (out_dir / "logs" / f"code_attempt{attempt}.py").write_text(code)
        attempt_info["code_extracted"] = True
        print(f"Code: {len(code)} chars, gen: {gen_time:.1f}s")

        stdout, stderr, rc = run_code(code)
        (out_dir / "logs" / f"stdout_attempt{attempt}.log").write_text(stdout)
        (out_dir / "logs" / f"stderr_attempt{attempt}.log").write_text(stderr)

        if rc == 0:
            print("Code executed successfully!")
            attempt_info["code_success"] = True
            attempt_info["success"] = True
            attempts.append(attempt_info)
            break
        else:
            error_lines = stderr.strip().split("\n")[-5:]
            error_str = "\n".join(error_lines)
            print(f"Failed (rc={rc}): {error_str[:300]}")
            attempt_info["code_success"] = False
            attempt_info["success"] = False
            attempt_info["error_preview"] = error_str[:300]
            attempts.append(attempt_info)
            if attempt < args.max_retries:
                conv.add_user(build_retry_prompt(stderr, attempt + 1))

    print("\n--- Running tests ---")
    test_results = run_tests(str(task_dir), str(out_dir))
    print(f"Tests: {json.dumps(test_results)}")

    # Analyze library choice in generated code
    all_codes = []
    for a in range(1, len(attempts) + 1):
        code_path = out_dir / "logs" / f"code_attempt{a}.py"
        if code_path.exists():
            all_codes.append(code_path.read_text())
    final_code = all_codes[-1] if all_codes else ""
    lib_analysis = {
        "uses_tls": bool(re.search(r"transitleastsquares|tls\.", final_code, re.I)),
        "uses_lightkurve": bool(re.search(r"lightkurve|lk\.", final_code, re.I)),
        "uses_scipy": bool(re.search(r"scipy", final_code, re.I)),
        "uses_astropy_ls": bool(re.search(r"LombScargle", final_code)),
    }

    result = {
        **test_results,
        "arm": args.arm,
        "total_gen_time_sec": round(conv.total_gen_time, 3),
        "total_gen_tokens": conv.total_gen_tokens,
        "attempts": attempts,
        "n_attempts": len(attempts),
        "final_code_success": any(a.get("success") for a in attempts),
        "skill_chars": len(skill_text) if args.arm == "full-skill" else 0,
        "library_analysis": lib_analysis,
    }
    if args.arm == "cartridge" and args.cartridge_dir:
        meta_path = Path(args.cartridge_dir) / "meta.json"
        if meta_path.exists():
            result["cartridge_meta"] = json.loads(meta_path.read_text())

    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFinal: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
