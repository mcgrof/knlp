#!/usr/bin/env python3
"""Evaluate cartridge vs full-skill on a SkillsBench task.

For single-turn tasks (citation-check, exoplanet, powerlifting):
1. Full-skill: include skill text in prompt, get model output
2. Cartridge: inject KV cache, exclude skill text from prompt, get model output
3. Compare outputs and run tests

Usage:
    python eval_cartridge.py \
        --model /workspace/models/Qwen2.5-7B-Instruct \
        --task-dir /workspace/skillsbench/tasks/citation-check \
        --cartridge-dir /workspace/cartridges/citation-check/default-50pct \
        --output-dir /workspace/tier1-results/citation-check/cartridge-default-50pct
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt(instruction: str, skill_text: str = None, task_data: str = None) -> str:
    """Build a chat prompt for the task."""
    parts = []
    if skill_text:
        parts.append(f"<skill>\n{skill_text}\n</skill>\n")
    parts.append(f"<instruction>\n{instruction}\n</instruction>\n")
    if task_data:
        parts.append(f"\n<data>\n{task_data}\n</data>\n")
    parts.append("\nProvide your answer as valid JSON. Output ONLY the JSON, no other text.")
    return "\n".join(parts)


def generate_with_model(
    model, tokenizer, prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
):
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": "You are a precise assistant. Follow instructions exactly and output only what is requested."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=False).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response, gen_time, len(new_tokens)


def generate_with_cartridge(
    model, tokenizer, prompt: str, cartridge_path: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
):
    """Generate with pre-loaded cartridge KV cache.

    The cartridge provides KV states for positions 0..budget_tokens-1.
    The new prompt tokens get positions budget_tokens..budget_tokens+prompt_len-1.
    """
    # Load cartridge
    cartridge_kv = torch.load(cartridge_path, map_location="cpu", weights_only=False)

    # Determine cartridge sequence length from first layer's key tensor
    cart_seq_len = cartridge_kv[0][0].shape[2]  # (batch, heads, seq_len, head_dim)

    messages = [
        {"role": "system", "content": "You are a precise assistant. Follow instructions exactly and output only what is requested."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=False).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    # Move cartridge KV to device and build DynamicCache
    from transformers import DynamicCache
    past_kv = DynamicCache()
    for layer_idx, (k, v) in enumerate(cartridge_kv):
        past_kv.update(k.to(model.device), v.to(model.device), layer_idx)

    # Create position_ids starting after the cartridge's sequence
    position_ids = torch.arange(
        cart_seq_len, cart_seq_len + prompt_len, dtype=torch.long, device=model.device
    ).unsqueeze(0)

    # Create cache_position for the new tokens
    cache_position = torch.arange(
        cart_seq_len, cart_seq_len + prompt_len, dtype=torch.long, device=model.device
    )

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=torch.ones(1, cart_seq_len + prompt_len, dtype=torch.long, device=model.device),
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_kv,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0

    new_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response, gen_time, len(new_tokens)


def extract_json(text: str) -> dict:
    """Extract JSON from model output, handling markdown fences."""
    import re
    # Try to find JSON in markdown code blocks
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        text = m.group(1)
    # Try to find JSON object
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return json.loads(text)


def run_tests(task_dir: str, result_dir: str) -> dict:
    """Run pytest on the task tests and return results."""
    test_dir = Path(task_dir) / "tests"
    result_dir = Path(result_dir)

    # Copy tests
    os.makedirs("/tests", exist_ok=True)
    for f in test_dir.iterdir():
        if f.is_file():
            import shutil
            shutil.copy2(f, f"/tests/{f.name}")

    # Run pytest
    result = subprocess.run(
        ["python3", "-m", "pytest", "/tests/test_outputs.py", "-rA", "-v", "--tb=short"],
        capture_output=True, text=True, cwd="/root"
    )

    log_path = result_dir / "logs" / "test_output.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr)

    # Parse results
    lines = result.stdout.split("\n")
    passed = sum(1 for l in lines if " PASSED" in l and "::" in l)
    failed = sum(1 for l in lines if " FAILED" in l and "::" in l)
    errors = sum(1 for l in lines if " ERROR" in l and "::" in l)
    total = passed + failed + errors
    rate = round(passed / max(total, 1), 3)

    return {
        "passed": passed,
        "failed": failed + errors,
        "total": total,
        "pass_rate": rate,
        "reward": 1.0 if passed == total and total > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--cartridge-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", choices=["full-skill", "no-skill", "cartridge"], default="cartridge")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    cart_dir = Path(args.cartridge_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    # Load instruction
    instruction = (task_dir / "instruction.md").read_text()

    # Load skill text
    skills_dir = task_dir / "environment" / "skills"
    skill_text = ""
    if skills_dir.exists():
        for skill_dir in sorted(skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                skill_text += skill_md.read_text() + "\n\n"

    # Load task-specific data (e.g., test.bib)
    task_data = ""
    env_dir = task_dir / "environment"
    for f in env_dir.iterdir():
        if f.is_file() and f.name not in ("Dockerfile",) and f.suffix in (".bib", ".txt", ".csv"):
            task_data += f"--- {f.name} ---\n{f.read_text()}\n\n"

    print(f"Task: {task_dir.name}")
    print(f"Arm: {args.arm}")
    print(f"Instruction: {len(instruction)} chars")
    print(f"Skill text: {len(skill_text)} chars")
    print(f"Task data: {len(task_data)} chars")

    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Build prompt and generate
    if args.arm == "full-skill":
        prompt = build_prompt(instruction, skill_text=skill_text, task_data=task_data)
        response, gen_time, n_tokens = generate_with_model(model, tokenizer, prompt)
    elif args.arm == "no-skill":
        prompt = build_prompt(instruction, skill_text=None, task_data=task_data)
        response, gen_time, n_tokens = generate_with_model(model, tokenizer, prompt)
    elif args.arm == "cartridge":
        # Prompt WITHOUT skill text; cartridge provides the skill context
        prompt = build_prompt(instruction, skill_text=None, task_data=task_data)
        cartridge_path = str(cart_dir / "cartridge.pt")
        response, gen_time, n_tokens = generate_with_cartridge(
            model, tokenizer, prompt, cartridge_path
        )

    print(f"\nGeneration: {gen_time:.2f}s, {n_tokens} tokens")
    print(f"Response preview: {response[:500]}")

    # Save raw response
    (out_dir / "logs" / "model_response.txt").write_text(response)

    # Try to extract and write answer
    try:
        answer = extract_json(response)
        answer_path = Path("/root/answer.json")
        with open(answer_path, "w") as f:
            json.dump(answer, f, indent=2)
        print(f"Answer written to {answer_path}")
        print(f"Answer: {json.dumps(answer, indent=2)}")
    except Exception as e:
        print(f"Failed to extract JSON: {e}")
        print(f"Writing raw response as answer...")
        Path("/root/answer.json").write_text(json.dumps({"error": str(e), "raw": response}))

    # Run tests
    print("\nRunning tests...")
    test_results = run_tests(str(task_dir), str(out_dir))
    print(f"Test results: {json.dumps(test_results)}")

    # Save combined result
    result = {
        **test_results,
        "arm": args.arm,
        "gen_time_sec": round(gen_time, 3),
        "gen_tokens": n_tokens,
        "prompt_chars": len(prompt),
        "skill_chars": len(skill_text) if args.arm == "full-skill" else 0,
    }
    if args.arm == "cartridge" and (cart_dir / "meta.json").exists():
        result["cartridge_meta"] = json.loads((cart_dir / "meta.json").read_text())

    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFinal result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
