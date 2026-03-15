#!/usr/bin/env python3
# BPA Large Model Validation: Gemma-2-27B + Llama-3.1-70B
#
# Creates RunPod pods, runs BPA KV-sensitivity experiments via SSH,
# retrieves results. Reuses the proven k_proj/v_proj hook approach.
#
# Usage:
#     source ~/.enhance-bash && source ~/envs/runpod/bin/activate
#     python3 scripts/bpa_large_model_gemma_llama.py

import json
import os
import subprocess
import sys
import time

import runpod

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
runpod.api_key = RUNPOD_API_KEY

RESULTS_DIR = "/data/knlp-key-results/bpa-large-model"
JSON_DIR = f"{RESULTS_DIR}/json"
LOG_DIR = f"{RESULTS_DIR}/logs"

EXPERIMENT_CODE = r"""
import gc
import json
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", None)


def quantize_tensor(tensor, bits):
    if bits >= 16:
        return tensor
    qmax = 2 ** (bits - 1) - 1
    scale = tensor.abs().amax(dim=-1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    quantized = (tensor / scale).round().clamp(-qmax, qmax)
    return quantized * scale


def make_proj_hook(state, quantize_fn, proj_type):
    def hook_fn(module, input, output):
        if not state["active"]:
            return output
        bits = state["k_bits"] if proj_type == "k" else state["v_bits"]
        if bits >= 16:
            return output
        return quantize_fn(output, bits)
    return hook_fn


def make_falcon_qkv_hook(state, quantize_fn, attn_module):
    # Hook for Falcon fused query_key_value projection.
    # Splits output into Q/K/V, quantizes K and V, rejoins.
    def hook_fn(module, input, output):
        if not state["active"]:
            return output
        k_bits = state["k_bits"]
        v_bits = state["v_bits"]
        if k_bits >= 16 and v_bits >= 16:
            return output

        n_heads = attn_module.num_heads
        n_kv = getattr(attn_module, "num_kv_heads", n_heads)
        head_dim = attn_module.head_dim
        new_arch = getattr(attn_module, "new_decoder_architecture", False)
        multi_query = getattr(attn_module, "multi_query", False)

        batch, seq_len, _ = output.shape

        if new_arch:
            # [B, T, n_kv * (q_per_kv + 2) * head_dim]
            q_per_kv = n_heads // n_kv
            qkv = output.view(batch, seq_len, n_kv, q_per_kv + 2, head_dim)
            q_part = qkv[:, :, :, :q_per_kv, :].clone()
            k_part = qkv[:, :, :, q_per_kv, :].clone()
            v_part = qkv[:, :, :, q_per_kv + 1, :].clone()

            if k_bits < 16:
                k_part = quantize_fn(k_part, k_bits)
            if v_bits < 16:
                v_part = quantize_fn(v_part, v_bits)

            qkv_out = torch.cat(
                [q_part, k_part.unsqueeze(3), v_part.unsqueeze(3)], dim=3
            )
            return qkv_out.view(batch, seq_len, -1)
        elif multi_query:
            # [B, T, n_heads * head_dim + 2 * head_dim]
            q_size = n_heads * head_dim
            q_part = output[:, :, :q_size].clone()
            k_part = output[:, :, q_size : q_size + head_dim].clone()
            v_part = output[:, :, q_size + head_dim :].clone()

            if k_bits < 16:
                k_part = quantize_fn(k_part, k_bits)
            if v_bits < 16:
                v_part = quantize_fn(v_part, v_bits)

            return torch.cat([q_part, k_part, v_part], dim=2)
        else:
            # Standard MHA: [B, T, 3 * hidden]
            third = output.shape[2] // 3
            q_part = output[:, :, :third].clone()
            k_part = output[:, :, third : 2 * third].clone()
            v_part = output[:, :, 2 * third :].clone()

            if k_bits < 16:
                k_part = quantize_fn(k_part, k_bits)
            if v_bits < 16:
                v_part = quantize_fn(v_part, v_bits)

            return torch.cat([q_part, k_part, v_part], dim=2)

    return hook_fn


def is_fused_qkv(attn):
    # Returns True if attention uses fused query_key_value
    return hasattr(attn, "query_key_value") and not hasattr(attn, "k_proj")


def install_hooks(layers, state, quantize_fn, k_only=False):
    # Install quantization hooks on all layers.
    # Handles both separate k/v proj and fused QKV architectures.
    hooks = []
    for layer in layers:
        attn = get_self_attn(layer)
        if is_fused_qkv(attn):
            h = attn.query_key_value.register_forward_hook(
                make_falcon_qkv_hook(state, quantize_fn, attn)
            )
            hooks.append(h)
        else:
            hk = attn.k_proj.register_forward_hook(
                make_proj_hook(state, quantize_fn, "k")
            )
            hooks.append(hk)
            if not k_only:
                hv = attn.v_proj.register_forward_hook(
                    make_proj_hook(state, quantize_fn, "v")
                )
                hooks.append(hv)
    return hooks


def get_layers(model):
    # Handle different model architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        if hasattr(model.model.decoder, "layers"):
            return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        f"Cannot find layers in model of type {type(model).__name__}"
    )


def get_self_attn(layer):
    # Handle different attention attribute names
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "self_attention"):
        return layer.self_attention
    if hasattr(layer, "attn"):
        return layer.attn
    raise AttributeError(
        f"Cannot find attention in layer of type {type(layer).__name__}"
    )


def run_all_tests(model_name, dtype_str="bfloat16", seq_len=2048,
                  n_prompts=5, cache_seq_lens=None,
                  context_len=2048, gen_tokens=50, n_repeats=3):
    if cache_seq_lens is None:
        cache_seq_lens = [2048, 4096, 8192]

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    print(f"Loading {model_name} in {dtype_str}...")
    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        token=HF_TOKEN,
    )
    model.eval()
    load_time = time.time() - t_load
    print(f"Loaded in {load_time:.0f}s")

    # Show device map
    if hasattr(model, "hf_device_map"):
        devices = set(str(v) for v in model.hf_device_map.values())
        print(f"Device map uses: {devices}")
        n_gpu = sum(
            1 for v in model.hf_device_map.values() if str(v) != "cpu"
        )
        n_cpu = sum(
            1 for v in model.hf_device_map.values() if str(v) == "cpu"
        )
        print(f"  GPU modules: {n_gpu}, CPU modules: {n_cpu}")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    layers = get_layers(model)
    n_layers = len(layers)
    n_params = sum(p.numel() for p in model.parameters())
    vocab_size = tokenizer.vocab_size

    mcfg = model.config
    n_kv_heads = getattr(mcfg, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(mcfg, "num_kv_heads", mcfg.num_attention_heads)
    head_dim = getattr(mcfg, "head_dim", None)
    if head_dim is None:
        head_dim = mcfg.hidden_size // mcfg.num_attention_heads

    print(f"Transformers version: {__import__('transformers').__version__}")

    # Probe k_proj output shape
    probe_state = {"shape": None}
    def probe_hook(module, input, output):
        probe_state["shape"] = list(output.shape)
        return output

    attn0 = get_self_attn(layers[0])
    fused = is_fused_qkv(attn0)
    probe_target = attn0.query_key_value if fused else attn0.k_proj
    h = probe_target.register_forward_hook(probe_hook)
    with torch.no_grad():
        tiny = torch.randint(100, 200, (1, 4)).to(model.device)
        _ = model(tiny)
    h.remove()
    probe_label = "query_key_value" if fused else "k_proj"
    print(f"{probe_label} output shape: {probe_state['shape']}")
    print(f"Fused QKV: {fused}")

    model_info = {
        "model": model_name,
        "n_params": n_params,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype_str,
        "gpu": gpu_name,
        "gpu_mem_gb": gpu_mem_gb,
        "load_time_s": load_time,
        "k_proj_shape": probe_state["shape"],
    }
    results = {"model_info": model_info}
    print(f"Model: {n_params/1e9:.1f}B params, {n_layers} layers, "
          f"GPU: {gpu_name} ({gpu_mem_gb:.0f}GB)")

    # ── Test 1: KV Precision Asymmetry ─────────────────────
    print("\n=== Test 1: KV Precision Asymmetry ===")
    torch.manual_seed(42)
    prompts = [
        torch.randint(100, vocab_size - 100, (seq_len,))
        for _ in range(n_prompts)
    ]

    kv_configs = [
        (16, 16, "K_FP16/V_FP16"),
        (16, 4, "K_FP16/V_INT4"),
        (8, 4, "K_INT8/V_INT4"),
        (4, 4, "K_INT4/V_INT4"),
    ]

    print("  Computing baseline...", flush=True)
    baseline_logits = []
    for i, prompt_ids in enumerate(prompts):
        print(f"    prompt {i+1}/{n_prompts}...", end=" ", flush=True)
        t0 = time.time()
        input_ids = prompt_ids.unsqueeze(0).to(model.device)
        with torch.no_grad():
            out = model(input_ids)
            baseline_logits.append(out.logits.float().cpu())
            del out
        torch.cuda.empty_cache()
        print(f"({time.time()-t0:.0f}s)")

    asym_results = []
    loss_fn = torch.nn.CrossEntropyLoss()
    state = {"active": False, "k_bits": 16, "v_bits": 16}

    for k_bits, v_bits, label in kv_configs:
        print(f"  {label}...", end=" ", flush=True)
        t0 = time.time()

        if k_bits >= 16 and v_bits >= 16:
            errs = [0.0] * n_prompts
            agrees = [1.0] * n_prompts
            ppl_deltas = [0.0] * n_prompts
        else:
            state["k_bits"] = k_bits
            state["v_bits"] = v_bits
            hooks = install_hooks(layers, state, quantize_tensor)

            errs, agrees, ppl_deltas = [], [], []
            for p_idx, prompt_ids in enumerate(prompts):
                input_ids = prompt_ids.unsqueeze(0).to(model.device)
                state["active"] = True
                with torch.no_grad():
                    out = model(input_ids)
                    logits_q = out.logits.float().cpu()
                    del out
                state["active"] = False

                logits_b = baseline_logits[p_idx]
                err = (logits_q - logits_b).abs().mean().item()
                pred_b = logits_b.argmax(dim=-1)
                pred_q = logits_q.argmax(dim=-1)
                agree = (pred_b == pred_q).float().mean().item()

                targets = input_ids[:, 1:].cpu()
                shift_b = logits_b[:, :-1, :].contiguous()
                shift_q = logits_q[:, :-1, :].contiguous()
                ppl_b = math.exp(loss_fn(
                    shift_b.view(-1, shift_b.size(-1)),
                    targets.view(-1),
                ).item())
                ppl_q = math.exp(loss_fn(
                    shift_q.view(-1, shift_q.size(-1)),
                    targets.view(-1),
                ).item())
                ppl_d = ((ppl_q - ppl_b) / ppl_b) * 100.0

                errs.append(err)
                agrees.append(agree)
                ppl_deltas.append(ppl_d)
                del logits_q
                torch.cuda.empty_cache()

            for h in hooks:
                h.remove()

        elapsed = time.time() - t0
        avg_err = sum(errs) / len(errs)
        avg_agree = sum(agrees) / len(agrees)
        avg_ppl_d = sum(ppl_deltas) / len(ppl_deltas)
        print(f"err={avg_err:.4f} agree={avg_agree:.4f} "
              f"ppl_d={avg_ppl_d:+.2f}% ({elapsed:.0f}s)")

        asym_results.append({
            "label": label,
            "k_bits": k_bits,
            "v_bits": v_bits,
            "avg_logit_error": avg_err,
            "avg_token_agreement": avg_agree,
            "avg_ppl_delta_pct": avg_ppl_d,
            "elapsed_s": elapsed,
        })

    results["kv_asymmetry"] = asym_results

    # ── Test 2: Ratio Classifier ───────────────────────────
    print("\n=== Test 2: Ratio Classifier ===")
    logit_errors = {}
    for bits in [6, 8]:
        print(f"  INT{bits} keys...", end=" ", flush=True)
        state["k_bits"] = bits
        state["v_bits"] = 16
        hooks = install_hooks(layers, state, quantize_tensor, k_only=True)

        errors = []
        for p_idx, prompt_ids in enumerate(prompts):
            input_ids = prompt_ids.unsqueeze(0).to(model.device)
            state["active"] = True
            with torch.no_grad():
                out = model(input_ids)
                logits_q = out.logits.float().cpu()
                del out
            state["active"] = False
            err = (logits_q - baseline_logits[p_idx]).abs().mean().item()
            errors.append(err)
            del logits_q
            torch.cuda.empty_cache()

        for h in hooks:
            h.remove()

        logit_errors[bits] = sum(errors) / len(errors)
        print(f"err={logit_errors[bits]:.4f}")

    ratio = logit_errors[6] / max(logit_errors[8], 1e-8)
    needs_fp16 = ratio > 3.0
    print(f"  Ratio: {ratio:.2f} (needs_fp16={needs_fp16})")

    results["ratio_classifier"] = {
        "logit_error_INT6": logit_errors[6],
        "logit_error_INT8": logit_errors[8],
        "ratio_INT6_INT8": ratio,
        "needs_fp16_keys": needs_fp16,
        "threshold": 3.0,
    }

    del baseline_logits
    gc.collect()
    torch.cuda.empty_cache()

    # ── Test 3: KV Cache Size Impact ───────────────────────
    print("\n=== Test 3: KV Cache Size ===")
    param_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    model_size_gb = param_bytes / 1e9

    cache_results = []
    for sl in cache_seq_lens:
        dtype_bytes = 2
        kv_per_tok = 2 * n_kv_heads * head_dim * dtype_bytes
        kv_total_fp16 = kv_per_tok * n_layers * sl
        kv_fp16_gb = kv_total_fp16 / 1e9
        kv_int4_gb = (kv_total_fp16 * 4 / 16) / 1e9
        kv_frac = kv_fp16_gb / (model_size_gb + kv_fp16_gb)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        mem_before = torch.cuda.memory_allocated() / 1e9

        try:
            torch.manual_seed(42)
            ids = torch.randint(100, vocab_size - 100, (1, sl)).to(
                model.device
            )
            with torch.no_grad():
                _ = model(ids)
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            mem_used = mem_peak - mem_before
            status = "ok"
            del ids
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                mem_peak, mem_used, status = -1, -1, "oom"
            else:
                raise
        torch.cuda.empty_cache()
        gc.collect()

        print(f"  L={sl}: KV_FP16={kv_fp16_gb:.2f}GB "
              f"KV_INT4={kv_int4_gb:.2f}GB "
              f"frac={kv_frac:.1%} status={status}")

        cache_results.append({
            "seq_len": sl,
            "kv_cache_fp16_gb": kv_fp16_gb,
            "kv_cache_int4_gb": kv_int4_gb,
            "kv_savings_gb": kv_fp16_gb - kv_int4_gb,
            "kv_fraction_of_total_fp16": kv_frac,
            "gpu_mem_peak_gb": mem_peak,
            "gpu_mem_used_gb": mem_used,
            "status": status,
        })

    results["kv_cache_size"] = {
        "model_size_gb": model_size_gb,
        "per_seq_len": cache_results,
    }

    # ── Test 4: Decode Latency ─────────────────────────────
    print("\n=== Test 4: Decode Latency ===")
    torch.manual_seed(42)
    context_ids = torch.randint(
        100, vocab_size - 100, (1, context_len)
    ).to(model.device)

    print("  Warmup...", flush=True)
    with torch.no_grad():
        _ = model.generate(
            context_ids[:, :64], max_new_tokens=5, do_sample=False,
        )
    torch.cuda.synchronize()

    lat_configs = [
        (16, 16, "FP16/FP16"),
        (16, 4, "FP16/INT4"),
        (8, 4, "INT8/INT4"),
    ]

    lat_results = []
    for k_bits, v_bits, label in lat_configs:
        print(f"  {label}...", end=" ", flush=True)
        hooks = []
        if k_bits < 16 or v_bits < 16:
            state["k_bits"] = k_bits
            state["v_bits"] = v_bits
            hooks = install_hooks(layers, state, quantize_fn=quantize_tensor)

        latencies = []
        for rep in range(n_repeats):
            torch.cuda.synchronize()
            state["active"] = k_bits < 16 or v_bits < 16
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model.generate(
                    context_ids, max_new_tokens=gen_tokens,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            state["active"] = False
            t1 = time.perf_counter()
            total_s = t1 - t0
            latencies.append({
                "ms_per_token": (total_s / gen_tokens) * 1000.0,
                "tokens_per_sec": gen_tokens / total_s,
            })

        for h in hooks:
            h.remove()

        mean_ms = sum(l["ms_per_token"] for l in latencies) / len(latencies)
        mean_tps = sum(l["tokens_per_sec"] for l in latencies) / len(latencies)
        print(f"{mean_ms:.1f} ms/tok ({mean_tps:.1f} tok/s)")

        lat_results.append({
            "label": label,
            "k_bits": k_bits,
            "v_bits": v_bits,
            "mean_ms_per_token": mean_ms,
            "mean_tokens_per_sec": mean_tps,
            "all_latencies": latencies,
        })

    results["decode_latency"] = lat_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_NAME", "")
    dtype_str = os.environ.get("DTYPE", "bfloat16")

    if not model_name:
        print("ERROR: MODEL_NAME env var required")
        sys.exit(1)

    short = model_name.split("/")[-1]
    print(f"\n{'#'*60}")
    print(f"# {model_name}")
    print(f"{'#'*60}")

    t0 = time.time()
    try:
        result = run_all_tests(model_name, dtype_str)
        result["total_elapsed_s"] = time.time() - t0
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        result = {"error": str(e)}

    safe = short.lower().replace("-", "_").replace(".", "_")
    out = f"/workspace/large_{safe}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    if "error" not in result:
        info = result["model_info"]
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"\n{short} ({info['n_params']/1e9:.1f}B, "
              f"D={info['n_layers']}, "
              f"loaded in {info['load_time_s']:.0f}s):")
        for r in result.get("kv_asymmetry", []):
            print(f"  {r['label']:20s} err={r['avg_logit_error']:.4f} "
                  f"agree={r['avg_token_agreement']:.4f} "
                  f"ppl_d={r['avg_ppl_delta_pct']:+.2f}%")
        rc = result.get("ratio_classifier", {})
        if rc:
            print(f"  Ratio: {rc['ratio_INT6_INT8']:.2f} "
                  f"(needs_fp16={rc['needs_fp16_keys']})")
        for r in result.get("kv_cache_size", {}).get("per_seq_len", []):
            print(f"  L={r['seq_len']:5d}: KV={r['kv_cache_fp16_gb']:.2f}GB "
                  f"({r['kv_fraction_of_total_fp16']:.1%})")
        for r in result.get("decode_latency", []):
            print(f"  Lat {r['label']:12s}: {r['mean_ms_per_token']:.1f} "
                  f"ms/tok ({r['mean_tokens_per_sec']:.1f} tok/s)")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("\nDONE")
"""

SSH_KEY = os.path.expanduser("~/.ssh/rqv")


def wait_for_pod_ready(pod_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus", "unknown")
        runtime = pod.get("runtime", {})
        if runtime:
            ports = runtime.get("ports", [])
            for p in ports:
                if p.get("privatePort") == 22 and p.get("ip"):
                    return p["ip"], p["publicPort"]
            if ports:
                port_info = ", ".join(
                    f"{p.get('privatePort')}->{p.get('publicPort')}" for p in ports
                )
                print(
                    f"  Pod {status}, ports: {port_info} " f"(waiting for port 22)...",
                    flush=True,
                )
            else:
                print(f"  Pod {status}, no ports yet...", flush=True)
        else:
            print(f"  Pod {status}, no runtime yet...", flush=True)
        time.sleep(15)
    raise TimeoutError(f"Pod {pod_id} not ready in {timeout}s")


def run_on_pod(ssh_ip, ssh_port, cmd, timeout=7200):
    ssh_cmd = [
        "ssh",
        "-i",
        SSH_KEY,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-p",
        str(ssh_port),
        f"root@{ssh_ip}",
        cmd,
    ]
    proc = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
    return proc


def scp_to_pod(ssh_ip, ssh_port, local_path, remote_path):
    scp_cmd = [
        "scp",
        "-i",
        SSH_KEY,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-P",
        str(ssh_port),
        local_path,
        f"root@{ssh_ip}:{remote_path}",
    ]
    subprocess.run(scp_cmd, check=True, timeout=60)


def scp_from_pod(ssh_ip, ssh_port, remote_path, local_path):
    scp_cmd = [
        "scp",
        "-i",
        SSH_KEY,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-P",
        str(ssh_port),
        f"root@{ssh_ip}:{remote_path}",
        local_path,
    ]
    subprocess.run(scp_cmd, check=True, timeout=120)


def run_model_on_pod(model_name, dtype_str, gpu_type, log_tag):
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ssh_pubkey_path = os.path.expanduser("~/.ssh/rqv.pub")
    with open(ssh_pubkey_path) as f:
        ssh_pubkey = f.read().strip()

    hf_token = ""
    hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(hf_token_path):
        with open(hf_token_path) as f:
            hf_token = f.read().strip()

    short = model_name.split("/")[-1]
    safe = short.lower().replace("-", "_").replace(".", "_")
    pod_name = f"bpa-{safe[:30]}"

    print(f"\nCreating {gpu_type} pod for {short}...")
    env_vars = {
        "PUBLIC_KEY": ssh_pubkey,
        "MODEL_NAME": model_name,
        "DTYPE": dtype_str,
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    pod = runpod.create_pod(
        name=pod_name,
        image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        gpu_type_id=gpu_type,
        cloud_type="ALL",
        gpu_count=1,
        container_disk_in_gb=200,
        volume_in_gb=0,
        start_ssh=True,
        ports="22/tcp",
        min_download=500,
        env=env_vars,
    )
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")

    log_file = f"{LOG_DIR}/pod_run_{log_tag}.log"
    result_file = f"large_{safe}.json"

    try:
        print("Waiting for pod to be ready...")
        ssh_ip, ssh_port = wait_for_pod_ready(pod_id, timeout=600)
        print(f"Pod ready: {ssh_ip}:{ssh_port}")

        print("Installing dependencies...")
        # Falcon needs transformers<5.0 (get_head_mask removed in 5.x)
        # Other models work fine with latest transformers.
        is_falcon = "falcon" in model_name.lower()
        if is_falcon:
            pip_cmd = (
                "pip install -q 'transformers<5.0' accelerate datasets "
                "huggingface_hub"
            )
        else:
            pip_cmd = (
                "pip install -q transformers accelerate datasets " "huggingface_hub"
            )
        r = run_on_pod(
            ssh_ip,
            ssh_port,
            pip_cmd,
            timeout=300,
        )
        print(f"  pip: {'OK' if r.returncode == 0 else 'FAILED'}")
        if r.returncode != 0:
            print(r.stderr[:500])

        # Login to HuggingFace for gated model access
        if hf_token:
            r = run_on_pod(
                ssh_ip,
                ssh_port,
                f"huggingface-cli login --token {hf_token}",
                timeout=60,
            )
            print(f"  HF login: {'OK' if r.returncode == 0 else 'FAILED'}")
            if r.returncode != 0:
                print(r.stderr[:300])

        # Upload experiment code
        tmp_script = "/tmp/bpa_experiment.py"
        with open(tmp_script, "w") as f:
            f.write(EXPERIMENT_CODE)
        scp_to_pod(ssh_ip, ssh_port, tmp_script, "/workspace/experiment.py")
        print("  Script uploaded.")

        # Run experiment
        print(f"\n{'='*60}")
        print(f"Running {short} on {gpu_type}...")
        print(f"{'='*60}\n")

        hf_env = f"HF_TOKEN='{hf_token}' " if hf_token else ""
        r = run_on_pod(
            ssh_ip,
            ssh_port,
            f"cd /workspace && "
            f"{hf_env}"
            f"MODEL_NAME='{model_name}' DTYPE='{dtype_str}' "
            f"python3 experiment.py 2>&1",
            timeout=7200,
        )

        output = r.stdout
        print(output)
        if r.stderr:
            print("STDERR:", r.stderr[:2000])

        # Save log
        with open(log_file, "w") as f:
            f.write(output)
            if r.stderr:
                f.write("\n\nSTDERR:\n" + r.stderr)
        print(f"Log saved: {log_file}")

        # Download result
        try:
            scp_from_pod(
                ssh_ip,
                ssh_port,
                f"/workspace/{result_file}",
                f"{JSON_DIR}/{result_file}",
            )
            print(f"Downloaded: {result_file}")

            # Split into per-test JSONs
            result_path = f"{JSON_DIR}/{result_file}"
            with open(result_path) as f:
                mresult = json.load(f)

            if "error" not in mresult:
                info = mresult.get("model_info", {})
                for test_name in [
                    "kv_asymmetry",
                    "ratio_classifier",
                    "kv_cache_size",
                    "decode_latency",
                ]:
                    if test_name in mresult:
                        out = f"{JSON_DIR}/large_{test_name}_{safe}.json"
                        with open(out, "w") as f:
                            json.dump(
                                {
                                    "experiment": f"large_{test_name}",
                                    "model_info": info,
                                    "results": mresult[test_name],
                                },
                                f,
                                indent=2,
                                default=str,
                            )
                        print(f"  Wrote: {out}")
                return mresult
            else:
                print(f"  ERROR in results: {mresult['error']}")
                return mresult

        except Exception as e:
            print(f"  Download failed: {e}")
            return {"error": str(e)}

    finally:
        print(f"\nTerminating pod {pod_id}...")
        try:
            runpod.terminate_pod(pod_id)
            print("Pod terminated.")
        except Exception as e:
            print(f"WARNING: Failed to terminate: {e}")
            print(f"MANUAL: runpod.terminate_pod('{pod_id}')")


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Gemma and Llama are 403-gated (HF agreement not accepted).
    # Substitutes: Falcon-40b (different arch, fused QKV, GQA)
    # and Yi-1.5-34B (LlamaForCausalLM, GQA, different training).
    # Yi already completed — only Falcon remaining.
    models = [
        {
            "name": "tiiuae/falcon-40b",
            "dtype": "bfloat16",
            "gpu": "NVIDIA H200",
            "tag": f"falcon40b_{ts}",
        },
    ]

    all_results = {}
    for m in models:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {m['name']}")
        print(f"# GPU: {m['gpu']}")
        print(f"{'#'*70}")

        result = run_model_on_pod(m["name"], m["dtype"], m["gpu"], m["tag"])
        all_results[m["name"]] = result

        if "error" in result:
            print(f"\n*** {m['name']} FAILED: {result['error']}")
        else:
            info = result["model_info"]
            print(
                f"\n*** {m['name']} COMPLETE: "
                f"{info['n_params']/1e9:.1f}B, D={info['n_layers']}"
            )

    # Save combined
    combined_path = f"{JSON_DIR}/large_model_gemma_llama.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results: {combined_path}")

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for mname, res in all_results.items():
        short = mname.split("/")[-1]
        if "error" in res:
            print(f"\n{short}: FAILED - {res['error'][:100]}")
            continue
        info = res["model_info"]
        print(f"\n{short} ({info['n_params']/1e9:.1f}B, " f"D={info['n_layers']}):")
        for r in res.get("kv_asymmetry", []):
            print(
                f"  {r['label']:20s} err={r['avg_logit_error']:.4f} "
                f"agree={r['avg_token_agreement']:.4f} "
                f"ppl_d={r['avg_ppl_delta_pct']:+.2f}%"
            )
        rc = res.get("ratio_classifier", {})
        if rc:
            print(
                f"  Ratio: {rc['ratio_INT6_INT8']:.2f} "
                f"(needs_fp16={rc['needs_fp16_keys']})"
            )

    print("\nALL DONE")


if __name__ == "__main__":
    main()
