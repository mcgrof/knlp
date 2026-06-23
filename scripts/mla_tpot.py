"""Brick F3: decode-isolated TPOT vs context for native MLA (DeepSeek-V2-Lite).

The smoke timed whole-generate wall (prefill + decode mixed). This isolates DECODE TPOT so the
MLA context slope is visible, not buried in prefill: for each context it times generating 1 token
(= TTFT, dominated by prefill) and N tokens, and reports TPOT = (t_N - t_1) / (N - 1). Run across
contexts and batch sizes; also report vLLM's KV-cache bytes/token and max concurrency. Compare the
measured slope to the Table-1 roofline floor (does MoE/kernel overhead eat the MLA gain?).
"""

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--contexts", default="4096,32768,65536,131072")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--gen", type=int, default=65)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=False,
        enable_prefix_caching=False,  # else t_N reuses t_1's cached prefill -> negative TPOT
    )
    cfg = llm.llm_engine.model_config.hf_config
    elem = getattr(cfg, "kv_lora_rank", 0) + getattr(cfg, "qk_rope_head_dim", 0)
    print(
        f"[mla-tpot] {args.model} MLA cache {elem} elem/tok/layer x{cfg.num_hidden_layers}"
    )

    def build(ctx):
        return ("def helper(x):\n    return x * 2  # utility line\n") * (ctx // 16)

    contexts = [int(c) for c in args.contexts.split(",")]
    batches = [int(b) for b in args.batches.split(",")]

    def timed(prompts, ntok):
        sp = SamplingParams(temperature=0.0, max_tokens=ntok, ignore_eos=True)
        t = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return time.time() - t, outs

    # warmup
    timed([build(2048)], 8)
    print(
        f"\n{'ctx':>8}{'batch':>7}{'TTFT(t1) s':>12}{'decode TPOT ms':>16}{'tok/s/seq':>11}"
    )
    print("-" * 54)
    rows = []
    for ctx in contexts:
        for bsz in batches:
            prompts = [build(ctx) for _ in range(bsz)]
            try:
                t1, _ = timed(prompts, 1)
                tn, outs = timed(prompts, args.gen)
            except Exception as e:
                print(f"{ctx:>8}{bsz:>7}  ERROR {type(e).__name__}: {str(e)[:50]}")
                continue
            n_in = len(outs[0].prompt_token_ids)
            tpot_ms = (tn - t1) / (args.gen - 1) * 1e3
            toks = 1000.0 / tpot_ms if tpot_ms > 0 else 0
            print(f"{n_in:>8}{bsz:>7}{t1:>12.2f}{tpot_ms:>16.2f}{toks:>11.1f}")
            rows.append((n_in, bsz, t1, tpot_ms))
    print("\n[mla-tpot] DONE")


if __name__ == "__main__":
    main()
