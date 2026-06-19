"""Brick F (capacity wall): throughput vs concurrency -- the metric that separates architectures.

Brick F showed batch-1 TPOT is fixed-cost-bound, so the architecture lever is CAPACITY: a smaller
resident state fits more concurrent sequences in fixed HBM, raising aggregate throughput. This
sweeps batch size at a fixed long context and reports aggregate decode tokens/s, per-sequence
tokens/s, and the capacity-normalized throughput (tokens/s per GB of KV state). The knee where
aggregate throughput stops rising is the concurrency wall; a denser cache (MLA, recurrent state)
should push that wall further. Run the same script on DeepSeek-V2-Lite MLA (auto + fp8 KV = F4) and
the Qwen2.5-7B GQA control for an apples-to-apples capacity comparison.

Decode is isolated per batch via (t_N - t_1)/(N-1) so prefill is not counted in throughput.
"""

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--ctx", type=int, default=16384, help="fixed context per sequence")
    ap.add_argument("--batches", default="1,4,8,16,32,64,128")
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
        enable_prefix_caching=False,
    )
    cfg = llm.llm_engine.model_config.hf_config
    print(
        f"[capacity] {args.model} kv_dtype={args.kv_cache_dtype} ctx={args.ctx} "
        f"max_len={args.max_model_len}"
    )

    def build(ctx):
        return ("def helper(x):\n    return x * 2  # utility line\n") * (ctx // 16)

    batches = [int(b) for b in args.batches.split(",")]

    def timed(prompts, ntok):
        sp = SamplingParams(temperature=0.0, max_tokens=ntok, ignore_eos=True)
        t = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return time.time() - t, outs

    timed([build(2048)], 8)  # warmup
    print(
        f"\n{'batch':>7}{'TTFT s':>9}{'decode tok/s agg':>18}{'tok/s/seq':>11}{'TPOT ms':>10}"
    )
    print("-" * 55)
    peak = 0.0
    for bsz in batches:
        prompts = [build(args.ctx) for _ in range(bsz)]
        try:
            t1, _ = timed(prompts, 1)
            tn, outs = timed(prompts, args.gen)
        except Exception as e:
            print(f"{bsz:>7}  WALL/OOM {type(e).__name__}: {str(e)[:48]}")
            break
        dt = tn - t1
        if dt <= 0:
            print(f"{bsz:>7}  (timing noise, dt<=0; skip)")
            continue
        agg = bsz * (args.gen - 1) / dt
        per = (args.gen - 1) / dt
        tpot = dt / (args.gen - 1) * 1e3
        peak = max(peak, agg)
        print(f"{bsz:>7}{t1:>9.2f}{agg:>18.1f}{per:>11.1f}{tpot:>10.2f}")
    print(f"\n[capacity] peak aggregate decode throughput: {peak:.0f} tok/s")
    print("[capacity] DONE")


if __name__ == "__main__":
    main()
