# KRI-FT: training a model to tolerate KV-cache routing

Long-context decoding is bottlenecked by KV-cache memory traffic — every
generated token reads the whole cache, so fewer KV bytes per token means
higher arithmetic intensity and more decode throughput before a server
saturates memory bandwidth. KV-cache *routing* (StreamingLLM, H2O, SnapKV,
Quest, and the KRI family) attacks this by keeping only a subset of cache
blocks per query. These routers are training-free: they run on an
off-the-shelf model and pick which blocks attention may see. The catch is
that the model was never trained to lose those blocks, so aggressive routing
degrades it.

KRI-FT asks the inverse question. Instead of building a better router, it
*fine-tunes the model* under a routing mask so the model learns to carry the
dropped information forward in its own hidden states. The router stays
training-free and unchanged; the model becomes the thing that tolerates it.
A KRI-FT checkpoint exports to a standard HuggingFace model with no custom
inference code.

## The finding

**KRI-FT helps models that lack native routing tolerance, and modern small
models already mostly have it.**

On GPT-2 small (124M) the win is large and reproduces across three seeds:

- **Up to 8× fewer global KV blocks** for the same quality — one block at
  1024-token context where vanilla needs eight, to stay within 0.03
  nats/token of full attention.
- **3–6× lower KL-to-full-cache** under every canonical KRI router (KRI-Q,
  KRI-Q-window, KRI-G, kv-sum), despite training on only one routing rule.
- **6.3× lower perplexity** than vanilla under naive recent-only routing at
  matched actual retention, holding across TinyStories, WikiText-103, and
  FineWeb-Edu.
- **No full-attention cost** (perplexity 9.50 vs 9.50 for the matched dense
  control, 9.62 vanilla).

The mechanism is direct: under aggressive routing, vanilla degrades by more
than two nats at late token positions while KRI-FT's late positions do
*better* than its early ones. The model learned to forward long-range
information into its recent hidden states. A matched dense-trained control
(same data, same steps, dense attention) stays tied with vanilla under
routing, which isolates the routing-aware training as the cause rather than
the extra data.

## The honest limit

Ported to a modern small model — SmolLM2-360M with RoPE and grouped-query
attention — the recipe transfers in direction but the magnitude collapses to
a few percent. The reason is the point, not a footnote: vanilla SmolLM2 is
*already* near the routing floor (loss gap ≈ 0 under KRI routing at low block
counts), so there is little left for KRI-FT to internalize. Modern small
models appear to ship with much of the routing tolerance KRI-FT installs into
GPT-2.

That makes KRI-FT a conditional tool. Its value is set by how
routing-intolerant the base model already is, which a cheap routing-tolerance
probe can estimate before any training is spent. Reach for it when a base
model degrades badly under routing; skip it when it does not.

## Reproduce

The code, full results, and every caveat are in the
[`gpt2-kri-ft/`](https://github.com/mcgrof/knlp/tree/main/gpt2-kri-ft)
directory. It runs on a single GPU with plain `matmul`+`softmax` attention
(no FlashAttention/Triton dependency) and exports a standard HuggingFace
checkpoint.

```bash
bash scripts/run_smoke.sh          # device check, tests, 200-step smoke
bash scripts/run_train_kri_gpt2.sh # full fine-tune from openai-community/gpt2
bash scripts/run_eval.sh           # the routing eval matrix
```

See the [interactive visualization](kri_ft_visualization.html) for the
routing-robustness curves, the block-budget reduction, the cross-router KL
gap, the memory-forwarding mechanism, and the GPT-2-vs-SmolLM2 narrowing.
