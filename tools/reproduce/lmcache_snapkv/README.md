# LMCache SnapKV reproduction

This harness turns LMCache's interactive SnapKV token-dropping notebook into a
non-interactive, revision-recorded KNLP run. It starts a real LMCache server and
vLLM instance, runs an unchanged baseline, clears LMCache, runs the same prompts
through `prefill -> modify(drop_tokens_fn) -> decode`, and writes accuracy and
decode-throughput measurements to JSON.

```sh
make defconfig-lmcache-snapkv
make
```

The default follows LMCache's small Colab profile: Qwen3-1.7B, ten roughly
1,024-token prompts, a 50% drop ratio, and 256 decode tokens. Adjust the Kconfig
values with `make menuconfig` for a larger GPU. The vLLM query-tensor patch is
applied from the selected LMCache revision and checked for compatibility before
the run starts.

This experiment validates the data path and measures its effect. It does not
make a transformed cache safe to share across unrelated requests. See
`docs/lmcache-snapkv-cache-identity.md` for that design requirement.
