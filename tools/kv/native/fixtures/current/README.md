# Frozen Qwen3 inputs

Separately archived frozen inputs support the current Qwen3-8B asymmetric KV recipe.
The external `current/awq` and `current/fp16` directories contain the actual prompt manifests and
GSM8K tasks used by the measured 8K/batch-32 comparisons. Decompress
`gsm8k-frozen.json.gz` and `scales.json.gz` once; keep `qwen.json.gz`
compressed. Each `manifest.json` binds both stored and decompressed bytes.

All numerical scale values are the original calibrated values. The dense
scale export removes two machine-specific loaded-module paths from its
provenance object. Its file hash therefore differs from the recorded original,
while its complete `scales` mapping is identical. The AWQ scale file is
byte-identical to the original. Both original and export hashes are recorded.
There is no recalibration or placeholder scale in this package.

The `measured-protocol.json` files retain the original engine settings,
source identities and input hashes. They describe the measurements, not a
new install. `tools.kv.native.current` records the new installer source and
file hashes separately before invoking the unchanged numerical entry point.

Prompt source: [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia),
revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, English articles, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
and [GFDL 1.3](https://www.gnu.org/licenses/fdl-1.3.html), as recorded in the
[pinned dataset card](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md).
The manifests retain article titles and IDs for attribution; these are tokenized
excerpts and synthetic retrieval records. Full article histories are available
at Wikipedia under the recorded titles. See [article attribution](ATTRIBUTION.md).
GSM8K source: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k),
revision `740312add88f781978c0658806c59bc2815b9866`, [MIT license](LICENSE-GSM8K.txt).
The task files contain the exact chat adaptation, parser, generation cap and
sample IDs; they are not an unmodified evaluation-harness leaderboard task.
Model weights remain separate downloads under their publishers' licenses.
