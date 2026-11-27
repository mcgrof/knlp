# Cross-family cache transfer and factual-state transfer

The two runners test whether one frozen language model can use internal
representations produced by another. Swarm Telepathy maps a prompt's
attention cache and lets the receiver decode from it. Plato's Key tests a
single residual-stream state at the last subject token. The Plato's Key
score is the fraction of interventions that make the receiver produce
the donor's fact. Cache quality uses the receiver's own answer score as
its baseline.

## Swarm Telepathy

The runner uses the authors' MIT-licensed [SwaT implementation](https://github.com/Ar-Kareem/SwaT/tree/34c7653c626c0c22ccf278b6628fa57dec4160d2).
It uses the release's mapper, cross-tokenizer alignment, model loading,
reconstruction and answer training, and evaluators. It loads this checkout
as an external dependency. The code release provides a training recipe and
data preparation scripts. The runner fits a new mapper for the paper's
Granite-3.3-8B-Instruct to SmolLM3-3B direction. See the [paper](https://arxiv.org/html/2610.15095v1).

Set up the author's requirements, model and dataset locations using their
README. Use their training and validation versions of ARC, HellaSwag,
WinoGrande, MMLU, HotpotQA, MuSiQue, QASPER, and QuALITY and the validation
version of SQuAD. The released checkout supplies the PG-19 and WikiText-103
text mix and GSM8K gold solutions. Use these model revisions for the
matched test.

| Model | Hub revision |
| --- | --- |
| `ibm-granite/granite-3.3-8b-instruct` | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` |
| `HuggingFaceTB/SmolLM3-3B` | `a07cc9a04f16550a088caea529712d1d335b0ac1` |

On a system with two CUDA devices and enough memory for the models and
activations, run the following from the knlp root. Use the same Python
environment for knlp and the SwaT requirements.

```bash
python -m research.kv_translate.swat \
  --upstream ../SwaT --sender /path/to/granite --receiver /path/to/smollm \
  --run /path/to/output
```

The default comparison uses 1,000 reconstruction updates, 2,000
answer-training updates, and a training context limit of 2,048 tokens.
The authors' full recipe uses 5,000 and 24,000 steps, a larger maximum
context and filtered receiver-generated GSM8K traces. For the full
recipe, use the command and model/dataset setup from the authors' release.
Use `--prepare` to inspect the resolved commands. A repeated command with
the same configuration reuses completed training stages and evaluations.
The paired comparison scores each cache against native receiver prefill
on the same examples with the same task flags. Its confidence intervals
resample the shared evaluation rows. Each task keeps its own score and
sample count.

## Plato's Key

The [authors' report](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key)
describes a scaled Procrustes map fit on training facts both models recall
and a hidden-state replacement at the last subject token. This independent
reconstruction uses [CounterFact](https://rome.baulab.info/)'s original,
true answers with the released [CounterFact data](https://rome.baulab.info/data/dsets/counterfact.json).
It uses Qwen2.5-3B and SmolLM3-3B-Base. Both revisions and the dataset
checksum are pinned in the runner.

```bash
curl -fL https://rome.baulab.info/data/dsets/counterfact.json \
  -o counterfact.json
python -m research.kv_translate.platos_key \
  --facts counterfact.json --output /path/to/output/forward.json
python -m research.kv_translate.platos_key \
  --facts counterfact.json --output /path/to/output/reverse.json --reverse
```

The runner scores a true-answer prefix only when every answer token agrees
with the greedy next-token prediction. Both models must recall a fact for
that fact to be eligible. The subject determines its split. Evaluation
uses pairs with the same prompt template and different subjects and
answers, and each subject is used once per split of paired swaps. Fit
rotations on the training facts. Select layers by validation swap success
and compare that selection with validation nearest-neighbor overlap.
On the held-out subjects, compare against the receiver's own donor state,
a fit with shuffled training correspondences, and the unchanged receiver.
This uses a bounded depth grid and a public fact set. Both choices affect
comparability with the authors' cross-model median.

When interpreting the two results together, use the same benchmarks and
model pair for quantitative rankings of cache algorithms. Cache repair
spends receiver prefill compute on selected layers or positions, while
Swarm Telepathy spends compute on a trained translator. The subject-state
intervention measures functional representation alignment independently
of a serving path.
