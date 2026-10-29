# Small-scale cartridge ladder

This directory holds a CPU-sized harness for cartridge research. A
cartridge is a small trained KV cache that stands in for a document in
front of a frozen reader (Eyuboglu et al., [Cartridges](https://arxiv.org/abs/2506.06266)).
Training one for an 8B reader on a clinical record costs about six
hours on a rented H100, which is too slow to answer algorithmic
questions one at a time. The harness here trains the same object, by
the same objective, on readers from 14M to 1B parameters on a CPU, so
that initialization, objective, attention and drift questions can be
screened in minutes and only the winner is confirmed on the big reader.

The reader must show a document advantage on the task, or there is
nothing to compress: the full-context accuracy has to sit well above
the no-context accuracy. Clinical question answering needs a reader of
several billion parameters for that, so the small rungs use a synthetic
record task that a small reader can answer from context and that is
scored exactly. A record lists entities with attribute values; the
questions ask for one value; forced-choice scoring compares the correct
value against the other entities' values for the same attribute. Two
record formats exist: `qa`, prose entries with question and answer
turns, and `kv`, one fact per line with the question as the fact's
prefix, which is the easier copy task for the smallest readers.

Training follows the Cartridges recipe in miniature. The teacher is the
reader with the record in context; the student is the reader with the
prefix in place of the record; the loss is the top-k context
distillation over self-study conversations about the record; the
prefix is initialized from the KV of the record's first tokens with the
first slot frozen as the attention sink. Informative positions are
those where the teacher's top token differs from the bare reader's,
the split the 8B work uses, and every held-out loss is reported on
that split and on the rest.

Instruments, all exact and all cheap enough to run every ten steps:

- held-out distillation loss, all, informative and rest positions;
- forced-choice accuracy and margin on held-out questions, with the
  full-context ceiling and the no-context floor from the same reader;
- attention mass on the prefix slots and on the sink per layer, at
  informative and rest positions (the reader runs with eager attention
  so the weights are available);
- key and value drift from the initialization, cosine and relative norm
  per layer;
- off-context damage: the loss on an unrelated record's text and the
  forced-choice accuracy on an unrelated record given in context while
  the prefix is loaded, which is the "off-context derailment" of
  [Cartridges++](https://arxiv.org/abs/2609.35621); `--generic-frac`
  trains with their data mix, a fraction of conversations about
  entities outside the record with the bare reader as teacher.

## Use

```
python3 smallcart.py run --model EleutherAI/pythia-160m --fmt kv \
    --init first_tokens --slots 64 --steps 200 --out out/first_tokens
python3 smallcart.py run --model EleutherAI/pythia-160m --fmt kv \
    --init chunks --slots 64 --steps 200 --out out/chunks
python3 smallcart.py run ... --generic-frac 0.05 --out out/mix5
python3 smallcart.py bench --model EleutherAI/pythia-160m --threads 8,16 \
    --out bench/host.json
```

`run` prints `VAL_CSV,<step>,<loss>,<loss_info>,<loss_rest>,<slot
mass info>,<slot mass rest>` every `--val-every` steps, with the
forced-choice and off-context numbers appended every
`--eval-choice-every` steps, and writes `result.json` with the whole
curve, the references and the final prefix. `--init` takes
`first_tokens`, `chunks` (Cartridges' chunk sampling), `other_doc` (the
first tokens of an unrelated record, the wrong-document control),
`random`, or `file:<prefix.pt>` to continue from a saved prefix.

`bench` times the training step of a fixed workload at the given thread
counts, plus a long-context forward and a plain matmul, and records the
CPU model and cache sizes from `lscpu`, so the same workload can be
compared across hosts. Use the same torch wheel on every host.

Everything runs in float32. `DEVICE=cuda` moves it to a GPU unchanged.

## What the ladder has to reproduce first

A small rung earns trust by reproducing the orderings measured on the
8B reader before it screens anything new: the first-tokens
initialization beats spread initializations; training lowers the
attention on the slots and moves it to the deep layers; the early
cartridge is generic (an unrelated document's early cartridge helps as
much as a fitted correction) and the initialization is forgotten within
a fraction of the run; keys stay near their start while values move;
steering attention onto the slots at inference hurts. Four of five is
the bar.
