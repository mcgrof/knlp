# Ministral 3: quality preservation not established

Ministral-3-8B-Instruct did not pass the frozen quality-preservation rule
against either native control. No performance confirmation followed.
This result concerns two specified prompt-and-parser tasks; it does not
establish a general loss of reasoning ability from KV quantization.

The test used one H200, the BF16 checkpoint converted to FP16 weights and
compute, and a fixed 40 GiB KV pool. K16/V8 stored FP16 keys and E4M3 FP8
values. Both native FlashInfer and actual FlashAttention 3 (FA3) stored
FP16 keys and values. All arms used the same checkpoint, tokenizer,
original model-specific scales, scheduler, graphs and task inputs.

The original GSM8K instruction produced bold answers that the frozen
strict parser usually rejected. Strict scores were 8, 9 and 11 of 1,319
for K16/V8, FlashInfer and FA3. Both native controls failed the required
50% accuracy floor. These are task-format failures, not evidence that the
model cannot solve the questions.

One explicit plain-text instruction was then tested on 16 training
questions with native FlashInfer: all 16 parsed, 14 were correct, and none
failed or truncated. A separate full task was frozen before scoring all
three arms. The questions, gold answers, parser, margins, model, scales,
sampling and scheduler stayed fixed. The instruction was:

> The final sentence must have this exact plain-text form, without Markdown,
> currency symbols, units or LaTeX: The answer is &lt;number&gt;.

| Separate plain-text task | K16/V8 | Native FlashInfer | Native FA3 |
| --- | ---: | ---: | ---: |
| Strict correct / 1,319 | 827 | 839 | 842 |
| Native-correct → candidate-wrong | — | 29 | 31 |
| Native-wrong → candidate-correct | — | 17 | 16 |
| One-sided 95% loss-probability upper bound | — | 2.986% | 3.159% |
| Required upper bound | — | <2% | <2% |

The loss probability uses all 1,319 paired questions as its denominator.
It is not the net accuracy change. Net changes were −0.910 percentage
points against FlashInfer and −1.137 points against FA3. Paired-question
bootstrap 95% intervals were [−1.895, +0.076] and [−2.125, −0.150] points.
The JSON also reports wider exact bounds for the net difference. None of
these estimates removes the task limitations below.

The frozen strict regular expression ends in an unescaped dot. For a
response ending in `The answer is 18` with no punctuation, it extracts
`1`. This flaw remains visible in the retained scores. The separately
recorded flexible extractor scored 875, 884 and 887; it is diagnostic and
does not replace the primary gate. The plain-text instruction also
changed response behavior, so the difference from the original task
cannot be attributed solely to extraction. No further prompt search or
post-outcome parser change was used to admit a performance run.

The already recorded flexible extractor gives a useful sensitivity check.
On the original instruction it scores 1,235, 1,232 and 1,234 for K16/V8,
FlashInfer and FA3. Paired loss counts are 10 and 11, with upper bounds
of 1.283% and 1.377%. This secondary result argues against interpreting
the original strict failure as broad quality harm. It does not replace
the primary task or admit a performance claim. On the separate plain-text
task, the same secondary extractor still has loss bounds of 2.637% and
2.812%. Thus the strict parser flaw alone does not explain that task's
failed preservation result. All comparisons reuse the original outputs;
no new generation or extractor selection is involved.

The 1K production checks passed for all three paths, including physical
storage, calibrated scales before graph capture, actual FA3 dispatch and
candidate routing. At both 1K and 8K, NLL and retrieval checks passed
against both controls; all arms retrieved 100/100 answers. Those passing
checks do not override the failed GSM8K preservation gate.

The complete archived quality result
contains model and source pins, counts, intervals, protocol settings and
raw evidence hashes. The [serving study](asymmetric-serving-study.md)
keeps quality, production qualification and performance evidence separate.
