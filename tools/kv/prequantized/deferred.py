"""Freeze uniform catalogues whose query arrives after cache persistence."""

import random
from bisect import bisect_right

from tools.kv.prequantized.contract import digest
from tools.kv.prequantized.fixtures import WORDS

INTRO = "Read this catalogue. Each record maps its key to a three-word value.\n"
FOOTER = "End of catalogue. Padding:"
QUESTION = (
    "\n\nQuestion: What is the exact value in record {key}? Reply with that value only."
)
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)


def render(tokenizer, model, catalogue, key):
    options = (
        {"enable_thinking": False}
        if model.startswith("Qwen/")
        else {"reasoning_effort": "low"}
    )
    question = QUESTION.format(key=key)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": catalogue + question}],
        tokenize=False,
        add_generation_prompt=True,
        **options,
    )
    return tokenize_prompt(tokenizer, text, question)


def tokenize_prompt(tokenizer, text, question):
    # Tokenize the combined template once. Offsets select a token boundary;
    # separately encoded strings can merge differently at the split.
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start = text.index(question)
    split = next(
        i for i, (_, end) in enumerate(encoded["offset_mapping"]) if end > start
    )
    if any(end > start for _, end in encoded["offset_mapping"][:split]):
        raise ValueError("query text entered catalogue prefill")
    return text, encoded["input_ids"], split


def record_text(key, value):
    return f"Record {key}: {value}. Archived for routine reference.\n"


def fixture(tokenizer, model, length, number, seed, dev=False):
    rng = random.Random(seed + length * 1000 + number)
    records, seen_keys, seen_values = [], set(), set()
    approximate_tokens = 0
    # Leave room for both chat formats and the question. No target exists yet.
    while True:
        key = f"REC-{rng.randrange(100000000, 1000000000)}"
        value = " ".join(rng.sample(WORDS, 3))
        if key in seen_keys or value in seen_values:
            continue
        width = len(tokenizer.encode(record_text(key, value), add_special_tokens=False))
        if approximate_tokens + width > length - 200:
            break
        records.append({"key": key, "value": value})
        seen_keys.add(key)
        seen_values.add(value)
        approximate_tokens += width
    catalogue = INTRO + "".join(record_text(r["key"], r["value"]) for r in records)
    depth = DEPTHS[number % 5]
    # Pick the query only after all record draws. All records have one format.
    encoded = tokenizer(
        catalogue, add_special_tokens=False, return_offsets_mapping=True
    )
    ends = [end for _, end in encoded["offset_mapping"]]
    offsets = [
        bisect_right(ends, catalogue.index(record_text(r["key"], r["value"])))
        for r in records
    ]
    index = min(range(len(records)), key=lambda i: abs(offsets[i] - length * depth))
    chosen = records[index]
    padding = 0
    for _ in range(8):
        content = catalogue + FOOTER + " ." * padding
        text, ids, split = render(tokenizer, model, content, chosen["key"])
        difference = length - len(ids)
        if difference == 0:
            break
        padding += difference
        if padding < 0:
            raise ValueError("catalogue exceeds logical prompt length")
    else:
        raise ValueError("padding did not reach exact combined token length")
    if text.count(chosen["value"]) != 1 or text.count(chosen["key"]) != 2:
        raise ValueError("target answer or key leaked")
    sample = {
        "id": f"{'dev-v3' if dev else 'retrieval-v3'}-{length}-{number:03d}",
        "kind": "dev" if dev else "retrieval",
        "length": length,
        "depth": depth,
        "generator_version": 3,
        "family": "retrieval_v3_deferred_query",
        "answer": chosen["value"],
        "target_key": chosen["key"],
        "target_index": index,
        "records": records,
        "catalogue": content,
        "catalogue_tokens": split,
        "query_tokens": length - split,
        "token_ids": ids,
        "prompt": text,
        "prompt_hash": digest(ids),
        "seed": seed,
        "target_token_offset": offsets[index],
    }
    return sample


def query_variant(tokenizer, model, sample, index, changed_value=None):
    record = sample["records"][index]
    catalogue = sample["catalogue"]
    if changed_value is not None:
        catalogue = catalogue.replace(
            record_text(record["key"], record["value"]),
            record_text(record["key"], changed_value),
        )
    # Keep the frozen chat frame. Some templates insert today's date, so
    # rendering again can change the persisted prefix across midnight.
    previous = sample["catalogue"] + QUESTION.format(key=sample["target_key"])
    if sample["prompt"].count(previous) != 1:
        raise ValueError("frozen prompt does not contain one catalogue and question")
    question = QUESTION.format(key=record["key"])
    text = sample["prompt"].replace(previous, catalogue + question, 1)
    text, ids, split = tokenize_prompt(tokenizer, text, question)
    result = {
        **sample,
        "target_key": record["key"],
        "target_index": index,
        "answer": changed_value or record["value"],
        "token_ids": ids,
        "prompt": text,
        "prompt_hash": digest(ids),
        "catalogue": catalogue,
        "length": len(ids),
        "catalogue_tokens": split,
        "query_tokens": len(ids) - split,
    }
    return result
