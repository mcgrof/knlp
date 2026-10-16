"""Document-separated NLL and exact-length retrieval fixtures."""

import random
import re

from tools.kv.prequantized.contract import digest

WORDS = (
    "amber",
    "birch",
    "cobalt",
    "dune",
    "elm",
    "fern",
    "granite",
    "hazel",
    "ivory",
    "jade",
    "kelp",
    "lilac",
    "maple",
    "nectar",
    "opal",
    "pearl",
    "quartz",
    "reed",
    "saffron",
    "topaz",
    "umber",
    "violet",
    "willow",
    "zenith",
)
PROMPTS = (
    "Explain why the sky is blue to a child.",
    "Write a Python function that removes duplicates while keeping order.",
    "What is the difference between a process and a thread?",
    "Give three practical ways to reduce household food waste.",
    "Describe how a refrigerator moves heat.",
    "Write a short story about a lost compass.",
    "Translate 'The train arrives tomorrow morning' into Spanish.",
    "Explain why multiplying two negative numbers gives a positive number.",
    "A rectangle has sides 7 and 9. What is its perimeter?",
    "Summarize the causes of tides.",
    "Explain how checksums detect corrupted files.",
    "Compare parliamentary and presidential systems.",
    "Write a polite message to reschedule a meeting.",
    "Explain the water cycle.",
    "Suggest a vegetarian dinner with beans and rice.",
    "Describe a binary search algorithm.",
    "Write a four-line poem about autumn.",
    "What happens to a helium balloon as it rises?",
    "Explain supply and demand using apples.",
    "Give an example of correlation without causation.",
    "Explain public-key encryption without advanced mathematics.",
    "Compare breadth-first and depth-first search.",
    "Describe how vaccines prepare the immune system.",
    "Write a simple SQL query to count rows per category.",
    "What is a scientific control group?",
    "Explain the difference between mass and weight.",
    "A shop discounts a 40-dollar item by 15 percent. What is its price?",
    "Describe the life cycle of a butterfly.",
    "Give two advantages and disadvantages of solar power.",
    "Write instructions for planting a tree.",
    "Explain why indexing speeds up a database query.",
    "Tell a short mystery set in a quiet library.",
)


def articles(rows):
    """WikiText article titles have one '=', subsection titles have more."""
    title = None
    body = []
    for row in rows:
        text = row["text"]
        if re.match(r"^\s*= [^=].*[^=] =\s*$", text):
            if title and body:
                yield {
                    "id": digest([title, "".join(body)]),
                    "title": title,
                    "text": title + "\n" + "".join(body),
                }
            title, body = text.strip(), []
        elif title:
            body.append(text)
    if title and body:
        yield {
            "id": digest([title, "".join(body)]),
            "title": title,
            "text": title + "\n" + "".join(body),
        }


def template_parts(tokenizer, model, question):
    marker = "CONTENT_MARKER_4f9caef9"
    options = (
        {"enable_thinking": False}
        if model.startswith("Qwen/")
        else {"reasoning_effort": "low"}
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker + "\n" + question}],
        tokenize=False,
        add_generation_prompt=True,
        **options,
    )
    before, after = text.split(marker)
    return tokenizer.encode(before, add_special_tokens=False), tokenizer.encode(
        after, add_special_tokens=False
    )


def retrieval(tokenizer, model, length, number, seed, dev=False):
    rng = random.Random(seed + length * 1000 + number + (10000000 if dev else 0))
    target_key = f"REF-{rng.randrange(100000000, 999999999)}"
    answer = " ".join(rng.sample(WORDS, 3))
    record = f"\nRecord {target_key}: {answer}. Archived for routine reference.\n"
    question = (
        f"What is the exact value in record {target_key}? Reply with that value only."
    )
    prefix, suffix = template_parts(tokenizer, model, question)
    intro = tokenizer.encode(
        "Read this catalogue of archive records. Each record has its own key and a three-word value.\n",
        add_special_tokens=False,
    )
    target = tokenizer.encode(record, add_special_tokens=False)
    available = length - len(prefix) - len(suffix) - len(target) - len(intro)
    if available < 0:
        raise ValueError("retrieval length does not fit the chat template")
    filler = []
    boundaries = [0]
    while len(filler) < available:
        value = " ".join(rng.sample(WORDS, 3))
        if value == answer:
            continue
        key = f"ARC-{rng.randrange(100000000, 999999999)}"
        filler.extend(
            tokenizer.encode(
                f"Record {key}: {value}. Archived for routine reference.\n",
                add_special_tokens=False,
            )
        )
        boundaries.append(len(filler))
    depth = (0.1, 0.3, 0.5, 0.7, 0.9)[number % 5]
    desired = min(available, max(0, round(length * depth) - len(prefix) - len(intro)))
    position = max(boundary for boundary in boundaries if boundary <= desired)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline) != 1:
        # SentencePiece can prepend a whitespace marker when encoding a
        # newline alone. Select only an ID that independently decodes to it.
        newline = next(
            ([token] for token in newline if tokenizer.decode([token]) == "\n"),
            None,
        )
        if newline is None:
            raise ValueError("exact-depth padding requires a single newline token")
    padding = newline * (desired - position)
    ids = (
        prefix
        + intro
        + filler[:position]
        + padding
        + target
        + filler[position : available - len(padding)]
        + suffix
    )
    assert len(ids) == length
    text = tokenizer.decode(ids)
    if text.count(answer) != 1 or text.count(target_key) != 2:
        raise ValueError("retrieval answer leaked or target key not unique")
    following = text.split(record.strip(), 1)[1].lstrip()
    if not following.startswith("Record "):
        raise ValueError("target record is followed by a broken distractor")
    return {
        "id": f"{'dev' if dev else 'retrieval'}-{length}-{number:03d}",
        "kind": "dev" if dev else "retrieval",
        "length": length,
        "depth": depth,
        "target_token_offset": len(prefix) + len(intro) + desired,
        "generator_version": 2,
        "answer": answer,
        "target_key": target_key,
        "token_ids": ids,
        "prompt": text,
        "prompt_hash": digest(ids),
        "seed": seed,
    }


def parse_final(tokenizer, ids, model):
    text = tokenizer.decode(ids, skip_special_tokens=False)
    if model.startswith("openai/"):
        if "<|channel|>final" not in text and "<|channel|>analysis" in text:
            return "", True
        if "<|channel|>final" in text:
            text = text.rsplit("<|channel|>final", 1)[1]
            text = text.split("<|message|>", 1)[-1]
    text = re.split(r"<\|(?:im_end|end|return|fim_suffix)\|>", text)[0]
    text = re.sub(r"<\|[^>]+\|>", "", text).strip()
    return text.strip(" \n\t.\"'`"), not bool(text)


def prepare_samples(
    tokenizer, matrix, model, dataset_revision, dataset_id="Salesforce/wikitext"
):
    from datasets import load_dataset

    lengths = matrix["quality"]["input_tokens"]
    maximum = max(lengths) + 129
    selected = {}
    inventories = {}
    if dataset_id == "wikimedia/wikipedia":
        rows = load_dataset(
            dataset_id,
            "20231101.en",
            split="train",
            streaming=True,
            revision=dataset_revision,
        )
        candidates = []
        seen = set()
        examined = 0
        for row in rows:
            examined += 1
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if len(ids) >= maximum and row["id"] not in seen:
                seen.add(row["id"])
                candidates.append(({"id": row["id"], "title": row["title"]}, ids))
            if len(candidates) == 128:
                break
        if len(candidates) < 80:
            raise ValueError(
                "alternative corpus needs at least 80 distinct long documents"
            )
        random.Random(20261013).shuffle(candidates)
        selected = {"train": candidates[:16], "test": candidates[16:80]}
        inventories = {
            "alternative": {
                "corpus": dataset_id,
                "configuration": "20231101.en",
                "examined": examined,
                "eligible_pool": len(candidates),
                "selection": "first 128 eligible documents, seeded permutation, disjoint 16 calibration and 64 evaluation",
                "reason": "WikiText-103 test has fewer than 32 distinct 8192-token articles",
            }
        }
    # Eligibility is assessed before any arm is evaluated.
    for split, count, seed in (
        () if selected else (("test", 64, 20261013), ("train", 16, 20261014))
    ):
        rows = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=split,
            revision=dataset_revision,
        )
        candidates = []
        for article in articles(rows):
            ids = tokenizer.encode(article["text"], add_special_tokens=False)
            if len(ids) >= maximum:
                candidates.append((article, ids))
        random.Random(seed).shuffle(candidates)
        inventories[split] = {"eligible_8192": len(candidates), "requested": count}
        if len(candidates) < (16 if split == "train" else 32):
            raise ValueError(
                f"WikiText {split} has only {len(candidates)} distinct eligible articles; freeze a documented alternative corpus before evaluation"
            )
        selected[split] = candidates[:count]
    samples = []
    for split, candidates in selected.items():
        for number, (article, ids) in enumerate(candidates):
            for length in lengths:
                window = ids[: length + 129]
                samples.append(
                    {
                        "id": f"{'cal' if split == 'train' else 'nll'}-{length}-{article['id']}",
                        "kind": "calibration" if split == "train" else "nll",
                        "length": length,
                        "article_id": article["id"],
                        "title": article["title"],
                        "token_ids": window,
                        "prompt_hash": digest(window),
                        "reserve": split == "test" and number >= 32,
                    }
                )
    for length in lengths:
        for number in range(256):
            sample = retrieval(tokenizer, model, length, number, 20261013)
            sample["reserve"] = number >= 100
            samples.append(sample)
        for number in range(5):
            samples.append(
                retrieval(tokenizer, model, length, number, 20261013, dev=True)
            )
    options = (
        {"enable_thinking": False}
        if model.startswith("Qwen/")
        else {"reasoning_effort": "low"}
    )
    for number, prompt in enumerate(PROMPTS):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **options,
        )
        samples.append(
            {
                "id": f"generation-{number:02d}",
                "kind": "generation",
                "length": len(ids),
                "token_ids": ids,
                "prompt_hash": digest(ids),
                "prompt": prompt,
            }
        )
    return (
        samples,
        inventories,
        [a[0]["id"] for a in selected["train"]],
        [a[0]["id"] for a in selected["test"]],
    )
