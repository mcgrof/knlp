# SPDX-License-Identifier: MIT
"""Freeze disjoint calibration, development and confirmation inputs on CPU."""
import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from tools.kv.native import gsm8k
from tools.kv.prequantized.fixtures import retrieval

MODELS = {
    "qwen3": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "float16"),
    "qwen25": (
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
        "bfloat16",
    ),
}
WIKI = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
GSM = "740312add88f781978c0658806c59bc2815b9866"
REGEX = r"(?i)the\s+answer\s+is\s*[:=]?\s*\**\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    model, revision, dtype = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    cfg = AutoConfig.from_pretrained(model, revision=revision)
    pool = []
    seen = set()
    examined = 0
    rows = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        revision=WIKI,
    )
    for row in rows:
        examined += 1
        if row["id"] in seen:
            continue
        ids = tok.encode(row["text"], add_special_tokens=False)
        if len(ids) < 8192 + 129:
            continue
        seen.add(row["id"])
        pool.append(
            dict(
                article_id=row["id"],
                title=row["title"],
                token_ids=ids[:8321],
                text_sha256=hashlib.sha256(row["text"].encode()).hexdigest(),
            )
        )
        if len(pool) == 384:
            break
    if len(pool) != 384:
        raise ValueError("Insufficient independent long documents")
    # The prior reference drew from the first 128 eligible articles. Exclude
    # that prefix before seeded sampling; no model outputs inform eligibility.
    candidates = pool[128:]
    random.Random(20261020).shuffle(candidates)
    selections = {
        "calibration": candidates[:32],
        "development": candidates[32:48],
        "confirmation": candidates[48:112],
    }
    samples = {stage: [] for stage in selections}
    for stage, docs in selections.items():
        for doc in docs:
            for length in [1024, 8192]:
                ids = doc["token_ids"][: length + 129]
                samples[stage].append(
                    dict(
                        id=f'{stage}-nll-{length}-{doc["article_id"]}',
                        kind="nll",
                        length=length,
                        article_id=doc["article_id"],
                        title=doc["title"],
                        text_sha256=doc["text_sha256"],
                        token_ids=ids,
                        prompt_hash=gsm8k.digest(ids),
                    )
                )
    for stage, count, seed in [
        ("development", 32, 20261021),
        ("confirmation", 200, 20261020),
    ]:
        for length in [1024, 8192]:
            for i in range(count):
                r = retrieval(tok, model, length, i, seed)
                r.update(
                    id=f"{stage}-retrieval-{length}-{i:03d}",
                    cluster_id=f"{stage}-{length}-{i:03d}",
                )
                samples[stage].append(r)
    train = list(load_dataset("openai/gsm8k", "main", split="train", revision=GSM))
    test = list(load_dataset("openai/gsm8k", "main", split="test", revision=GSM))
    if len(test) != 1319:
        raise ValueError("GSM8K test size changed")

    def question(row, i, split):
        gold = gsm8k.number(row["answer"].split("####")[-1])
        if gold is None:
            raise ValueError("Bad benchmark gold")
        return dict(
            id=f"{split}-{i:04d}",
            kind="gsm8k",
            gold=gold,
            **gsm8k.render(tok, row["question"], final_instruction=True),
        )

    samples["development"] += [
        question(train[i], i, "train")
        for i in random.Random(20261020).sample(range(len(train)), 64)
    ]
    samples["confirmation"] += [question(r, i, "test") for i, r in enumerate(test)]
    identities = {s: [x["article_id"] for x in docs] for s, docs in selections.items()}
    assert (
        sum(map(len, identities.values()))
        == len(set(sum(identities.values(), [])))
        == 112
    )
    ret = [
        r["prompt_hash"]
        for stage in samples.values()
        for r in stage
        if r["kind"] == "retrieval"
    ]
    assert len(ret) == len(set(ret)) == 464
    data = dict(
        schema=1,
        model=model,
        revision=revision,
        dtype=dtype,
        config=cfg.to_dict(),
        samples=samples,
        document_ids=identities,
        wiki_revision=WIKI,
        gsm_revision=GSM,
        gsm_parser="last_numeric_answer_label_v1",
        gsm_regex=REGEX,
        parser_source="unchanged from tools.kv.recent.prepare and run",
        selection=dict(
            examined=examined, eligible=384, exclude_initial_eligible=128, seed=20261020
        ),
        framing="Raw NLL without added special tokens; official chat template for GSM/retrieval; thinking disabled",
        gsm_exposure="Previously exposed standard benchmark, not a newly unseen holdout",
    )
    raw = gzip.compress(json.dumps(data, allow_nan=False).encode(), mtime=0)
    (a.out / "inputs.json.gz").write_bytes(raw)
    tok.save_pretrained(a.out / "tokenizer")
    identity = dict(
        model=model,
        revision=revision,
        dtype=dtype,
        inputs_sha256=hashlib.sha256(raw).hexdigest(),
        stage_counts={s: len(rows) for s, rows in samples.items()},
        document_ids=identities,
    )
    (a.out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity), flush=True)


if __name__ == "__main__":
    main()
