# SPDX-License-Identifier: GPL-2.0
"""Independent reconstruction of Plato's Key using factual recall on CounterFact."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import torch

from research.kv_translate.published.paired import compare

COUNTERFACT_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
MODELS = {
    "Qwen/Qwen2.5-3B": "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
    "HuggingFaceTB/SmolLM3-3B-Base": "d78a42f79198603e614095753484a04c10c2b940",
}
DEPTHS = (0.05, 0.15, 0.25, 0.35, 0.45)


@dataclass(frozen=True)
class Fact:
    id: int
    template: str
    subject: str
    answer: str

    @property
    def prompt(self):
        return self.template.format(self.subject).rstrip()

    @property
    def subject_end(self):
        return self.template.index("{}") + len(self.subject)

    @property
    def split(self):
        group = (
            int.from_bytes(
                hashlib.sha256(self.subject.casefold().encode()).digest()[:8], "big"
            )
            % 10
        )
        return "train" if group < 6 else "validation" if group < 8 else "test"


def load_facts(path, limit, seed):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != COUNTERFACT_SHA256:
        raise ValueError("CounterFact SHA-256 differs from the registered dataset")
    facts = {}
    for row in json.loads(data):
        rewrite = row["requested_rewrite"]
        template = rewrite["prompt"]
        subject = rewrite["subject"]
        answer = " " + rewrite["target_true"]["str"].strip()
        key = (template, subject.casefold())
        if template.count("{}") == 1 and subject and answer.strip():
            facts.setdefault(key, Fact(row["case_id"], template, subject, answer))
    result = list(facts.values())
    random.Random(seed).shuffle(result)
    return result[:limit]


@dataclass
class Rotation:
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    axes: torch.Tensor
    scale: torch.Tensor

    def __call__(self, x):
        return (
            x.float() - self.source_mean
        ) @ self.axes * self.scale + self.target_mean


def fit_rotation(source, target):
    """Fit centered, scaled semi-orthogonal Procrustes on row vectors."""
    source, target = source.float(), target.float()
    if (
        source.ndim != 2
        or target.ndim != 2
        or len(source) != len(target)
        or len(source) < 2
    ):
        raise ValueError(
            "Procrustes requires two aligned matrices with at least two rows"
        )
    x_mean, y_mean = source.mean(0), target.mean(0)
    x, y = source - x_mean, target - y_mean
    spread = torch.linalg.vector_norm(x)
    target_spread = torch.linalg.vector_norm(y)
    if not (
        torch.isfinite(spread)
        and spread > 0
        and torch.isfinite(target_spread)
        and target_spread > 0
    ):
        raise ValueError("Degenerate calibration spread")
    driver = "gesvd" if x.device.type == "cuda" else None
    u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False, driver=driver)
    return Rotation(x_mean, y_mean, u @ vh, target_spread / spread)


def tokenize_fact(tokenizer, fact, answer):
    prompt = tokenizer(fact.prompt, return_offsets_mapping=True)
    full = tokenizer(fact.prompt + answer, add_special_tokens=True)["input_ids"]
    prefix = prompt["input_ids"]
    if full[: len(prefix)] != prefix or len(full) == len(prefix):
        return None
    subject = next(
        (
            i
            for i, (start, end) in enumerate(prompt["offset_mapping"])
            if start < fact.subject_end <= end
        ),
        None,
    )
    if subject is None:
        return None
    return full[:-1], subject, len(prefix) - 1, full[len(prefix) :]


class FactModel:
    def __init__(self, model_id, batch_size, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        revision = MODELS[model_id]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .eval()
            .to(device)
        )
        self.layers = self.model.model.layers
        self.depths = sorted(
            {round(depth * (len(self.layers) - 1)) for depth in DEPTHS}
        )
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.tokenizations = {}

    def encode(self, fact, answer):
        key = (fact.id, answer)
        if key not in self.tokenizations:
            self.tokenizations[key] = tokenize_fact(self.tokenizer, fact, answer)
        return self.tokenizations[key]

    @torch.inference_mode()
    def run(self, requests, capture=False, swap_layer=None, swap=None):
        """Score greedy exact answer prefixes with one masked forward per batch."""
        known = torch.zeros(len(requests), dtype=torch.bool)
        captures = defaultdict(list)
        for start in range(0, len(requests), self.batch_size):
            end = min(start + self.batch_size, len(requests))
            valid = [
                (offset, encoded)
                for offset, (fact, answer) in enumerate(requests[start:end], start)
                if (encoded := self.encode(fact, answer)) is not None
            ]
            width = max((len(row[0]) for _, row in valid), default=1)
            ids = torch.full((len(valid), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            positions = []
            for i, (_, (tokens, subject, _, _)) in enumerate(valid):
                ids[i, : len(tokens)] = torch.tensor(tokens)
                mask[i, : len(tokens)] = 1
                positions.append(subject)
            captured = {}
            if valid:
                ids, mask = ids.to(self.device), mask.to(self.device)
                rows = torch.arange(len(valid), device=self.device)
                subjects = torch.tensor(positions, device=self.device)
                hooks = []

                def capture_layer(layer):
                    def record(_module, _args, output):
                        captured[layer] = output[rows, subjects].float().cpu()

                    return record

                def replace(_module, _args, output):
                    updated = output.clone()
                    updated[rows, subjects] = swap[[i for i, _ in valid]].to(
                        self.device, dtype=output.dtype
                    )
                    return updated

                if capture:
                    hooks.extend(
                        self.layers[i].register_forward_hook(capture_layer(i))
                        for i in self.depths
                    )
                if swap_layer is not None:
                    hooks.append(self.layers[swap_layer].register_forward_hook(replace))
                try:
                    tokens = (
                        self.model(input_ids=ids, attention_mask=mask, use_cache=False)
                        .logits.argmax(-1)
                        .cpu()
                    )
                finally:
                    for handle in hooks:
                        handle.remove()
                for i, (offset, (_, _, prediction, expected)) in enumerate(valid):
                    known[offset] = (
                        tokens[i, prediction : prediction + len(expected)].tolist()
                        == expected
                    )
            if capture:
                for layer in self.depths:
                    result = torch.zeros(end - start, self.model.config.hidden_size)
                    if valid:
                        result[[index - start for index, _ in valid]] = captured[layer]
                    captures[layer].append(result)
        return known, {layer: torch.cat(parts) for layer, parts in captures.items()}


def swap_pairs(facts, indices, limit, valid):
    groups = defaultdict(list)
    for i in indices:
        groups[facts[i].template].append(i)
    selected, used = [], set()
    for i in indices:
        donor = facts[i]
        identity = donor.subject.casefold()
        if identity in used:
            continue
        for j in groups[donor.template]:
            recipient = facts[j]
            other = recipient.subject.casefold()
            if (
                other == identity
                or other in used
                or recipient.answer.casefold() == donor.answer.casefold()
                or not valid(recipient, donor.answer)
            ):
                continue
            selected.append((i, j))
            used.update((identity, other))
            break
        if len(selected) == limit:
            break
    return selected


def neighbor_overlap(x, y, k=10):
    n = len(x)
    k = min(n - 1, k)
    if n <= 1:
        raise ValueError("Neighbor overlap needs at least two rows")

    def neighbors(z):
        z = torch.nn.functional.normalize(z.float(), dim=-1)
        scores = z @ z.T
        scores.fill_diagonal_(float("-inf"))
        return scores.topk(k).indices

    a, b = neighbors(x), neighbors(y)
    return float((a[:, :, None] == b[:, None, :]).any(2).float().mean())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--validation-pairs", type=int, default=128)
    parser.add_argument("--test-pairs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--reverse", action="store_true")
    return parser.parse_args(argv)


def evaluate(facts, source, target, source_states, target_states, common, args):
    train = common["train"][: args.train_limit]
    val = swap_pairs(facts, common["validation"], args.validation_pairs, target.encode)
    test = swap_pairs(facts, common["test"], args.test_pairs, target.encode)
    if min(len(train), len(val), len(test)) < 2:
        raise ValueError("Too few jointly recalled facts for a split")
    print(
        f"Calibration {len(train)}, validation pairs {len(val)}, test pairs {len(test)}",
        flush=True,
    )

    def fit(a, b, ids=train):
        return fit_rotation(
            source_states[a][ids].to(target.device),
            target_states[b][train].to(target.device),
        )

    def score(pairs, layer, replacement):
        requests = [(facts[j], facts[i].answer) for i, j in pairs]
        return target.run(requests, swap_layer=layer, swap=replacement)[0].tolist()

    def mapped(pairs, a, b, ids=train):
        return fit(a, b, ids)(
            source_states[a][[i for i, _ in pairs]].to(target.device)
        ).cpu()

    val_ids = common["validation"]
    selection = []
    for a in source.depths:
        for b in target.depths:
            agreement = neighbor_overlap(
                source_states[a][val_ids], target_states[b][val_ids]
            )
            accuracy = sum(score(val, b, mapped(val, a, b))) / len(val)
            selection.append(
                {
                    "source_layer": a,
                    "target_layer": b,
                    "validation_exact": accuracy,
                    "mknn": agreement,
                }
            )
            print(
                f"Layers {a}->{b}: validation={accuracy:.4f} mKNN={agreement:.4f}",
                flush=True,
            )
    best = max(selection, key=lambda row: row["validation_exact"])
    geometry = max(selection, key=lambda row: row["mknn"])
    a, b = best["source_layer"], best["target_layer"]
    g, h = geometry["source_layer"], geometry["target_layer"]
    shuffled = random.Random(args.seed).sample(train, len(train))
    donor_ids = [i for i, _ in test]
    arms = {
        "unmodified": score(test, None, None),
        "self_swap": score(test, b, target_states[b][donor_ids]),
        "rotation": score(test, b, mapped(test, a, b)),
        "shuffled_fit": score(test, b, mapped(test, a, b, shuffled)),
        "geometry_selected": score(test, h, mapped(test, g, h)),
    }
    paired = {}
    for baseline in ("self_swap", "shuffled_fit", "geometry_selected"):
        stats = compare(
            dict(enumerate(arms[baseline])), dict(enumerate(arms["rotation"]))
        )
        paired[baseline] = {
            k: stats[k] for k in ("difference", "difference_interval_95")
        }

    return {
        "calibration_facts": len(train),
        "validation_pairs": len(val),
        "selected": best,
        "geometry_selected": geometry,
        "selection_grid": selection,
        "n": len(test),
        "correct": {name: sum(values) for name, values in arms.items()},
        "rotation_paired_differences": paired,
        "rows": [
            {
                "donor": facts[i].id,
                "recipient": facts[j].id,
                **{name: values[ix] for name, values in arms.items()},
            }
            for ix, (i, j) in enumerate(test)
        ],
    }


def main(argv=None):
    args = parse_args(argv)
    if (
        min(
            args.batch_size,
            args.limit,
            args.train_limit,
            args.validation_pairs,
            args.test_pairs,
        )
        <= 0
    ):
        raise ValueError("Sample limits and batch size must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    facts = load_facts(args.facts, args.limit, args.seed)
    models = list(MODELS)
    if args.reverse:
        models.reverse()
    source, target = (FactModel(name, args.batch_size, args.device) for name in models)
    requests = [(fact, fact.answer) for fact in facts]
    source_correct, source_states = source.run(requests, capture=True)
    print(f"{models[0]} recalled {int(source_correct.sum())}/{len(facts)}", flush=True)
    target_correct, target_states = target.run(requests, capture=True)
    print(f"{models[1]} recalled {int(target_correct.sum())}/{len(facts)}", flush=True)
    common = defaultdict(list)
    for i in (source_correct & target_correct).nonzero().flatten().tolist():
        common[facts[i].split].append(i)
    result = evaluate(facts, source, target, source_states, target_states, common, args)
    result.update(
        models={name: MODELS[name] for name in models},
        direction=models,
        dataset_sha256=COUNTERFACT_SHA256,
        facts_screened=len(facts),
        source_recalled=int(source_correct.sum()),
        target_recalled=int(target_correct.sum()),
        jointly_recalled_by_split={k: len(v) for k, v in common.items()},
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {k: result[k] for k in ("selected", "geometry_selected", "n", "correct")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
