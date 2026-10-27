# SPDX-License-Identifier: MIT
"""Dataset generator checks. Tests that need the pinned tokenizer skip when
its files are not in the local Hugging Face cache."""

from __future__ import annotations

import pytest

from research.context_cart import dataset as D
from research.context_cart import prompts as P


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            revision="c1899de289a04d12100db370d81485cdf75e47ca",
            local_files_only=True,
        )
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"pinned tokenizer unavailable: {error}")
    return tokenizer


def test_normalize_and_exact_match():
    assert D.normalize_answer("Team Ridge.", "team") == "ridge"
    assert D.normalize_answer(" Locker 117 ", "locker") == "117"
    assert D.normalize_answer("badge code K4P-7Q2M", "badge") == "k4p-7q2m"
    assert D.exact_match("Varo Delmin.", "Varo Delmin", "mentor")
    assert not D.exact_match("K4P-7Q2N", "K4P-7Q2M", "badge")
    assert D.exact_match("Mira Oakes, Team Ridge.", "Mira Oakes, Ridge", "team")
    assert D.exact_match("Mira Oakes; 417", "Mira Oakes, 417", "locker")
    assert not D.exact_match("Ridge", "Mira Oakes, Ridge", "team")
    assert not D.exact_match("Thea Vance, Ridge", "Mira Oakes, Ridge", "team")


def test_worlds_are_exact_length_verified_and_deterministic(tok):
    T = D.Tokenizer(tok)
    worlds = D.generate_pool("validation", T, D.NameRegistry(), count=6)
    again = D.generate_pool("validation", T, D.NameRegistry(), count=6)
    for w, w2 in zip(worlds, again):
        assert w.n_tokens == D.SOURCE_TOKENS
        assert tok(w.source_text, add_special_tokens=False)["input_ids"] == w.source_ids
        assert w.to_dict() == w2.to_dict()
        families = sorted(q["family"] for q in w.questions)
        assert families == sorted(D.FAMILIES * 2)
        for q in w.questions:
            for part in q["answer"].split(", "):
                assert part in w.source_text
            assert len(tok(q["answer"], add_special_tokens=False)["input_ids"]) <= 14
            n = len(P.prompt_ids(tok, P.user_message("learned", q["text"]))) + len(
                P.answer_ids(tok, q["answer"])
            )
            assert n <= 128
        assert 0.2 < w.filler_fraction < 0.6
    names = [n for w in worlds for n in w.live]
    assert len(names) == len(set(names))


def test_pools_disjoint_and_shifted_templates(tok):
    T = D.Tokenizer(tok)
    registry = D.NameRegistry()
    ordinary = D.generate_pool("test_ordinary", T, registry, count=3)
    shifted = D.generate_pool("test_shifted", T, registry, count=3)
    assert all(w.stratum == "ordinary" for w in ordinary)
    assert all(w.stratum == "shifted" for w in shifted)
    assert all(w.template_set == "B" for w in shifted)
    assert not (
        {w.source_digest for w in ordinary} & {w.source_digest for w in shifted}
    )
    codes = [f["badge"] for w in ordinary + shifted for f in w.live.values()]
    assert len(codes) == len(set(codes))


def test_updates_follow_originals_and_answers_are_latest(tok):
    T = D.Tokenizer(tok)
    world = D.generate_pool("train", T, D.NameRegistry(), count=1)[0]
    for update in world.updates:
        base = world.source_text.index(f"{update['name']}")
        correction = next(
            r
            for r in world.records
            if r["rid"] == f"update:{update['name']}:{update['attribute']}"
        )
        original = next(
            r
            for r in world.records
            if r["rid"] == f"live:{update['name']}:{update['attribute']}"
        )
        assert world.source_text.index(correction["text"]) > world.source_text.index(
            original["text"]
        )
        assert base >= 0
    for q in world.questions:
        if q["family"] == "updated":
            update = next(
                u
                for u in world.updates
                if u["name"] == q["entity"] and u["attribute"] == q["attribute"]
            )
            assert q["answer"] == update["new"] != update["old"]
        if q["family"] == "linked":
            mentor, value = q["answer"].split(", ")
            assert mentor == world.live[q["entity"]]["mentor"]
            assert value == world.live[mentor][q["attribute"]]
            assert len(q["support"]) == 2


def test_diagnostic_fit_and_eval_questions(tok):
    T = D.Tokenizer(tok)
    world = D.generate_pool("diagnostic", T, D.NameRegistry(), count=2)[1]
    assert len(world.fit_questions) == 32 and len(world.questions) == 8
    fit_families = {f: 0 for f in D.FAMILIES}
    for q in world.fit_questions:
        fit_families[q["family"]] += 1
    assert all(v == 8 for v in fit_families.values())
    fit_texts = {q["text"] for q in world.fit_questions}
    assert not any(q["text"] in fit_texts for q in world.questions)
    assert all(q["template"] == "C" for q in world.questions)
    # every record an evaluation answer depends on is supervised by a fit
    # question, so no unseen fact can move the evaluation
    assert D.uncovered_support(world.fit_questions, world.questions) == {}
    covered = D.fit_support(world.fit_questions)
    for q in world.questions:
        assert set(q["support"]) <= covered
    # the counterexample the coverage rule exists for: a two-hop question on
    # a record no fit question touches is reported, not accepted
    linked = next(q for q in world.questions if q["family"] == "linked")
    fit = [q for q in world.fit_questions if linked["support"][1] not in q["support"]]
    missing = D.uncovered_support(fit, [linked])
    assert missing == {linked["qid"]: [linked["support"][1]]}


def test_derangement_and_questions_only(tok):
    T = D.Tokenizer(tok)
    worlds = [
        w.to_dict()
        for w in D.generate_pool("calibration", T, D.NameRegistry(), count=10)
    ]
    mapping = D.derangement(worlds, "unit")
    by_id = {w["world_id"]: w for w in worlds}
    assert set(mapping) == set(by_id) and set(mapping.values()) == set(by_id)
    for receiver, donor in mapping.items():
        assert receiver != donor
        assert D.wrong_world_differs(by_id[receiver], by_id[donor])
    rows = D.questions_only(worlds)
    assert len(rows) == 80
    assert all("answer" not in r and "support" not in r for r in rows)
    panel = D.validation_panel(worlds, count=4)
    assert len(panel) == 16
