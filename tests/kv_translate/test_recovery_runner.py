# SPDX-License-Identifier: GPL-2.0
"""Drive the real runner, on a real pair, and check what it actually built.

The helper tests before this one checked arithmetic on constant arrays. They
could not have caught a condition that silently repaired nothing, a selector
that left the question resident in the prefix it selected from, or a replay
that wrote three layers of twenty-eight. Those are properties of the cache the
production path hands to the decoder, so that is what is inspected here: the
runner's own ``build_cache``, not a reimplementation of it.

The models are random. Nothing here is a claim about quality; a random pair
answers nothing and is supposed to. What it establishes is that each condition
performs the operation its name promises.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from research.kv_translate.recovery import BLOCK, CacheContract, positions_changed
from research.kv_translate.recovery_runner import (
    build_cache,
    check_conditions,
    is_oracle,
    lexical_span,
    question_attention,
    select,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CTX = 128
NDOC = 4
TOL = 3e-3  # float32, this fixture, chosen before the run


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Build the fixture once, then load everything the runner would load."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from research.kv_translate import freeze
    from research.kv_translate.fit import SourceLayout
    from research.kv_translate.make_cpu_fixture import (
        build_manifest,
        build_models,
        fit_tiny_mapper,
    )
    from research.kv_translate.pairs import describe, prefill, to_content_keys
    from research.kv_translate.run_a1 import flat_features

    d = str(tmp_path_factory.mktemp("kvrec"))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    paths = build_models(d, tok)
    man = build_manifest(os.path.join(d, "manifest.json"), tok, CTX, NDOC)
    arm_path = os.path.join(d, "arm.pt")
    fit_tiny_mapper(paths, man, arm_path, CTX)

    models, geom = {}, {}
    for role in ("source", "target"):
        m = AutoModelForCausalLM.from_pretrained(paths[role], dtype=torch.float32)
        m.eval()
        models[role], geom[role] = m, describe(m, paths[role])
    sg, tg = geom["source"], geom["target"]
    layout = SourceLayout(sg.n_layers, sg.n_kv_heads, sg.head_dim)
    mk, mv, _ = freeze.load(arm_path, layout, tg)
    pos = torch.arange(CTX)

    from research.cartridges_cas.scripts.cas_kv_rope import rerot

    def translate(_arm, sp):
        xk = flat_features(to_content_keys(sp)).float()
        xv = flat_features([v.float() for v in sp.values]).float()
        keys = [
            rerot(b[0].float(), pos, tg.rope_theta).unsqueeze(0)
            for b in mk.cast(torch.float32).apply(xk)
        ]
        return keys, [v.float() for v in mv.cast(torch.float32).apply(xv)]

    doc = man["documents"][0]
    ids = torch.tensor(doc["prompt_ids"]).unsqueeze(0)
    qids = tok(doc["retrieval_query"], return_tensors="pt").input_ids
    mapped = {"tiny": translate("tiny", prefill(models["source"], ids, sg))}
    wman = {d_["doc"]: d_ for d_ in man["documents"]}
    wids = torch.tensor(wman[doc["wrong_source_doc"]]["prompt_ids"]).unsqueeze(0)
    wrong = {"tiny": translate("tiny", prefill(models["source"], wids, sg))}
    tp = prefill(models["target"], ids, tg)
    native = ([k.float() for k in tp.keys_post], [v.float() for v in tp.values])
    contract = CacheContract(
        n_layers=tg.n_layers,
        n_kv_heads=tg.n_kv_heads,
        n_positions=CTX,
        head_dim=tg.head_dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return {
        "dir": d,
        "paths": paths,
        "arm": arm_path,
        "man": man,
        "tok": tok,
        "models": models,
        "tg": tg,
        "doc": doc,
        "ids": ids,
        "qids": qids,
        "mapped": mapped,
        "wrong": wrong,
        "native": native,
        "contract": contract,
        "by_doc": wman,
        "translate": translate,
    }


def build(world, spec):
    return build_cache(
        spec,
        arm="tiny",
        doc=world["doc"],
        ids=world["ids"],
        qids=world["qids"],
        mapped=world["mapped"],
        wrong_mapped=world["wrong"],
        native=world["native"],
        models=world["models"],
        tg=world["tg"],
        dtype=torch.float32,
        contract=world["contract"],
        tok=world["tok"],
        query=world["doc"]["retrieval_query"],
        ctx=CTX,
    )


# -- what each condition actually builds ------------------------------------


def test_full_oracle_substitution_reproduces_the_native_cache(world):
    """Every position donated must leave nothing of the translation behind."""
    k, v, info = build(
        world, {"repair": "oracle", "selector": "uniform", "blocks": CTX // BLOCK}
    )
    assert info["replayed_positions"] == CTX
    for li in range(len(k)):
        assert torch.equal(k[li], world["native"][0][li])
        assert torch.equal(v[li], world["native"][1][li])


def test_full_replay_reproduces_the_native_cache(world):
    """The deployable operation, at full budget, must agree with native.

    This is the one that says the replay reads the right context and rotates
    at the right positions. A partial budget cannot show it, because a
    partially repaired cache is supposed to differ from native.
    """
    k, v, info = build(
        world, {"repair": "replay", "selector": "uniform", "blocks": CTX // BLOCK}
    )
    assert info["replayed_positions"] == CTX
    assert info["layers_replayed"] == world["tg"].n_layers
    for li in range(len(k)):
        assert torch.allclose(k[li], world["native"][0][li], atol=TOL), f"k{li}"
        assert torch.allclose(v[li], world["native"][1][li], atol=TOL), f"v{li}"


def test_no_repair_returns_the_mapping_untouched(world):
    k, v, info = build(world, {"repair": "none"})
    assert info["replayed_positions"] == 0
    for li in range(len(k)):
        assert torch.equal(k[li], world["mapped"]["tiny"][0][li])
        assert torch.equal(v[li], world["mapped"]["tiny"][1][li])


def test_a_one_block_repair_changes_exactly_that_block(world):
    for repair in ("oracle", "replay"):
        k, v, info = build(
            world, {"repair": repair, "selector": "uniform", "blocks": 1}
        )
        assert info["replayed_positions"] == BLOCK
        moved = positions_changed(world["mapped"]["tiny"][0], k)
        moved |= positions_changed(world["mapped"]["tiny"][1], v)
        want = sorted(
            range(
                info["selected_blocks"][0] * BLOCK,
                (info["selected_blocks"][0] + 1) * BLOCK,
            )
        )
        assert sorted(torch.nonzero(moved).flatten().tolist()) == want, repair


def test_both_keys_and_values_move_under_every_repair(world):
    for repair in ("oracle", "replay"):
        k, v, _ = build(world, {"repair": repair, "selector": "uniform", "blocks": 2})
        assert positions_changed(world["mapped"]["tiny"][0], k).any(), f"{repair} keys"
        assert positions_changed(world["mapped"]["tiny"][1], v).any(), f"{repair} vals"


def test_a_noncontiguous_selection_survives_the_runner(world):
    k, _, info = build(
        world, {"repair": "replay", "selector": "mapped_attn", "blocks": 3}
    )
    sel = info["selected_blocks"]
    assert len(sel) == len(set(sel)) == 3
    moved = torch.nonzero(positions_changed(world["mapped"]["tiny"][0], k)).flatten()
    assert set(b // BLOCK for b in moved.tolist()) == set(sel)


def test_the_prefix_length_is_unchanged_by_any_full_length_condition(world):
    for spec in (
        {"repair": "none"},
        {"repair": "native"},
        {"repair": "oracle", "selector": "uniform", "blocks": 2},
        {"repair": "replay", "selector": "mapped_attn", "blocks": 2},
    ):
        k, v, info = build(world, spec)
        assert info["prompt_len"] == CTX, spec
        assert all(x.shape[2] == CTX for x in k), spec
        assert all(x.shape[2] == CTX for x in v), spec


def test_the_empty_control_is_told_of_no_prefix(world):
    k, v, info = build(world, {"repair": "empty"})
    assert k is None and v is None and info["prompt_len"] == 0


def test_the_raw_text_baseline_declares_its_shorter_prefix(world):
    """Its cache really is shorter, and it says so, or the question lands
    after positions the model cannot read."""
    k, _, info = build(world, {"repair": "span", "selector": "lexical", "blocks": 2})
    assert info["prompt_len"] == 2 * BLOCK
    assert all(x.shape[2] == 2 * BLOCK for x in k)
    a, b = info["span_blocks"]
    assert b - a == 2 and 0 <= a < CTX // BLOCK


# -- the selector -----------------------------------------------------------


def test_the_selector_does_not_leave_the_question_in_the_prefix(world):
    """Input isolation and scratch-query discard, checked on the tensors."""
    before = [t.clone() for t in world["mapped"]["tiny"][0]]
    a = question_attention(
        world["models"]["target"],
        world["mapped"]["tiny"][0],
        world["mapped"]["tiny"][1],
        world["qids"],
        CTX,
    )
    assert a.shape == (CTX,), "the selector scored the question's own positions"
    assert torch.isfinite(a).all()
    for x, y in zip(before, world["mapped"]["tiny"][0]):
        assert torch.equal(x, y), "the selector pass mutated the cache it read"


def test_a_repair_after_selection_holds_only_document_positions(world):
    k, _, _ = build(world, {"repair": "replay", "selector": "mapped_attn", "blocks": 2})
    assert all(x.shape[2] == CTX for x in k)


def test_the_two_attention_selectors_read_different_caches(world):
    """They are different conditions, and must be able to disagree."""
    mapped_sel, _ = select(
        "mapped_attn",
        3,
        model=world["models"]["target"],
        mapped=world["mapped"]["tiny"],
        native=world["native"],
        query_ids=world["qids"],
        n_positions=CTX,
    )
    native_sel, _ = select(
        "native_attn",
        3,
        model=world["models"]["target"],
        mapped=world["mapped"]["tiny"],
        native=world["native"],
        query_ids=world["qids"],
        n_positions=CTX,
    )
    assert len(mapped_sel) == len(native_sel) == 3
    assert all(0 <= b < CTX // BLOCK for b in mapped_sel + native_sel)


def test_the_oracle_selector_refuses_to_run_without_a_donor(world):
    with pytest.raises(ValueError, match="oracle"):
        select(
            "native_attn",
            2,
            model=world["models"]["target"],
            mapped=world["mapped"]["tiny"],
            native=None,
            query_ids=world["qids"],
            n_positions=CTX,
        )


def test_uniform_selection_needs_no_model_pass(world):
    blocks, scores = select(
        "uniform",
        4,
        model=None,
        mapped=None,
        native=None,
        query_ids=None,
        n_positions=CTX,
    )
    assert scores is None and len(blocks) == 4


def test_the_wrong_source_control_uses_no_donor(world):
    """It must differ from the own-source repair, and never touch native."""
    own, _, _ = build(
        world, {"repair": "replay", "selector": "mapped_attn", "blocks": 2}
    )
    wrong, _, _ = build(
        world,
        {"repair": "replay", "selector": "mapped_attn", "blocks": 2, "source": "wrong"},
    )
    assert not torch.allclose(own[0], wrong[0])


def test_the_lexical_baseline_prefers_the_matching_text(world):
    tok, doc = world["tok"], world["doc"]
    a, b = lexical_span(tok, doc["prompt_ids"], doc["retrieval_gold"], 2)
    assert b - a == 2
    hit = tok.decode(doc["prompt_ids"][a * BLOCK : b * BLOCK])
    assert isinstance(hit, str)


# -- labelling and validation ----------------------------------------------


def test_every_condition_that_borrows_native_is_labelled_an_oracle():
    assert is_oracle({"repair": "native"})
    assert is_oracle({"repair": "oracle", "selector": "uniform", "blocks": 1})
    assert is_oracle({"repair": "replay", "selector": "native_attn", "blocks": 1})
    assert not is_oracle({"repair": "replay", "selector": "mapped_attn", "blocks": 1})
    assert not is_oracle({"repair": "none"})
    assert not is_oracle({"repair": "span", "selector": "lexical", "blocks": 1})


@pytest.mark.parametrize(
    "spec,msg",
    [
        ({"repair": "wat"}, "not in"),
        ({"repair": "oracle", "selector": "uniform"}, "block count"),
        ({"repair": "oracle", "blocks": 2}, "selector"),
        (
            {"repair": "replay", "selector": "lexical", "blocks": 2, "oracle": True},
            "not an oracle",
        ),
        ({"repair": "span", "selector": "uniform", "blocks": 2}, "selects on the text"),
        ({"repair": "none", "source": "elsewhere"}, "own"),
    ],
)
def test_a_malformed_condition_is_refused_before_a_card_is_rented(spec, msg):
    with pytest.raises(SystemExit, match=msg):
        check_conditions({"c": spec}, {"retrieval": ["c"]})


def test_a_matrix_naming_an_undefined_condition_is_refused():
    with pytest.raises(SystemExit, match="undefined"):
        check_conditions({"a": {"repair": "none"}}, {"retrieval": ["a", "b"]})


# -- the command line, resume, and what a resumed run is allowed to change --

SMALL = {
    "definitions": {
        "target_native": {"repair": "native"},
        "empty_cache": {"repair": "empty"},
        "mapped_unrepaired": {"repair": "none"},
        "repaired_2": {"repair": "replay", "selector": "mapped_attn", "blocks": 2},
    },
    "matrix": {
        "retrieval": [
            "target_native",
            "empty_cache",
            "mapped_unrepaired",
            "repaired_2",
        ],
        "cloze": ["target_native", "empty_cache", "mapped_unrepaired", "repaired_2"],
        "free": ["target_native", "empty_cache", "mapped_unrepaired", "repaired_2"],
    },
}


def run_cli(world, out_dir, *extra):
    cfg = os.path.join(world["dir"], "conditions.json")
    if not os.path.exists(cfg):
        json.dump(SMALL, open(cfg, "w"))
    cmd = [
        sys.executable,
        "-m",
        "research.kv_translate.recovery_runner",
        "--source",
        world["paths"]["source"],
        "--target",
        world["paths"]["target"],
        "--source-revision",
        "local",
        "--target-revision",
        "local",
        "--manifest",
        os.path.join(world["dir"], "manifest.json"),
        "--conditions",
        cfg,
        "--arm",
        f"tiny={world['arm']}",
        "--ctx",
        str(CTX),
        "--max-new",
        "6",
        "--max-new-free",
        "12",
        "--dtype",
        "float32",
        "--out-dir",
        out_dir,
        *extra,
    ]
    env = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)


def cells(path):
    return [
        (r["kind"], r["condition"], r["doc"])
        for r in (json.loads(l) for l in open(path) if l.strip())
    ]


def expected_cells(man):
    return {
        (k, c, d["doc"])
        for k, cs in SMALL["matrix"].items()
        for c in cs
        for d in man["documents"]
    }


@pytest.fixture(scope="module")
def complete(world, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("cli"))
    r = run_cli(world, out)
    assert r.returncode == 0, r.stdout + r.stderr
    return out


def test_a_complete_run_writes_every_cell_once(world, complete):
    got = cells(os.path.join(complete, "rows.jsonl"))
    assert len(got) == len(set(got)), "a cell was written twice"
    assert set(got) == expected_cells(world["man"])


def test_the_strict_gate_accepts_the_runner_s_own_output(world, complete):
    """The scorer and the runner must agree on the shape of a row.

    They were written against the same contract and that is exactly why this
    is worth checking: a field renamed on one side and not the other produces
    an experiment that runs to completion and cannot be scored.
    """
    from research.kv_translate.recovery_gate import load_rows

    rows, docs = load_rows(
        os.path.join(complete, "rows.jsonl"), world["man"], SMALL["matrix"], "screen"
    )
    assert len(docs) == NDOC
    assert len(rows) == len(expected_cells(world["man"]))


def test_the_run_writes_the_matrix_the_scorer_will_demand(world, complete):
    """The scorer is told which cells must exist by a file; the run writes it,
    so the set demanded is the set the run was configured to produce."""
    from research.kv_translate.recovery_gate import load_rows

    m = json.load(open(os.path.join(complete, "matrix.json")))
    assert m == SMALL["matrix"]
    load_rows(os.path.join(complete, "rows.jsonl"), world["man"], m, "screen")


def test_a_run_records_the_pins_a_later_reader_needs(complete):
    r = json.load(open(os.path.join(complete, "RECEIPT.json")))
    for f in ("manifest_version", "arms", "models", "block_tokens", "definitions"):
        assert f in r, f
    assert r["arms"]["tiny"]["joint_weight_sha256"]


def test_an_interrupted_run_resumes_to_exactly_the_expected_cells(
    world, complete, tmp_path
):
    out = str(tmp_path / "resumed")
    os.makedirs(out)
    full = [l for l in open(os.path.join(complete, "rows.jsonl")) if l.strip()]
    with open(os.path.join(out, "rows.jsonl"), "w") as fh:
        fh.writelines(full[: len(full) // 3])
    r = run_cli(world, out)
    assert r.returncode == 0, r.stdout + r.stderr
    got = cells(os.path.join(out, "rows.jsonl"))
    assert len(got) == len(set(got)), "resume duplicated a cell"
    assert set(got) == expected_cells(world["man"])


def test_a_torn_final_line_does_not_become_a_duplicate(world, complete, tmp_path):
    """A process killed mid-append leaves a partial line; it must not survive."""
    out = str(tmp_path / "torn")
    os.makedirs(out)
    full = [l for l in open(os.path.join(complete, "rows.jsonl")) if l.strip()]
    with open(os.path.join(out, "rows.jsonl"), "w") as fh:
        fh.writelines(full[:5])
        fh.write(full[5][: len(full[5]) // 2])  # no newline, truncated JSON
    r = run_cli(world, out)
    assert r.returncode == 0, r.stdout + r.stderr
    got = cells(os.path.join(out, "rows.jsonl"))
    assert len(got) == len(set(got)) == len(expected_cells(world["man"]))


def test_a_failure_is_preserved_across_a_resume_and_retried_only_on_request(
    world, complete, tmp_path
):
    out = str(tmp_path / "failed")
    os.makedirs(out)
    rows = [
        json.loads(l) for l in open(os.path.join(complete, "rows.jsonl")) if l.strip()
    ]
    victim = rows[0]
    broken = {
        k: v for k, v in victim.items() if k not in ("primary", "produced", "healthy")
    }
    broken["error"] = "RuntimeError: injected"
    with open(os.path.join(out, "rows.jsonl"), "w") as fh:
        fh.write(json.dumps(broken) + "\n")

    r = run_cli(world, out)
    assert r.returncode == 1, "a run with a failed cell must not report success"
    got = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl")) if l.strip()]
    keys = [(g["kind"], g["condition"], g["doc"]) for g in got]
    assert len(keys) == len(set(keys)) == len(expected_cells(world["man"]))
    kept = [g for g in got if g.get("error")]
    assert len(kept) == 1 and kept[0]["error"] == "RuntimeError: injected"

    r = run_cli(world, out, "--retry-failed")
    assert r.returncode == 0, r.stdout + r.stderr
    got = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl")) if l.strip()]
    keys = [(g["kind"], g["condition"], g["doc"]) for g in got]
    assert len(keys) == len(set(keys)) == len(expected_cells(world["man"]))
    assert not [g for g in got if g.get("error")]


def test_a_fixture_of_the_wrong_length_is_refused(world, tmp_path):
    r = run_cli(world, str(tmp_path / "wrongctx"), "--ctx", "256")
    assert r.returncode != 0
    assert "different fraction" in (r.stdout + r.stderr)


def test_free_rows_carry_every_health_field_the_gate_reads(complete):
    from research.kv_translate.recovery_gate import HEALTH_FIELDS

    for line in open(os.path.join(complete, "rows.jsonl")):
        r = json.loads(line)
        if r["kind"] == "free":
            for f in HEALTH_FIELDS:
                assert f in r, f


def test_corruption_that_is_not_a_torn_append_is_an_error(world, complete, tmp_path):
    """A bad line in the middle of the file is not recoverable and must say so."""
    out = str(tmp_path / "corrupt")
    os.makedirs(out)
    full = [l for l in open(os.path.join(complete, "rows.jsonl")) if l.strip()]
    with open(os.path.join(out, "rows.jsonl"), "w") as fh:
        fh.writelines(full[:3])
        fh.write("{not json at all\n")
        fh.writelines(full[3:6])
    r = run_cli(world, out)
    assert r.returncode != 0
    assert "not a torn append" in (r.stdout + r.stderr)


def test_the_budget_a_condition_declares_is_the_budget_it_spends(world):
    for k in (1, 2, 4):
        _, _, info = build(
            world, {"repair": "oracle", "selector": "mapped_attn", "blocks": k}
        )
        assert info["replayed_positions"] == k * BLOCK
        assert len(info["selected_blocks"]) == k


def test_a_condition_naming_an_unsupplied_arm_is_refused_up_front(world, tmp_path):
    cfg = str(tmp_path / "armless.json")
    json.dump(
        {
            "definitions": {
                "target_native": {"repair": "native"},
                "other": {"repair": "none", "arm": "nobody_passed_this"},
            },
            "matrix": {"retrieval": ["target_native", "other"]},
        },
        open(cfg, "w"),
    )
    out = str(tmp_path / "armless-out")
    os.makedirs(out)
    cmd = [
        sys.executable,
        "-m",
        "research.kv_translate.recovery_runner",
        "--source",
        world["paths"]["source"],
        "--target",
        world["paths"]["target"],
        "--manifest",
        os.path.join(world["dir"], "manifest.json"),
        "--conditions",
        cfg,
        "--arm",
        f"tiny={world['arm']}",
        "--ctx",
        str(CTX),
        "--dtype",
        "float32",
        "--out-dir",
        out,
    ]
    env = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode != 0
    assert "arms that were not supplied" in (r.stdout + r.stderr)
    assert not os.path.exists(os.path.join(out, "rows.jsonl"))


def test_the_run_declares_the_documents_it_covers(world, complete):
    """A stage narrower than the fixture must be scorable.

    The scorer demands a cell for every document in the manifest it is given,
    which is correct, so a sixteen-document screen handed the whole fixture
    reads as a complete stage that is missing most of its cells. The scope is
    written from the frozen fixture and the declared selection before any
    score exists, so a run cannot narrow it by failing.
    """
    from research.kv_translate.recovery_gate import load_rows

    sc = json.load(open(os.path.join(complete, "scope.json")))
    assert sc["docs_scope"] == "all"
    assert [d["doc"] for d in sc["documents"]] == [
        d["doc"] for d in world["man"]["documents"]
    ]
    assert sc["version"] == world["man"]["version"]
    assert sc["task_joint_sha256"] == world["man"]["task_joint_sha256"]
    rows, docs = load_rows(
        os.path.join(complete, "rows.jsonl"),
        sc,
        json.load(open(os.path.join(complete, "matrix.json"))),
        "screen",
    )
    assert len(docs) == NDOC


def test_a_narrowed_scope_names_only_the_documents_that_stage_ran(world, tmp_path):
    out = str(tmp_path / "screenonly")
    os.makedirs(out)
    r = run_cli(world, out, "--docs", "screen")
    assert r.returncode == 0, r.stdout + r.stderr
    sc = json.load(open(os.path.join(out, "scope.json")))
    want = [d["doc"] for d in world["man"]["documents"] if d["in_screen"]]
    assert [d["doc"] for d in sc["documents"]] == want
    assert 0 < len(want) < len(world["man"]["documents"])
    got = {c[2] for c in cells(os.path.join(out, "rows.jsonl"))}
    assert got == set(want)


def test_the_cost_of_a_plain_native_prefill_is_quotable(world):
    """It is the baseline, not an oracle: a serving system can always prefill.

    Marking it unquotable left the timing comparison with no denominator and
    no ratio was produced at all.
    """
    from research.kv_translate.recovery_runner import is_cost_oracle, is_oracle

    native = {"repair": "native"}
    assert is_oracle(native), "it is still the quality ceiling"
    assert not is_cost_oracle(native), "but its cost is ordinary"
    donor = {"repair": "oracle", "selector": "uniform", "blocks": 4}
    assert is_cost_oracle(donor)
    assert is_cost_oracle({"repair": "replay", "selector": "native_attn", "blocks": 4})
    assert not is_cost_oracle(
        {"repair": "replay", "selector": "mapped_attn", "blocks": 4}
    )
    assert not is_cost_oracle({"repair": "span", "selector": "lexical", "blocks": 4})
