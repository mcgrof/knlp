# SPDX-License-Identifier: MIT
import pytest
import torch

from tools.kv.native.routing_oracle import raw_suffix, require_equal, require_placement


def test_raw_suffix_retains_zero_repeats_remap_tail_and_decode_token_offset():
    expected = raw_suffix(
        [0, 2, 35, 52],
        [17, 33, 17],
        [[7, 6, 0], [3, 0, 2], [5, 3, 0]],
        decode_requests=1,
    )
    assert expected == {
        "lengths": [33, 17],
        "source_ids": [3, 0, 2, 5, 3],
        "query_offset": 2,
        "query_indptr": [0, 33, 50],
        "page_indptr": [0, 3, 5],
        "tails": [1, 1],
    }


def test_fixed_faults_are_rejected_from_independent_expected_mapping():
    old = raw_suffix(
        [0, 2, 35, 52],
        [17, 33, 17],
        [[7, 6, 0], [3, 0, 2], [5, 3, 0]],
        decode_requests=1,
    )
    new = raw_suffix(
        [0, 2, 35, 52],
        [17, 33, 17],
        [[6, 7, 0], [2, 3, 0], [0, 5, 0]],
        decode_requests=1,
    )
    ids = torch.tensor(old["source_ids"])
    for wrong in (
        torch.tensor([7, 6, 3, 0, 2]),
        torch.tensor([7, 0, 2, 5, 3]),
        torch.tensor(new["source_ids"]),
    ):
        with pytest.raises(RuntimeError):
            require_equal(wrong, ids, "injected page fault")
    q = torch.zeros(52, 3, 4)
    out = torch.zeros_like(q)
    require_placement(q[2:], out[2:], q, out, 2)
    with pytest.raises(RuntimeError):
        require_placement(q[2:], out[1:-1], q, out, 2)


def test_nonfinite_equality_rejected():
    x = torch.tensor([float("inf")])
    with pytest.raises(RuntimeError):
        require_equal(x, x, "nonfinite")


def test_actual_forward_and_run_finalizer_create_one_complete_receipt(tmp_path):
    """Execute the actual recorder control flow with its exclusive file writer."""
    import ast
    import inspect
    import json
    from types import SimpleNamespace

    from tools.kv.native import routing_oracle
    from tools.kv.native.instrument import save

    tree = ast.parse(inspect.getsource(routing_oracle.validate_routing_layers))
    context = tree.body[0]
    forward = next(
        n
        for n in context.body
        if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    run = next(
        n for n in context.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    finalizer = next(n for n in run.body if isinstance(n, ast.Try)).finalbody
    code = compile(
        ast.fix_missing_locations(ast.Module(body=finalizer, type_ignores=[])),
        "actual-run-finalizer",
        "exec",
    )
    rows = []
    current = []
    scope = {
        "rows": rows,
        "current": current,
        "destination": tmp_path,
        "torch": torch,
        "save": save,
        "_save_forward_receipt": routing_oracle._save_forward_receipt,
    }

    def old_forward(*args, **kwargs):
        scope["row"] = {
            "layer": "layer-0",
            "routing_equivalence_passed": True,
            "forward_placement_verified": False,
        }
        exec(code, scope)  # noqa: S102 - execute this repository's extracted finalizer
        return args[7]

    scope["old_forward"] = old_forward
    exec(  # noqa: S102 - execute this repository's extracted forward
        compile(
            ast.fix_missing_locations(ast.Module(body=[forward], type_ignores=[])),
            "actual-forward",
            "exec",
        ),
        scope,
    )
    q = torch.ones(1, 2)
    out = torch.empty_like(q)
    scope["forward"](
        None,
        SimpleNamespace(layer_name="layer-0"),
        q,
        None,
        None,
        None,
        SimpleNamespace(_routing_oracle=True, num_actual_tokens=1, num_decode_tokens=0),
        out,
    )
    assert not current
    assert len(list(tmp_path.glob("call-*.json"))) == 1
    receipt = json.loads((tmp_path / "call-0000.json").read_text())
    assert (
        receipt["routing_equivalence_passed"] and receipt["forward_placement_verified"]
    )
    assert receipt["decode_tokens_checked"] == 0
