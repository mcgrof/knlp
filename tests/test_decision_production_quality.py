# SPDX-License-Identifier: MIT
"""Protect reference scoring and teacher-token identity at the HTTP boundary."""

import asyncio
import json

import pytest

from tools.kv.decision.production_quality import body, cohort, score, teacher_rows


REGEX = r"(?i)the\s+answer\s+is\s*[:=]?\s*\**\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)"


class Tokenizer:
    def __init__(self, text):
        self.text = text

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return self.text


def natural(text, *, tokens=None, reason="stop", stop=None):
    return {
        "choices": [
            {
                "token_ids": [1, 2] if tokens is None else tokens,
                "text": text,
                "finish_reason": reason,
                "stop_reason": stop,
            }
        ]
    }


def math_sample():
    return {"id": "fixed-1", "kind": "gsm8k", "gold": "18", "token_ids": [7, 8]}


def test_last_answer_and_length_cap_keep_reference_semantics():
    text = "The answer is 6. Correction: The answer is 18."
    row = score(
        math_sample(),
        natural(text, tokens=[1] * 2048, reason="length"),
        {"gsm_regex": REGEX},
        Tokenizer(text),
        [9],
    )
    assert row["correct"] and row["truncated"] and not row["eos"]
    assert row["answer"] == "18" and row["output_tokens"] == 2048
    text = "18"
    row = score(
        math_sample(), natural(text), {"gsm_regex": REGEX}, Tokenizer(text), [9]
    )
    assert not row["correct"] and row["parser_failure"]


@pytest.mark.parametrize(
    "bad", ["text", "early_cap", "unfrozen_stop", "internal_eos", "finish"]
)
def test_corrupt_response_evidence_is_not_scored(bad):
    text = "The answer is 18"
    response = natural(text)
    choice = response["choices"][0]
    if bad == "text":
        choice["text"] = "The answer is 19"
    elif bad == "early_cap":
        choice["finish_reason"] = "length"
    elif bad == "unfrozen_stop":
        choice["stop_reason"] = "\n"
    elif bad == "internal_eos":
        choice["token_ids"] = [9, 1]
    else:
        choice["finish_reason"] = "abort"
    with pytest.raises(ValueError):
        score(math_sample(), response, {"gsm_regex": REGEX}, Tokenizer(text), [9])


def test_raw_eos_retained_but_excluded_from_reference_output_count():
    text = "The answer is 18"
    row = score(
        math_sample(),
        natural(text, tokens=[1, 2, 9], stop=9),
        {"gsm_regex": REGEX},
        Tokenizer(text),
        [9],
    )
    assert row["raw_api_token_ids"] == [1, 2, 9]
    assert row["tokens"] == [1, 2] and row["eos"]


def teacher_fixture(path):
    sample = {
        "id": "doc",
        "kind": "nll",
        "length": 2,
        "article_id": "17",
        "token_ids": list(range(131)),
    }
    rows = [
        {
            "sample_id": "doc",
            "request_id": "request-doc",
            "step": i,
            "target": i + 2,
            "nll": 999.0 if i == 0 else i / 128,
        }
        for i in range(129)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return sample, rows


def test_teacher_scores_exactly_the_128_decode_targets(tmp_path):
    path = tmp_path / "teacher.jsonl"
    sample, rows = teacher_fixture(path)
    result = teacher_rows([sample], path)[0]
    assert result["nll"] == [i / 128 for i in range(1, 129)]
    assert result["mean_nll"] == 64.5 / 128
    request = body(sample, [9])
    assert request["prompt"] == [0, 1]
    assert request["max_tokens"] == 129 and request["ignore_eos"]
    assert request["stop_token_ids"] == []


@pytest.mark.parametrize("bad", ["target", "duplicate", "request", "nonfinite"])
def test_teacher_corruption_rejected(tmp_path, bad):
    path = tmp_path / "teacher.jsonl"
    sample, rows = teacher_fixture(path)
    if bad == "target":
        rows[64]["target"] += 1
    elif bad == "duplicate":
        rows[64]["step"] = 63
    elif bad == "request":
        rows[64]["request_id"] = "replacement"
    else:
        rows[64]["nll"] = float("nan")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        teacher_rows([sample], path)


def test_failed_quality_http_attempt_is_retained_and_not_retried(tmp_path):
    from aiohttp import web

    async def run():
        offered = []

        async def reply(request):
            value = await request.json()
            offered.append(value["request_id"])
            if value["request_id"].endswith("bad"):
                return web.json_response({"error": "fixture failure"}, status=503)
            return web.json_response(natural("The answer is 18"))

        app = web.Application()
        app.router.add_post("/v1/completions", reply)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        path = tmp_path / "attempts.jsonl"
        try:
            with pytest.raises(RuntimeError, match="all attempts retained"):
                await cohort(
                    [{**math_sample(), "id": i} for i in ("good", "bad")],
                    [9],
                    path,
                    url=f"http://127.0.0.1:{port}",
                )
        finally:
            await runner.cleanup()
        assert sorted(offered) == ["decision-bad", "decision-good"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 2 and sum(r["error"] is not None for r in rows) == 1

    asyncio.run(run())
