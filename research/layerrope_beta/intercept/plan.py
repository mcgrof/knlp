# SPDX-License-Identifier: MIT
"""Freeze the tied magnitude-intercept grid before any outcome is seen."""

INTERCEPTS = (0.0, -0.1, -0.25, -0.5)
BETAS = (0.0, -0.1, -0.2, -0.3, -0.5)
LRS = {"R": 0.003, "M": 0.003, "L": 0.01, "ML": 0.01}
SCREEN_SEED = 77911
CONFIRM_SEEDS = (77912, 77913, 77914)
PROBE_SEED = 77990
AUTHOR_POINT = (-0.25, -0.2)
ANCHOR = (0.0, 0.0)


def tag(value):
    return str(value).replace("-", "minus").replace(".", "p")


def row(arm, alpha_out=None, beta=None):
    if alpha_out is None:
        name = arm
    else:
        name = f"{arm}-a-{tag(alpha_out)}-beta-{tag(beta)}"
    return dict(id=name, arm=arm, alpha_out=alpha_out, beta=beta, lr=LRS[arm])


def job(r, stage, seed):
    return dict(
        id=f"{stage}-{r['id']}-seed{seed}",
        row=r["id"],
        arm=r["arm"],
        alpha_out=r["alpha_out"],
        beta=r["beta"],
        stage=stage,
        steps=2000,
        lr=r["lr"],
        seed=seed,
    )


def screen_rows():
    rows = [row("R"), row("M")]
    rows += [
        row(arm, a, beta) for arm in ("L", "ML") for a in INTERCEPTS for beta in BETAS
    ]
    return rows


def primary(result):
    points = [p["ce"] for p in result["curve"] if p["step"] >= 1000]
    if result.get("status") != "complete" or len(points) != 11:
        return None
    return sum(points) / len(points)


def select(scores):
    """Pick the lowest screen CE per gain arm; ties prefer the smaller change."""
    chosen = {}
    for arm in ("L", "ML"):
        candidates = [
            (value, abs(r["alpha_out"]), abs(r["beta"]), r["id"])
            for r, value in scores
            if r["arm"] == arm and value is not None
        ]
        if not candidates:
            raise RuntimeError("No valid screen row for " + arm)
        chosen[arm] = min(candidates)[3]
    return chosen


def confirm_rows(chosen):
    rows = [row("R"), row("M")]
    for arm in ("L", "ML"):
        named = {row(arm, *ANCHOR)["id"]: row(arm, *ANCHOR)}
        named.setdefault(row(arm, *AUTHOR_POINT)["id"], row(arm, *AUTHOR_POINT))
        for candidate in screen_rows():
            if candidate["id"] == chosen[arm]:
                named.setdefault(candidate["id"], candidate)
        rows += named.values()
    return rows


def freeze():
    return dict(
        schema="layerrope-magnitude-intercept-grid.v1",
        frozen_before_outcomes=True,
        question="Does tuning the tied LayerRoPE log-magnitude intercepts with the depth slope improve LayerRoPE and mHC + LayerRoPE story-model learning at fixed tokens?",
        source="Author suggestion: alpha_out = -alpha_in = a in {-0.5,-0.25,-0.1}, beta in {-0.5,-0.3,-0.1,0}; best reported a=-0.25, beta=-0.2 on DepthBench",
        intercepts=list(INTERCEPTS),
        intercept_semantics="Initial in_attn/in_mlp alpha = -a and out_attn/out_mlp alpha = a; rotation intercepts and base frequency 100 unchanged; all schedule parameters trainable",
        betas=list(BETAS),
        beta_semantics="Equal magnitude and rotation initial slopes at all four sites, as in the earlier full-gain study",
        geometry=dict(depth=32, width=128, heads=2, ffn_width=352, context=256),
        learning_rates=LRS,
        lr_basis="Fixed-token full-gain study selections; no new learning-rate pilot",
        updates=2000,
        training_tokens_per_update=2048,
        warmup_steps=100,
        optimizer=dict(type="AdamW", betas=[0.9, 0.95], eps=1e-8, weight_decay=0.0),
        screen=dict(
            seed=SCREEN_SEED,
            rows=screen_rows(),
            measurement="Curve CE every 100 updates and final clean CE; no quantization grid",
        ),
        selection="Per gain arm, lowest screen primary CE; ties prefer smaller |a| then smaller |beta|; nonfinite or incomplete rows are ineligible",
        confirmation=dict(
            seeds=list(CONFIRM_SEEDS),
            rows="R, M, the a=0/beta=0 anchor, the author point a=-0.25/beta=-0.2 and each arm's screen winner",
            measurement="Full-gain study endpoints: distributions, 51 KV cache cases and floating/integer interventions",
        ),
        primary="Mean curve CE at updates 1000, 1100, ..., 2000",
        decision_rule="The tuned intercept helps an arm if its confirmed winner beats that arm's a=0/beta=0 anchor on primary CE in all three paired seeds with mean gain at least 0.01 nats; otherwise report no confirmed gain at this scale",
        limits=[
            "Small scratch-trained story models at depth 32 and width 128",
            "Fixed learning rates; the a-shift can move the best rate",
            "One screen seed selects; three fresh seeds confirm",
            "Equal tokens, not equal training time",
            "Different GPU and runtime from the earlier H100 runs; anchors are re-measured, not reused",
        ],
    )
