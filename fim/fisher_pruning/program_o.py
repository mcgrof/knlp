"""Post-training pruning-signal evaluation on HF checkpoints.

Open-weight lane of the factored-Fisher pruning program: given a
finished HuggingFace causal-LM checkpoint (Pythia/GPT-NeoX first,
generic transformer block stacks otherwise), synthesize fresh
saliency statistics from a short frozen-weight calibration pass and
measure one-shot unstructured pruning regret against magnitude and
Wanda. No optimizer step is ever taken; weights stay frozen.

Calibration reuses FactorAccumulator (kfac_capture.py), which
accumulates per nn.Linear:

    A = E[x x^T]        input second moment          [in, in]
    G = E[g g^T]        output-gradient second moment [out, out]
    D = E[g_i^2 x_j^2]  exact elementwise diagonal of the
                        per-token empirical Fisher    [out, in]

g is the per-token loss gradient at the layer output; the
accumulator rescales autograd's mean-loss gradients by the row
count. The HF loss averages over B*(T-1) shifted label positions
while B*T token rows pass through each layer, so the captured
factors carry a uniform T/(T-1) scale relative to strict per-label
normalization; a uniform per-run scale cannot change any score
ranking or mask.

Score arms (per-weight importance, higher = keep):

  random      uniform noise; negative control.
  magnitude   |w|.
  fresh_q025  |w| * (D + eps)^0.25 — exact diagonal empirical
              Fisher from the calibration pass, quarter power.
  fresh_q050  |w| * (D + eps)^0.50 — same statistic, half power
              (the classic sqrt-Fisher / OBD-style weighting).
  kron_q025   |w| * (G_ii A_jj + eps)^0.25 — rank-1 Kronecker
              (factored) approximation of D, quarter power.
  kron_q050   same, half power.
  kfac_obs    w^2 / [(G + lam I)^-1_ii (A + lam I)^-1_jj] with
              lam = kappa * tr/dim, kappa = 1e-2 — OBS-style
              score; the only arm using off-diagonal curvature.
  wanda       |w| * sqrt(diag A)_j broadcast over rows. Wanda
              scores |w_ij| * ||x_j||_2 over calibration tokens;
              sqrt(A_jj) = ||x_j||_2 / sqrt(n_tokens) differs
              only by a per-run constant, so the score is exact
              from A's diagonal — no separate capture needed.

Masks are per-layer uniform sparsity (per_layer_mask). Evaluation
is mean token-level cross-entropy + perplexity on fixed held-out
batch sets drawn from document ranges disjoint from calibration
(two disjoint eval sets give a noise floor). Pristine weights are
restored between arms.

Design contract: knlp-key-results/fisher-factored-pruning-20260820/
CLOUD_WAVE_PLAN.md, "Program O". One command:

  python3 fim/fisher_pruning/program_o.py prune-eval --config CFG
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.kfac_capture import (  # noqa: E402
    FactorAccumulator,
    damped_inverse_diag,
    per_layer_mask,
)

EPS = 1e-8

# The four 2-D hidden matmuls of a GPT-NeoX block. Embeddings
# (embed_in) and the lm head (embed_out — an nn.Linear in HF's
# GPTNeoXForCausalLM) are never pruning targets.
GPTNEOX_BLOCK_LINEARS = (
    "attention.query_key_value",
    "attention.dense",
    "mlp.dense_h_to_4h",
    "mlp.dense_4h_to_h",
)

# Name fragments that disqualify a Linear in the generic fallback
# even when it sits inside a block stack.
_EXCLUDE_FRAGMENTS = ("embed", "head", "wte", "wpe")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return bool(out)
    except Exception:
        return None


def _own_code_sha256() -> str:
    h = hashlib.sha256()
    for name in ("kfac_capture.py", "program_o.py"):
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def _sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def _jsonl(path: Path, event: Dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def _manifest(cfg: Dict, extra: Dict) -> Dict:
    man = {
        "code_commit": _git_commit(),
        "code_dirty": _git_dirty(),
        "lane_code_sha256": _own_code_sha256(),
        "config_hash": _sha256_json(cfg),
        "config": cfg,
        "created_unix": time.time(),
    }
    man.update(extra)
    return man


def load_hf_model(
    model_id: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    revision: Optional[str] = None,
):
    """Load a causal LM + tokenizer for calibration/eval.

    attn_implementation="eager" keeps the attention graph plain so
    output-gradient hooks see unfused per-layer tensors.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def discover_target_linears(model: nn.Module) -> Dict[str, nn.Linear]:
    """Per-block 2-D hidden matmul nn.Linear layers, by name.

    GPT-NeoX gets the explicit four-per-block list. Any other
    architecture falls back to: every nn.Linear living under an
    nn.ModuleList (the block stack), excluding names that look
    like embeddings or heads. Embedding matrices and lm_head /
    embed_out sit outside the block stack and are never selected.
    """
    named = list(model.named_modules())
    out: Dict[str, nn.Linear] = {}
    if any(n.endswith("attention.query_key_value") for n, _ in named):
        for n, m in named:
            if isinstance(m, nn.Linear) and n.endswith(GPTNEOX_BLOCK_LINEARS):
                out[n] = m
    else:
        stack_prefixes = [
            n + "." for n, m in named if isinstance(m, nn.ModuleList) and n
        ]
        for n, m in named:
            if not isinstance(m, nn.Linear):
                continue
            if not any(n.startswith(p) for p in stack_prefixes):
                continue
            if any(f in seg for seg in n.split(".") for f in _EXCLUDE_FRAGMENTS):
                continue
            out[n] = m
    if not out:
        raise RuntimeError("no target linears discovered")
    return out


def stream_token_batches(
    tokenizer,
    *,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    seed: int,
    skip_docs: int = 0,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    split: str = "train",
    text_key: str = "text",
    shuffle_buffer: int = 10_000,
) -> Iterator[torch.Tensor]:
    """Stream documents into packed [batch, seq_len] token blocks.

    Documents are tokenized without special tokens, joined with a
    single EOS separator, and packed contiguously; each yielded
    block is int64 [batch_size, seq_len]. The stream is shuffled
    with `seed` then advanced by `skip_docs`, so the draw is
    deterministic given (seed, skip_docs). Disjointness contract:
    with the SAME seed, two draws whose [skip_docs, skip_docs +
    docs_consumed) windows do not overlap see disjoint document
    sets — leave a generous skip gap between calibration and eval
    (packed batches consume at most a few docs per batch).
    """
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset(
        dataset_name, dataset_config, split=split, streaming=True
    )
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if skip_docs:
        ds = ds.skip(skip_docs)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id for packing")
    need = seq_len * batch_size
    buf: List[int] = []
    yielded = 0
    for doc in ds:
        buf.extend(tokenizer(doc[text_key], add_special_tokens=False)["input_ids"])
        buf.append(eos)
        while len(buf) >= need and yielded < n_batches:
            block = torch.tensor(buf[:need], dtype=torch.long)
            del buf[:need]
            yield block.view(batch_size, seq_len)
            yielded += 1
        if yielded >= n_batches:
            return
    raise RuntimeError(f"stream exhausted after {yielded}/{n_batches} batches")


def calibrate(
    model: nn.Module,
    targets: Dict[str, nn.Linear],
    batches: Iterable[torch.Tensor],
):
    """Accumulate A/G/D factors over calibration batches.

    Forward with labels=input_ids for the mean LM loss, backward
    for output gradients; the accumulator handles the per-token
    rescale and fp32 accumulation (the model may stay bf16).
    Returns (factors, n_tokens); factors are on CPU.
    """
    model.eval()
    device = next(model.parameters()).device
    acc = FactorAccumulator(model, list(targets))
    acc.attach()
    try:
        for i, x in enumerate(batches):
            x = x.to(device)
            out = model(input_ids=x, labels=x)
            model.zero_grad(set_to_none=True)
            out.loss.backward()
            acc.count_batch(x.numel())
            if i % 10 == 0:
                print(f"calib batch {i} loss {out.loss.item():.4f}", flush=True)
    finally:
        acc.detach()
        model.zero_grad(set_to_none=True)
    return acc.finalize(), acc.n_tokens


def build_scores(
    weights: Dict[str, torch.Tensor],
    factors: Dict[str, Dict[str, torch.Tensor]],
    obs_kappa: float = 1e-2,
    random_seed: int = 1234,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """arm -> {module name -> score tensor [out, in]} (cpu fp32).

    Arm definitions are in the module docstring.
    """
    rng = torch.Generator().manual_seed(random_seed)
    arm_names = (
        "random",
        "magnitude",
        "fresh_q025",
        "fresh_q050",
        "kron_q025",
        "kron_q050",
        "kfac_obs",
        "wanda",
    )
    scores: Dict[str, Dict[str, torch.Tensor]] = {a: {} for a in arm_names}
    for n, w in weights.items():
        f = factors[n]
        d = f["D"].float()
        diag_g = torch.diagonal(f["G"]).float()
        diag_a = torch.diagonal(f["A"]).float()
        kron = torch.outer(diag_g, diag_a)
        scores["random"][n] = torch.rand(w.shape, generator=rng)
        scores["magnitude"][n] = w.abs()
        scores["fresh_q025"][n] = w.abs() * (d + EPS) ** 0.25
        scores["fresh_q050"][n] = w.abs() * (d + EPS) ** 0.5
        scores["kron_q025"][n] = w.abs() * (kron + EPS) ** 0.25
        scores["kron_q050"][n] = w.abs() * (kron + EPS) ** 0.5
        inv_g = damped_inverse_diag(f["G"], obs_kappa).float()
        inv_a = damped_inverse_diag(f["A"], obs_kappa).float()
        denom = torch.clamp(torch.outer(inv_g, inv_a), min=1e-30)
        scores["kfac_obs"][n] = w.pow(2) / denom
        col_norm = torch.clamp(diag_a, min=0.0).sqrt()
        scores["wanda"][n] = w.abs() * col_norm.unsqueeze(0)
    return scores


def _apply_masks(
    model: nn.Module,
    targets: Dict[str, nn.Linear],
    pristine: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
) -> None:
    """Restore pristine target weights, then zero masked entries."""
    with torch.no_grad():
        for name, mod in targets.items():
            w = pristine[name].to(device=mod.weight.device, dtype=mod.weight.dtype)
            mod.weight.copy_(w)
            mask = masks.get(name)
            if mask is not None:
                mod.weight.mul_(
                    mask.to(device=mod.weight.device, dtype=mod.weight.dtype)
                )


def _actual_sparsity(masks: Dict[str, torch.Tensor]) -> float:
    total = sum(m.numel() for m in masks.values())
    kept = sum(m.sum().item() for m in masks.values())
    return 1.0 - kept / total


@torch.no_grad()
def eval_mean_ce(model: nn.Module, batches: Iterable[torch.Tensor]) -> float:
    """Mean token-level cross-entropy over fixed eval batches.

    HF's loss is the mean over the B*(T-1) shifted label tokens of
    one batch; batches are re-weighted by their label-token count
    so the result is the exact corpus-mean token CE.
    """
    device = next(model.parameters()).device
    tot_loss = 0.0
    tot_tokens = 0
    for x in batches:
        x = x.to(device)
        out = model(input_ids=x, labels=x)
        n = x.shape[0] * (x.shape[1] - 1)
        tot_loss += out.loss.float().item() * n
        tot_tokens += n
    if tot_tokens == 0:
        raise RuntimeError("no eval tokens")
    return tot_loss / tot_tokens


def cmd_prune_eval(
    cfg: Dict,
    device: str = "cuda",
    model: Optional[nn.Module] = None,
    tokenizer=None,
    batch_fn: Optional[Callable[[int, int], Iterable[torch.Tensor]]] = None,
) -> None:
    """One-shot prune-eval over all score arms and sparsities.

    batch_fn(skip_docs, n_batches) -> iterable of [B, T] token
    blocks; injectable for tests, defaults to the fineweb-edu
    streaming packer. model/tokenizer are likewise injectable;
    by default the checkpoint named in cfg is loaded in bf16.
    """
    out_dir = Path(cfg["out_dir"])
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    log = results_dir / "programO.jsonl"
    if model is None:
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
        model, tokenizer = load_hf_model(
            cfg["model_id"], device, dtype=dtype, revision=cfg.get("revision")
        )
    else:
        model.to(device)
        model.eval()
    targets = discover_target_linears(model)
    dcfg = cfg["data"]
    if batch_fn is None:

        def batch_fn(skip_docs: int, n_batches: int):
            return stream_token_batches(
                tokenizer,
                seq_len=dcfg["seq_len"],
                batch_size=dcfg["batch_size"],
                n_batches=n_batches,
                seed=dcfg["seed"],
                skip_docs=skip_docs,
                dataset_name=dcfg.get("dataset_name", "HuggingFaceFW/fineweb-edu"),
                dataset_config=dcfg.get("dataset_config", "sample-10BT"),
                split=dcfg.get("split", "train"),
            )

    ccfg = cfg["calibration"]
    ecfg = cfg["eval"]
    # Fixed batch sets: drawn once, identical tokens for every arm.
    calib_batches = list(batch_fn(ccfg.get("skip_docs", 0), ccfg["n_batches"]))
    eval_sets = [list(batch_fn(skip, ecfg["n_batches"])) for skip in ecfg["skip_docs"]]
    pristine = {n: m.weight.detach().clone().cpu() for n, m in targets.items()}
    t0 = time.time()
    factors, n_calib_tokens = calibrate(model, targets, calib_batches)
    weights = {n: m.weight.detach().float().cpu() for n, m in targets.items()}
    scores = build_scores(
        weights,
        factors,
        obs_kappa=cfg.get("obs_kappa", 1e-2),
        random_seed=cfg.get("random_arm_seed", 1234),
    )
    sparsities = cfg.get("sparsities", [0.3, 0.5, 0.7])
    _jsonl(
        log,
        _manifest(
            cfg,
            {
                "event": "start",
                "model_id": cfg.get("model_id"),
                "revision": cfg.get("revision"),
                "device": device,
                "arms": sorted(scores),
                "targets": sorted(targets),
                "sparsities": sparsities,
                "n_calib_tokens": n_calib_tokens,
                "n_eval_sets": len(eval_sets),
            },
        ),
    )

    def eval_cell(tag: Dict) -> None:
        for i, batches in enumerate(eval_sets):
            val_loss = eval_mean_ce(model, batches)
            _jsonl(
                log,
                dict(
                    tag,
                    event="eval",
                    eval_set=i,
                    val_loss=val_loss,
                    val_ppl=math.exp(val_loss),
                ),
            )
            print(
                f"{tag} eval_set={i} loss={val_loss:.4f} "
                f"ppl={math.exp(val_loss):.2f}",
                flush=True,
            )

    _apply_masks(model, targets, pristine, {})
    eval_cell({"arm": "dense", "sparsity": 0.0})
    for arm, arm_scores in scores.items():
        for sparsity in sparsities:
            masks = {n: per_layer_mask(s, sparsity) for n, s in arm_scores.items()}
            _apply_masks(model, targets, pristine, masks)
            eval_cell(
                {
                    "arm": arm,
                    "sparsity": sparsity,
                    "actual_sparsity": _actual_sparsity(masks),
                }
            )
    _apply_masks(model, targets, pristine, {})  # leave the model dense
    _jsonl(log, {"event": "done", "elapsed_s": time.time() - t0})
    print("done", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prune-eval"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.command == "prune-eval":
        cmd_prune_eval(cfg, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
