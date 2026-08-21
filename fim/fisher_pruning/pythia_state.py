"""Pythia GPT-NeoX optimizer-state converter for the pruning lane.

EleutherAI published the full training state of the Pythia suite as
GPT-NeoX (DeeperSpeed) checkpoints, e.g.
EleutherAI/neox-ckpt-pythia-160m-deduped-v1 at global_step143000.
Those checkpoints carry the final Adam moments (exp_avg /
exp_avg_sq) of the actual pretraining run — the "stale optimizer
state" that Program O's fresh-calibration arms are compared
against. This module reassembles that state and maps it onto the
HuggingFace GPTNeoXForCausalLM parameter namespace so it can score
the same target linears program_o prunes.

Checkpoint format (verified on the real 160m-deduped shards):

  layer_XX-model_00-model_states.pt   fp16 module weights, one
      file per pipeline stage: 00 = word embedding, (i+2) =
      transformer layer i, (n_layers+3) = final layernorm,
      (n_layers+4) = the lm head (final_linear).
  mp_rank_00_model_states.pt          run metadata only (module
      and optimizer entries are None).
  zero_pp_rank_{r}_mp_rank_00_optim_states.pt  for r in
      0..world-1: the legacy DeepSpeed ZeRO stage-1
      sub-partition format. Each shard pickles

      {"optimizer_state_dict":
          {"loss_scaler": <DynamicLossScaler>,
           "dynamic_loss_scale": bool, "overflow": bool,
           "zero_stage": 1, "partition_count": world,
           "num_comm_intervals_per_group": [ints],
           "base_optimizer_state":
               [ [ {"exp_avg": flat fp32, "exp_avg_sq": flat
                    fp32} per sub-partition ] per group ],
           "local_sub_partitions_of_fp32_groups":
               [ [ flat fp32 master weights per sub-partition ]
                 per group ]},
       "param_shapes": OrderedDict "P.name" -> torch.Size}

The flattened per-group buffer is split into world *
num_comm_intervals equal sub-partitions; rank r holds
sub-partition (interval * world + r) for each interval.
Concatenating in that order and splitting by param_shapes order
recovers the per-parameter tensors; trailing padding (if any)
after the last parameter is dropped.

param_shapes is one flat OrderedDict over both groups, keyed
"<pipe_index>.<neox name>" (e.g. "2.attention.query_key_value.
weight"). Group membership is not stored; GPT-NeoX builds two
Adam groups — weight-decayed 2-D matrices and non-decayed 1-D
params (biases, layernorms) — each in model parameter order.
That split is reconstructed here and validated against the flat
buffer lengths (and, end to end, against the fp16 module weights:
fp16(master) equals the stored layer weights exactly).

CLI:

  python3 fim/fisher_pruning/pythia_state.py convert \
      --shard-dir D --hf-model EleutherAI/pythia-160m-deduped \
      --out OUT.pt [--device cpu] [--force]
"""

import argparse
import hashlib
import importlib.machinery
import re
import subprocess
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fim.fisher_pruning.program_o import (  # noqa: E402
    EPS,
    discover_target_linears,
)

SHARD_RE = re.compile(r"zero_pp_rank_(\d+)_mp_rank_00_optim_states\.pt$")

# NeoX-side buffers with no HF parameter. "attention.dense.bias"
# must survive, so the bias fragments are anchored to "attention.".
DROPPED_LEAF_FRAGMENTS = (
    "rotary_emb.inv_freq",
    "attention.bias",
    "attention.masked_bias",
)

# Relative tolerance for the fp16-roundtrip identity check. The HF
# repo weights are fp32 upcasts of the fp16 training weights, so
# fp16(master) upcast to fp32 should match them exactly; the slack
# only absorbs benign serialization noise.
VALIDATE_RTOL = 1e-6


def _install_deepspeed_stubs() -> None:
    """Make the shard pickles loadable without deepspeed installed.

    The optimizer shards pickle deepspeed objects (DynamicLossScaler
    et al.) by reference. Their contents are irrelevant here, so if
    deepspeed is absent, register an importer that fabricates empty
    stand-in classes for any deepspeed.* lookup.
    """
    if "deepspeed" in sys.modules:
        return
    try:
        import deepspeed  # noqa: F401

        return
    except ImportError:
        pass

    class _Stub:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            self._state = state

    class _AutoModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (_Stub,), {"__module__": self.__name__})
            setattr(self, name, cls)
            return cls

    class _Loader:
        def create_module(self, spec):
            mod = _AutoModule(spec.name)
            mod.__path__ = []
            return mod

        def exec_module(self, mod):
            pass

    class _Finder:
        def find_spec(self, name, path=None, target=None):
            if name == "deepspeed" or name.startswith("deepspeed."):
                spec = importlib.machinery.ModuleSpec(name, _Loader())
                spec.submodule_search_locations = []
                return spec
            return None

    sys.meta_path.insert(0, _Finder())


def _load_shard(path: Path) -> Dict:
    _install_deepspeed_stubs()
    # Trusted EleutherAI pickles; they contain non-tensor objects,
    # so weights_only loading cannot be used.
    return torch.load(path, map_location="cpu", weights_only=False)


def _shard_paths(shard_dir: Path) -> List[Path]:
    by_rank = {}
    for p in shard_dir.iterdir():
        m = SHARD_RE.match(p.name)
        if m:
            by_rank[int(m.group(1))] = p
    if not by_rank:
        raise FileNotFoundError(f"no zero_pp_rank_*_optim_states.pt in {shard_dir}")
    world = max(by_rank) + 1
    missing = sorted(set(range(world)) - set(by_rank))
    if missing:
        raise FileNotFoundError(f"missing optimizer shards for ranks {missing}")
    return [by_rank[r] for r in range(world)]


def _group_param_lists(
    param_shapes: "OrderedDict[str, torch.Size]", n_groups: int
) -> List[List[Tuple[str, torch.Size]]]:
    """Assign the flat param_shapes to optimizer param groups.

    One group takes everything. Two groups follow GPT-NeoX's Adam
    grouping: 2-D+ matrices (weight decay) then 1-D params (no
    decay), each in param_shapes order. Anything else is a format
    this converter has never seen.
    """
    items = [(n, torch.Size(s)) for n, s in param_shapes.items()]
    if n_groups == 1:
        return [items]
    if n_groups == 2:
        return [
            [(n, s) for n, s in items if len(s) >= 2],
            [(n, s) for n, s in items if len(s) == 1],
        ]
    raise ValueError(f"cannot infer param grouping for {n_groups} param groups")


def _split_flat(
    flat: torch.Tensor,
    entries: List[Tuple[str, torch.Size]],
    sub_partition_len: int,
    what: str,
) -> "OrderedDict[str, torch.Tensor]":
    total = sum(s.numel() for _, s in entries)
    if flat.numel() < total:
        raise ValueError(
            f"{what}: flat buffer has {flat.numel()} elements but the "
            f"group's parameters need {total} — wrong group assignment "
            f"or corrupt shards"
        )
    leftover = flat.numel() - total
    if leftover >= sub_partition_len:
        raise ValueError(
            f"{what}: {leftover} trailing elements exceed one "
            f"sub-partition ({sub_partition_len}) — parameters are "
            f"missing from the reconstructed group order"
        )
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    offset = 0
    for name, shape in entries:
        n = shape.numel()
        out[name] = flat[offset : offset + n].view(shape).clone()
        offset += n
    return out


def reassemble_zero1(shard_dir: Union[str, Path]) -> Dict:
    """Rebuild per-parameter optimizer state from ZeRO-1 shards.

    Loads every zero_pp_rank_*_optim_states.pt under shard_dir,
    concatenates each group's flat exp_avg / exp_avg_sq / fp32
    master sub-partitions in (comm interval, rank) order, and
    splits by param_shapes into named tensors. Returns

      {"exp_avg": {neox_name: tensor}, "exp_avg_sq": {...},
       "master_weights": {...}, "param_shapes": OrderedDict}

    with neox_name the "<pipe_index>.<name>" keys of param_shapes.
    """
    paths = _shard_paths(Path(shard_dir))
    world = len(paths)

    param_shapes = None
    n_groups = None
    # per group: list of per-sub-partition tensors from each rank
    per_rank: Dict[str, List[List[List[torch.Tensor]]]] = {}
    for rank, path in enumerate(paths):
        shard = _load_shard(path)
        osd = shard["optimizer_state_dict"]
        zero_stage = osd.get("zero_stage")
        if zero_stage != 1:
            raise ValueError(f"{path.name}: zero_stage={zero_stage}, expected 1")
        pc = osd.get("partition_count")
        if pc is not None and pc != world:
            raise ValueError(
                f"{path.name}: partition_count={pc} but {world} shard files found"
            )
        if param_shapes is None:
            param_shapes = shard["param_shapes"]
            n_groups = len(osd["base_optimizer_state"])
            per_rank = {
                k: [[] for _ in range(n_groups)]
                for k in ("exp_avg", "exp_avg_sq", "master")
            }
        elif list(shard["param_shapes"]) != list(param_shapes):
            raise ValueError(f"{path.name}: param_shapes disagree with rank 0")
        if len(osd["base_optimizer_state"]) != n_groups:
            raise ValueError(f"{path.name}: param group count disagrees with rank 0")
        for gi in range(n_groups):
            subs = osd["base_optimizer_state"][gi]
            fp32_subs = osd["local_sub_partitions_of_fp32_groups"][gi]
            if len(subs) != len(fp32_subs):
                raise ValueError(
                    f"{path.name} group {gi}: {len(subs)} moment "
                    f"sub-partitions vs {len(fp32_subs)} fp32 sub-partitions"
                )
            per_rank["exp_avg"][gi].append([s["exp_avg"] for s in subs])
            per_rank["exp_avg_sq"][gi].append([s["exp_avg_sq"] for s in subs])
            per_rank["master"][gi].append(list(fp32_subs))

    group_lists = _group_param_lists(param_shapes, n_groups)
    result = {
        "exp_avg": OrderedDict(),
        "exp_avg_sq": OrderedDict(),
        "master_weights": OrderedDict(),
        "param_shapes": param_shapes,
    }
    out_key = {
        "exp_avg": "exp_avg",
        "exp_avg_sq": "exp_avg_sq",
        "master": "master_weights",
    }
    for gi, entries in enumerate(group_lists):
        n_intervals = {len(ranks) for ranks in per_rank["master"][gi]}
        if len(n_intervals) != 1:
            raise ValueError(f"group {gi}: ranks hold unequal sub-partition counts")
        n_intervals = n_intervals.pop()
        sub_len = per_rank["master"][gi][0][0].numel()
        for kind in ("exp_avg", "exp_avg_sq", "master"):
            ranks = per_rank[kind][gi]
            lens = {t.numel() for r in ranks for t in r}
            if lens != {sub_len}:
                raise ValueError(
                    f"group {gi} {kind}: unequal sub-partition lengths {sorted(lens)}"
                )
            # Legacy ZeRO-1 lays sub-partitions out interval-major:
            # sub-partition (interval * world + rank).
            flat = torch.cat(
                [ranks[r][i] for i in range(n_intervals) for r in range(world)]
            )
            result[out_key[kind]].update(
                _split_flat(flat, entries, sub_len, f"group {gi} {kind}")
            )
    return result


def build_layer_offset_map(n_layers: int) -> Dict[int, str]:
    """Pipeline-stage index -> HF module prefix for GPT-NeoX."""
    offsets = {0: "gpt_neox.embed_in"}
    for i in range(n_layers):
        offsets[i + 2] = f"gpt_neox.layers.{i}"
    offsets[n_layers + 3] = "gpt_neox.final_layer_norm"
    offsets[n_layers + 4] = "embed_out"
    return offsets


def neox_to_hf_name(neox_name: str, layer_offset_map: Dict[int, str]) -> Optional[str]:
    """Map "<pipe>.<neox leaf>" to the HF parameter name.

    Returns None for NeoX buffers with no HF parameter (rotary
    inv_freq, causal-mask bias buffers). Transformer-layer leaves
    keep their names; the embedding / final-norm / lm-head stages
    replace the NeoX module name (word_embeddings, norm,
    final_linear) with the HF prefix.
    """
    pipe_str, leaf = neox_name.split(".", 1)
    pipe = int(pipe_str)
    if any(frag in leaf for frag in DROPPED_LEAF_FRAGMENTS):
        return None
    prefix = layer_offset_map.get(pipe)
    if prefix is None:
        raise KeyError(f"pipeline stage {pipe} ({neox_name!r}) has no HF mapping")
    if prefix.startswith("gpt_neox.layers."):
        return f"{prefix}.{leaf}"
    # Single-module stages: drop the NeoX module name, keep the
    # attribute path (".weight" / ".bias").
    return f"{prefix}.{leaf.split('.', 1)[1]}"


def map_state_to_hf(
    state: Dict[str, torch.Tensor], n_layers: int
) -> Dict[str, torch.Tensor]:
    """{neox_name: tensor} -> {hf_name: tensor}, dropping buffers."""
    offsets = build_layer_offset_map(n_layers)
    out: Dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        hf = neox_to_hf_name(name, offsets)
        if hf is not None:
            out[hf] = tensor
    return out


def infer_n_layers(param_shapes: "OrderedDict[str, torch.Size]") -> int:
    """Layer count from the pipeline indices in param_shapes.

    The lm head sits at stage n_layers + 4 (embedding, then an
    unused stage, n_layers transformer stages, an unused stage,
    the final norm, the head).
    """
    return max(int(n.split(".", 1)[0]) for n in param_shapes) - 4


def validate_against_hf(
    mapped_master_weights: Dict[str, torch.Tensor],
    hf_model_or_state_dict: Union[nn.Module, Dict[str, torch.Tensor]],
    atol: float = 0.0,
) -> Dict:
    """Check reassembled fp32 masters against the HF weights.

    The HF repo weights are fp32 upcasts of the fp16 training
    weights, and those fp16 weights were casts of the fp32 masters
    — so fp16(master) upcast to fp32 must equal the HF tensor. The
    check is |hf - fp32(fp16(master))| <= atol + 1e-6 * |hf|
    elementwise; a correct reassembly passes with atol = 0.
    """
    if isinstance(hf_model_or_state_dict, nn.Module):
        hf = {n: p.detach() for n, p in hf_model_or_state_dict.named_parameters()}
    else:
        hf = {
            n: t
            for n, t in hf_model_or_state_dict.items()
            if not any(frag in n for frag in DROPPED_LEAF_FRAGMENTS)
        }
    per_tensor = {}
    missing_in_hf = sorted(set(mapped_master_weights) - set(hf))
    missing_in_mapped = sorted(set(hf) - set(mapped_master_weights))
    ok_all = not missing_in_hf and not missing_in_mapped
    for name, master in mapped_master_weights.items():
        ref = hf.get(name)
        if ref is None:
            continue
        if tuple(ref.shape) != tuple(master.shape):
            per_tensor[name] = {
                "ok": False,
                "max_abs_diff": None,
                "error": f"shape {tuple(master.shape)} vs hf {tuple(ref.shape)}",
            }
            ok_all = False
            continue
        ref32 = ref.detach().float().cpu()
        roundtrip = master.half().float()
        diff = (ref32 - roundtrip).abs()
        ok = bool((diff <= atol + VALIDATE_RTOL * ref32.abs()).all())
        per_tensor[name] = {"ok": ok, "max_abs_diff": diff.max().item()}
        ok_all = ok_all and ok
    return {
        "pass": ok_all,
        "atol": atol,
        "rtol": VALIDATE_RTOL,
        "n_tensors": len(per_tensor),
        "missing_in_hf": missing_in_hf,
        "missing_in_mapped": missing_in_mapped,
        "per_tensor": per_tensor,
    }


def stale_adam_scores(
    mapped_exp_avg_sq: Dict[str, torch.Tensor],
    hf_model: nn.Module,
    q: float,
) -> Dict[str, torch.Tensor]:
    """|w| * (exp_avg_sq + eps)^q for Program O's target linears.

    The stale-Adam analogue of Program O's fresh_q arms:
    exp_avg_sq is the training run's own second-moment estimate,
    standing in for the fresh calibration Fisher diagonal. Keys are
    the module names discover_target_linears returns.
    """
    targets = discover_target_linears(hf_model)
    scores: Dict[str, torch.Tensor] = {}
    for name, mod in targets.items():
        v = mapped_exp_avg_sq.get(name + ".weight")
        if v is None:
            raise KeyError(f"no exp_avg_sq mapped for target {name}.weight")
        w = mod.weight.detach().float().cpu()
        assert tuple(v.shape) == tuple(
            w.shape
        ), f"{name}: exp_avg_sq shape {tuple(v.shape)} != weight {tuple(w.shape)}"
        scores[name] = w.abs() * (v.float().cpu() + EPS) ** q
    return scores


def _sha256_file(path: Path, max_bytes: int = 1 << 24) -> str:
    """Hash of the first max_bytes plus the file size."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(max_bytes))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()


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


def cmd_convert(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM

    shard_dir = Path(args.shard_dir)
    state = reassemble_zero1(shard_dir)
    n_layers = infer_n_layers(state["param_shapes"])
    print(
        f"reassembled {len(state['master_weights'])} params, "
        f"{n_layers} transformer layers",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, revision=args.revision, dtype=torch.float32
    )
    model.to(args.device)
    model.eval()
    hf_layers = model.config.num_hidden_layers
    if hf_layers != n_layers:
        raise ValueError(
            f"checkpoint has {n_layers} layers but {args.hf_model} has {hf_layers}"
        )

    mapped_v = map_state_to_hf(state["exp_avg_sq"], n_layers)
    mapped_master = map_state_to_hf(state["master_weights"], n_layers)
    report = validate_against_hf(mapped_master, model, atol=args.atol)
    worst = max(
        (
            t["max_abs_diff"]
            for t in report["per_tensor"].values()
            if t["max_abs_diff"] is not None
        ),
        default=None,
    )
    print(
        f"validation pass={report['pass']} n={report['n_tensors']} "
        f"worst max|diff|={worst}",
        flush=True,
    )
    if not report["pass"] and not args.force:
        print("validation FAILED — not saving (use --force to override)")
        return 1

    manifest = {
        "shard_dir": str(shard_dir),
        "shard_sha256": {p.name: _sha256_file(p) for p in _shard_paths(shard_dir)},
        "hf_model": args.hf_model,
        "hf_revision": args.revision,
        "n_layers": n_layers,
        "code_commit": _git_commit(),
        "created_unix": time.time(),
    }
    torch.save(
        {"exp_avg_sq": mapped_v, "validation": report, "manifest": manifest},
        args.out,
    )
    print(f"saved {args.out}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    cv = sub.add_parser("convert", help="reassemble, map to HF, validate, save")
    cv.add_argument("--shard-dir", required=True)
    cv.add_argument("--hf-model", required=True)
    cv.add_argument("--out", required=True)
    cv.add_argument("--device", default="cpu")
    cv.add_argument("--revision", default=None)
    cv.add_argument("--atol", type=float, default=0.0)
    cv.add_argument(
        "--force", action="store_true", help="save even if validation fails"
    )
    args = ap.parse_args()
    if args.command == "convert":
        return cmd_convert(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
