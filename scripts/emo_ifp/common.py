"""Shared helpers for the EMO expert-pool experiments.

EMO (allenai/Emo_1b14b_1T) stores each expert as three separate tensors,
``model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight``. In the legacy
routing path used by the released config, the shared experts are the last
``num_shared_experts`` indices of each layer, so with 128 experts and one
shared expert, expert 127 is shared and experts 0..126 are routed.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch

EXPERT_MATS = ("gate_proj", "up_proj", "down_proj")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def tensor_sha256(t):
    """Hash the raw bytes of a tensor in its own dtype."""
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def file_sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def git_rev(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def runtime_identity():
    """Record the software and device identity of the current process."""
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "kernel": platform.release(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_rev": git_rev(os.path.dirname(os.path.abspath(__file__)))
        or os.environ.get("EMO_IFP_CODE_REV"),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["device_name"] = p.name
        info["device_total_bytes"] = p.total_memory
        info["gcn_arch"] = getattr(p, "gcnArchName", None)
    return info


def expert_tensor_name(layer, expert, mat):
    return f"model.layers.{layer}.mlp.experts.{expert}.{mat}.weight"


def stacked_name(layer, which):
    """Name of a packed layer's stacked routed weights (w13 or w2)."""
    return f"model.layers.{layer}.mlp.routed_{which}"


def load_model(model_dir, device="cuda", dtype=torch.bfloat16):
    """Load EMO with its pinned remote code onto one device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    # Router logits are only needed for the auxiliary loss; disable them so
    # restricted layers with fewer experts do not trip the loss code.
    model.config.output_router_logits = False
    return model, tok


def prune_layer_inplace(moe, keep, num_shared):
    """Restrict one EmoSparseMoeBlock to ``keep`` original expert indices.

    Mirrors EMO's ``prune_moe_layer_inplace``: the expert list and router
    rows are subset, so the routed softmax renormalizes over the kept routed
    experts. ``keep`` must list routed experts first and shared experts last,
    which preserves the legacy convention that shared experts are the final
    indices of the layer.
    """
    keep = list(keep)
    moe.experts = torch.nn.ModuleList([moe.experts[i] for i in keep])
    w = moe.gate.weight.data[keep, :].clone()
    moe.gate.weight = torch.nn.Parameter(w, requires_grad=False)
    moe.gate.out_features = len(keep)
    moe.num_experts = len(keep)
    moe.num_shared_experts = num_shared
    moe.top_k = min(moe.top_k, len(keep))


def apply_pool(model, pool):
    """Apply a frozen pool manifest (see calibrate_pool.py) to a full model."""
    n_shared = pool["num_shared_experts"]
    for li, layer in enumerate(model.model.layers):
        keep = pool["layers"][str(li)]["routed"] + pool["layers"][str(li)]["shared"]
        prune_layer_inplace(layer.mlp, keep, n_shared)
    torch.cuda.empty_cache()


def mem_snapshot():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
    }
