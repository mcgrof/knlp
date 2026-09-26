"""The four arms of the normalized top-2 study, and the checks that bind them.

Every arm is built here so that no caller can assemble one by hand and get the
configuration subtly wrong. After construction each model is interrogated for
the k it actually uses and the readout policy its mixers actually implement, so
a copied top-1 arm cannot pass itself off as the top-2 candidate.

The three memory arms share one freshly initialised parameter state per seed:
they have identical parameter structure, since k changes how many banks are
read and not how many exist. The baseline is a different architecture and is
initialised under the same seed contract.
"""
import hashlib
import importlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SRC = os.environ.get("MATCHED_MICRO_SOURCE", os.path.dirname(HERE))

# id -> (trainer arm, generated module, expected k, expected readout policy)
ARMS = {
    "G0": dict(trainer_arm="gdn3", module=None, topk=None, readout=None,
               family="gdn",
               what="the contemporary baseline, five recurrent heads"),
    "K1": dict(trainer_arm="mom3_k1", module="mom_renorm", topk=1,
               readout="renormalized", family="mom",
               what="one routed memory plus shared; the coefficient is one"),
    "K2": dict(trainer_arm="mom3", module="mom_renorm", topk=2,
               readout="renormalized", family="mom",
               what="two routed memories plus shared, relative weights"),
    "U2": dict(trainer_arm="mom3", module="mom_uniform", topk=2,
               readout="uniform", family="mom",
               what="two routed memories plus shared, constant half weights"),
}
ORDER_SEED_A = ["G0", "K1", "K2", "U2"]
ORDER_SEED_B = ["U2", "K2", "K1", "G0"]


def sha_text(p):
    return hashlib.sha256(io.open(p, "rb").read()).hexdigest()


def source_hashes():
    out = {}
    for n in ("_pinned_mom_c2.py", "_pinned_mom_r1.py", "make_modules.py",
              "mom_renorm.py", "mom_uniform.py", "arms.py"):
        p = os.path.join(HERE, n)
        if os.path.exists(p):
            out[n] = sha_text(p)
    for n in ("matched_micro_train.py", "matched_micro_data.py"):
        p = os.path.join(TRAIN_SRC, n)
        if os.path.exists(p):
            out[n] = sha_text(p)
    return out


def _install(module_name):
    """Point the library's mixer at one of our generated policies."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import fla.layers.mom as stock
    importlib.reload(stock)
    if module_name is None:
        return None
    mod = importlib.import_module(module_name)
    importlib.reload(mod)
    stock.MomAttention = mod.MomAttention
    return mod


def state_fingerprint(model):
    """Content hash of every parameter, so two builds can be compared."""
    h = hashlib.sha256()
    per = {}
    for n, p in sorted(model.named_parameters()):
        b = p.detach().cpu().contiguous().numpy().tobytes()
        h.update(n.encode()); h.update(b)
        per[n] = dict(shape=tuple(p.shape),
                      sha256=hashlib.sha256(b).hexdigest()[:16])
    return h.hexdigest(), per


def param_counts(model):
    tot = sum(p.numel() for p in model.parameters())
    emb = 0
    for n, p in model.named_parameters():
        if "embed" in n or "wte" in n or "lm_head" in n:
            emb += p.numel()
    return dict(total=tot, embedding=emb, nonembedding=tot - emb)


def build(arm_id, seed, device, state=None):
    """Build one arm. `state` shares a fresh initialisation across memory arms."""
    import torch
    if TRAIN_SRC not in sys.path:
        sys.path.insert(0, TRAIN_SRC)
    import matched_micro_train as mmt
    spec = ARMS[arm_id]
    mod = _install(spec["module"])
    # build_arm reseeds from the trainer's own contract, so setting only the
    # global generator here would be overwritten and every seed would match
    mmt.CONTRACT["seed"] = seed
    torch.manual_seed(seed)
    model = mmt.build_arm(spec["trainer_arm"], torch.device(device))
    model.train()
    if state is not None:
        model.load_state_dict(state)
    checks = verify(model, arm_id, mod)
    return model, checks


def verify(model, arm_id, mod):
    """Interrogate the built model rather than trusting its label."""
    spec = ARMS[arm_id]
    out = dict(arm=arm_id, trainer_arm=spec["trainer_arm"],
               module=spec["module"], expected_topk=spec["topk"],
               expected_readout=spec["readout"])
    mixers = [b.mixer for b, k in zip(model.blocks, model.kinds) if k == "M"]
    out["memory_layers"] = len(mixers)
    if spec["family"] == "gdn":
        out["memory_layers_expected_zero"] = (len(mixers) == 0)
        out["passed"] = len(mixers) == 0
        return out
    assert mixers, "a memory arm with no memory layers"
    ks = sorted({int(m.topk) for m in mixers})
    nm = sorted({int(m.num_memories) for m in mixers})
    sh = sorted({bool(m.shared_mem) for m in mixers})
    out.update(topk_found=ks, num_memories_found=nm, shared_found=sh,
               module_policy=getattr(mod, "READOUT_POLICY", None),
               module_pinned_source=getattr(mod, "PINNED_SOURCE_SHA256", None),
               mixer_class_module=type(mixers[0]).__module__)
    out["passed"] = bool(
        ks == [spec["topk"]] and nm == [4] and sh == [True]
        and getattr(mod, "READOUT_POLICY", None) == spec["readout"]
        and type(mixers[0]).__module__ == spec["module"])
    return out


def readout_probe(model, device, batch=2, length=96):
    """Read the coefficients the forward actually applied.

    Turns on the generated capture hook, runs one forward, and returns the
    selected coefficients per memory layer. This is how the policy is checked
    functionally rather than by reading the source.
    """
    import torch
    mixers = [b.mixer for b, k in zip(model.blocks, model.kinds) if k == "M"]
    for m in mixers:
        m._capture_routing = True
    ids = torch.randint(0, 50257, (batch, length), device=device)
    was = model.training
    model.eval()
    with torch.no_grad():
        model(ids)
    if was:
        model.train()
    out = []
    for i, m in enumerate(mixers):
        w = getattr(m, "_last_routing_weights", None)
        m._capture_routing = False
        if w is None:
            out.append(None)
            continue
        f = w.float()
        out.append(dict(layer=i, shape=tuple(f.shape),
                        sum_min=float(f.sum(-1).min()),
                        sum_max=float(f.sum(-1).max()),
                        value_min=float(f.min()), value_max=float(f.max()),
                        spread_max=float((f.max(-1).values - f.min(-1).values).max())))
    return out
