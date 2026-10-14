# SPDX-License-Identifier: MIT
"""Teacher/student runner for the subliminal-learning MNIST experiment.

For every model key (one seed, or one member of the reference cohort) it
saves the untouched initialization W0 and an independent alternate
initialization W0_other, trains a teacher from W0 on MNIST digits using
outputs 0:10 only, and then distills four students on the same fixed
noise images with the same minibatch order:

  aux_shared    copy of W0,        target: teacher outputs 10:10+k
  full_shared   copy of W0,        target: all teacher outputs
  aux_other     copy of W0_other,  target: teacher outputs 10:10+k
  full_other    copy of W0_other,  target: all teacher outputs

Accuracy is always read from outputs 0:10 on the MNIST test split. The
primary transfer measure is the aux_shared accuracy minus the accuracy
of W0 itself.

Subcommands: doctor (read-only checks), probe (measured timing of a few
steps, projected to the full run), run (the experiment; resumable at
epoch boundaries under the wall-clock budget).
"""

import argparse
import csv
import copy
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from . import recipe as cfgmod
from .data import epoch_order, load_mnist, make_noise, tensor_digest
from .models import build, count_params_per_model, init_model
from .numerical_kl import LEGACY_KL, kl_rows

NUM_DIGITS = cfgmod.NUM_DIGITS
STUDENTS = ("aux_shared", "full_shared", "aux_other", "full_other")
EXIT_INCOMPLETE = 3


# ───────────────────────────── losses ──────────────────────────────────────
def teacher_loss(logits, y, protocol, cohort):
    """Cross-entropy on the digit outputs; returns (loss, per-model CE mean).

    logits [P, b, >=10], y [P, b]. upstream_batched_v1 divides each
    model's per-image mean by the logical cohort, which is what flattening
    M models into one cross_entropy mean does in the reference script.
    """
    p, b = y.shape
    ce = F.cross_entropy(
        logits[..., :NUM_DIGITS].reshape(p * b, NUM_DIGITS),
        y.reshape(p * b),
        reduction="none",
    ).view(p, b)
    ce_mean = ce.mean(1)
    if protocol == "upstream_batched_v1":
        return ce_mean.sum() / cohort, ce_mean
    return ce_mean.sum(), ce_mean


def student_loss(student_logits, teacher_logits, protocol, cohort, loss_mode=LEGACY_KL):
    """KL(teacher || student) on already-sliced logits [P, b, K].

    Returns (loss, per-model KL summed over images, per-model per-image
    mean KL). upstream_batched_v1 matches kl_div(..., reduction=
    "batchmean") applied to an [M, B, K] tensor: the sum over images and
    outputs divided by M, not by M * B.
    """
    rows = kl_rows(student_logits, teacher_logits, loss_mode)
    kl_sum = rows.sum(1)
    kl_mean = rows.mean(1)
    if protocol == "upstream_batched_v1":
        return kl_sum.sum() / cohort, kl_sum, kl_mean
    return kl_mean.sum(), kl_sum, kl_mean


def output_index(kind, recipe):
    if kind.startswith("aux"):
        return list(range(NUM_DIGITS, recipe.total_outputs))
    return list(range(recipe.total_outputs))


def make_optimizer(model, recipe):
    return torch.optim.Adam(
        model.parameters(),
        lr=recipe.lr,
        betas=recipe.adam_betas,
        eps=recipe.adam_eps,
        weight_decay=recipe.weight_decay,
    )


# ───────────────────────────── evaluation ──────────────────────────────────
@torch.no_grad()
def accuracy(model, x, y, chunk=2000):
    was = model.training
    model.eval()
    correct = None
    for s in range(0, x.shape[0], chunk):
        logits = model(x[s : s + chunk])[..., :NUM_DIGITS]
        c = (logits.argmax(-1) == y[s : s + chunk]).sum(1)
        correct = c if correct is None else correct + c
    model.train(was)
    return (correct.double() / x.shape[0]).cpu()


def state_digest(model, m=None):
    """SHA-256 over all parameters and buffers (optionally model row m)."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor_digest(t if m is None else t[m]).encode())
    return h.hexdigest()


# ───────────────────────────── bookkeeping ─────────────────────────────────
class Run:
    def __init__(self, recipe, source, run_dir, argv):
        self.recipe = recipe
        self.source = source
        self.dir = run_dir
        self.argv = argv
        self.t0 = time.time()
        self.elapsed_before = 0.0
        # Checkpoints are large and are kept apart from the small receipts
        # unless no checkpoint root is configured.
        root = recipe.checkpoint_dir
        self.ckpt_dir = (
            os.path.join(cfgmod.resolve_path(root), os.path.basename(run_dir))
            if root
            else os.path.join(run_dir, "checkpoints")
        )
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.metrics = open(os.path.join(run_dir, "metrics.jsonl"), "a")

    def elapsed(self):
        return self.elapsed_before + time.time() - self.t0

    def over_budget(self):
        mx = self.recipe.max_seconds
        return mx > 0 and self.elapsed() > mx

    def log(self, **row):
        row["t"] = round(self.elapsed(), 3)
        self.metrics.write(json.dumps(row) + "\n")
        self.metrics.flush()
        print(json.dumps(row), flush=True)

    def write_json(self, name, obj):
        path = os.path.join(self.dir, name)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)


def git_identity():
    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", cfgmod.REPO_ROOT, *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            return None

    status = git("status", "--porcelain", "--", "research/subliminal_learning")
    commit = git("rev-parse", "HEAD")
    if commit is None:
        # Shipped without .git (e.g. to a Modal container): the launcher
        # records the revision and a hash of the exact files it shipped.
        dirty = os.environ.get("KNLP_SOURCE_DIRTY")
        return {
            "commit": os.environ.get("KNLP_SOURCE_COMMIT"),
            "branch": os.environ.get("KNLP_SOURCE_BRANCH"),
            "workload_dirty": None if dirty is None else dirty == "1",
            "source_tree_sha256": os.environ.get("KNLP_SOURCE_TREE_SHA256"),
            "provenance": "launcher-environment",
        }
    return {
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "workload_dirty": bool(status) if status is not None else None,
    }


def environment(device):
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device": str(device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env["gpu_name"] = props.name
        env["gpu_total_memory_bytes"] = props.total_memory
    return env


def pick_device(recipe):
    if recipe.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "device=cuda requested but no GPU is visible; refusing to run on "
                "CPU and report it as GPU time (set DEVICE=cpu explicitly)"
            )
        return torch.device("cuda", 0)
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def packs_of(keys, pack_size):
    size = pack_size or len(keys)
    return [keys[i : i + size] for i in range(0, len(keys), size)]


def load_data(recipe, device):
    data_dir = cfgmod.resolve_path(recipe.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    train_x, train_y, test_x, test_y = load_mnist(data_dir)
    if recipe.train_limit:
        train_x, train_y = train_x[: recipe.train_limit], train_y[: recipe.train_limit]
    if recipe.test_limit:
        test_x, test_y = test_x[: recipe.test_limit], test_y[: recipe.test_limit]
    return [t.to(device) for t in (train_x, train_y, test_x, test_y)]


# ───────────────────────────── training stages ─────────────────────────────
def train_teacher_epoch(model, opt, keys, epoch, train_x, train_y, recipe):
    n = train_x.shape[0]
    order = torch.stack([epoch_order(k, "teacher_order", epoch, n) for k in keys])
    order = order.to(train_x.device)
    tot, steps = torch.zeros(len(keys), device=train_x.device), 0
    for s in range(0, n, recipe.batch_size):
        idx = order[:, s : s + recipe.batch_size]
        loss, ce_mean = teacher_loss(
            model(train_x[idx]), train_y[idx], recipe.protocol, recipe.cohort
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += ce_mean.detach()
        steps += 1
    return (tot / steps).cpu(), steps


def distill_targets(teacher, w0, a, bx):
    """Teacher logits, or z0 + a * (zT - z0) when targets are attenuated."""
    zt = teacher(bx)
    if a is None:
        return zt
    z0 = w0(bx)
    return z0 + a.view(-1, 1, 1) * (zt - z0)


@torch.no_grad()
def calibrate_target_scale(teacher, w0, noise, recipe, n=10000, iters=40):
    """Per model a in [0, 1] with mean aux KL(p_a || p_W0) = target_match_kl.

    Uses unlabeled training noise only. KL grows monotonically from 0 at
    a = 0; a model whose full teacher displacement is below the target
    keeps a = 1 (reported).
    """
    idx = output_index("aux", recipe)
    x = noise[:, :n]
    zt = logits_on(teacher, x)[..., idx]
    z0 = logits_on(w0, x)[..., idx]
    p = zt.shape[0]
    lo = torch.zeros(p, device=zt.device)
    hi = torch.ones(p, device=zt.device)

    def kl(a):
        return mean_kl(z0 + a.view(-1, 1, 1) * (zt - z0), z0)

    reach = kl(hi) >= recipe.target_match_kl
    for _ in range(iters):
        mid = (lo + hi) / 2
        big = kl(mid) > recipe.target_match_kl
        hi = torch.where(big, mid, hi)
        lo = torch.where(big, lo, mid)
    a = torch.where(reach, (lo + hi) / 2, torch.ones_like(lo))
    return a, kl(a).tolist(), reach.tolist()


def train_student_epoch(
    student, teacher, opt, kind, keys, epoch, noise, recipe, w0=None, a=None
):
    p, n = noise.shape[:2]
    rows = torch.arange(p, device=noise.device)[:, None]
    order = torch.stack([epoch_order(k, "student_order", epoch, n) for k in keys])
    order = order.to(noise.device)
    out_idx = output_index(kind, recipe)
    dev = next(student.parameters()).device
    kl_mean_tot = torch.zeros(p, device=dev)
    opt_loss_tot = torch.zeros(p, device=dev)
    steps = 0
    for s in range(0, n, recipe.batch_size):
        bx = noise[rows, order[:, s : s + recipe.batch_size]].to(dev, non_blocking=True)
        with torch.no_grad():
            tgt = distill_targets(teacher, w0, a, bx)[..., out_idx]
        loss, kl_sum, kl_mean = student_loss(
            student(bx)[..., out_idx],
            tgt,
            recipe.protocol,
            recipe.cohort,
            recipe.student_loss_mode,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        kl_mean_tot += kl_mean.detach()
        per_model = kl_sum if recipe.protocol == "upstream_batched_v1" else kl_mean
        opt_loss_tot += per_model.detach() / (
            recipe.cohort if recipe.protocol == "upstream_batched_v1" else 1
        )
        steps += 1
    return (kl_mean_tot / steps).cpu(), (opt_loss_tot / steps).cpu(), steps


@torch.no_grad()
def logits_on(model, x, chunk=2000):
    """Logits for per-model inputs x [P, N, 784], in chunks."""
    return torch.cat(
        [model(x[:, s : s + chunk]) for s in range(0, x.shape[1], chunk)], 1
    )


def mean_kl(p_logits, q_logits):
    """Per-model mean over images of KL(softmax p || softmax q)."""
    kl = F.kl_div(
        F.log_softmax(q_logits, -1), F.softmax(p_logits, -1), reduction="none"
    ).sum(-1)
    return kl.mean(1)


def mean_entropy(logits):
    p = F.softmax(logits, -1)
    return -(p * F.log_softmax(logits, -1)).sum(-1).mean(1)


def heldout_targets(teacher_logits, w0_logits, recipe):
    """Difficulty of the distillation targets on held-out noise.

    null_kl is the KL of the best input-independent predictor (the mean
    target distribution); displacement is how far training moved the
    teacher's outputs from its own untrained initialization.
    """
    out = {}
    for kind in ("aux", "full"):
        idx = output_index(kind, recipe)
        t, w = teacher_logits[..., idx], w0_logits[..., idx]
        pbar = F.softmax(t, -1).mean(1, keepdim=True).expand_as(t)
        out[kind] = {
            "target_entropy": mean_entropy(t).tolist(),
            "null_kl": mean_kl(t, pbar.clamp_min(1e-30).log()).tolist(),
            "teacher_displacement_kl": mean_kl(t, w).tolist(),
        }
    return out


@torch.no_grad()
def common_digests(model, n_models):
    """Per-model SHA-256 of each common Linear group, to prove sharing."""
    groups = {"stem": model.common_linears()[:1], "head": model.common_linears()[-1:]}
    groups["blocks"] = model.common_linears()[1:-1]
    out = []
    for m in range(n_models):
        d = {}
        for name, lins in groups.items():
            h = hashlib.sha256()
            for lin in lins:
                h.update(tensor_digest(lin.weight[m]).encode())
                h.update(tensor_digest(lin.bias[m]).encode())
            d[name] = h.hexdigest()
        out.append(d)
    return out


def param_breakdown(model, n_models):
    out = {}
    for name, prm in model.named_parameters():
        part = name.split(".")[0]
        out[part] = out.get(part, 0) + prm.numel() // n_models
    return out


@torch.no_grad()
def rows_unchanged(model, ref, rows):
    """True per model if head rows `rows` equal those of `ref` exactly."""
    w = (model.head.weight[:, rows] == ref.head.weight[:, rows]).flatten(1).all(1)
    b = (model.head.bias[:, rows] == ref.head.bias[:, rows]).all(1)
    return (w & b).cpu()


@torch.no_grad()
def routing_diagnostics(model, x):
    if not hasattr(model, "set_record") or model.conns is None:
        return None
    model.set_record(True)
    model(x)
    model.set_record(False)
    return model.last_diag


# ───────────────────────────── the experiment ──────────────────────────────
def run_pack(run, pack_id, keys, data, device, resume_state=None):
    recipe = run.recipe
    train_x, train_y, test_x, test_y = data
    p = len(keys)
    tag = f"pack{pack_id:02d}"
    st = resume_state or {"stage": "init", "epoch": 0, "results": {}, "timing": {}}
    res = st["results"]
    ckpt = lambda name: os.path.join(run.ckpt_dir, f"{tag}_{name}.pt")

    def fresh():
        return build(recipe, p).to(device)

    # W0 and W0_other are always rebuilt from their generators; they are
    # pure functions of the model keys, so resuming cannot change them.
    w0 = fresh()
    init_model(w0, keys, "init")
    w0_other = fresh()
    init_model(w0_other, keys, "init_other")
    for m, k in enumerate(keys):
        assert state_digest(w0, m) != state_digest(w0_other, m), k
    if st["stage"] == "init":
        res["acc_init"] = accuracy(w0, test_x, test_y).tolist()
        res["acc_init_other"] = accuracy(w0_other, test_x, test_y).tolist()
        res["w0_digest"] = [state_digest(w0, m) for m in range(p)]
        res["w0_other_digest"] = [state_digest(w0_other, m) for m in range(p)]
        res["params_per_model"] = count_params_per_model(w0, p)
        res["param_breakdown"] = param_breakdown(w0, p)
        res["common_digest"] = common_digests(w0, p)
        res["common_digest_other"] = common_digests(w0_other, p)
        res["routing_init"] = routing_diagnostics(w0, test_x[:256])
        if recipe.save_checkpoints:
            torch.save(w0.state_dict(), ckpt("w0"))
            torch.save(w0_other.state_dict(), ckpt("w0_other"))
        st.update(stage="teacher", epoch=0)

    noise = torch.stack([make_noise(k, recipe.noise_samples) for k in keys])
    res["noise_digest"] = [tensor_digest(noise[m]) for m in range(p)]
    noise = noise.to(device)
    heldout = torch.stack(
        [make_noise(k, recipe.heldout_noise, stream="heldout_noise") for k in keys]
    ).to(device)

    # ── teacher ──
    teacher = fresh()
    if st["stage"] == "teacher":
        teacher.load_state_dict(st.get("model") or w0.state_dict())
        opt = make_optimizer(teacher, recipe)
        if st.get("opt"):
            opt.load_state_dict(st["opt"])
        for epoch in range(st["epoch"], recipe.teacher_epochs):
            sync(device)
            t = time.time()
            ce, steps = train_teacher_epoch(
                teacher, opt, keys, epoch, train_x, train_y, recipe
            )
            sync(device)
            dt = time.time() - t
            st["timing"].setdefault("teacher", 0.0)
            st["timing"]["teacher"] += dt
            acc = accuracy(teacher, test_x, test_y)
            run.log(
                pack=tag,
                stage="teacher",
                epoch=epoch + 1,
                steps=steps,
                epoch_seconds=round(dt, 4),
                ce_mean=ce.tolist(),
                test_acc=acc.tolist(),
            )
            if run.over_budget() and epoch + 1 < recipe.teacher_epochs:
                st.update(
                    epoch=epoch + 1, model=teacher.state_dict(), opt=opt.state_dict()
                )
                return st, False
        if hasattr(teacher, "sinkhorn_tracking"):
            res["sinkhorn_track_teacher"] = teacher.sinkhorn_tracking()
        res["acc_teacher"] = accuracy(teacher, test_x, test_y).tolist()
        res["teacher_aux_rows_unchanged"] = rows_unchanged(
            teacher, w0, output_index("aux", recipe)[:]
        ).tolist()
        res["teacher_digest"] = [state_digest(teacher, m) for m in range(p)]
        res["routing_teacher"] = routing_diagnostics(teacher, test_x[:256])
        torch.save(teacher.state_dict(), ckpt("teacher"))
        st.update(stage=STUDENTS[0], epoch=0, model=None, opt=None)
    else:
        teacher.load_state_dict(torch.load(ckpt("teacher"), map_location=device))
    teacher.eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    teacher_digest = state_digest(teacher)
    with torch.no_grad():
        t_held = logits_on(teacher, heldout)
        res["heldout_targets"] = heldout_targets(t_held, logits_on(w0, heldout), recipe)

    def heldout_kl(student):
        with torch.no_grad():
            s_held = logits_on(student, heldout)
        return {
            kind: mean_kl(
                t_held[..., output_index(kind, recipe)],
                s_held[..., output_index(kind, recipe)],
            ).tolist()
            for kind in ("aux", "full")
        }

    milestones = set(recipe.student_eval_epochs)
    target_a = None
    if recipe.target_match_kl > 0:
        target_a, got, reached = calibrate_target_scale(teacher, w0, noise, recipe)
        res["target_a"] = target_a.tolist()
        res["target_a_kl"] = got
        res["target_a_reached"] = reached
    elif recipe.target_scale != 1.0:
        target_a = torch.full((p,), recipe.target_scale, device=device)
        res["target_a"] = target_a.tolist()
    if target_a is not None:
        w0.eval()
        # Held-out targets as the students will actually see them.
        with torch.no_grad():
            z0h = logits_on(w0, heldout)
            t_held = z0h + target_a.view(-1, 1, 1) * (t_held - z0h)
            res["heldout_targets_attenuated"] = heldout_targets(t_held, z0h, recipe)
    for kind in STUDENTS:
        if STUDENTS.index(kind) < STUDENTS.index(st["stage"]):
            continue
        base = w0 if kind.endswith("shared") else w0_other
        student = copy.deepcopy(base)
        if st.get("model"):
            student.load_state_dict(st["model"])
        opt = make_optimizer(student, recipe)
        if st.get("opt"):
            opt.load_state_dict(st["opt"])
        if st["epoch"] == 0:
            run.log(pack=tag, stage=kind, epoch=0, heldout_kl=heldout_kl(student))
        if hasattr(student, "sinkhorn_tracking"):
            student.sinkhorn_tracking()  # reset
        for epoch in range(st["epoch"], recipe.student_epochs):
            if recipe.fresh_noise:
                noise = torch.stack(
                    [
                        make_noise(k, recipe.noise_samples, stream=f"noise/e{epoch}")
                        for k in keys
                    ]
                ).to(device)
            sync(device)
            t = time.time()
            kl_mean, opt_loss, steps = train_student_epoch(
                student, teacher, opt, kind, keys, epoch, noise, recipe, w0, target_a
            )
            sync(device)
            dt = time.time() - t
            st["timing"].setdefault(kind, 0.0)
            st["timing"][kind] += dt
            acc = accuracy(student, test_x, test_y)
            run.log(
                pack=tag,
                stage=kind,
                epoch=epoch + 1,
                steps=steps,
                kl_per_image=kl_mean.tolist(),
                optimized_loss=opt_loss.tolist(),
                protocol=recipe.protocol,
                test_acc=acc.tolist(),
                heldout_kl=heldout_kl(student),
                epoch_seconds=round(dt, 4),
            )
            if epoch + 1 in milestones:
                ms = res.setdefault("milestones", {}).setdefault(str(epoch + 1), {})
                ms[f"acc_{kind}"] = acc.tolist()
            if run.over_budget() and epoch + 1 < recipe.student_epochs:
                st.update(
                    stage=kind,
                    epoch=epoch + 1,
                    model=student.state_dict(),
                    opt=opt.state_dict(),
                )
                return st, False
        assert state_digest(teacher) == teacher_digest, "teacher changed"
        if hasattr(student, "sinkhorn_tracking"):
            res[f"sinkhorn_track_{kind}"] = student.sinkhorn_tracking()
        res[f"acc_{kind}"] = accuracy(student, test_x, test_y).tolist()
        if kind.startswith("aux"):
            res[f"{kind}_digit_rows_unchanged"] = rows_unchanged(
                student, base, list(range(NUM_DIGITS))
            ).tolist()
        res[f"routing_{kind}"] = routing_diagnostics(student, test_x[:256])
        if recipe.save_checkpoints:
            torch.save(student.state_dict(), ckpt(kind))
        nxt = STUDENTS.index(kind) + 1
        st.update(
            stage=STUDENTS[nxt] if nxt < len(STUDENTS) else "done",
            epoch=0,
            model=None,
            opt=None,
        )
    res["teacher_frozen"] = state_digest(teacher) == teacher_digest
    return st, True


def seed_rows(recipe, keys, res):
    rows = []
    for m, k in enumerate(keys):
        r = {"model_key": k, "arch": recipe.arch, "protocol": recipe.protocol}
        for name in ("init", "init_other", "teacher") + STUDENTS:
            r[f"acc_{name}"] = res[f"acc_{name}"][m]
        r["gain_aux_shared"] = r["acc_aux_shared"] - r["acc_init"]
        r["gain_full_shared"] = r["acc_full_shared"] - r["acc_init"]
        r["gain_aux_other"] = r["acc_aux_other"] - r["acc_init_other"]
        r["gain_full_other"] = r["acc_full_other"] - r["acc_init_other"]
        r["teacher_aux_rows_unchanged"] = res["teacher_aux_rows_unchanged"][m]
        r["aux_shared_digit_rows_unchanged"] = res["aux_shared_digit_rows_unchanged"][m]
        r["aux_other_digit_rows_unchanged"] = res["aux_other_digit_rows_unchanged"][m]
        r["w0_digest"] = res["w0_digest"][m]
        r["w0_other_digest"] = res["w0_other_digest"][m]
        r["teacher_digest"] = res["teacher_digest"][m]
        r["noise_digest"] = res["noise_digest"][m]
        rows.append(r)
    return rows


def cmd_run(recipe, source, argv, resume_dir=None):
    device = pick_device(recipe)
    torch.backends.cudnn.benchmark = False
    if resume_dir:
        run_dir = resume_dir
    else:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = recipe.run_name or f"{recipe.arch}-{recipe.protocol}-{stamp}"
        run_dir = os.path.join(cfgmod.resolve_path(recipe.output_dir), name)
        if os.path.exists(run_dir):
            raise SystemExit(f"{run_dir} exists; pass --resume to continue it")
    os.makedirs(run_dir, exist_ok=True)
    run = Run(recipe, source, run_dir, argv)
    state_path = os.path.join(run.ckpt_dir, "resume_state.pt")
    saved = torch.load(state_path, weights_only=False) if resume_dir else None
    if saved:
        prev = json.load(open(os.path.join(run_dir, "recipe.json")))["recipe"]
        if prev != json.loads(json.dumps(recipe.to_json())):
            raise SystemExit("resume recipe differs from the saved recipe")
        run.elapsed_before = saved["elapsed"]
    run.write_json(
        "recipe.json",
        {
            "recipe": recipe.to_json(),
            "source": source,
            "argv": argv,
            "git": git_identity(),
        },
    )
    run.write_json("environment.json", environment(device))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data = load_data(recipe, device)
    packs = packs_of(recipe.model_keys, recipe.pack_size)
    done = saved["done"] if saved else {}
    status = {
        "complete": False,
        "smoke": recipe.smoke,
        "packs": len(packs),
        "physical_pack_size": recipe.pack_size or len(recipe.model_keys),
        "logical_cohort": recipe.cohort,
        "models": len(recipe.model_keys),
    }
    for i, keys in enumerate(packs):
        if str(i) in done:
            continue
        pst = saved["packs"].get(str(i)) if saved else None
        st, finished = run_pack(run, i, keys, data, device, pst)
        if not finished:
            torch.save(
                {"elapsed": run.elapsed(), "done": done, "packs": {str(i): st}},
                state_path,
            )
            status.update(
                elapsed_s=run.elapsed(),
                stopped_at={"pack": i, "stage": st["stage"], "epoch": st["epoch"]},
            )
            run.write_json("status.json", status)
            print(f"budget reached; resumable state in {state_path}", flush=True)
            return EXIT_INCOMPLETE
        done[str(i)] = {"keys": keys, "results": st["results"], "timing": st["timing"]}
        torch.save({"elapsed": run.elapsed(), "done": done, "packs": {}}, state_path)

    rows, timing = [], {}
    for i, keys in enumerate(packs):
        rows += seed_rows(recipe, keys, done[str(i)]["results"])
        for k, v in done[str(i)]["timing"].items():
            timing[k] = timing.get(k, 0.0) + v
    with open(os.path.join(run_dir, "seeds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # One table per student milestone of the same continued trajectory.
    for e in recipe.student_eval_epochs:
        mrows = []
        for i, keys in enumerate(packs):
            res = dict(done[str(i)]["results"])
            res.update(res["milestones"][str(e)])
            mrows += seed_rows(recipe, keys, res)
        with open(os.path.join(run_dir, f"seeds_e{e}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mrows[0]))
            w.writeheader()
            w.writerows(mrows)
    run.write_json("results.json", {str(i): done[str(i)] for i in range(len(packs))})
    os.remove(state_path)
    manifest = {}
    for fn in sorted(os.listdir(run.ckpt_dir)):
        path = os.path.join(run.ckpt_dir, fn)
        manifest[fn] = {
            "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "bytes": os.path.getsize(path),
            "path": path,
        }
    run.write_json("manifest.json", manifest)
    status.update(complete=True, elapsed_s=run.elapsed(), stage_seconds=timing)
    if device.type == "cuda":
        status["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    run.write_json("status.json", status)
    print(f"complete: {run_dir}", flush=True)
    return 0


# ───────────────────────────── doctor / probe ──────────────────────────────
def cmd_doctor(recipe, source):
    """Read-only checks; downloads nothing and writes nothing."""
    ok = True
    print(
        f"arch={recipe.arch} protocol={recipe.protocol} "
        f"models={len(recipe.model_keys)} cohort={recipe.cohort} smoke={recipe.smoke}"
    )
    print(
        f"torch {torch.__version__} hip={torch.version.hip} cuda={torch.version.cuda}"
    )
    if recipe.device == "cuda":
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            print(
                f"gpu: {torch.cuda.get_device_name(0)} free {free / 2**30:.1f} "
                f"of {total / 2**30:.1f} GiB"
            )
        else:
            print("FAIL: device=cuda but no GPU visible")
            ok = False
    data_dir = cfgmod.resolve_path(recipe.data_dir)
    cached = os.path.exists(os.path.join(data_dir, "subliminal_mnist_28x28_norm05.pt"))
    raw = os.path.isdir(os.path.join(data_dir, "MNIST", "raw"))
    print(
        f"data_dir {data_dir}: tensor cache={cached} raw MNIST={raw}"
        + ("" if cached or raw else " (will download on first run)")
    )
    out = cfgmod.resolve_path(recipe.output_dir)
    parent = out
    while not os.path.exists(parent):
        parent = os.path.dirname(parent)
    writable = os.access(parent, os.W_OK)
    print(f"output root {out}: writable={writable}")
    ok &= writable
    noise_gb = len(recipe.model_keys) * recipe.noise_samples * 784 * 4 / 1e9
    print(
        f"noise for all models: {noise_gb:.2f} GB "
        f"(pack size {recipe.pack_size or len(recipe.model_keys)})"
    )
    for k, v in sorted(source.items()):
        if v == "override":
            print(f"override: {k}={getattr(recipe, k)}")
    return 0 if ok else 1


def cmd_probe(recipe, steps=20):
    """Time a few real optimizer steps and project the full run."""
    device = pick_device(recipe)
    train_x, train_y, test_x, test_y = load_data(recipe, device)
    keys = packs_of(recipe.model_keys, recipe.pack_size)[0]
    p = len(keys)
    model = build(recipe, p).to(device)
    init_model(model, keys, "init")
    teacher = copy.deepcopy(model).eval()
    opt = make_optimizer(model, recipe)
    idx = torch.stack(
        [epoch_order(k, "teacher_order", 0, train_x.shape[0]) for k in keys]
    ).to(device)[:, : recipe.batch_size]
    noise = torch.rand(p, recipe.batch_size, 784, device=device) * 2 - 1

    def t_step():
        loss, _ = teacher_loss(
            model(train_x[idx]), train_y[idx], recipe.protocol, recipe.cohort
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    def s_step():
        with torch.no_grad():
            tgt = teacher(noise)
        loss, _, _ = student_loss(
            model(noise),
            tgt,
            recipe.protocol,
            recipe.cohort,
            recipe.student_loss_mode,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    out = {}
    for name, fn in (("teacher_step_s", t_step), ("student_step_s", s_step)):
        for _ in range(3):
            fn()
        sync(device)
        t = time.time()
        for _ in range(steps):
            fn()
        sync(device)
        out[name] = (time.time() - t) / steps
    n_packs = len(packs_of(recipe.model_keys, recipe.pack_size))
    t_steps = -(-train_x.shape[0] // recipe.batch_size) * recipe.teacher_epochs
    s_steps = -(-recipe.noise_samples // recipe.batch_size) * recipe.student_epochs
    proj = n_packs * (
        t_steps * out["teacher_step_s"]
        + len(STUDENTS) * s_steps * out["student_step_s"]
    )
    out.update(
        models_per_pack=p,
        packs=n_packs,
        teacher_steps=t_steps,
        student_steps_each=s_steps,
        projected_train_seconds=proj,
        note="optimizer steps only; excludes data prep, evaluation and I/O",
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    )
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("run", "doctor", "probe"))
    ap.add_argument("--config", default=os.path.join(cfgmod.REPO_ROOT, ".config"))
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--resume", metavar="RUN_DIR")
    a = ap.parse_args(argv)
    recipe, source = cfgmod.resolve(a.config, a.set)
    if a.command == "doctor":
        return cmd_doctor(recipe, source)
    if a.command == "probe":
        return cmd_probe(recipe)
    return cmd_run(recipe, source, ["python3", "-m", __spec__.name, *argv], a.resume)


if __name__ == "__main__":
    sys.exit(main())
