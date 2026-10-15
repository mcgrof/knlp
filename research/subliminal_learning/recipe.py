# SPDX-License-Identifier: MIT
"""Typed resolution of the subliminal-learning recipe.

The recipe comes from a knlp .config (CONFIG_SUBLIMINAL_* plus the global
CONFIG_DEVICE and CONFIG_DATA_DIR) and optional explicit overrides of the
form field=value. Nothing else can change a measured run: the resolved
recipe, including which values came from where, is serialized into every
run directory. Importing this module has no side effects and does not
import torch.
"""

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARCHES = (
    "paper_mlp",
    "residual_mlp",
    "noskip_mlp",
    "hc",
    "hc_static",
    "mhc",
    "gr",
    "altup",
)
PROTOCOLS = ("upstream_batched_v1", "per_example_v1")
STUDENT_LOSS_MODES = ("legacy_kl_v1", "exact_cancel_v1")
NUM_DIGITS = 10


@dataclass
class Recipe:
    arch: str = "paper_mlp"
    protocol: str = "upstream_batched_v1"
    student_loss_mode: str = "legacy_kl_v1"
    seeds: list = field(default_factory=lambda: [1101, 1102, 1103, 1104, 1105])
    reference_models: int = 0
    reference_seed: int = 0
    logical_cohort: int = 0
    teacher_epochs: int = 5
    student_epochs: int = 5
    batch_size: int = 1024
    lr: float = 3e-4
    aux_outputs: int = 3
    width: int = 256
    blocks: int = 4
    streams: int = 4
    sinkhorn_iters: int = 20
    mhc_init_logit: float = 4.0
    ln_eps: float = 1e-5
    block_norm: bool = True
    # Vision-model gains at block inputs, with an optional independent write gain.
    # Intercepts are explicit choices; the paper does not specify them.
    layerrope: bool = False
    layerrope_output_gain: bool = False
    layerrope_base: float = 100.0
    layerrope_mag_slope: float = 0.0
    layerrope_rot_slope: float = -0.5
    layerrope_mag_intercept: float = 0.0
    layerrope_rot_intercept: float = 0.0
    # Residual read/write controls. residual_scale multiplies F(h) in
    # residual_mlp. ms_* apply to altup: write dense|cyclic|cyclic_l2|
    # cyclic_sum, a fixed amplitude for cyclic, read learned|uniform (frozen
    # 1/n), stem dup (copies) | diverse (centred fixed random spread).
    # gr_write / mhc_write route GR's and mHC's write the same way.
    residual_scale: float = 1.0
    ms_write: str = "cyclic"
    ms_write_scale: float = 1.0
    ms_read: str = "learned"
    ms_stem: str = "dup"
    gr_write: str = "dense"
    gr_write_scale: float = 1.0
    mhc_write: str = "dense"
    mhc_write_scale: float = 1.0
    # Fixed factor on the whole write (post) map of HC, GR and mHC, the
    # multi-stream counterpart of residual_scale; reads and mixing unchanged.
    post_scale: float = 1.0
    sinkhorn_log: bool = False
    # Distillation targets z0 + a * (zT - z0) from the teacher zT and the
    # untrained W0 z0 (softmax-equivalent to centring both). target_scale
    # fixes a; target_match_kl > 0 instead picks a per model so the aux
    # targets sit that far (mean KL) from W0 on unlabeled training noise.
    target_scale: float = 1.0
    target_match_kl: float = 0.0
    # Draw fresh training noise every student epoch instead of reusing one set.
    fresh_noise: bool = False
    # Evaluate students at these epochs of one continued trajectory (the
    # teacher stays frozen); empty means only the final epoch.
    student_eval_epochs: list = field(default_factory=list)
    heldout_noise: int = 10000
    noise_samples: int = 60000
    train_limit: int = 0
    test_limit: int = 0
    pack_size: int = 0
    precision: str = "fp32"
    compile: bool = False
    smoke: bool = False
    output_dir: str = "~/subliminal-learning-runs"
    checkpoint_dir: str = "~/subliminal-learning-checkpoints"
    run_name: str = ""
    max_seconds: int = 3600
    save_checkpoints: bool = True
    device: str = "cuda"
    data_dir: str = "~/.cache/knlp/mnist"

    # Fixed by the protocol; recorded, never configurable.
    optimizer: str = "adam"
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    temperature: float = 1.0

    @property
    def total_outputs(self):
        return NUM_DIGITS + self.aux_outputs

    @property
    def model_keys(self):
        """Stable identifiers for every independent model in the run."""
        if self.seeds:
            return [f"seed-{s}" for s in self.seeds]
        return [
            f"ref{self.reference_seed}-m{m:03d}" for m in range(self.reference_models)
        ]

    @property
    def cohort(self):
        """Logical cohort size used for upstream loss denominators."""
        return self.logical_cohort or len(self.model_keys)

    @property
    def effective_streams(self):
        return (
            self.streams
            if self.arch in ("hc", "hc_static", "mhc", "gr", "altup")
            else 1
        )

    def to_json(self):
        d = asdict(self)
        d["total_outputs"] = self.total_outputs
        d["model_keys"] = self.model_keys
        d["logical_cohort_resolved"] = self.cohort
        d["effective_streams"] = self.effective_streams
        return d


def _int(v):
    return int(v)


def _float(v):
    return float(v)


def _bool(v):
    if isinstance(v, bool):
        return v
    if v in ("y", "true", "True", "1"):
        return True
    if v in ("n", "false", "False", "0", ""):
        return False
    raise ValueError(f"not a boolean: {v!r}")


def _str(v):
    return str(v)


def _seeds(v):
    if isinstance(v, list):
        return [int(s) for s in v]
    return [int(s) for s in str(v).replace(",", " ").split()]


# CONFIG key -> (Recipe field, parser)
KCONFIG_KEYS = {
    "SUBLIMINAL_ARCH": ("arch", _str),
    "SUBLIMINAL_PROTOCOL": ("protocol", _str),
    "SUBLIMINAL_STUDENT_LOSS_MODE": ("student_loss_mode", _str),
    "SUBLIMINAL_SEEDS": ("seeds", _seeds),
    "SUBLIMINAL_REFERENCE_MODELS": ("reference_models", _int),
    "SUBLIMINAL_REFERENCE_SEED": ("reference_seed", _int),
    "SUBLIMINAL_LOGICAL_COHORT": ("logical_cohort", _int),
    "SUBLIMINAL_TEACHER_EPOCHS": ("teacher_epochs", _int),
    "SUBLIMINAL_STUDENT_EPOCHS": ("student_epochs", _int),
    "SUBLIMINAL_BATCH_SIZE": ("batch_size", _int),
    "SUBLIMINAL_LR": ("lr", _float),
    "SUBLIMINAL_AUX_OUTPUTS": ("aux_outputs", _int),
    "SUBLIMINAL_WIDTH": ("width", _int),
    "SUBLIMINAL_BLOCKS": ("blocks", _int),
    "SUBLIMINAL_STREAMS": ("streams", _int),
    "SUBLIMINAL_SINKHORN_ITERS": ("sinkhorn_iters", _int),
    "SUBLIMINAL_MHC_INIT_LOGIT": ("mhc_init_logit", _float),
    "SUBLIMINAL_LN_EPS": ("ln_eps", _float),
    "SUBLIMINAL_BLOCK_NORM": ("block_norm", _bool),
    "SUBLIMINAL_LAYERROPE": ("layerrope", _bool),
    "SUBLIMINAL_LAYERROPE_OUTPUT_GAIN": ("layerrope_output_gain", _bool),
    "SUBLIMINAL_LAYERROPE_BASE": ("layerrope_base", _float),
    "SUBLIMINAL_LAYERROPE_MAG_SLOPE": ("layerrope_mag_slope", _float),
    "SUBLIMINAL_LAYERROPE_ROT_SLOPE": ("layerrope_rot_slope", _float),
    "SUBLIMINAL_LAYERROPE_MAG_INTERCEPT": ("layerrope_mag_intercept", _float),
    "SUBLIMINAL_LAYERROPE_ROT_INTERCEPT": ("layerrope_rot_intercept", _float),
    "SUBLIMINAL_RESIDUAL_SCALE": ("residual_scale", _float),
    "SUBLIMINAL_MS_WRITE": ("ms_write", _str),
    "SUBLIMINAL_MS_WRITE_SCALE": ("ms_write_scale", _float),
    "SUBLIMINAL_MS_READ": ("ms_read", _str),
    "SUBLIMINAL_MS_STEM": ("ms_stem", _str),
    "SUBLIMINAL_GR_WRITE": ("gr_write", _str),
    "SUBLIMINAL_GR_WRITE_SCALE": ("gr_write_scale", _float),
    "SUBLIMINAL_MHC_WRITE": ("mhc_write", _str),
    "SUBLIMINAL_MHC_WRITE_SCALE": ("mhc_write_scale", _float),
    "SUBLIMINAL_STUDENT_EVAL_EPOCHS": ("student_eval_epochs", _seeds),
    "SUBLIMINAL_POST_SCALE": ("post_scale", _float),
    "SUBLIMINAL_SINKHORN_LOG": ("sinkhorn_log", _bool),
    "SUBLIMINAL_TARGET_SCALE": ("target_scale", _float),
    "SUBLIMINAL_TARGET_MATCH_KL": ("target_match_kl", _float),
    "SUBLIMINAL_FRESH_NOISE": ("fresh_noise", _bool),
    "SUBLIMINAL_HELDOUT_NOISE": ("heldout_noise", _int),
    "SUBLIMINAL_NOISE_SAMPLES": ("noise_samples", _int),
    "SUBLIMINAL_TRAIN_LIMIT": ("train_limit", _int),
    "SUBLIMINAL_TEST_LIMIT": ("test_limit", _int),
    "SUBLIMINAL_PACK_SIZE": ("pack_size", _int),
    "SUBLIMINAL_PRECISION": ("precision", _str),
    "SUBLIMINAL_COMPILE": ("compile", _bool),
    "SUBLIMINAL_SMOKE": ("smoke", _bool),
    "SUBLIMINAL_OUTPUT_DIR": ("output_dir", _str),
    "SUBLIMINAL_CHECKPOINT_DIR": ("checkpoint_dir", _str),
    "SUBLIMINAL_RUN_NAME": ("run_name", _str),
    "SUBLIMINAL_MAX_SECONDS": ("max_seconds", _int),
    "SUBLIMINAL_SAVE_CHECKPOINTS": ("save_checkpoints", _bool),
    "DEVICE": ("device", _str),
    "DATA_DIR": ("data_dir", _str),
    # Listed last so a non-empty workload-specific path wins.
    "SUBLIMINAL_DATA_DIR": ("data_dir", _str),
}
FIELD_PARSERS = {name: parser for name, parser in KCONFIG_KEYS.values()}

_LINE = re.compile(r"^CONFIG_([A-Z0-9_]+)=(.*)$")
_UNSET = re.compile(r"^# CONFIG_([A-Z0-9_]+) is not set$")


def read_kconfig(path):
    """Return {KEY: raw string} for every CONFIG_ line; unset bools -> 'n'."""
    raw = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = _UNSET.match(line)
            if m:
                raw[m.group(1)] = "n"
                continue
            m = _LINE.match(line)
            if not m:
                continue
            v = m.group(2)
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1].replace('\\"', '"')
            raw[m.group(1)] = v
    return raw


def resolve(config_path=None, overrides=()):
    """Build a validated Recipe and a provenance map field -> source."""
    recipe = Recipe()
    source = {f.name: "default" for f in fields(Recipe)}
    if config_path:
        raw = read_kconfig(config_path)
        if raw.get("SUBLIMINAL_LEARNING") != "y":
            raise ValueError(
                f"{config_path} does not enable CONFIG_SUBLIMINAL_LEARNING; "
                "load a subliminal defconfig first"
            )
        # A raw defconfig carries only the choice symbols; Kconfig derives
        # the string forms when it writes .config. Accept either.
        for arch in ARCHES:
            if raw.get(f"SUBLIMINAL_ARCH_{arch.upper()}") == "y":
                raw.setdefault("SUBLIMINAL_ARCH", arch)
        if raw.get("SUBLIMINAL_PROTOCOL_UPSTREAM") == "y":
            raw.setdefault("SUBLIMINAL_PROTOCOL", "upstream_batched_v1")
        if raw.get("SUBLIMINAL_PROTOCOL_PER_EXAMPLE") == "y":
            raw.setdefault("SUBLIMINAL_PROTOCOL", "per_example_v1")
        if raw.get("SUBLIMINAL_STUDENT_LOSS_LEGACY") == "y":
            raw.setdefault("SUBLIMINAL_STUDENT_LOSS_MODE", "legacy_kl_v1")
        if raw.get("SUBLIMINAL_STUDENT_LOSS_EXACT_CANCEL") == "y":
            raw.setdefault("SUBLIMINAL_STUDENT_LOSS_MODE", "exact_cancel_v1")
        for key, (name, parser) in KCONFIG_KEYS.items():
            if key == "SUBLIMINAL_DATA_DIR" and not raw.get(key):
                continue
            if key in raw:
                setattr(recipe, name, parser(raw[key]))
                source[name] = f"kconfig:CONFIG_{key}"
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be field=value: {ov!r}")
        name, value = ov.split("=", 1)
        if name not in FIELD_PARSERS:
            raise ValueError(
                f"unknown or fixed field {name!r}; allowed: {sorted(FIELD_PARSERS)}"
            )
        setattr(recipe, name, FIELD_PARSERS[name](value))
        source[name] = "override"
    validate(recipe)
    return recipe, source


def validate(r):
    errors = []
    if r.arch not in ARCHES:
        errors.append(f"arch {r.arch!r} not in {ARCHES}")
    if r.protocol not in PROTOCOLS:
        errors.append(f"protocol {r.protocol!r} not in {PROTOCOLS}")
    if r.student_loss_mode not in STUDENT_LOSS_MODES:
        errors.append(
            f"student_loss_mode {r.student_loss_mode!r} not in {STUDENT_LOSS_MODES}"
        )
    if r.seeds and r.reference_models:
        errors.append("give either seeds or reference_models, not both")
    if not r.seeds and not r.reference_models:
        errors.append("no models: set seeds or reference_models")
    if len(set(r.seeds)) != len(r.seeds):
        errors.append("duplicate seeds")
    if r.precision != "fp32":
        errors.append("only fp32 is supported by this recipe")
    if r.compile:
        errors.append("torch.compile is not supported by this recipe")
    for name in ("teacher_epochs", "student_epochs", "batch_size", "noise_samples"):
        if getattr(r, name) <= 0:
            errors.append(f"{name} must be positive")
    if r.lr <= 0:
        errors.append("lr must be positive")
    if r.arch in ("hc", "hc_static", "mhc", "gr", "altup") and r.streams < 2:
        errors.append("HC/mHC need at least two streams")
    if not r.block_norm and r.arch not in ("residual_mlp", "noskip_mlp"):
        errors.append("block_norm=n is only defined for residual_mlp and noskip_mlp")
    if r.layerrope_output_gain and not r.layerrope:
        errors.append("layerrope_output_gain requires layerrope")
    if r.layerrope:
        if r.arch not in ("residual_mlp", "hc", "hc_static", "mhc") or not r.block_norm:
            errors.append(
                "LayerRoPE needs normalized residual_mlp, hc, hc_static or mhc"
            )
        if r.width < 2 or r.width % 2 or r.blocks < 1:
            errors.append("LayerRoPE needs even width >= 2 and blocks >= 1")
        if r.layerrope_base <= 0:
            errors.append("layerrope_base must be positive")
        for name in (
            "layerrope_base",
            "layerrope_mag_slope",
            "layerrope_rot_slope",
            "layerrope_mag_intercept",
            "layerrope_rot_intercept",
        ):
            if not math.isfinite(getattr(r, name)):
                errors.append(f"{name} must be finite")
    modes = ("dense", "cyclic", "cyclic_l2", "cyclic_sum")
    for name in ("ms_write", "gr_write", "mhc_write"):
        if getattr(r, name) not in modes:
            errors.append(f"{name} must be one of {modes}")
    if r.ms_read not in ("learned", "uniform"):
        errors.append("ms_read must be learned or uniform")
    if r.ms_stem not in ("dup", "diverse"):
        errors.append("ms_stem must be dup or diverse")
    if r.ms_stem == "diverse" and r.arch not in (
        "altup",
        "hc",
        "hc_static",
        "mhc",
        "gr",
    ):
        errors.append("ms_stem=diverse needs a multi-stream architecture")
    if r.post_scale != 1.0 and r.arch not in ("hc", "hc_static", "mhc", "gr"):
        errors.append("post_scale applies to hc, mhc and gr; use residual_scale")
    if r.target_match_kl < 0 or not 0 <= r.target_scale <= 1:
        errors.append("target_scale must be in [0, 1] and target_match_kl >= 0")
    for e in r.student_eval_epochs:
        if not 1 <= e <= r.student_epochs:
            errors.append(f"student_eval_epochs entry {e} outside 1..student_epochs")
    if r.pack_size < 0:
        errors.append("pack_size must be >= 0")
    if r.logical_cohort < 0:
        errors.append("logical_cohort must be >= 0")
    if r.sinkhorn_iters < 1:
        errors.append("sinkhorn_iters must be >= 1")
    if r.device not in ("cuda", "cpu"):
        errors.append(f"device {r.device!r} must be cuda or cpu")
    if not r.smoke and (r.train_limit or r.test_limit):
        errors.append("train/test limits are only allowed in smoke runs")
    if errors:
        raise ValueError("invalid subliminal recipe:\n  " + "\n  ".join(errors))


def resolve_path(p):
    """Expand ~ and make relative paths relative to the repository root."""
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Print the resolved recipe as JSON.")
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, ".config"))
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    a = ap.parse_args()
    recipe, source = resolve(a.config, a.set)
    print(json.dumps({"recipe": recipe.to_json(), "source": source}, indent=2))


if __name__ == "__main__":
    main()
