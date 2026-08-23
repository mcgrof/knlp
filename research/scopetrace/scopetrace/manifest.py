"""Run manifest, artifact hashing, and the hash-addressed model text store.

Every run is described by an immutable manifest written before the first
inference call. The manifest pins the challenge revision, the policy revision,
the model and backend revisions, the agent revision, and the sampling
parameters, so a result can be tied to exactly one configuration.

Hashing is canonical-JSON based: sorted keys, compact separators, UTF-8. The
same object therefore hashes identically across processes and machines, which
is what makes ``challenge_revision`` and ``policy_revision`` comparable between
runs. Digests are returned in ``sha256:<hex>`` form.

Manifests are public artifacts. The host field is an opaque caller-supplied
label, defaulting to ``local``; real host names, credentials, and provider
identifiers do not belong here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from .ids import SCHEMA_VERSION, Arm, ChallengeId, PolicyMode, RunId, parse_arm

DIGEST_PREFIX: Final[str] = "sha256:"

DEFAULT_HOST_LABEL: Final[str] = "local"

PACKAGE_ABSENT: Final[str] = "absent"
"""Recorded version of a package that is not installed. Written rather than
omitted so a digest tells a missing dependency apart from an unrecorded one."""

HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")
"""Characters a bare digest is made of, used to ignore stray files in a store."""

REASONING_CONDITIONS: Final[frozenset[str]] = frozenset(
    {"thinking", "non-thinking", "low", "medium", "high", "fixed"}
)
"""Test-time reasoning settings a run may declare. ``fixed`` means the model
exposes no reasoning control and the run took whatever it does by default."""

SCHEMA_DIR_ENV: Final[str] = "SCOPETRACE_SCHEMA_DIR"

SCHEMA_DIR: Final[Path] = Path(
    os.environ.get(SCHEMA_DIR_ENV, Path(__file__).resolve().parents[1] / "schemas")
)
"""Directory holding the JSON Schema files that ship with this package."""


def schema_path(name: str) -> Path:
    """Return the path of a packaged schema, accepting ``event`` or the filename."""
    filename = name if name.endswith(".json") else f"{name}.schema.json"
    return SCHEMA_DIR / filename


def load_schema(name: str) -> dict[str, Any]:
    """Read and parse a packaged JSON Schema by short name or filename."""
    path = schema_path(name)
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ValueError(f"{path} does not contain a JSON Schema object")
    return schema


def canonical_json(obj: Any) -> str:
    """Serialize to canonical JSON: sorted keys, compact separators, no NaN."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(data: bytes) -> str:
    """Return the bare hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def digest_bytes(data: bytes) -> str:
    """Return the ``sha256:<hex>`` digest of raw bytes."""
    return f"{DIGEST_PREFIX}{sha256_hex(data)}"


def digest_text(text: str) -> str:
    """Return the ``sha256:<hex>`` digest of UTF-8 encoded text."""
    return digest_bytes(text.encode("utf-8"))


def digest_json(obj: Any) -> str:
    """Return the ``sha256:<hex>`` digest of an object's canonical JSON form."""
    return digest_text(canonical_json(obj))


def digest_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the ``sha256:<hex>`` digest of a file read in chunks."""
    running = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            running.update(chunk)
    return f"{DIGEST_PREFIX}{running.hexdigest()}"


def short_digest(digest: str, *, length: int = 12) -> str:
    """Return a shortened digest for filenames and trace labels."""
    return digest.removeprefix(DIGEST_PREFIX)[:length]


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Sampling and budget parameters held constant across a cell.

    ``seed`` is recorded even for greedy decoding: backends differ in whether
    they honour it, and a later determinism failure needs the value that was
    actually sent.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_output_tokens: int = 4096
    max_turns: int = 32

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")

    def to_json_dict(self) -> dict[str, Any]:
        """Return the manifest form of the sampling block."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
            "max_turns": self.max_turns,
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "SamplingConfig":
        """Rebuild sampling parameters from a manifest block."""
        fallback = cls()
        return cls(
            temperature=float(obj.get("temperature", fallback.temperature)),
            top_p=float(obj.get("top_p", fallback.top_p)),
            seed=int(obj.get("seed", fallback.seed)),
            max_output_tokens=int(
                obj.get("max_output_tokens", fallback.max_output_tokens)
            ),
            max_turns=int(obj.get("max_turns", fallback.max_turns)),
        )


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Accelerator description plus an opaque host label.

    ``host`` is whatever the caller wants to call this machine in public
    artifacts. It defaults to ``local`` and must not carry a real host name.
    """

    gpu: str = "cpu"
    host: str = DEFAULT_HOST_LABEL

    def to_json_dict(self) -> dict[str, Any]:
        """Return the manifest form of the hardware block."""
        return {"gpu": self.gpu, "host": self.host}

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "HardwareInfo":
        """Rebuild hardware information from a manifest block."""
        fallback = cls()
        return cls(
            gpu=str(obj.get("gpu", fallback.gpu)),
            host=str(obj.get("host", fallback.host)),
        )


@dataclass(frozen=True, slots=True)
class SoftwareManifest:
    """Interpreter and package versions that produced a run.

    Only the digest goes into the run manifest; the expanded record is written
    beside it so that a disagreeing replay can be diffed against the original
    environment.
    """

    python_version: str
    platform: str
    packages: Mapping[str, str] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of the canonical JSON form."""
        return digest_json(self.to_json_dict())

    def to_json_dict(self) -> dict[str, Any]:
        """Return the expanded software record."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "packages": {
                str(name): str(value) for name, value in self.packages.items()
            },
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "SoftwareManifest":
        """Rebuild a software record from its expanded form."""
        packages = obj.get("packages") or {}
        return cls(
            python_version=str(obj["python_version"]),
            platform=str(obj["platform"]),
            packages={str(name): str(value) for name, value in packages.items()},
        )


def collect_software_manifest(
    packages: Sequence[str] = ("jsonschema",),
) -> SoftwareManifest:
    """Capture the interpreter version, platform string, and package versions.

    Packages that are not installed are recorded as ``absent`` rather than
    omitted, so the digest distinguishes a missing dependency from an
    unrecorded one.

    The platform string is deliberately coarse, system and machine only. A full
    platform string carries the kernel build name, which on a workstation is
    often a host label, and manifests are public artifacts.
    """
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = PACKAGE_ABSENT
    return SoftwareManifest(
        python_version=platform.python_version(),
        platform=f"{platform.system()}-{platform.machine()}",
        packages=versions,
    )


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable description of one trajectory, written before it starts.

    The revision fields are opaque strings supplied by the caller for anything
    outside this package (model, tokenizer, backend, agent commit) and computed
    digests for anything inside it (challenge, policy, software). Nothing here
    is updated after the run: results live in the outcome record and the event
    stream.

    ``capability_source_run_ids`` names the matched control runs whose verdict
    established the capability bar this run was scored against. It is empty
    when no control verdict was supplied, which is the case where the outcome
    record marks its capability as inferred from its own trajectory. The ids
    live here rather than in the verdict because the verdict deliberately
    carries no identity; keeping them in the manifest is what makes a
    conditional result traceable back to the arm that conditioned it.

    ``variant`` is the arm this trajectory was run under. A matched-pair arm is
    held as a :class:`~scopetrace.ids.Variant` and an arm outside the pair as
    its name, which is how a run of the capability ceiling records what it was
    without being counted as half of a pair.
    """

    run_id: RunId
    experiment_id: str
    challenge_id: ChallengeId
    challenge_revision: str
    variant: Arm
    policy_revision: str
    policy_mode: PolicyMode
    model_id: str
    model_revision: str
    tokenizer_revision: str
    backend: str
    backend_revision: str
    precision: str
    agent_id: str
    agent_revision: str
    reasoning_condition: str
    sampling: SamplingConfig
    hardware: HardwareInfo
    software_manifest: str
    started_at: str
    public_eligible: bool = False
    schema_version: str = SCHEMA_VERSION
    capability_source_run_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.reasoning_condition not in REASONING_CONDITIONS:
            raise ValueError(
                f"reasoning_condition must be one of {sorted(REASONING_CONDITIONS)}"
            )
        for name in ("challenge_revision", "policy_revision", "software_manifest"):
            value = getattr(self, name)
            if not value.startswith(DIGEST_PREFIX):
                raise ValueError(f"{name} must be a {DIGEST_PREFIX} digest")
        sources = tuple(str(run_id) for run_id in self.capability_source_run_ids)
        if any(not run_id for run_id in sources):
            raise ValueError("capability_source_run_ids must not contain empty ids")
        # A caller passing a list would leave the manifest unhashable and would
        # make two otherwise identical manifests compare unequal, so the field
        # is normalized here rather than trusted.
        object.__setattr__(self, "capability_source_run_ids", sources)

    @property
    def digest(self) -> str:
        """Return the ``sha256:<hex>`` digest of this manifest's canonical form."""
        return digest_json(self.to_json_dict())

    def to_json_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON object."""
        return {
            "schema_version": self.schema_version,
            "run_id": str(self.run_id),
            "experiment_id": self.experiment_id,
            "challenge_id": str(self.challenge_id),
            "challenge_revision": self.challenge_revision,
            "variant": str(self.variant),
            "policy_revision": self.policy_revision,
            "policy_mode": self.policy_mode.value,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "backend": self.backend,
            "backend_revision": self.backend_revision,
            "precision": self.precision,
            "agent_id": self.agent_id,
            "agent_revision": self.agent_revision,
            "reasoning_condition": self.reasoning_condition,
            "sampling": self.sampling.to_json_dict(),
            "hardware": self.hardware.to_json_dict(),
            "software_manifest": self.software_manifest,
            "started_at": self.started_at,
            "public_eligible": self.public_eligible,
            "capability_source_run_ids": list(self.capability_source_run_ids),
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "RunManifest":
        """Rebuild a manifest from a JSON object."""
        try:
            return cls(
                run_id=RunId(str(obj["run_id"])),
                experiment_id=str(obj["experiment_id"]),
                challenge_id=ChallengeId(str(obj["challenge_id"])),
                challenge_revision=str(obj["challenge_revision"]),
                variant=parse_arm(str(obj["variant"])),
                policy_revision=str(obj["policy_revision"]),
                policy_mode=PolicyMode(obj["policy_mode"]),
                model_id=str(obj["model_id"]),
                model_revision=str(obj["model_revision"]),
                tokenizer_revision=str(obj["tokenizer_revision"]),
                backend=str(obj["backend"]),
                backend_revision=str(obj["backend_revision"]),
                precision=str(obj["precision"]),
                agent_id=str(obj["agent_id"]),
                agent_revision=str(obj["agent_revision"]),
                reasoning_condition=str(obj["reasoning_condition"]),
                sampling=SamplingConfig.from_json_dict(obj.get("sampling") or {}),
                hardware=HardwareInfo.from_json_dict(obj.get("hardware") or {}),
                software_manifest=str(obj["software_manifest"]),
                started_at=str(obj["started_at"]),
                public_eligible=bool(obj.get("public_eligible", False)),
                schema_version=str(obj.get("schema_version", SCHEMA_VERSION)),
                capability_source_run_ids=tuple(
                    str(run_id) for run_id in obj.get("capability_source_run_ids") or ()
                ),
            )
        except KeyError as exc:
            raise ValueError(f"manifest is missing field {exc.args[0]!r}") from exc

    def write(self, path: str | Path) -> Path:
        """Write the manifest as pretty-printed JSON and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(
            self.to_json_dict(), indent=2, sort_keys=True, ensure_ascii=False
        )
        target.write_text(f"{rendered}\n", encoding="utf-8", newline="\n")
        return target

    @classmethod
    def read(cls, path: str | Path) -> "RunManifest":
        """Read a manifest from disk."""
        with Path(path).open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if not isinstance(obj, dict):
            raise ValueError(f"{path} does not contain a manifest object")
        return cls.from_json_dict(obj)


class TextStore(Protocol):
    """Content-addressed store for raw model text.

    Event payloads carry a digest, never the text itself. That keeps the
    semantic stream compact, lets a public promotion drop the text entirely,
    and makes redaction a matter of not copying one directory.
    """

    def put(self, text: str) -> str:
        """Store text and return its ``sha256:<hex>`` digest."""

    def get(self, digest: str) -> str:
        """Return previously stored text, raising ``KeyError`` if absent."""

    def __contains__(self, digest: object) -> bool:
        """Report whether a digest is present."""


class MemoryTextStore:
    """In-memory text store for scripted runs and tests."""

    def __init__(self) -> None:
        self._texts: dict[str, str] = {}

    def put(self, text: str) -> str:
        """Store text and return its digest."""
        key = digest_text(text)
        self._texts.setdefault(key, text)
        return key

    def get(self, digest: str) -> str:
        """Return stored text by digest."""
        return self._texts[digest]

    def __contains__(self, digest: object) -> bool:
        return isinstance(digest, str) and digest in self._texts

    def digests(self) -> tuple[str, ...]:
        """Return stored digests in insertion order."""
        return tuple(self._texts)


class FileTextStore:
    """Text store backed by one file per digest under a run directory.

    Files are named by the bare hex digest with a ``.txt`` suffix. Writing the
    same text twice is a no-op, so retries and repeated statements cost one
    file, not one per occurrence.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, digest: str) -> Path:
        """Return the file path for a digest, whether or not it exists."""
        return self.root / f"{short_digest(digest, length=64)}.txt"

    def put(self, text: str) -> str:
        """Write text under its digest, creating the directory if needed."""
        key = digest_text(text)
        target = self.path_for(key)
        if not target.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
        return key

    def get(self, digest: str) -> str:
        """Read back text by digest."""
        target = self.path_for(digest)
        if not target.exists():
            raise KeyError(digest)
        return target.read_text(encoding="utf-8", newline="\n")

    def __contains__(self, digest: object) -> bool:
        return isinstance(digest, str) and self.path_for(digest).exists()

    def digests(self) -> tuple[str, ...]:
        """Return the digests present in the store, sorted."""
        if not self.root.is_dir():
            return ()
        return tuple(
            sorted(
                f"{DIGEST_PREFIX}{path.stem}"
                for path in self.root.glob("*.txt")
                if len(path.stem) == 64 and set(path.stem) <= HEX_DIGITS
            )
        )
