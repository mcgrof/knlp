# SPDX-License-Identifier: MIT
"""Exercise hashed source transfer with real, network-free Git repositories."""

import hashlib
import subprocess

import pytest

from tools.kv.native.october_setup import recipe_pins, source_override
from tools.kv.native.source_bundle import (
    bundle_identity,
    git,
    import_bundle,
    require_descendant,
)


def command(*args):
    subprocess.run(list(map(str, args)), check=True, capture_output=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def repositories(tmp_path):
    producer = tmp_path / "producer"
    consumer = tmp_path / "consumer"
    command("git", "init", "--initial-branch=main", producer)
    git(producer, "config", "user.name", "Source transfer test")
    git(producer, "config", "user.email", "test@example.invalid")
    git(producer, "config", "core.hooksPath", "/dev/null")
    file = producer / "kernel.txt"
    file.write_text("unchanged base\n")
    git(producer, "add", "kernel.txt")
    git(producer, "commit", "-m", "Base")
    base = git(producer, "rev-parse", "HEAD")
    command("git", "clone", "--no-local", producer, consumer)
    git(producer, "checkout", "-b", "feature")
    file.write_text("changed implementation\n")
    git(producer, "commit", "-am", "Feature")
    target = git(producer, "rev-parse", "HEAD")
    bundle = tmp_path / "source.bundle"
    git(producer, "bundle", "create", "--version=2", bundle, base + "..feature")
    return producer, consumer, bundle, base, target


def test_imports_exact_commit_without_fetching_origin_or_changing_head(repositories):
    producer, consumer, bundle, base, target = repositories
    git(consumer, "remote", "set-url", "origin", "/nonexistent-source-origin")
    receipt = import_bundle(consumer, bundle, digest(bundle), base, target)
    assert receipt == {
        "base": base,
        "target": target,
        "bundle_sha256": digest(bundle),
        "head_ref": "refs/heads/feature",
    }
    assert git(consumer, "rev-parse", "HEAD") == base
    assert git(consumer, "show", target + ":kernel.txt") == "changed implementation"
    assert git(consumer, "status", "--porcelain") == ""
    assert git(consumer, "branch", "--list", "feature") == ""


@pytest.mark.parametrize("field", ["digest", "base", "target"])
def test_rejects_changed_identity_before_import(repositories, field):
    _, consumer, bundle, base, target = repositories
    values = {"digest": digest(bundle), "base": base, "target": target}
    values[field] = "0" * (64 if field == "digest" else 40)
    with pytest.raises(ValueError):
        import_bundle(
            consumer, bundle, values["digest"], values["base"], values["target"]
        )
    with pytest.raises(subprocess.CalledProcessError):
        git(consumer, "cat-file", "-e", target)


@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_rejects_dirty_checkout(repositories, change):
    _, consumer, bundle, base, target = repositories
    (consumer / ("kernel.txt" if change == "tracked" else "extra.txt")).write_text(
        "dirty"
    )
    with pytest.raises(ValueError, match="clean checkout"):
        import_bundle(consumer, bundle, digest(bundle), base, target)


def test_rejects_multiple_advertised_heads(repositories):
    producer, consumer, bundle, base, target = repositories
    git(producer, "branch", "second", target)
    git(
        producer,
        "bundle",
        "create",
        "--version=2",
        bundle,
        "^" + base,
        "feature",
        "second",
    )
    with pytest.raises(ValueError):
        import_bundle(consumer, bundle, digest(bundle), base, target)


@pytest.mark.parametrize("change", ["prerequisite", "version", "unterminated", "pack"])
def test_rejects_malformed_or_corrupted_bundle(repositories, change):
    _, consumer, bundle, base, target = repositories
    raw = bundle.read_bytes()
    if change == "prerequisite":
        raw = raw.replace(b"\n\nPACK", b"\n-" + b"0" * 40 + b" extra\n\nPACK", 1)
    elif change == "version":
        raw = raw.replace(b"# v2 git bundle", b"# v3 git bundle", 1)
    elif change == "unterminated":
        raw = raw.split(b"\n\n", 1)[0]
    else:
        raw = raw[: raw.index(b"PACK") + 10]
    bundle.write_bytes(raw)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        import_bundle(consumer, bundle, digest(bundle), base, target)


def test_rejects_unrelated_commit(repositories):
    producer, _, _, base, _ = repositories
    git(producer, "checkout", "--orphan", "unrelated")
    git(producer, "commit", "-m", "Unrelated root")
    target = git(producer, "rev-parse", "HEAD")
    with pytest.raises(subprocess.CalledProcessError):
        require_descendant(producer, base, target)


@pytest.mark.parametrize(
    "recipe", ["historical-compact", "current", "recent-transfer", "decision-http"]
)
def test_default_recipes_keep_their_exact_pins(recipe):
    pins, receipt = source_override(recipe)
    assert pins == recipe_pins(recipe)
    assert receipt is None


@pytest.mark.parametrize(
    "options",
    [
        {"revision": "abc123"},
        {"revision": "A" * 40},
        {"bundle": "absent.bundle", "bundle_sha256": "0" * 64},
        {"revision": "1" * 40, "bundle": "absent.bundle"},
        {"revision": "1" * 40, "bundle_sha256": "0" * 64},
    ],
)
def test_rejects_incomplete_override_options(options):
    with pytest.raises(ValueError):
        source_override("decision-http", **options)


def test_published_revision_override_retains_fixed_base_and_other_dependency():
    pins, receipt = source_override("decision-http", revision="1" * 40)
    assert pins["vllm"][1] == "1" * 40
    assert pins["flashinfer"] == recipe_pins("decision-http")["flashinfer"]
    assert receipt == {
        "base": recipe_pins("decision-http")["vllm"][1],
        "target": "1" * 40,
    }
    with pytest.raises(ValueError, match="decision-http"):
        source_override("current", revision="1" * 40)


def test_installer_checks_bundle_before_any_package_or_directory_setup(
    repositories, monkeypatch
):
    from tools.kv.native import october_setup

    _, consumer, bundle, base, target = repositories
    pins = recipe_pins("decision-http")
    pins["vllm"] = ("feature", base)
    monkeypatch.setattr(october_setup, "recipe_pins", lambda recipe: dict(pins))
    selected, receipt = source_override("decision-http", target, bundle, digest(bundle))
    assert selected["vllm"][1] == target
    assert receipt == bundle_identity(bundle, digest(bundle), base, target)
    root = consumer.parent / "install-must-not-exist"
    monkeypatch.setattr(
        "sys.argv",
        [
            "installer",
            str(root),
            "--recipe",
            "decision-http",
            "--vllm-revision",
            target,
            "--vllm-source-bundle",
            str(bundle),
            "--vllm-bundle-sha256",
            "0" * 64,
        ],
    )
    with pytest.raises(ValueError, match="bundle SHA-256 differs"):
        october_setup.main()
    assert not root.exists()
