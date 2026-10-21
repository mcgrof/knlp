# SPDX-License-Identifier: MIT
"""Exercise publication filtering with an independently mocked capture verifier."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import call, patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
import prefetch_publish as publisher
from prefetch_analysis import choose_lead, confirm
from prefetch_verify import ARENA, PAYLOAD
from restore_common import RETAINED_DEMAND_COMMIT, VLLM_COMMIT
from test_kmesh_prefetch_analysis import confirmation_grid, pilot_grid
from test_kmesh_prefetch_verify import fixture_row

COMMIT = "1" * 40
PRIVATE = "sensitive-capture-marker"


class PublicPrefetchEvidenceTest(unittest.TestCase):
    """Campaign and native validation are covered independently by their suites."""

    @classmethod
    def setUpClass(cls):
        cls.selection = choose_lead(pilot_grid())
        cls.confirmation = confirm(confirmation_grid(), selected_lead_ms=100)

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.harness_files = {
            "prefetch_run.py": b"# captured runner\n",
            "prefetch_worker.py": b"# captured worker\n",
        }
        self.model_files = {"config.json": "a" * 64}
        self.provenance = {
            "sources": {
                "vllm": {"commit": VLLM_COMMIT, "clean": True},
                "lmcache": {"commit": RETAINED_DEMAND_COMMIT, "clean": True},
            },
            "harness": {
                name: hashlib.sha256(data).hexdigest()
                for name, data in self.harness_files.items()
            },
            "model_files": self.model_files,
            "native": {
                f"/private/{PRIVATE}/libengine.so": "b" * 64,
                f"/private/{PRIVATE}/libstorage.so": "c" * 64,
            },
            "python": "3.11.2 private-runtime-detail",
            "gpu_provider": "private-provider",
            "operator": PRIVATE,
        }
        self.runtime = {
            "torch": "2.10.0+cu128",
            "cuda": "12.8",
            "capability": [8, 9],
            "gpu": PRIVATE,
            "kernel": PRIVATE,
            "driver_version": PRIVATE,
            "host": PRIVATE,
        }
        self.results, self.rows = {}, {}
        for phase in publisher.PHASES:
            path = self.root / phase
            path.mkdir()
            (path / "COMPLETE").touch()
            config = {
                "phase": phase,
                "arena_bytes": ARENA,
                "decode_tokens": 8,
                "max_num_seqs": 9,
                "external_singleflight": True,
                "family_prefetch": True,
                "retained_fill": True,
                "selected_lead_ms": 100 if phase == "confirmation" else None,
                "model": f"/private/{PRIVATE}/model",
            }
            self.write(path / "CONFIG.json", config)
            self.write(path / "provenance.json", self.provenance)
            self.write(path / "runtime.json", self.runtime)
            blocks, rows = [], []
            for mode in ("staged", "direct"):
                name = f"{phase}-{mode}-n4-b00"
                cell = path / name
                cell.mkdir()
                block_config = {"mode": mode, "fanouts": [4], "block_id": name}
                for sample in (-3, 0):
                    row = fixture_row(block_config, sample, "early", 100)
                    row["private"] = {"request": PRIVATE, "path": f"/private/{PRIVATE}"}
                    row["memory"]["private"] = PRIVATE
                    row["integrity"] = PRIVATE
                    rows.append(row)
                if phase == "qualification":
                    unused = fixture_row(block_config, 0, "early", 250)
                    unused["rows"] = []
                    unused["private"] = PRIVATE
                    self.write(cell / "unused.json", unused)
                blocks.append(
                    {
                        "block_id": name,
                        "mode": mode,
                        "fanout": 4,
                        "process": {
                            "pid": 111,
                            "boot_id": PRIVATE,
                            "start_ticks": "222",
                        },
                        "child_continuations_checked": 8,
                        "warm_continuations_checked": 2,
                        "chosen_logprob_comparisons": 66,
                        "maximum_logprob_delta": 0,
                        "private": PRIVATE,
                    }
                )
            result = {
                "phase": phase,
                "accepted": True,
                "blocks": len(blocks),
                "verified_blocks": blocks,
                "overlap_joins_by_path": {"staged": 1, "direct": 1, "private": PRIVATE},
                "private": PRIVATE,
            }
            if phase == "pilot":
                result["selection"] = deepcopy(self.selection) | {
                    "private": PRIVATE,
                    "scope": PRIVATE,
                }
            elif phase == "confirmation":
                analysis = deepcopy(self.confirmation)
                analysis.update(private=PRIVATE, confidence_scope=PRIVATE)
                for cell in analysis["cells"].values():
                    cell["private"] = PRIVATE
                result["confirmation"] = analysis
            self.results[phase], self.rows[phase] = result, rows
            self.write(path / "RESULT.json", result)
        patcher = patch.object(
            publisher, "verify_capture", side_effect=self.verify_capture
        )
        self.verifier = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(publisher, "git_bytes", side_effect=self.git_bytes)
        self.git = patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def build(self):
        return publisher.build(
            self.root, harness_commit=COMMIT, repository=self.repository
        )

    def verify_capture(self, path):
        if not (path / "COMPLETE").exists():
            raise ValueError("partial capture")
        return deepcopy(self.results[path.name]), deepcopy(self.rows[path.name])

    def git_bytes(self, repository, *args):
        self.assertEqual(repository, self.repository)
        if args == ("cat-file", "-t", COMMIT):
            return b"commit\n"
        if args == ("ls-tree", "-r", "--name-only", COMMIT, "--", publisher.HARNESS):
            return (
                "\n".join(publisher.HARNESS + name for name in self.harness_files)
                + "\n"
            ).encode()
        if len(args) == 2 and args[0] == "show":
            prefix = f"{COMMIT}:{publisher.HARNESS}"
            self.assertTrue(args[1].startswith(prefix))
            name = args[1][len(prefix) :]
            if name == "model-files.json":
                return json.dumps(self.model_files).encode()
            return self.harness_files[name]
        self.fail(f"unexpected Git invocation: {args}")

    def change_all(self, filename, change):
        for phase in publisher.PHASES:
            path = self.root / phase / filename
            value = json.loads(path.read_text())
            change(value)
            self.write(path, value)

    def test_complete_export_is_whitelisted_and_preserves_scientific_scalars(self):
        result = self.build()
        serialized = json.dumps(result, allow_nan=False)
        self.assertNotIn(PRIVATE, serialized)
        for key in (
            "request_id",
            "declaration_identity",
            "boot_id",
            "pid",
            "block_id",
            "driver_version",
            "kernel",
            "operator",
        ):
            self.assertNotIn('"' + key + '"', serialized)
        self.assertNotIn("/private/", serialized)
        self.assertEqual(
            result["runtime"],
            {"torch": "2.10.0+cu128", "cuda": "12.8", "compute_capability": [8, 9]},
        )
        self.assertEqual(
            set(result["native_sha256_by_basename"]), {"libengine.so", "libstorage.so"}
        )
        self.assertEqual(
            result["sources"],
            {
                "knlp_harness_commit": COMMIT,
                "vllm": VLLM_COMMIT,
                "lmcache": RETAINED_DEMAND_COMMIT,
            },
        )
        self.assertEqual(result["model"]["files_sha256"], self.model_files)
        self.assertEqual(result["configuration"]["ancestor_bytes"], 48 << 20)
        self.assertEqual(result["configuration"]["engine_configured_bytes"], 512 << 20)
        self.assertEqual(
            result["configuration"]["external_configured_bytes"], 512 << 20
        )
        self.assertEqual(result["pilot"]["selection"]["selected_lead_ms"], 100)
        self.assertEqual(
            result["confirmation"]["confirmation"]["selected_lead_ms"], 100
        )
        self.assertEqual(len(result["samples"]), 12)
        row = result["samples"][0]
        self.assertEqual(row["backend_chunk_reads"], 16)
        self.assertEqual(row["backend_payload_bytes"], PAYLOAD)
        self.assertEqual(row["preparation"]["warm_request_count"], 1)
        self.assertEqual(row["preparation"]["warm_decode_tokens"], 1)
        self.assertEqual(row["preparation"]["engine_warm_ms"], 97)
        self.assertEqual(row["clock"]["a_deadline_first_token_ms"], 20)
        self.assertEqual(len(row["clock"]["all_request_deadline_first_token_ms"]), 4)
        self.assertEqual(row["memory"]["external_backed_bytes"], ARENA)
        self.assertEqual(len(result["unused_hint_controls"]), 2)
        self.assertTrue(
            all(
                r["unused_hints_charged"] == 1 and r["child_count"] == 0
                for r in result["unused_hint_controls"]
            )
        )
        self.assertEqual(
            result["validation"]["qualification"]["child_continuations_checked"], 16
        )
        self.assertEqual(
            result["validation"]["qualification"]["maximum_logprob_delta"], 0
        )
        self.assertEqual(
            self.verifier.call_args_list,
            [call(self.root / phase) for phase in publisher.PHASES],
        )
        for name in self.harness_files:
            self.assertIn(
                call(self.repository, "show", f"{COMMIT}:{publisher.HARNESS}{name}"),
                self.git.call_args_list,
            )
        self.assertTrue(result["limits"])

    def test_partial_or_failed_raw_phase_is_refused(self):
        for phase in publisher.PHASES:
            with self.subTest(phase=phase):
                complete = self.root / phase / "COMPLETE"
                complete.unlink()
                with self.assertRaisesRegex(ValueError, "partial capture"):
                    self.build()
                complete.touch()
        (self.root / "confirmation" / "RESULT.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete or unauthenticated"):
            self.build()

    def test_claimed_result_must_equal_recomputed_raw_result(self):
        path = self.root / "pilot" / "RESULT.json"
        result = json.loads(path.read_text())
        result["blocks"] += 1
        self.write(path, result)
        with self.assertRaisesRegex(ValueError, "claimed result differs"):
            self.build()

    def test_raw_provenance_mismatch_is_refused_even_for_nonpublic_field(self):
        path = self.root / "confirmation" / "provenance.json"
        value = json.loads(path.read_text())
        value["python"] = "different private runtime"
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "raw provenance changed"):
            self.build()

    def test_raw_runtime_mismatch_is_refused_even_for_nonpublic_field(self):
        path = self.root / "pilot" / "runtime.json"
        self.write(path, self.runtime | {"kernel": "changed-private-kernel"})
        with self.assertRaisesRegex(ValueError, "raw runtime changed"):
            self.build()

    def test_dirty_or_wrong_component_pin_is_refused(self):
        for component in ("vllm", "lmcache"):
            for mutation in ({"clean": False}, {"commit": "0" * 40}):
                with self.subTest(component=component, mutation=mutation):
                    self.change_all(
                        "provenance.json",
                        lambda p: p["sources"][component].update(mutation),
                    )
                    with self.assertRaisesRegex(ValueError, "component source differs"):
                        self.build()
                    self.change_all(
                        "provenance.json",
                        lambda p: p.update(
                            sources=deepcopy(self.provenance["sources"])
                        ),
                    )

    def test_missing_extra_or_changed_harness_hash_is_refused(self):
        for change in (
            lambda p: p["harness"].pop("prefetch_run.py"),
            lambda p: p["harness"].update({"extra.py": "0" * 64}),
            lambda p: p["harness"].update({"prefetch_worker.py": "0" * 64}),
        ):
            with self.subTest(change=change):
                self.change_all("provenance.json", change)
                with self.assertRaisesRegex(ValueError, "captured harness differs"):
                    self.build()
                self.change_all(
                    "provenance.json",
                    lambda p: p.update(harness=deepcopy(self.provenance["harness"])),
                )

    def test_git_source_bytes_are_authenticated_instead_of_local_files(self):
        self.harness_files["prefetch_worker.py"] += b"# different source\n"
        with self.assertRaisesRegex(ValueError, "captured harness differs"):
            self.build()

    def test_model_manifest_mismatch_is_refused(self):
        self.change_all(
            "provenance.json",
            lambda p: p["model_files"].update({"config.json": "0" * 64}),
        )
        with self.assertRaisesRegex(ValueError, "captured model hashes differ"):
            self.build()

    def test_native_basename_collision_or_invalid_digest_is_refused(self):
        self.change_all(
            "provenance.json",
            lambda p: p["native"].update({"/different/private/libengine.so": "d" * 64}),
        )
        with self.assertRaisesRegex(ValueError, "basenames are ambiguous"):
            self.build()
        self.change_all(
            "provenance.json",
            lambda p: p.update(native={"/private/libengine.so": "not-a-digest"}),
        )
        with self.assertRaisesRegex(ValueError, "invalid SHA256"):
            self.build()

    def test_full_commit_id_is_required_and_git_failure_is_refused(self):
        for commit in ("abc123", "A" * 40, "main", "/private/source"):
            with self.subTest(commit=commit):
                with self.assertRaisesRegex(ValueError, "full lowercase"):
                    publisher.build(
                        self.root, harness_commit=commit, repository=self.repository
                    )
        self.git.side_effect = subprocess.CalledProcessError(128, ["git"])
        with self.assertRaisesRegex(ValueError, "incomplete or unauthenticated"):
            self.build()

    def test_runtime_version_and_capability_cannot_carry_paths_or_strings(self):
        for field, value in (
            ("torch", f"/private/{PRIVATE}"),
            ("torch", f"2.10.0+{PRIVATE}"),
            ("cuda", None),
            ("cuda", f"12.8-{PRIVATE}"),
            ("capability", [PRIVATE, 9]),
        ):
            with self.subTest(field=field):
                self.change_all("runtime.json", lambda r: r.update({field: value}))
                with self.assertRaisesRegex(
                    ValueError, "invalid runtime|invalid compute"
                ):
                    self.build()
                self.change_all(
                    "runtime.json", lambda r: r.update(deepcopy(self.runtime))
                )

    def test_release_nightly_and_source_runtime_versions_remain_public(self):
        for version in (
            "2.10.0",
            "2.10.0+cu128",
            "2.11.0.dev20261020+cu130",
            "2.11.0a0+git1234abcd",
        ):
            with self.subTest(version=version):
                runtime = publisher.public_runtime(self.runtime | {"torch": version})
                self.assertEqual(runtime["torch"], version)

    def test_confirmation_cannot_change_selected_pilot_lead(self):
        path = self.root / "confirmation" / "CONFIG.json"
        config = json.loads(path.read_text()) | {"selected_lead_ms": 250}
        self.write(path, config)
        with self.assertRaisesRegex(ValueError, "frozen pilot lead"):
            self.build()

    def test_configuration_budget_and_nonfinite_sample_are_refused(self):
        path = self.root / "pilot" / "CONFIG.json"
        config = json.loads(path.read_text())
        self.write(path, config | {"arena_bytes": ARENA // 2})
        with self.assertRaisesRegex(ValueError, "fixed profile"):
            self.build()
        self.write(path, config)
        self.rows["pilot"][0]["memory"]["external_payload_byte_seconds_upper_bound"] = (
            float("nan")
        )
        with self.assertRaisesRegex(ValueError, "invalid public scalar"):
            self.build()


if __name__ == "__main__":
    unittest.main()
