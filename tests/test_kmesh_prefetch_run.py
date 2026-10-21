# SPDX-License-Identifier: MIT
"""Refuse changed admission evidence before any real device or process work."""

import builtins
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
import prefetch_run as runner


def provenance():
    return {
        "sources": {"vllm": {"commit": "v"}, "lmcache": {"commit": "l"}},
        "model_files": {"weights": "model-sha"},
        "harness": {"prefetch_run.py": "harness-sha"},
        "native": {"extension.so": "native-sha"},
        "python": "frozen-python",
        "gpu_provider": "vulkan_rm",
    }


def runtime():
    # This is the JSON representation of a prior admitted process, not an
    # in-memory copy of a tuple returned by the device API.
    return {
        "torch": "fixture-torch",
        "cuda": "fixture-cuda",
        "gpu": "fixture-device",
        "capability": [8, 9],
        "kernel": "fixture-kernel",
        "driver_version": "fixture-driver",
    }


def record(directory, phase, *, accepted=True, sources=None, device=None):
    directory.mkdir()
    result = {"phase": phase, "accepted": accepted}
    if phase == "pilot":
        result["selection"] = {"selected_lead_ms": 100}
    for name, value in (
        ("RESULT.json", result),
        ("provenance.json", sources if sources is not None else provenance()),
        ("runtime.json", device if device is not None else runtime()),
    ):
        (directory / name).write_text(json.dumps(value))
    return directory


class StopAfterAdmission(Exception):
    """The oracle boundary is reached without executing any child process."""


@contextmanager
def admission_case(phase="pilot", *, allow_runtime=False):
    """Provide only CPU stand-ins; fail if an early gate imports the runtime."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        args = SimpleNamespace(
            phase=phase,
            output=root / "output",
            model=root / "model",
            vllm_source=root / "vllm",
            lmcache_source=root / "lmcache",
            lmcache_commit="fixture-commit",
            gpu_provider="vulkan_rm",
            qualified=None,
            pilot=None,
        )
        fake = ModuleType("torch")
        fake.__version__ = "fixture-torch"
        fake.version = SimpleNamespace(cuda="fixture-cuda")
        fake.cuda = SimpleNamespace(
            get_device_name=Mock(return_value="fixture-device"),
            get_device_capability=Mock(return_value=(8, 9)),
        )
        actual_import = builtins.__import__
        imports = []

        def cpu_import(name, *positional, **keywords):
            if name == "torch":
                imports.append(name)
                if not allow_runtime:
                    raise AssertionError("device runtime imported before admission")
                return fake
            return actual_import(name, *positional, **keywords)

        read_text = Path.read_text

        def cpu_read(path, *positional, **keywords):
            if path == Path("/sys/module/nvidia/version"):
                return "fixture-driver\n"
            return read_text(path, *positional, **keywords)

        def verify_record(directory):
            return json.loads((directory / "RESULT.json").read_text()), []

        with (
            patch.dict(runner.os.environ),
            patch.object(runner, "freeze_sources", return_value=provenance()),
            patch.object(runner, "verify_capture", side_effect=verify_record) as verify,
            patch("builtins.__import__", side_effect=cpu_import),
            patch.object(Path, "read_text", new=cpu_read),
            patch.object(
                runner.os,
                "uname",
                return_value=SimpleNamespace(release="fixture-kernel"),
            ),
            patch.object(runner, "execute", side_effect=StopAfterAdmission) as execute,
            patch.object(
                runner, "launch", side_effect=AssertionError("unexpected child process")
            ) as launch,
            patch.object(
                runner, "serving", side_effect=AssertionError("unexpected serving")
            ) as serving,
        ):
            yield SimpleNamespace(
                root=root,
                args=args,
                runtime=fake,
                verify=verify,
                imports=imports,
                execute=execute,
                launch=launch,
                serving=serving,
            )


class SourceAdmissionTest(unittest.TestCase):
    def test_every_frozen_provenance_field_is_required_to_match(self):
        for key in provenance():
            with self.subTest(field=key):
                old, new = provenance(), provenance()
                new[key] = "different"
                with self.assertRaisesRegex(ValueError, key):
                    runner.compatible(old, new)

    def test_provider_manifest_hashes_the_actual_imported_binaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            here, model = root / "harness", root / "model"
            here.mkdir()
            model.mkdir()
            (model / "weights").write_bytes(b"pinned model")
            model_sha = hashlib.sha256(b"pinned model").hexdigest()
            (here / "model-files.json").write_text(json.dumps({"weights": model_sha}))
            (here / "prefetch_run.py").write_bytes(b"frozen harness")
            sources = [root / "vllm", root / "lmcache"]
            source_binary = sources[0] / "vllm" / "attention.so"
            source_binary.parent.mkdir(parents=True)
            source_binary.write_bytes(b"attention native")
            native = {}
            for name in ("lmcache_rust_raw_block_io", "lmcache._vulkan_rm"):
                path = root / (name + ".so")
                path.write_bytes(name.encode())
                native[name] = path
            args = SimpleNamespace(
                model=model,
                vllm_source=sources[0],
                lmcache_source=sources[1],
                lmcache_commit="expected-lmcache",
                gpu_provider="vulkan_rm",
            )

            def imported(name):
                return SimpleNamespace(__file__=str(native[name]))

            with (
                patch.object(runner, "HERE", here),
                patch.object(
                    runner, "source_identity", return_value={"clean": True}
                ) as identify,
                patch.object(
                    runner.importlib, "import_module", side_effect=imported
                ) as load,
            ):
                direct = runner.freeze_sources(args)
                self.assertEqual(
                    identify.call_args_list,
                    [
                        call(sources[0], runner.VLLM_COMMIT),
                        call(sources[1], "expected-lmcache"),
                    ],
                )
                self.assertEqual(load.call_args_list, [call(name) for name in native])
                for path in (source_binary, *native.values()):
                    self.assertEqual(direct["native"][str(path)], runner.sha256(path))
                args.gpu_provider = "native"
                load.reset_mock()
                staged = runner.freeze_sources(args)
                load.assert_called_once_with("lmcache_rust_raw_block_io")
                self.assertNotIn(str(native["lmcache._vulkan_rm"]), staged["native"])
                self.assertEqual(staged["gpu_provider"], "native")
                with self.assertRaisesRegex(ValueError, "native|gpu_provider"):
                    runner.compatible(direct, staged)
                args.gpu_provider = "vulkan_rm"
                native["lmcache._vulkan_rm"].write_bytes(b"rebuilt provider")
                rebuilt = runner.freeze_sources(args)
                with self.assertRaisesRegex(ValueError, "native"):
                    runner.compatible(direct, rebuilt)

                def missing_provider(name):
                    if name == "lmcache._vulkan_rm":
                        raise ImportError("required provider is absent")
                    return imported(name)

                load.side_effect = missing_provider
                with self.assertRaisesRegex(ImportError, "required provider is absent"):
                    runner.freeze_sources(args)
                (model / "weights").write_bytes(b"changed model")
                load.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "pinned model artifact"):
                    runner.freeze_sources(args)
                load.assert_not_called()

    def test_wrong_source_stops_before_native_import_or_model_access(self):
        args = SimpleNamespace(
            vllm_source=Path("v"), lmcache_source=Path("l"), lmcache_commit="l"
        )
        with (
            patch.object(
                runner,
                "source_identity",
                side_effect=RuntimeError("source identity mismatch"),
            ),
            patch.object(runner.importlib, "import_module") as load,
        ):
            with self.assertRaisesRegex(RuntimeError, "source identity mismatch"):
                runner.freeze_sources(args)
            load.assert_not_called()


class PhaseAdmissionTest(unittest.TestCase):
    def assert_no_work(self, case):
        case.execute.assert_not_called()
        case.launch.assert_not_called()
        case.serving.assert_not_called()

    def test_pilot_without_qualification_refuses_before_runtime_import(self):
        with admission_case() as case:
            with self.assertRaisesRegex(ValueError, "require a qualified capture"):
                runner.main(case.args)
            self.assertEqual(case.imports, [])
            self.assert_no_work(case)

    def test_unaccepted_or_wrong_phase_qualification_refuses_before_runtime(self):
        for phase, accepted in (("qualification", False), ("pilot", True)):
            with self.subTest(phase=phase, accepted=accepted), admission_case() as case:
                case.args.qualified = record(
                    case.root / "qualified", phase, accepted=accepted
                )
                with self.assertRaisesRegex(
                    ValueError, "qualification was not accepted"
                ):
                    runner.main(case.args)
                self.assertEqual(case.imports, [])
                self.assert_no_work(case)

    def test_changed_source_or_provider_qualification_refuses_before_runtime(self):
        for key in ("sources", "native", "gpu_provider"):
            with self.subTest(field=key), admission_case() as case:
                prior = provenance()
                prior[key] = "different"
                case.args.qualified = record(
                    case.root / "qualified", "qualification", sources=prior
                )
                with self.assertRaisesRegex(ValueError, key):
                    runner.main(case.args)
                self.assertEqual(case.imports, [])
                self.assert_no_work(case)

    def test_confirmation_requires_accepted_pilot_before_runtime(self):
        for pilot_state in (None, False, "wrong-phase"):
            with (
                self.subTest(pilot=pilot_state),
                admission_case("confirmation") as case,
            ):
                case.args.qualified = record(case.root / "qualified", "qualification")
                if pilot_state is not None:
                    case.args.pilot = record(
                        case.root / "pilot",
                        "qualification" if pilot_state == "wrong-phase" else "pilot",
                        accepted=pilot_state is not False,
                    )
                with self.assertRaisesRegex(
                    ValueError, "requires the frozen|pilot was not accepted"
                ):
                    runner.main(case.args)
                self.assertEqual(case.imports, [])
                self.assert_no_work(case)

    def test_confirmation_checks_pilot_provenance_independently(self):
        with admission_case("confirmation") as case:
            case.args.qualified = record(case.root / "qualified", "qualification")
            prior = provenance() | {"gpu_provider": "native"}
            case.args.pilot = record(case.root / "pilot", "pilot", sources=prior)
            with self.assertRaisesRegex(ValueError, "gpu_provider"):
                runner.main(case.args)
            self.assertEqual(case.imports, [])
            self.assert_no_work(case)

    def test_raw_admission_recomputation_mismatch_refuses_before_runtime(self):
        for phase, changed in (("pilot", "qualification"), ("confirmation", "pilot")):
            with self.subTest(phase=phase), admission_case(phase) as case:
                case.args.qualified = record(case.root / "qualified", "qualification")
                if phase == "confirmation":
                    case.args.pilot = record(case.root / "pilot", "pilot")

                def mismatched(directory):
                    result = json.loads((directory / "RESULT.json").read_text())
                    if result["phase"] == changed:
                        result["accepted"] = False
                    return result, []

                case.verify.side_effect = mismatched
                with self.assertRaisesRegex(ValueError, "raw offline verification"):
                    runner.main(case.args)
                self.assertEqual(case.imports, [])
                self.assert_no_work(case)

    def test_invalid_raw_qualification_refuses_before_runtime(self):
        with admission_case() as case:
            case.args.qualified = record(case.root / "qualified", "qualification")
            case.verify.side_effect = ValueError("raw capture is incomplete")
            with self.assertRaisesRegex(ValueError, "raw capture is incomplete"):
                runner.main(case.args)
            self.assertEqual(case.imports, [])
            self.assert_no_work(case)

    def test_json_roundtrip_of_matching_runtime_reaches_oracle_boundary(self):
        with admission_case(allow_runtime=True) as case:
            case.args.qualified = record(case.root / "qualified", "qualification")
            with self.assertRaises(StopAfterAdmission):
                runner.main(case.args)
            self.assertEqual(case.imports, ["torch"])
            case.runtime.cuda.get_device_capability.assert_called_once_with()
            case.execute.assert_called_once()
            case.launch.assert_not_called()
            case.serving.assert_not_called()
            self.assertEqual(
                json.loads((case.args.output / "runtime.json").read_text()), runtime()
            )

    def test_each_runtime_change_refuses_before_oracle_or_serving(self):
        for key in runtime():
            with self.subTest(field=key), admission_case(allow_runtime=True) as case:
                prior = runtime()
                prior[key] = [7, 5] if key == "capability" else "different"
                case.args.qualified = record(
                    case.root / "qualified", "qualification", device=prior
                )
                with self.assertRaisesRegex(
                    ValueError, "runtime, driver or device changed"
                ):
                    runner.main(case.args)
                self.assert_no_work(case)

    def test_matching_confirmation_keeps_the_frozen_pilot_selection(self):
        with admission_case("confirmation", allow_runtime=True) as case:
            case.args.qualified = record(case.root / "qualified", "qualification")
            case.args.pilot = record(case.root / "pilot", "pilot")
            with self.assertRaises(StopAfterAdmission):
                runner.main(case.args)
            config = json.loads((case.args.output / "CONFIG.json").read_text())
            self.assertEqual(config["selected_lead_ms"], 100)
            self.assertEqual(config["blocks_per_path_fanout"], 10)
            case.execute.assert_called_once()
            case.launch.assert_not_called()
            case.serving.assert_not_called()

    def test_confirmation_checks_both_admitted_runtimes(self):
        with admission_case("confirmation", allow_runtime=True) as case:
            case.args.qualified = record(case.root / "qualified", "qualification")
            prior = runtime() | {"driver_version": "different"}
            case.args.pilot = record(case.root / "pilot", "pilot", device=prior)
            with self.assertRaisesRegex(
                ValueError, "runtime, driver or device changed"
            ):
                runner.main(case.args)
            self.assert_no_work(case)


if __name__ == "__main__":
    unittest.main()
