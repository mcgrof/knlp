# SPDX-License-Identifier: MIT
"""Check saved campaign admission independently of native block verification."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "tools/reproduce/kmesh_restore")
)
import prefetch_campaign as campaign
from prefetch_analysis import choose_lead, confirm
from prefetch_verify import CASES
from test_kmesh_prefetch_analysis import trial


class CampaignEvidenceTest(unittest.TestCase):
    """Native receipts are mocked here; their full checks have separate tests."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.receipts = {}
        patcher = patch.object(campaign, "verify_block", side_effect=self.verify_cell)
        self.verifier = patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def change_config(self, cell, **changes):
        path = cell / "config.json"
        self.write(path, json.loads(path.read_text()) | changes)

    def verify_cell(self, cell):
        return deepcopy(self.receipts[cell.name])

    def make(self, phase="qualification"):
        for path in self.root.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        self.receipts.clear()
        blocks = {"qualification": 1, "pilot": 2, "confirmation": 10}[phase]
        samples = {"qualification": 1, "pilot": 5, "confirmation": 20}[phase]
        fanouts = [4, 8] if phase == "confirmation" else [4]
        config = {
            "phase": phase,
            "model": "/frozen/model",
            "oracle": "../oracle/oracle.json",
            "arena_bytes": 512 << 20,
            "decode_tokens": 8,
            "external_singleflight": True,
            "family_prefetch": True,
            "retained_fill": True,
            "max_num_seqs": 9,
            "selected_lead_ms": 100 if phase == "confirmation" else None,
            "gpu_provider": "consumer",
            "blocks_per_path_fanout": blocks,
            "measured_groups_per_arm_block": samples,
            "warmups_per_arm_block": 3,
            "arrival_offset_ms": 250,
        }
        self.write(self.root / "CONFIG.json", config)
        for block in range(blocks):
            for mode in ("staged", "direct"):
                for fanout in fanouts:
                    name = f"{phase}-{mode}-n{fanout}-b{block:02d}"
                    cell = self.root / name
                    cell.mkdir()
                    self.write(
                        cell / "config.json",
                        config
                        | {
                            "mode": mode,
                            "fanouts": [fanout],
                            "block_index": block,
                            "block_id": name,
                            "serving_samples": samples,
                        },
                    )
                    cases = (
                        [("demand", 0), ("early", 100)]
                        if phase == "confirmation"
                        else CASES
                    )
                    rows = []
                    for sample in range(-3, samples):
                        for arm, lead in cases:
                            effect = 3 * (block + 1) if arm == "early" else 0
                            row = trial(
                                mode=mode,
                                fanout=fanout,
                                block=block,
                                sample=sample,
                                arm=arm,
                                lead=lead,
                                ready=(
                                    lead >= 100 or (lead == 25 and mode == "staged")
                                ),
                                first_ms=100 + 10 * block - effect,
                                group_ms=400 + 10 * block - 2 * effect,
                                phase=phase,
                            )
                            row["block_id"] = name
                            row["before"] = {
                                "prefetch_controller": {"overlap_joined_total": 0}
                            }
                            row["after"] = {
                                "prefetch_controller": {"overlap_joined_total": 1}
                            }
                            rows.append(row)
                    index = len(self.receipts)
                    self.receipts[name] = {
                        "block_id": name,
                        "phase": phase,
                        "mode": mode,
                        "fanout": fanout,
                        "process": {
                            "pid": 1000 + index,
                            "boot_id": "boot-a",
                            "start_ticks": str(100 + index),
                            "block_id": name,
                        },
                        "rows": rows,
                    }
        if phase == "qualification":
            for mode in ("staged", "direct"):
                self.write(
                    self.root / f"corruption-{mode}-rejected.json",
                    {"returncode": 1, "expected_payload_rejection": True},
                )
                (self.root / f"corruption-{mode}.log").write_text(
                    "ValueError: restored GPU consumer bytes differ\n"
                )
        self.verifier.reset_mock()
        return config

    def first_cell(self):
        return self.root / next(iter(self.receipts))

    def test_complete_qualification_retains_raw_rows_and_summarizes_both_paths(self):
        self.make()
        with (
            patch.object(campaign, "choose_lead") as selection,
            patch.object(campaign, "confirm") as confirmation,
        ):
            result, rows = campaign.verify_capture(self.root)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["phase"], "qualification")
        self.assertEqual(result["blocks"], 2)
        self.assertEqual(self.verifier.call_count, 2)
        self.assertEqual(len(rows), 2 * 4 * len(CASES))
        self.assertEqual({r["mode"] for r in rows}, {"staged", "direct"})
        self.assertEqual(result["overlap_joins_by_path"], {"staged": 32, "direct": 32})
        self.assertTrue(all("rows" not in b for b in result["verified_blocks"]))
        self.assertIn("rows", rows[0])
        selection.assert_not_called()
        confirmation.assert_not_called()

    def test_every_phase_rejects_missing_or_extra_process_block(self):
        for phase in ("qualification", "pilot", "confirmation"):
            for mutation in ("missing", "extra"):
                with self.subTest(phase=phase, mutation=mutation):
                    self.make(phase)
                    if mutation == "missing":
                        shutil.rmtree(self.first_cell())
                    else:
                        (self.root / f"{phase}-direct-n4-b99").mkdir()
                    with self.assertRaisesRegex(ValueError, "missing or extra"):
                        campaign.verify_capture(self.root)
                    self.verifier.assert_not_called()

    def test_process_incarnation_reuse_rejected_even_with_distinct_block_ids(self):
        self.make()
        first, second = self.receipts.values()
        second["process"].update(
            {k: first["process"][k] for k in ("boot_id", "pid", "start_ticks")}
        )
        with self.assertRaisesRegex(ValueError, "incarnation reused"):
            campaign.verify_capture(self.root)

    def test_pid_reuse_with_new_start_or_boot_is_a_distinct_incarnation(self):
        for changed in ("start_ticks", "boot_id"):
            with self.subTest(changed=changed):
                self.make()
                first, second = self.receipts.values()
                second["process"].update(
                    {k: first["process"][k] for k in ("boot_id", "pid", "start_ticks")}
                )
                second["process"][changed] += "1"
                campaign.verify_capture(self.root)

    def test_campaign_and_both_trial_failure_receipt_spellings_rejected(self):
        for receipt in (
            "FAILURE.json",
            "failed-s0-demand-g0-u0.json",
            "s0-demand-failed.json",
        ):
            with self.subTest(receipt=receipt):
                self.make()
                parent = self.root if receipt == "FAILURE.json" else self.first_cell()
                self.write(parent / receipt, {"error": "trial failed"})
                with self.assertRaisesRegex(ValueError, "failed trial"):
                    campaign.verify_capture(self.root)
                self.verifier.assert_not_called()

    def test_frozen_sampling_plan_cannot_change(self):
        for key, value in (
            ("blocks_per_path_fanout", 2),
            ("measured_groups_per_arm_block", 2),
            ("warmups_per_arm_block", 0),
            ("arrival_offset_ms", 251),
        ):
            with self.subTest(key=key):
                config = self.make()
                self.write(self.root / "CONFIG.json", config | {key: value})
                with self.assertRaisesRegex(ValueError, "frozen sampling plan"):
                    campaign.verify_capture(self.root)
                self.verifier.assert_not_called()

    def test_each_block_must_match_declared_position_and_sampling(self):
        for key, value in (
            ("mode", "other"),
            ("fanouts", [8]),
            ("block_id", "other"),
            ("block_index", 1),
            ("phase", "pilot"),
            ("serving_samples", 5),
        ):
            with self.subTest(key=key):
                self.make()
                self.change_config(self.first_cell(), **{key: value})
                with self.assertRaisesRegex(ValueError, "declared campaign"):
                    campaign.verify_capture(self.root)

    def test_each_invariant_configuration_field_is_pinned(self):
        for key, value in (
            ("model", "/other/model"),
            ("arena_bytes", 256 << 20),
            ("decode_tokens", 9),
            ("external_singleflight", False),
            ("family_prefetch", False),
            ("retained_fill", False),
            ("max_num_seqs", 10),
            ("selected_lead_ms", 250),
            ("gpu_provider", "other"),
        ):
            with self.subTest(key=key):
                self.make()
                self.change_config(self.first_cell(), **{key: value})
                with self.assertRaisesRegex(
                    ValueError, "configuration changed: " + key
                ):
                    campaign.verify_capture(self.root)

    def test_qualification_requires_overlap_in_each_path(self):
        for mode in ("staged", "direct"):
            with self.subTest(mode=mode):
                self.make()
                for receipt in self.receipts.values():
                    if receipt["mode"] == mode:
                        for row in receipt["rows"]:
                            row["after"]["prefetch_controller"][
                                "overlap_joined_total"
                            ] = 0
                with self.assertRaisesRegex(ValueError, "never joined"):
                    campaign.verify_capture(self.root)

    def test_native_corruption_rejection_is_required_in_each_path(self):
        for mode in ("staged", "direct"):
            for mutation in ("exit", "flag", "log"):
                with self.subTest(mode=mode, mutation=mutation):
                    self.make()
                    path = self.root / f"corruption-{mode}-rejected.json"
                    receipt = json.loads(path.read_text())
                    if mutation == "exit":
                        receipt["returncode"] = 0
                    elif mutation == "flag":
                        receipt["expected_payload_rejection"] = False
                    else:
                        (self.root / f"corruption-{mode}.log").write_text(
                            "unrelated failure"
                        )
                    self.write(path, receipt)
                    with self.assertRaisesRegex(ValueError, "deliberate-corruption"):
                        campaign.verify_capture(self.root)

    def test_native_block_failure_propagates_before_analysis(self):
        self.make("pilot")
        self.verifier.side_effect = ValueError("native route or drain failed")
        with patch.object(campaign, "choose_lead") as selection:
            with self.assertRaisesRegex(ValueError, "native route or drain"):
                campaign.verify_capture(self.root)
        selection.assert_not_called()

    def test_pilot_invokes_real_readiness_selection_on_all_process_rows(self):
        self.make("pilot")
        with (
            patch.object(campaign, "choose_lead", wraps=choose_lead) as selection,
            patch.object(campaign, "confirm") as confirmation,
        ):
            result, rows = campaign.verify_capture(self.root)
        self.assertEqual(result["blocks"], 4)
        self.assertEqual(self.verifier.call_count, 4)
        self.assertEqual(result["selection"]["selected_lead_ms"], 100)
        self.assertTrue(result["selection"]["readiness_rule_satisfied"])
        selection.assert_called_once_with(rows)
        confirmation.assert_not_called()
        self.assertEqual(len(rows), 4 * 8 * len(CASES))

    def test_confirmation_invokes_real_cluster_analysis_with_frozen_lead(self):
        self.make("confirmation")
        with (
            patch.object(campaign, "choose_lead") as selection,
            patch.object(campaign, "confirm", wraps=confirm) as confirmation,
        ):
            result, rows = campaign.verify_capture(self.root)
        self.assertEqual(result["blocks"], 40)
        self.assertEqual(self.verifier.call_count, 40)
        confirmation.assert_called_once_with(rows, selected_lead_ms=100)
        selection.assert_not_called()
        analysis = result["confirmation"]
        self.assertEqual(analysis["selected_lead_ms"], 100)
        self.assertEqual(
            set(analysis["cells"]), {"staged-n4", "direct-n4", "staged-n8", "direct-n8"}
        )
        for cell in analysis["cells"].values():
            self.assertEqual(cell["process_blocks"], 10)
            self.assertEqual(cell["measured_groups_per_arm"], 200)
            self.assertTrue(cell["engineering_rule_satisfied"])

    def test_request_export_contains_every_child_and_warmup_but_no_warm_request(self):
        self.make()
        _, rows = campaign.verify_capture(self.root)
        path = self.root / "requests.csv"
        campaign.export_csv(path, rows)
        with path.open(newline="") as stream:
            exported = list(csv.DictReader(stream))
        expected_ids = {child["request_id"] for row in rows for child in row["rows"]}
        warm_ids = {warm["request_id"] for row in rows for warm in row["warm_rows"]}
        self.assertEqual(len(exported), len(rows) * 4)
        self.assertEqual({r["request_id"] for r in exported}, expected_ids)
        self.assertFalse(expected_ids & warm_ids)
        self.assertEqual({r["warmup"] for r in exported}, {"True", "False"})
        self.assertEqual({r["mode"] for r in exported}, {"staged", "direct"})
        self.assertEqual(float(exported[0]["deadline_first_token_ms"]), 100)
        self.assertEqual(float(exported[0]["ttft_ms"]), 3)

    def test_export_rechecks_fixed_arrival_before_writing_children(self):
        self.make()
        _, rows = campaign.verify_capture(self.root)
        rows[0]["scheduled_arrival_ns"] += 1
        with self.assertRaisesRegex(ValueError, "fixed origin"):
            campaign.export_csv(self.root / "requests.csv", rows)


if __name__ == "__main__":
    unittest.main()
