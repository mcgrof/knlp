"""No-allocation checks for restart, selection, scope and convergence rules."""

import unittest

from .coordinator import (
    allowed_failure,
    choose_lrs,
    matching_runs,
    ordered_jobs,
)
from .analyze import target


class Controls(unittest.TestCase):
    def test_lr_failures_and_exact_ties(self):
        jobs = [
            {"id": str(lr), "variant": "V00", "lr": lr}
            for lr in (0.0003, 0.001, 0.003, 0.01)
        ]
        rs = {
            j["id"]: {
                "status": "complete",
                "validation": [{"step": s, "ce": 2} for s in (300, 400, 500)],
            }
            for j in jobs
        }
        rs["0.0003"] = {"status": "failed", "error": "nonfinite loss/output"}
        self.assertEqual(choose_lrs(jobs, rs)["V00"]["lr"], 0.001)
        rs["0.001"]["validation"] = [{"step": 300, "ce": 0}]
        self.assertEqual(choose_lrs(jobs, rs)["V00"]["lr"], 0.003)
        for r in rs.values():
            r["status"] = "failed"
        self.assertIsNone(choose_lrs(jobs, rs)["V00"]["lr"])

    def test_ambiguous_submission_never_duplicates(self):
        r = {
            "command": [
                "python",
                "-m",
                "research.mo_variants.job",
                "--job-id",
                "J",
            ]
        }
        self.assertEqual(matching_runs([r], "J"), [r])
        self.assertEqual(matching_runs([r], "other"), [])
        with self.assertRaises(AssertionError):
            matching_runs([r, r], "J")

    def test_only_lr_nonfinite_is_recoverable(self):
        for stage, expected in (
            ("A-pilot", True),
            ("A-discovery", False),
            ("A-qualification", False),
        ):
            self.assertEqual(
                allowed_failure(
                    {"stage": stage},
                    {"status": "failed", "error": "nonfinite gradient"},
                ),
                expected,
            )
        for error in (
            "disk watermark",
            "foreign GPU owners",
            "broken invariant",
            "nonfinite invariant",
        ):
            self.assertFalse(
                allowed_failure(
                    {"stage": "A-pilot"}, {"status": "failed", "error": error}
                )
            )

    def test_convergence_two_consecutive_evaluations(self):
        r = {
            "validation": [
                {
                    "step": s,
                    "ce": v,
                    "tokens": s * 2048,
                    "training_seconds": s / 2,
                    "end_to_end_seconds": s / 2 + 20,
                }
                for s, v in ((100, 2.9), (200, 3.1), (300, 2.9), (400, 2.8))
            ],
            "completed_step": 400,
            "training_seconds": 200,
            "end_to_end_seconds": 220,
            "optimizer_step_flops_estimate": 10,
        }
        self.assertEqual(target(r, 3)["step"], 400)
        self.assertFalse(target(r, 3)["censored"])
        self.assertTrue(target(r, 2)["censored"])
        r["validation"] = [r["validation"][0], r["validation"][3]]
        self.assertTrue(target(r, 3)["censored"])

    def test_scope_and_full_manifest(self):
        jobs = []
        for i in range(32):
            v = f"V{i:02d}"
            jobs.append({"id": f"q{v}", "stage": "A-qualification", "variant": v})
            jobs.extend(
                {"id": f"p{v}{lr}", "stage": "A-pilot", "variant": v, "lr": lr}
                for lr in (0.0003, 0.001, 0.003, 0.01)
            )
            jobs.extend(
                {
                    "id": f"d{v}{s}",
                    "stage": "A-discovery",
                    "variant": v,
                    "seed": s,
                }
                for s in (76311, 76312, 76313)
            )
            jobs.append({"id": f"b{v}", "stage": "B", "variant": v})
        q, p, d = ordered_jobs({"jobs": jobs})
        self.assertEqual((len(q), len(p), len(d)), (32, 128, 96))
        self.assertTrue(all(j["seed"] == 76311 for j in d[:32]))
        self.assertTrue(all(j["lr"] == 0.0003 for j in p[:32]))
        self.assertTrue(all(j["stage"].startswith("A-") for j in q + p + d))


if __name__ == "__main__":
    unittest.main()
