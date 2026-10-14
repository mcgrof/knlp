"""Evaluate frozen cache-dependence controls before quantized retrieval."""

CRITERION = {
    "catalogues_per_length": 5,
    "queries_per_catalogue": 3,
    "min_query_exact_accuracy": 0.8,
    "min_changed_value_exact_accuracy": 0.8,
    "min_original_target_exact_accuracy": 0.8,
    "max_zero_catalogue_exact_accuracy": 0.2,
    "min_zero_catalogue_accuracy_drop": 0.6,
    "max_malformed_or_truncated": 0,
    "independent_cache_copies": True,
    "corruption_preserves_lengths_positions_masks": True,
}


def sensitivity_verdict(rows, criterion):
    groups = []
    for length in (2048, 8192):
        group = [row for row in rows if row["length"] == length]
        counts = {}
        for condition, expected in (
            ("query", 15),
            ("zero", 5),
            ("changed", 5),
            ("concatenated", 5),
        ):
            selected = [r for r in group if r["condition"] == condition]
            if len(selected) != expected:
                raise ValueError("incomplete workload sensitivity controls")
            counts[condition] = sum(r["result"]["correct"] for r in selected) / expected
        original = [
            r for r in group if r["condition"] == "query" and r["original_target"]
        ]
        original_accuracy = sum(r["result"]["correct"] for r in original) / 5
        bad_format = sum(
            r["result"]["malformed"] or r["result"]["truncated"]
            for r in group
            if r["condition"] != "zero"
        )
        passed = (
            counts["query"] >= criterion["min_query_exact_accuracy"]
            and counts["changed"] >= criterion["min_changed_value_exact_accuracy"]
            and original_accuracy >= criterion["min_original_target_exact_accuracy"]
            and counts["zero"] <= criterion["max_zero_catalogue_exact_accuracy"]
            and original_accuracy - counts["zero"]
            >= criterion["min_zero_catalogue_accuracy_drop"]
            and bad_format <= criterion["max_malformed_or_truncated"]
        )
        groups.append(
            {
                "length": length,
                "accuracy": counts,
                "original_accuracy": original_accuracy,
                "malformed_or_truncated": bad_format,
                "passed": passed,
            }
        )
    return {
        "status": "passed" if all(g["passed"] for g in groups) else "failed",
        "groups": groups,
    }
