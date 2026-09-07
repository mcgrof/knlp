"""Tests for the combined live UFO acceptance verdict."""

from rl.flight.run_ufo_acceptance import build_acceptance_verdict


def inputs():
    summary = {"frames": 20_000, "rejected": 0}
    analysis = {
        "sequence_gaps": 0,
        "monotonic_errors": 0,
        "requested_action_frames": 20_000,
        "applied_action_frames": 20_000,
        "simulator_total_wrench_frames": 20_000,
        "aerodynamic_wrench_frames": 20_000,
        "vehicle_mass_frames": 20_000,
        "vehicle_inertia_frames": 20_000,
    }
    response = {"mass_and_inertia_gate_passed": True}
    model_match = {"model_match_gate_passed": True}
    return summary, analysis, response, model_match


def test_complete_machine_evidence_passes_without_claiming_human_gates():
    verdict = build_acceptance_verdict(*inputs(), expected_frames=20_000)

    assert verdict["machine_gate_passed"]
    assert verdict["human_gates"] == {
        "handling_safe": None,
        "audio_clean_in_headset": None,
        "halo_visible_without_artifacts": None,
    }


def test_missing_diagnostics_fail_the_combined_gate():
    summary, analysis, response, model_match = inputs()
    analysis["aerodynamic_wrench_frames"] = 19_999

    verdict = build_acceptance_verdict(
        summary,
        analysis,
        response,
        model_match,
        expected_frames=20_000,
    )

    assert verdict["transport_gate_passed"]
    assert not verdict["diagnostic_coverage_gate_passed"]
    assert not verdict["machine_gate_passed"]


def test_target_mismatch_fails_the_combined_gate():
    summary, analysis, response, model_match = inputs()
    model_match["model_match_gate_passed"] = False

    verdict = build_acceptance_verdict(
        summary,
        analysis,
        response,
        model_match,
        expected_frames=20_000,
    )

    assert not verdict["target_model_gate_passed"]
    assert not verdict["machine_gate_passed"]
