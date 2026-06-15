from __future__ import annotations

import csv
from pathlib import Path

from ai_open_experiments.claims_report import run as run_claims_report
from ai_open_experiments.evidence_package import (
    build_evidence_package,
    claim_traceability,
    metric,
    read_csv,
    table_paired_tests,
)


RESULTS_DIR = Path("results/full_suite_2026-06-08")
EVIDENCE_DIR = RESULTS_DIR / "evidence_package"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_metric_matches_hand_checked_confusion_counts() -> None:
    precision, recall, f1 = metric([1, 1, 1, 0, 0], [1, 0, 1, 1, 0])
    assert round(precision, 4) == 0.6667
    assert round(recall, 4) == 0.6667
    assert round(f1, 4) == 0.6667


def test_paired_table_preserves_claimed_grove_advantage() -> None:
    stat_rows = read_csv(RESULTS_DIR / "statistical_comparisons.csv")
    paired = table_paired_tests(stat_rows)
    by_metric = {row["metric"]: row for row in paired}

    f1 = by_metric["f1"]
    iou = by_metric["iou_at_0_5"]
    assert f1["reference_system"] == "GROVE_full_paper_archived"
    assert f1["comparator_system"] == "baseline_direct_qwen35_27b"
    assert f1["paired_difference"].startswith("+")
    assert float(f1["p_value"]) < 0.001
    assert iou["interpretation"] == "GROVE higher"


def test_claim_traceability_artifacts_exist() -> None:
    claims = load_csv(EVIDENCE_DIR / "claim_traceability_matrix.csv")
    assert len(claims) >= 6
    artifact_names = {path.name for path in EVIDENCE_DIR.rglob("*") if path.is_file()}
    artifact_names.update({path.name for path in (EVIDENCE_DIR / "figures").glob("*")})

    for row in claims:
        assert row["claim"]
        assert row["traceable_result"]
        for artifact in [part.strip() for part in row["evidence_artifacts"].split(";")]:
            assert artifact in artifact_names


def test_not_run_rows_do_not_report_metrics() -> None:
    rows = load_csv(EVIDENCE_DIR / "table_1_component_ablation_completed.csv")
    not_run_rows = [row for row in rows if row["Status"] == "NOT_RUN"]
    assert not_run_rows
    for row in not_run_rows:
        assert row["Missing Reason"]
        assert row["Micro F1"] == ""
        assert row["95% CI"] == ""


def test_evidence_package_regenerates_to_tmpdir(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence_package"
    build_evidence_package(RESULTS_DIR, output_dir, n_bootstrap=25, seed=20260608)

    expected = {
        "table_1_full_vs_key_ablations.csv",
        "table_2_paired_tests.csv",
        "table_4_per_category_performance.csv",
        "claim_traceability_matrix.csv",
        "README.md",
        "captions.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    assert (output_dir / "figures" / "figure_1_modular_ablation_map.png").exists()


def test_claims_report_regenerates_to_tmpdir(tmp_path: Path) -> None:
    copied_results = tmp_path / "full_suite"
    copied_results.mkdir()
    for name in [
        "ablation_results_completed.csv",
        "grounding_sensitivity_multisystem",
    ]:
        source = RESULTS_DIR / name
        target = copied_results / name
        if source.is_dir():
            import shutil

            shutil.copytree(source, target)
        else:
            target.write_bytes(source.read_bytes())
    copied_evidence = copied_results / "evidence_package"
    copied_evidence.mkdir()
    for name in [
        "table_2_paired_tests.csv",
        "table_4_per_category_performance.csv",
        "compliance_ablation_nohazard_fp.csv",
        "no_other_ablation.csv",
    ]:
        (copied_evidence / name).write_bytes((EVIDENCE_DIR / name).read_bytes())

    run_claims_report(copied_results)
    output_dir = copied_evidence / "claims_analysis"
    assert (output_dir / "defensible_claims.csv").exists()
    assert (output_dir / "strongest_defensible_claims.html").exists()
    assert (output_dir / "source_notes.md").exists()


def test_claim_traceability_function_has_no_empty_claims() -> None:
    table1 = load_csv(EVIDENCE_DIR / "table_1_full_vs_key_ablations.csv")
    table2 = load_csv(EVIDENCE_DIR / "table_2_paired_tests.csv")
    table3 = load_csv(EVIDENCE_DIR / "table_3_threshold_sensitivity.csv")
    table4 = load_csv(EVIDENCE_DIR / "table_4_per_category_performance.csv")

    rows = claim_traceability(table1, table2, table4, table3)
    assert rows
    assert all(row["claim"] and row["evidence_artifacts"] and row["caveat"] for row in rows)
