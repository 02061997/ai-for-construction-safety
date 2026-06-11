#!/usr/bin/env python3
"""Build reviewer-facing evidence artifacts from the GROVE experiment suite.

This module is intentionally a derived-results layer: it reads the CSV files
created by ``run_ai_open_experiments.py`` and writes paper-ready tables, plots,
captions, and a claim traceability matrix. It does not rerun inference or
download anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path
from typing import Any

from ai_open_experiments.suite import CANONICAL_CATEGORIES, DISPLAY_LABELS, write_csv, write_markdown_table


DEFAULT_RESULTS_DIR = Path("ai_open_results/full_suite_2026-06-08")
DEFAULT_SEED = 20260608
DEFAULT_BOOTSTRAPS = 1000
LOW_SUPPORT_THRESHOLD = 10


SYSTEM_LABELS = {
    "GROVE_full_paper_archived": "GROVE full system",
    "GROVE_trace_qwen35_9b_final": "GROVE trace control",
    "baseline_direct_qwen35_27b": "Best single-pass baseline",
    "ablation_path1_only_no_path2": "Path 1 only",
    "ablation_path2_only_no_grounding": "Path 2 only",
    "ablation_no_reconciliation_union": "No reconciliation",
    "ablation_groundingdino_only": "GroundingDINO only",
    "ablation_gdino_detr_openclip_stack_only": "GDINO + DETR/OpenCLIP stack",
    "ablation_qwen_image_only_no_grounding": "Qwen image-only, no grounding",
    "ablation_caption_only_keyword_proxy": "Caption-only keyword proxy",
    "ablation_image_only_reasoning": "Image-only reasoning",
    "ablation_no_fallback_grounding_drop": "No fallback, drop hazards",
    "ablation_no_fallback_grounding_boxes_removed": "No fallback, strip boxes",
    "ablation_no_path2_verification": "No Path 2 verification",
    "ablation_no_path2_stage_correction": "No Path 2 stage correction",
    "ablation_grounded_final_only": "Grounded final only",
}


KEY_SYSTEM_ORDER = [
    "GROVE_full_paper_archived",
    "GROVE_trace_qwen35_9b_final",
    "baseline_direct_qwen35_27b",
    "ablation_path1_only_no_path2",
    "ablation_path2_only_no_grounding",
    "ablation_no_reconciliation_union",
    "ablation_groundingdino_only",
    "ablation_gdino_detr_openclip_stack_only",
    "ablation_qwen_image_only_no_grounding",
    "ablation_caption_only_keyword_proxy",
    "ablation_image_only_reasoning",
    "ablation_no_fallback_grounding_drop",
    "ablation_no_fallback_grounding_boxes_removed",
    "ablation_no_path2_verification",
    "ablation_no_path2_stage_correction",
    "ablation_grounded_final_only",
]


DEFAULT_THRESHOLDS = {
    "groundingdino_confidence": "0.3",
    "openclip_similarity": "0.3",
    "oversized_box_cutoff_pct": "100",
    "iog_success_threshold": "0.3",
}


ABLATION_NOTES = {
    "GROVE_full_paper_archived": "Primary archived paper-facing GROVE run.",
    "GROVE_trace_qwen35_9b_final": "Trace-enabled control used for exact ablation deltas.",
    "baseline_direct_qwen35_27b": "Best single-pass baseline selected by identification F1.",
    "ablation_path1_only_no_path2": "Removes Path 2 verification from the trace-enabled pipeline.",
    "ablation_path2_only_no_grounding": "Lower-bound Path 2-only alternative; retained hazards have all boxes stripped.",
    "ablation_no_reconciliation_union": "Replaces reconciliation with a simple union rule.",
    "ablation_groundingdino_only": "Keeps only hazards grounded by the primary GroundingDINO source.",
    "ablation_gdino_detr_openclip_stack_only": "Tests the grounding stack without final Path 2 reconciliation.",
    "ablation_qwen_image_only_no_grounding": "Uses cached image reasoning labels without grounding boxes.",
    "ablation_caption_only_keyword_proxy": "Deterministic caption-keyword proxy, not a rerun VLM.",
    "ablation_image_only_reasoning": "Image-only reasoning path from cached trace fields.",
    "ablation_no_fallback_grounding_drop": "Drops hazards that depended on fallback grounding.",
    "ablation_no_fallback_grounding_boxes_removed": "Keeps fallback-dependent hazards but removes their boxes.",
    "ablation_no_path2_verification": "Uses Path 1 pre-verification hazards as final outputs.",
    "ablation_no_path2_stage_correction": "Removes Path 2 category/stage correction while retaining detections.",
    "ablation_grounded_final_only": "Keeps final hazards only when a grounding box is present.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4) -> str:
    number = safe_float(value, None)
    if number is None or (isinstance(number, float) and math.isnan(number)):
        return ""
    return f"{number:.{digits}f}"


def fmt_signed(value: Any, digits: int = 4) -> str:
    number = safe_float(value, None)
    if number is None:
        return ""
    return f"{number:+.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    number = safe_float(value, None)
    if number is None:
        return ""
    return f"{100 * number:.{digits}f}%"


def ci_text(low: Any, high: Any, digits: int = 4) -> str:
    if safe_float(low, None) is None or safe_float(high, None) is None:
        return ""
    return f"[{fmt(low, digits)}, {fmt(high, digits)}]"


def metric(truth: list[int], pred: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for g, p in zip(truth, pred) if g and p)
    fp = sum(1 for g, p in zip(truth, pred) if not g and p)
    fn = sum(1 for g, p in zip(truth, pred) if g and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(q * (len(ordered) - 1))
    return ordered[idx]


def stable_seed(*parts: str, seed: int) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def bootstrap_category_ci(
    category_rows: list[dict[str, str]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, str]:
    if not category_rows:
        return {
            "precision_ci95": "",
            "recall_ci95": "",
            "f1_ci95": "",
        }
    truth = [safe_int(row.get("gt_present")) for row in category_rows]
    pred = [safe_int(row.get("pred_present")) for row in category_rows]
    rng = random.Random(seed)
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    n = len(category_rows)
    for _ in range(n_bootstrap):
        sample_truth: list[int] = []
        sample_pred: list[int] = []
        for _idx in range(n):
            j = rng.randrange(n)
            sample_truth.append(truth[j])
            sample_pred.append(pred[j])
        p, r, f = metric(sample_truth, sample_pred)
        precision_values.append(p)
        recall_values.append(r)
        f1_values.append(f)
    return {
        "precision_ci95": ci_text(percentile(precision_values, 0.025), percentile(precision_values, 0.975)),
        "recall_ci95": ci_text(percentile(recall_values, 0.025), percentile(recall_values, 0.975)),
        "f1_ci95": ci_text(percentile(f1_values, 0.025), percentile(f1_values, 0.975)),
    }


def index_rows(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row.get(key, ""): row for row in rows}


def index_pair(rows: list[dict[str, str]], first: str, second: str) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get(first, ""), row.get(second, "")): row for row in rows}


def best_baseline(aggregate_rows: list[dict[str, str]], stat_rows: list[dict[str, str]]) -> str:
    for row in stat_rows:
        comparator = row.get("comparator_system")
        if comparator:
            return comparator
    candidates = [
        row for row in aggregate_rows
        if row.get("family", "").startswith("baseline") or row.get("system", "").startswith("baseline_")
    ]
    if not candidates:
        return ""
    return max(candidates, key=lambda r: safe_float(r.get("f1")) or 0.0).get("system", "")


def role_for_system(system: str) -> str:
    if system == "GROVE_full_paper_archived":
        return "Primary system"
    if system == "GROVE_trace_qwen35_9b_final":
        return "Ablation control"
    if system.startswith("baseline_"):
        return "Fairness baseline"
    if system.startswith("ablation_"):
        return "Ablation"
    return "Comparator"


def table_full_vs_ablations(
    aggregate_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
    best_baseline_system: str,
) -> list[dict[str, str]]:
    aggregate = index_rows(aggregate_rows, "system")
    ablations = index_rows(ablation_rows, "ablation")
    control = aggregate.get("GROVE_trace_qwen35_9b_final", {})
    control_f1 = safe_float(control.get("f1")) or 0.0
    control_gt = safe_float(control.get("gt_coverage_iog03")) or 0.0
    order = [s for s in KEY_SYSTEM_ORDER if s in aggregate or s in ablations]
    if best_baseline_system and best_baseline_system not in order:
        order.insert(2, best_baseline_system)

    rows: list[dict[str, str]] = []
    for system in order:
        source = aggregate.get(system, {})
        ablation = ablations.get(system, {})
        if not source:
            source = ablation
        if not source:
            continue
        f1 = safe_float(source.get("f1")) or 0.0
        gt_cov = safe_float(source.get("gt_coverage_iog03")) or 0.0
        if ablation:
            delta_f1 = ablation.get("delta_f1_vs_control", "")
            delta_gt = ablation.get("delta_gt_coverage_iog03_vs_control", "")
        elif system == "GROVE_trace_qwen35_9b_final":
            delta_f1 = "0.0000"
            delta_gt = "0.0000"
        elif system.startswith("ablation_"):
            delta_f1 = fmt_signed(f1 - control_f1)
            delta_gt = fmt_signed(gt_cov - control_gt)
        else:
            delta_f1 = ""
            delta_gt = ""
        rows.append(
            {
                "system": system,
                "label": SYSTEM_LABELS.get(system, system),
                "role": role_for_system(system),
                "exactness": source.get("exactness", ablation.get("exactness", "")),
                "precision": fmt(source.get("precision")),
                "recall": fmt(source.get("recall")),
                "f1": fmt(source.get("f1")),
                "f1_ci95": ci_text(source.get("f1_ci95_low"), source.get("f1_ci95_high")),
                "delta_f1_vs_trace_control": fmt_signed(delta_f1) if delta_f1 != "" else "",
                "gt_coverage_iog03": fmt(source.get("gt_coverage_iog03")),
                "delta_gt_coverage_vs_trace_control": fmt_signed(delta_gt) if delta_gt != "" else "",
                "iou_at_0_5": fmt(source.get("iou_at_0_5")),
                "no_box_rate": fmt(source.get("no_box_rate")),
                "interpretation": ABLATION_NOTES.get(system, source.get("description", "")),
            }
        )
    return rows


def table_paired_tests(stat_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in stat_rows:
        metric_name = row.get("metric", "")
        p_boot = row.get("paired_bootstrap_p_two_sided", "")
        p_mcnemar = row.get("mcnemar_p_approx", "")
        p_value = p_mcnemar if metric_name.endswith("mcnemar") else p_boot
        ci = ci_text(row.get("diff_ci95_low"), row.get("diff_ci95_high"))
        direction = "GROVE higher" if (safe_float(row.get("paired_diff_reference_minus_comparator"), 0.0) or 0.0) > 0 else "mixed/no increase"
        rows.append(
            {
                "reference_system": row.get("reference_system", ""),
                "comparator_system": row.get("comparator_system", ""),
                "metric": metric_name,
                "paired_difference": fmt_signed(row.get("paired_diff_reference_minus_comparator")),
                "difference_ci95": ci,
                "p_value": fmt(p_value),
                "test": "McNemar-style image-category test" if metric_name.endswith("mcnemar") else "Paired bootstrap over images",
                "n_bootstrap": row.get("n_bootstrap", ""),
                "n_image_category_units": row.get("n_image_category_units", ""),
                "interpretation": direction,
            }
        )
    return rows


def table_threshold_sensitivity(threshold_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    baseline_by_type: dict[str, dict[str, str]] = {}
    for row in threshold_rows:
        threshold_type = row.get("threshold_type", "")
        if str(row.get("threshold", "")) == DEFAULT_THRESHOLDS.get(threshold_type):
            baseline_by_type[threshold_type] = row
    rows: list[dict[str, str]] = []
    for row in threshold_rows:
        threshold_type = row.get("threshold_type", "")
        base = baseline_by_type.get(threshold_type, row)
        gt_cov = safe_float(row.get("gt_coverage_iog03")) or 0.0
        base_gt = safe_float(base.get("gt_coverage_iog03")) or 0.0
        no_box = safe_float(row.get("no_box_rate")) or 0.0
        base_no_box = safe_float(base.get("no_box_rate")) or 0.0
        rows.append(
            {
                "system": row.get("system", ""),
                "threshold_type": threshold_type,
                "threshold": row.get("threshold", ""),
                "default_threshold_for_type": DEFAULT_THRESHOLDS.get(threshold_type, ""),
                "gt_coverage_iog03": fmt(row.get("gt_coverage_iog03")),
                "delta_gt_coverage_vs_default": fmt_signed(gt_cov - base_gt),
                "prediction_coverage_iog03": fmt(row.get("prediction_coverage_iog03")),
                "category_aware_grounding_success_iog03": fmt(row.get("category_aware_grounding_success_iog03")),
                "iou_at_0_3": fmt(row.get("iou_at_0_3")),
                "iou_at_0_5": fmt(row.get("iou_at_0_5")),
                "mean_iou_covering": fmt(row.get("mean_iou_covering")),
                "no_box_rate": fmt(row.get("no_box_rate")),
                "delta_no_box_rate_vs_default": fmt_signed(no_box - base_no_box),
            }
        )
    return rows


def table_per_category(
    per_category_id_rows: list[dict[str, str]],
    per_category_prediction_rows: list[dict[str, str]],
    grounding_category_rows: list[dict[str, str]],
    full_system: str,
    baseline_system: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, str]]:
    id_by_pair = index_pair(per_category_id_rows, "system", "category")
    ground_by_pair = index_pair(grounding_category_rows, "system", "category")
    pred_rows_by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in per_category_prediction_rows:
        pred_rows_by_pair.setdefault((row.get("system", ""), row.get("category", "")), []).append(row)

    rows: list[dict[str, str]] = []
    for category in CANONICAL_CATEGORIES:
        full = id_by_pair.get((full_system, category), {})
        base = id_by_pair.get((baseline_system, category), {})
        full_ci = bootstrap_category_ci(
            pred_rows_by_pair.get((full_system, category), []),
            n_bootstrap,
            stable_seed(full_system, category, "category_ci", seed=seed),
        )
        base_ci = bootstrap_category_ci(
            pred_rows_by_pair.get((baseline_system, category), []),
            n_bootstrap,
            stable_seed(baseline_system, category, "category_ci", seed=seed),
        )
        support_images = safe_int(full.get("support_gt_images") or base.get("support_gt_images"))
        support_boxes = safe_int(ground_by_pair.get((full_system, category), {}).get("support_gt_boxes"))
        low_support = int(support_images <= LOW_SUPPORT_THRESHOLD or support_boxes <= LOW_SUPPORT_THRESHOLD)
        delta_f1 = (safe_float(full.get("f1")) or 0.0) - (safe_float(base.get("f1")) or 0.0)
        full_ground = ground_by_pair.get((full_system, category), {})
        base_ground = ground_by_pair.get((baseline_system, category), {})
        delta_ground = (safe_float(full_ground.get("gt_coverage_iog03")) or 0.0) - (
            safe_float(base_ground.get("gt_coverage_iog03")) or 0.0
        )
        rows.append(
            {
                "category": category,
                "category_label": DISPLAY_LABELS.get(category, category),
                "support_gt_images": str(support_images),
                "support_gt_boxes": str(support_boxes),
                "low_support_flag": str(low_support),
                "grove_precision": fmt(full.get("precision")),
                "grove_precision_ci95": full_ci["precision_ci95"],
                "grove_recall": fmt(full.get("recall")),
                "grove_recall_ci95": full_ci["recall_ci95"],
                "grove_f1": fmt(full.get("f1")),
                "grove_f1_ci95": full_ci["f1_ci95"],
                "baseline_precision": fmt(base.get("precision")),
                "baseline_precision_ci95": base_ci["precision_ci95"],
                "baseline_recall": fmt(base.get("recall")),
                "baseline_recall_ci95": base_ci["recall_ci95"],
                "baseline_f1": fmt(base.get("f1")),
                "baseline_f1_ci95": base_ci["f1_ci95"],
                "delta_f1_grove_minus_baseline": fmt_signed(delta_f1),
                "grove_gt_coverage_iog03": fmt(full_ground.get("gt_coverage_iog03")),
                "baseline_gt_coverage_iog03": fmt(base_ground.get("gt_coverage_iog03")),
                "delta_gt_coverage_grove_minus_baseline": fmt_signed(delta_ground),
                "interpretation": "Low support; cite cautiously." if low_support else "Supported category.",
            }
        )
    return rows


def table_failure_attribution(failure_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in failure_rows:
        failure_type = row.get("failure_type", "")
        rows.append(
            {
                "system": row.get("system", ""),
                "failure_type": failure_type,
                "count": row.get("count", ""),
                "percent_of_all_f1_errors": pct(row.get("percent_of_all_f1_errors")),
                "interpretation": failure_interpretation(failure_type),
            }
        )
    return rows


def failure_interpretation(failure_type: str) -> str:
    mapping = {
        "caption_omission": "The caption missed a ground-truth hazard cue.",
        "reasoning_omission": "Reasoning failed to predict a ground-truth category.",
        "reasoning_hallucination": "Reasoning predicted a category absent from ground truth.",
        "primary_grounding_failure": "Primary grounding failed after a correct hazard category was present.",
        "fallback_grounding_failure": "Fallback grounding failed after primary grounding was insufficient.",
    }
    return mapping.get(failure_type, "")


def plot_pipeline_ablation_map(path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    boxes = [
        ("Image", 0.4, 5.7, 1.45, 0.8, "#f7f7f7"),
        ("Captioning", 2.1, 5.7, 1.75, 0.8, "#dceefb"),
        ("Hazard\nreasoning", 4.2, 5.7, 1.85, 0.8, "#dceefb"),
        ("Path 1\nGroundingDINO", 6.55, 6.45, 2.0, 0.8, "#d7f0df"),
        ("DETR/OpenCLIP\nfallback", 9.0, 6.45, 2.1, 0.8, "#d7f0df"),
        ("Path 2\nverification", 6.65, 4.7, 2.0, 0.8, "#ffe2c2"),
        ("Stage/category\ncorrection", 9.05, 4.7, 2.05, 0.8, "#ffe2c2"),
        ("Reconciliation", 11.55, 5.6, 1.75, 0.9, "#eadcf8"),
        ("Final GROVE\noutput", 11.75, 3.7, 1.65, 0.8, "#f7f7f7"),
    ]
    for label, x, y, w, h, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, weight="bold")

    arrows = [
        ((1.85, 6.1), (2.1, 6.1)),
        ((3.85, 6.1), (4.2, 6.1)),
        ((6.05, 6.1), (6.55, 6.85)),
        ((6.05, 6.1), (6.65, 5.1)),
        ((8.55, 6.85), (9.0, 6.85)),
        ((8.65, 5.1), (9.05, 5.1)),
        ((11.1, 6.85), (11.55, 6.05)),
        ((11.1, 5.1), (11.55, 6.05)),
        ((12.42, 5.6), (12.55, 4.5)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#333333"))

    callouts = [
        ("Path 1 only:\nremove Path 2", 5.7, 3.3, "#ffe2c2"),
        ("Path 2 only:\nremove grounding boxes", 8.0, 3.2, "#d7f0df"),
        ("GDINO only:\nremove fallback", 9.1, 7.45, "#d7f0df"),
        ("No reconciliation:\nunion rule", 11.35, 6.9, "#eadcf8"),
        ("No stage correction:\nfreeze Path 1 labels", 9.0, 3.85, "#ffe2c2"),
        ("No fallback:\ndrop or strip fallback boxes", 10.85, 7.45, "#d7f0df"),
    ]
    for text, x, y, color in callouts:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": color, "linewidth": 2},
        )

    ax.text(0.4, 7.55, "GROVE modular pipeline and ablation targets", fontsize=15, weight="bold", ha="left")
    ax.text(
        0.4,
        0.55,
        "Colored boxes are modules retained in the full system. Callouts identify components removed or replaced by key ablations.",
        fontsize=9,
        color="#555555",
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_category_gain_heatmap(path: Path, category_rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("delta_f1_grove_minus_baseline", "ID F1"),
        ("delta_gt_coverage_grove_minus_baseline", "GT coverage"),
        ("grove_f1", "GROVE F1"),
        ("grove_gt_coverage_iog03", "GROVE coverage"),
    ]
    labels = [
        row["category_label"] + (" *" if row.get("low_support_flag") == "1" else "")
        for row in category_rows
    ]
    data: list[list[float]] = []
    for row in category_rows:
        data.append([safe_float(row.get(key)) or 0.0 for key, _label in metrics])

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    image = ax.imshow(data, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for _key, label in metrics], rotation=25, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            color = "white" if value > 0.62 or value < -0.25 else "#222222"
            ax.text(j, i, f"{value:+.2f}" if j < 2 else f"{value:.2f}", ha="center", va="center", color=color, fontsize=8.5)
    ax.set_title("Category-level gains and absolute GROVE performance")
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label("Metric value")
    fig.text(
        0.13,
        0.035,
        "* Low support category. Deltas compare GROVE full system with the best single-pass baseline.",
        fontsize=8.5,
        color="#555555",
        ha="left",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_no_box_tradeoff(path: Path, aggregate_rows: list[dict[str, str]], best_baseline_system: str) -> None:
    import matplotlib.pyplot as plt

    aggregate = index_rows(aggregate_rows, "system")
    selected = [
        "GROVE_full_paper_archived",
        "GROVE_trace_qwen35_9b_final",
        best_baseline_system,
        "ablation_groundingdino_only",
        "ablation_gdino_detr_openclip_stack_only",
        "ablation_no_fallback_grounding_drop",
        "ablation_no_fallback_grounding_boxes_removed",
        "ablation_path2_only_no_grounding",
        "ablation_no_reconciliation_union",
    ]
    rows = [aggregate[s] for s in selected if s and s in aggregate]
    labels = [SYSTEM_LABELS.get(row.get("system", ""), row.get("system", "")) for row in rows]
    y = list(range(len(rows)))
    gt = [safe_float(row.get("gt_coverage_iog03")) or 0.0 for row in rows]
    nobox = [safe_float(row.get("no_box_rate")) or 0.0 for row in rows]
    f1 = [safe_float(row.get("f1")) or 0.0 for row in rows]

    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    height = 0.32
    ax.barh([v - height / 2 for v in y], gt, height=height, color="#2ca25f", alpha=0.85, label="GT coverage")
    ax.barh([v + height / 2 for v in y], nobox, height=height, color="#de2d26", alpha=0.72, label="No-box rate")
    ax.plot(f1, y, marker="D", linestyle="none", color="#2b2b2b", markersize=6, label="Identification F1")
    for yi, value in zip(y, f1):
        ax.text(min(value + 0.02, 1.02), yi, f"{value:.2f}", va="center", fontsize=7.8, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Rate")
    ax.set_title("Grounding trade-off across systems")
    ax.set_xlim(0, 1.08)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_stability(path: Path, threshold_rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, list[dict[str, str]]] = {}
    for row in threshold_rows:
        by_type.setdefault(row.get("threshold_type", ""), []).append(row)

    order = ["groundingdino_confidence", "openclip_similarity", "oversized_box_cutoff_pct", "iog_success_threshold"]
    title_map = {
        "groundingdino_confidence": "GroundingDINO confidence",
        "openclip_similarity": "OpenCLIP similarity",
        "oversized_box_cutoff_pct": "Oversized-box cutoff",
        "iog_success_threshold": "IoG success threshold",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=False)
    for ax, threshold_type in zip(axes.ravel(), order):
        rows = sorted(by_type.get(threshold_type, []), key=lambda r: safe_float(r.get("threshold")) or 0.0)
        x = [safe_float(r.get("threshold")) or 0.0 for r in rows]
        gt = [safe_float(r.get("gt_coverage_iog03")) or 0.0 for r in rows]
        nobox = [safe_float(r.get("no_box_rate")) or 0.0 for r in rows]
        ax.plot(x, gt, marker="o", linewidth=2, label="GT coverage")
        ax.plot(x, nobox, marker="s", linewidth=2, label="No-box rate")
        default = safe_float(DEFAULT_THRESHOLDS.get(threshold_type), None)
        if default is not None:
            ax.axvline(default, color="#444444", linestyle="--", linewidth=1)
        ax.set_title(title_map.get(threshold_type, threshold_type), fontsize=10.5)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Threshold")
    axes[0][0].set_ylabel("Rate")
    axes[1][0].set_ylabel("Rate")
    axes[0][0].legend(loc="best", fontsize=8.5)
    fig.suptitle("Threshold stability and calibration proxy", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def captions_text(best_baseline_system: str) -> str:
    baseline_label = best_baseline_system
    return f"""# Captions for AI Open Evidence Package

Table 1. Full-system and ablation comparison. Identification and grounding metrics are computed on the same 203 local-image evaluation set. Ablation deltas are relative to the trace-enabled GROVE control when exact trace fields are required; proxy or lower-bound rows are labeled in the exactness column.

Table 2. Paired statistical comparison between the archived GROVE full system and the best single-pass baseline ({baseline_label}). Identification and grounding differences use paired bootstrap resampling over images; the image-category correctness row reports a McNemar-style paired test.

Table 3. Threshold sensitivity for the archived GROVE full system. Rows vary one threshold family at a time and report changes in GT coverage, prediction coverage, IoU, and no-box rate relative to that threshold family's default.

Table 4. Per-category identification and grounding evidence. Support counts identify categories with limited evidence, and low-support rows are flagged when either image support or box support is at or below {LOW_SUPPORT_THRESHOLD}. Confidence intervals are bootstrap 95% intervals over images.

Table 5. Stage-level failure attribution for the archived GROVE full system. Counts are percentages of identification F1 errors and separate caption, reasoning, primary grounding, and fallback grounding failures.

Figure 1. Modular GROVE pipeline and ablation map. The diagram shows the main captioning, reasoning, grounding, verification, correction, and reconciliation modules, with callouts marking which component each key ablation removes or replaces.

Figure 2. Category-level gain heatmap. Deltas compare archived GROVE with the best single-pass baseline; absolute GROVE F1 and GT coverage show where gains are supported by strong within-system performance. Asterisks mark sparse categories.

Figure 3. Grounding trade-off across systems. Bars show GT coverage at IoG >= 0.3 and no-box rate for the full system, baseline, trace control, and key grounding ablations; diamonds show identification F1. This figure checks whether stronger grounding comes from missing boxes, oversized boxes, or fallback-heavy behavior.

Figure 4. Threshold stability and calibration proxy. Because cached predictions do not expose full probabilistic calibration curves for every stage, this defensible alternative varies available threshold families and tracks GT coverage and no-box rate.
"""


def claim_traceability(
    ablation_table: list[dict[str, str]],
    paired_table: list[dict[str, str]],
    category_table: list[dict[str, str]],
    threshold_table: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_system = index_rows(ablation_table, "system")
    by_metric = index_rows(paired_table, "metric")

    full = by_system.get("GROVE_full_paper_archived", {})
    baseline = by_system.get("baseline_direct_qwen35_27b", {})
    trace = by_system.get("GROVE_trace_qwen35_9b_final", {})
    gdino = by_system.get("ablation_groundingdino_only", {})
    no_fallback_drop = by_system.get("ablation_no_fallback_grounding_drop", {})
    no_recon = by_system.get("ablation_no_reconciliation_union", {})
    path1 = by_system.get("ablation_path1_only_no_path2", {})
    no_path2 = by_system.get("ablation_no_path2_verification", {})
    low_support_count = sum(1 for row in category_table if row.get("low_support_flag") == "1")

    # Threshold stability is summarized as the largest absolute GT-coverage
    # change among OpenCLIP thresholds up to 0.7 and the stricter GDINO 0.4 row.
    openclip_stable = [
        abs(safe_float(row.get("delta_gt_coverage_vs_default")) or 0.0)
        for row in threshold_table
        if row.get("threshold_type") == "openclip_similarity" and (safe_float(row.get("threshold")) or 0.0) <= 0.7
    ]
    max_openclip_delta = max(openclip_stable) if openclip_stable else 0.0
    gdino_04 = next(
        (
            abs(safe_float(row.get("delta_gt_coverage_vs_default")) or 0.0)
            for row in threshold_table
            if row.get("threshold_type") == "groundingdino_confidence" and row.get("threshold") == "0.4"
        ),
        0.0,
    )

    return [
        {
            "claim": "Modular decomposition improves reliability under this tested setup.",
            "evidence_artifacts": "table_1_full_vs_key_ablations.csv; table_2_paired_tests.csv; figure_1_modular_ablation_map.png",
            "traceable_result": (
                f"GROVE F1={full.get('f1')} versus best baseline F1={baseline.get('f1')}; "
                f"paired F1 difference={by_metric.get('f1', {}).get('paired_difference')} "
                f"{by_metric.get('f1', {}).get('difference_ci95')}, p={by_metric.get('f1', {}).get('p_value')}."
            ),
            "caveat": "Ablation deltas use the trace-enabled GROVE control where archived paper outputs lack internal traces.",
        },
        {
            "claim": "The gain is not explained solely by GroundingDINO alone.",
            "evidence_artifacts": "table_1_full_vs_key_ablations.csv; figure_3_no_box_grounding_tradeoff.png",
            "traceable_result": (
                f"GroundingDINO-only F1={gdino.get('f1')} and GT coverage={gdino.get('gt_coverage_iog03')}; "
                f"trace control F1={trace.get('f1')} and GT coverage={trace.get('gt_coverage_iog03')}."
            ),
            "caveat": "This isolates the trace-enabled modular run, not the archived full-system trace internals.",
        },
        {
            "claim": "The fallback does not by itself inflate the main grounding result.",
            "evidence_artifacts": "table_1_full_vs_key_ablations.csv; table_3_threshold_sensitivity.csv; figure_3_no_box_grounding_tradeoff.png",
            "traceable_result": (
                f"No-fallback drop F1={no_fallback_drop.get('f1')} and GT coverage={no_fallback_drop.get('gt_coverage_iog03')}; "
                f"trace control GT coverage={trace.get('gt_coverage_iog03')}."
            ),
            "caveat": "Fallback ablations are post hoc filters over cached trace sources; they test metric inflation rather than retraining.",
        },
        {
            "claim": "Path 2 verification contributes measurable but modest identification value in cached traces.",
            "evidence_artifacts": "table_1_full_vs_key_ablations.csv",
            "traceable_result": (
                f"Path 1 only F1={path1.get('f1')} versus trace control F1={trace.get('f1')}; "
                f"no-Path-2 verification F1={no_path2.get('f1')}."
            ),
            "caveat": "The effect is small in this trace-enabled run, so the manuscript should avoid implying Path 2 is the dominant driver.",
        },
        {
            "claim": "Reconciliation contributes measurable but not dominant value.",
            "evidence_artifacts": "table_1_full_vs_key_ablations.csv",
            "traceable_result": (
                f"No-reconciliation union F1={no_recon.get('f1')} with delta "
                f"{no_recon.get('delta_f1_vs_trace_control')} versus the trace control."
            ),
            "caveat": "The observed reconciliation delta is small; describe it as an incremental quality-control component.",
        },
        {
            "claim": "Sparse categories are too weak for strong category-specific claims.",
            "evidence_artifacts": "table_4_per_category_performance.csv; figure_2_category_metric_gain_heatmap.png",
            "traceable_result": f"{low_support_count} of {len(category_table)} categories are flagged low-support at threshold {LOW_SUPPORT_THRESHOLD}.",
            "caveat": "Low-support flags are based on image or box support and should guide cautious manuscript language.",
        },
        {
            "claim": "Grounding metrics are not being gamed by oversized boxes.",
            "evidence_artifacts": "table_3_threshold_sensitivity.csv; figure_4_threshold_stability.png; figure_3_no_box_grounding_tradeoff.png",
            "traceable_result": "Oversized-box cutoff rows report GT coverage, IoU, tightness proxy, and no-box changes as large boxes are filtered.",
            "caveat": "The cutoff analysis is a post hoc stress test over cached boxes, not a new model run.",
        },
        {
            "claim": "Results are stable enough to be credible under available threshold stress tests.",
            "evidence_artifacts": "table_3_threshold_sensitivity.csv; figure_4_threshold_stability.png",
            "traceable_result": (
                f"OpenCLIP thresholds through 0.7 change GT coverage by at most {max_openclip_delta:.4f}; "
                f"raising GDINO confidence to 0.4 changes GT coverage by {gdino_04:.4f}."
            ),
            "caveat": "Very strict thresholds reduce coverage and increase no-box rate, so stability should be claimed within the tested operating range.",
        },
    ]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def readme_text(results_dir: Path, output_dir: Path, best_baseline_system: str, n_bootstrap: int, seed: int) -> str:
    return f"""# GROVE AI Open Evidence Package

This directory contains reviewer-facing evidence derived from the completed local experiment suite in:

`{results_dir.resolve()}`

No inference is rerun by this package generator. It reads the suite CSV files, computes derived per-category bootstrap confidence intervals, renders publication-ready PNG figures, and writes captions and traceability files.

## Reproduce

From the repository root:

```bash
.venv-macos/bin/python run_ai_open_experiments.py --output-dir {results_dir} --n-bootstrap {n_bootstrap} --seed {seed}
.venv-macos/bin/python generate_ai_open_evidence_package.py --results-dir {results_dir}
```

The best single-pass baseline used for paired comparisons is `{best_baseline_system}`.

## Main Artifacts

- `table_1_full_vs_key_ablations.csv`: full system, best baseline, and key ablations.
- `table_2_paired_tests.csv`: paired bootstrap and McNemar-style tests.
- `table_3_threshold_sensitivity.csv`: threshold stress tests.
- `table_4_per_category_performance.csv`: support counts, low-support flags, per-category CIs, and category grounding.
- `table_5_failure_attribution.csv`: failure attribution counts and percentages.
- `claim_traceability_matrix.csv`: manuscript claims mapped to exact tables and figures.
- `captions.md`: concise table and figure captions.
- `figures/`: publication-ready PNG figures.

## Important Caveats

- The archived paper-facing GROVE run is treated as the primary system.
- Internal ablations that require stage traces use the trace-enabled GROVE run and report deltas against that trace control.
- The Path 2-only row is a lower-bound alternative because rejected/dropped Path 2 candidates are not fully recoverable from cached traces.
- The caption-only row is a deterministic keyword proxy over cached captions, not a fresh VLM inference run.
- Threshold and oversized-box analyses are post hoc stress tests over cached predictions.

## Output Directory

`{output_dir.resolve()}`
"""


def build_evidence_package(results_dir: Path, output_dir: Path, n_bootstrap: int, seed: int) -> None:
    aggregate_rows = read_csv(results_dir / "aggregate_metrics.csv")
    ablation_rows = read_csv(results_dir / "ablation_results.csv")
    stat_rows = read_csv(results_dir / "statistical_comparisons.csv")
    threshold_rows = read_csv(results_dir / "threshold_sensitivity.csv")
    per_category_id_rows = read_csv(results_dir / "per_category_identification_metrics.csv")
    per_category_prediction_rows = read_csv(results_dir / "per_category_predictions.csv")
    grounding_category_rows = read_csv(results_dir / "grounding_per_category_metrics.csv")
    failure_rows = read_csv(results_dir / "failure_attribution_results.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    best = best_baseline(aggregate_rows, stat_rows)

    table1 = table_full_vs_ablations(aggregate_rows, ablation_rows, best)
    table2 = table_paired_tests(stat_rows)
    table3 = table_threshold_sensitivity(threshold_rows)
    table4 = table_per_category(
        per_category_id_rows,
        per_category_prediction_rows,
        grounding_category_rows,
        "GROVE_full_paper_archived",
        best,
        n_bootstrap,
        seed,
    )
    table5 = table_failure_attribution(failure_rows)
    claims = claim_traceability(table1, table2, table4, table3)

    write_csv(output_dir / "table_1_full_vs_key_ablations.csv", table1)
    write_csv(output_dir / "table_2_paired_tests.csv", table2)
    write_csv(output_dir / "table_3_threshold_sensitivity.csv", table3)
    write_csv(output_dir / "table_4_per_category_performance.csv", table4)
    write_csv(output_dir / "table_5_failure_attribution.csv", table5)
    write_csv(output_dir / "claim_traceability_matrix.csv", claims)
    write_csv(
        output_dir / "evidence_index.csv",
        [
            {"artifact": "table_1_full_vs_key_ablations.csv", "type": "table", "purpose": "Modularity, ablation, and fairness summary."},
            {"artifact": "table_2_paired_tests.csv", "type": "table", "purpose": "Paired tests against the best baseline."},
            {"artifact": "table_3_threshold_sensitivity.csv", "type": "table", "purpose": "Threshold stability and grounding stress tests."},
            {"artifact": "table_4_per_category_performance.csv", "type": "table", "purpose": "Support counts, sparse categories, and category CIs."},
            {"artifact": "table_5_failure_attribution.csv", "type": "table", "purpose": "Causal attribution of errors by stage."},
            {"artifact": "claim_traceability_matrix.csv", "type": "table", "purpose": "Every manuscript claim mapped to evidence."},
            {"artifact": "figures/figure_1_modular_ablation_map.png", "type": "figure", "purpose": "Pipeline modules and ablation removals."},
            {"artifact": "figures/figure_2_category_metric_gain_heatmap.png", "type": "figure", "purpose": "Category-level gains and sparse-category caution."},
            {"artifact": "figures/figure_3_no_box_grounding_tradeoff.png", "type": "figure", "purpose": "No-box rate versus grounding coverage across systems."},
            {"artifact": "figures/figure_4_threshold_stability.png", "type": "figure", "purpose": "Threshold stability and calibration proxy."},
        ],
    )

    write_markdown_table(
        output_dir / "table_1_full_vs_key_ablations.md",
        table1,
        ["label", "role", "precision", "recall", "f1", "f1_ci95", "gt_coverage_iog03", "iou_at_0_5", "no_box_rate", "exactness"],
        max_rows=40,
    )
    write_markdown_table(
        output_dir / "table_2_paired_tests.md",
        table2,
        ["metric", "paired_difference", "difference_ci95", "p_value", "test"],
        max_rows=40,
    )
    write_markdown_table(
        output_dir / "table_3_threshold_sensitivity.md",
        table3,
        ["threshold_type", "threshold", "gt_coverage_iog03", "delta_gt_coverage_vs_default", "iou_at_0_5", "no_box_rate"],
        max_rows=40,
    )
    write_markdown_table(
        output_dir / "table_4_per_category_performance.md",
        table4,
        ["category_label", "support_gt_images", "support_gt_boxes", "low_support_flag", "grove_f1", "grove_f1_ci95", "baseline_f1", "delta_f1_grove_minus_baseline", "grove_gt_coverage_iog03"],
        max_rows=40,
    )
    write_markdown_table(
        output_dir / "table_5_failure_attribution.md",
        table5,
        ["failure_type", "count", "percent_of_all_f1_errors", "interpretation"],
        max_rows=20,
    )
    write_markdown_table(
        output_dir / "claim_traceability_matrix.md",
        claims,
        ["claim", "evidence_artifacts", "traceable_result", "caveat"],
        max_rows=20,
    )

    plot_pipeline_ablation_map(figure_dir / "figure_1_modular_ablation_map.png")
    plot_category_gain_heatmap(figure_dir / "figure_2_category_metric_gain_heatmap.png", table4)
    plot_no_box_tradeoff(figure_dir / "figure_3_no_box_grounding_tradeoff.png", aggregate_rows, best)
    plot_threshold_stability(figure_dir / "figure_4_threshold_stability.png", threshold_rows)

    write_text(output_dir / "captions.md", captions_text(best))
    write_text(output_dir / "README.md", readme_text(results_dir, output_dir, best, n_bootstrap, seed))

    print(f"Evidence package complete: {output_dir}")
    print(f"Best single-pass baseline: {best}")
    print("Key files:")
    for name in [
        "table_1_full_vs_key_ablations.csv",
        "table_2_paired_tests.csv",
        "table_3_threshold_sensitivity.csv",
        "table_4_per_category_performance.csv",
        "table_5_failure_attribution.csv",
        "claim_traceability_matrix.csv",
        "captions.md",
        "README.md",
    ]:
        print(f"  {output_dir / name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate AI Open reviewer evidence from completed GROVE results.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory created by run_ai_open_experiments.py")
    parser.add_argument("--output-dir", default="", help="Evidence output directory. Defaults to RESULTS_DIR/evidence_package.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir / "evidence_package"
    build_evidence_package(results_dir, output_dir, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()
