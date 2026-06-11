#!/usr/bin/env python3
"""Multi-system grounding sensitivity for the GROVE AI Open package.

All metrics are recomputed from local cached prediction JSONL files and the
local COCO annotations. No inference or network access is used.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from ai_open_experiments.suite import (
    build_systems,
    collect_predictions,
    iog,
    iop,
    iou,
    load_ground_truth,
    load_image_names,
    write_csv,
    write_markdown_table,
)


IOG_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
IOU_THRESHOLDS = [0.3, 0.4, 0.5]
IOP_THRESHOLDS = [0.3, 0.5]

DISPLAY_SYSTEMS = {
    "GROVE_full_paper_archived": "GROVE full",
    "baseline_direct_qwen35_9b": "Qwen 3.5 9B single-pass",
    "baseline_direct_qwen35_27b": "Qwen 3.5 27B single-pass",
    "baseline_direct_llama32_vision_11b_apr2026": "Llama 3.2 Vision 11B",
    "baseline_direct_gemma4_31b": "Gemma 4 31B",
}

NAMED_SYSTEM_ORDER = [
    "GROVE_full_paper_archived",
    "baseline_direct_qwen35_9b",
    "baseline_direct_qwen35_27b",
    "baseline_direct_llama32_vision_11b_apr2026",
    "baseline_direct_gemma4_31b",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def valid_predictions(hazards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "bbox_xyxy": hazard["bbox_xyxy"],
            "category": hazard.get("canonical_category", ""),
            "score": safe_float(hazard.get("bbox_confidence"), 0.0),
        }
        for hazard in hazards
        if hazard.get("bbox_xyxy")
    ]


def greedy_category_match(
    gt_items: list[dict[str, Any]],
    pred_items: list[dict[str, Any]],
    overlap_fn: Callable[[list[float], list[float]], float],
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_item in enumerate(gt_items):
        for pred_idx, pred_item in enumerate(pred_items):
            if gt_item.get("category") != pred_item.get("category"):
                continue
            score = overlap_fn(gt_item["bbox_xyxy"], pred_item["bbox_xyxy"])
            if score >= threshold:
                candidates.append((score, gt_idx, pred_idx))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, score))
    return matches


def compute_threshold_metrics(
    by_image: dict[str, list[dict[str, Any]]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    family: str,
    threshold: float,
) -> dict[str, Any]:
    overlap_fns: dict[str, Callable[[list[float], list[float]], float]] = {
        "iog": iog,
        "iou": iou,
        "iop": iop,
    }
    overlap_fn = overlap_fns[family]
    total_gt = 0
    total_pred = 0
    total_hazards = 0
    no_box_hazards = 0
    gt_covered = 0
    pred_covered = 0
    pred_contained = 0
    category_matches = 0
    matched_iou: list[float] = []
    matched_iop: list[float] = []

    for image_name in image_names:
        gt_items = gt_boxes.get(image_name, [])
        hazards = by_image.get(image_name, [])
        pred_items = valid_predictions(hazards)
        total_hazards += len(hazards)
        no_box_hazards += sum(1 for hazard in hazards if not hazard.get("bbox_xyxy"))
        total_gt += len(gt_items)
        total_pred += len(pred_items)

        for gt_item in gt_items:
            scores = [
                overlap_fn(gt_item["bbox_xyxy"], pred_item["bbox_xyxy"])
                for pred_item in pred_items
            ]
            if scores and max(scores) >= threshold:
                gt_covered += 1

        for pred_item in pred_items:
            family_scores = [
                overlap_fn(gt_item["bbox_xyxy"], pred_item["bbox_xyxy"])
                for gt_item in gt_items
            ]
            if family_scores and max(family_scores) >= threshold:
                pred_covered += 1
                best_idx = max(range(len(family_scores)), key=family_scores.__getitem__)
                best_gt = gt_items[best_idx]
                matched_iou.append(iou(best_gt["bbox_xyxy"], pred_item["bbox_xyxy"]))
                matched_iop.append(iop(best_gt["bbox_xyxy"], pred_item["bbox_xyxy"]))
            containment_scores = [
                iop(gt_item["bbox_xyxy"], pred_item["bbox_xyxy"])
                for gt_item in gt_items
            ]
            if containment_scores and max(containment_scores) >= threshold:
                pred_contained += 1

        category_matches += len(greedy_category_match(gt_items, pred_items, overlap_fn, threshold))

    category_recall = category_matches / total_gt if total_gt else 0.0
    category_precision = category_matches / total_pred if total_pred else 0.0
    return {
        "overlap_family": family,
        "threshold": threshold,
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "predicted_hazards": total_hazards,
        "no_box_hazards": no_box_hazards,
        "no_box_rate": no_box_hazards / total_hazards if total_hazards else 0.0,
        "gt_coverage_recall": gt_covered / total_gt if total_gt else 0.0,
        "pred_to_gt_coverage": pred_covered / total_pred if total_pred else 0.0,
        "gt_to_pred_containment": pred_contained / total_pred if total_pred else 0.0,
        "category_aware_grounding_recall": category_recall,
        "category_aware_grounding_precision": category_precision,
        "box_label_grounding_f1": harmonic(category_precision, category_recall),
        "mean_iou": sum(matched_iou) / len(matched_iou) if matched_iou else 0.0,
        "mean_iop_tightness": sum(matched_iop) / len(matched_iop) if matched_iop else 0.0,
    }


def read_aggregate(results_dir: Path) -> list[dict[str, str]]:
    with (results_dir / "aggregate_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_systems(
    aggregate_rows: list[dict[str, str]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[list[str], dict[str, str]]:
    baseline_rows = [row for row in aggregate_rows if row.get("family") == "baseline_single_pass"]
    best_id = max(baseline_rows, key=lambda row: safe_float(row.get("f1"), -1.0))["system"] if baseline_rows else ""
    best_gt = max(
        baseline_rows,
        key=lambda row: safe_float(row.get("gt_coverage_iog03"), -1.0),
    )["system"] if baseline_rows else ""

    roles: dict[str, list[str]] = {}
    for system in NAMED_SYSTEM_ORDER:
        roles.setdefault(system, []).append("Named comparison")
    if best_id:
        roles.setdefault(best_id, []).append("Best single-pass by identification F1")
    if best_gt:
        roles.setdefault(best_gt, []).append("Best single-pass by GT coverage")

    selected: list[str] = []
    for system in NAMED_SYSTEM_ORDER + [best_id, best_gt]:
        if system and system not in selected:
            selected.append(system)
    return selected, {system: "; ".join(roles.get(system, [])) for system in selected}


def build_long_rows(
    selected: list[str],
    roles: dict[str, str],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grids = {
        "iog": IOG_THRESHOLDS,
        "iou": IOU_THRESHOLDS,
        "iop": IOP_THRESHOLDS,
    }
    for system in selected:
        if system not in predictions:
            rows.append(
                {
                    "system": system,
                    "system_label": DISPLAY_SYSTEMS.get(system, system),
                    "selection_role": roles.get(system, ""),
                    "status": "NOT_RUN",
                    "not_run_reason": "Usable cached prediction trace was not found.",
                }
            )
            continue
        for family, thresholds in grids.items():
            for threshold in thresholds:
                metrics = compute_threshold_metrics(
                    predictions[system],
                    gt_boxes,
                    image_names,
                    family,
                    threshold,
                )
                rows.append(
                    {
                        "system": system,
                        "system_label": DISPLAY_SYSTEMS.get(system, system),
                        "selection_role": roles.get(system, ""),
                        "status": "COMPUTED",
                        "not_run_reason": "",
                        **{
                            key: fmt(value) if isinstance(value, float) else value
                            for key, value in metrics.items()
                        },
                    }
                )
    return rows


def metric_at(
    long_rows: list[dict[str, Any]],
    system: str,
    family: str,
    threshold: float,
    metric_name: str,
) -> str:
    for row in long_rows:
        if (
            row.get("system") == system
            and row.get("overlap_family") == family
            and math.isclose(safe_float(row.get("threshold")), threshold)
        ):
            return str(row.get(metric_name, ""))
    return ""


def system_status(long_rows: list[dict[str, Any]], system: str) -> tuple[str, str]:
    rows = [row for row in long_rows if row.get("system") == system]
    if not rows:
        return "NOT_RUN", "System was not selected."
    computed = [row for row in rows if row.get("status") == "COMPUTED"]
    if computed:
        return "COMPUTED", ""
    return "NOT_RUN", rows[0].get("not_run_reason", "")


def build_wide_tables(
    selected: list[str],
    roles: dict[str, str],
    long_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    iog_rows: list[dict[str, Any]] = []
    iou_rows: list[dict[str, Any]] = []
    iop_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    for system in selected:
        status, reason = system_status(long_rows, system)
        label = DISPLAY_SYSTEMS.get(system, system)
        common = {
            "system": system,
            "System": label,
            "selection_role": roles.get(system, ""),
            "status": status,
            "not_run_reason": reason,
        }
        iog_row = dict(common)
        for threshold in IOG_THRESHOLDS:
            iog_row[f"IoG@{threshold:.1f}"] = metric_at(
                long_rows, system, "iog", threshold, "gt_coverage_recall"
            )
        iog_row["No-box Rate"] = metric_at(long_rows, system, "iog", 0.3, "no_box_rate")
        iog_rows.append(iog_row)

        iou_row = dict(common)
        for threshold in IOU_THRESHOLDS:
            iou_row[f"IoU@{threshold:.1f}"] = metric_at(
                long_rows, system, "iou", threshold, "gt_coverage_recall"
            )
        iou_row["Mean IoU"] = metric_at(long_rows, system, "iou", 0.3, "mean_iou")
        iou_row["Mean Tightness"] = metric_at(long_rows, system, "iou", 0.3, "mean_iop_tightness")
        iou_row["No-box Rate"] = metric_at(long_rows, system, "iou", 0.3, "no_box_rate")
        iou_rows.append(iou_row)

        iop_row = dict(common)
        for threshold in IOP_THRESHOLDS:
            iop_row[f"IoP@{threshold:.1f}"] = metric_at(
                long_rows, system, "iop", threshold, "gt_coverage_recall"
            )
        iop_row["Pred Containment@0.3"] = metric_at(
            long_rows, system, "iop", 0.3, "gt_to_pred_containment"
        )
        iop_row["Mean Tightness"] = metric_at(long_rows, system, "iop", 0.3, "mean_iop_tightness")
        iop_row["No-box Rate"] = metric_at(long_rows, system, "iop", 0.3, "no_box_rate")
        iop_rows.append(iop_row)

        box_rows.append(
            {
                **common,
                "Box+Label F1@IoG0.3": metric_at(
                    long_rows, system, "iog", 0.3, "box_label_grounding_f1"
                ),
                "Box+Label F1@IoG0.5": metric_at(
                    long_rows, system, "iog", 0.5, "box_label_grounding_f1"
                ),
                "Box+Label F1@IoU0.3": metric_at(
                    long_rows, system, "iou", 0.3, "box_label_grounding_f1"
                ),
                "Box+Label F1@IoU0.5": metric_at(
                    long_rows, system, "iou", 0.5, "box_label_grounding_f1"
                ),
            }
        )
    return iog_rows, iou_rows, iop_rows, box_rows


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_latex_table(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
    caption: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "c" * (len(columns) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize" if len(columns) > 9 else r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
    ]
    if len(columns) > 9:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            " & ".join(latex_escape(column) for column in columns) + r" \\",
            r"\midrule",
        ]
    )
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(column, "")) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if len(columns) > 9:
        lines.append(r"}")
    lines.extend([r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_multisystem(
    path_png: Path,
    path_pdf: Path,
    long_rows: list[dict[str, Any]],
    selected: list[str],
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#5477C4", "#CC6F47", "#386411", "#8A3A6F", "#736422", "#7A828F"]
    markers = ["o", "s", "^", "D", "P", "X"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True, sharey=True)
    for idx, system in enumerate(selected):
        rows = sorted(
            [
                row for row in long_rows
                if row.get("system") == system
                and row.get("overlap_family") == "iog"
                and row.get("status") == "COMPUTED"
            ],
            key=lambda row: safe_float(row.get("threshold")),
        )
        if not rows:
            continue
        x = [safe_float(row.get("threshold")) for row in rows]
        coverage = [safe_float(row.get("gt_coverage_recall")) for row in rows]
        box_f1 = [safe_float(row.get("box_label_grounding_f1")) for row in rows]
        label = DISPLAY_SYSTEMS.get(system, system)
        axes[0].plot(
            x, coverage, color=colors[idx % len(colors)], marker=markers[idx % len(markers)],
            linewidth=1.8, markersize=5, label=label,
        )
        axes[1].plot(
            x, box_f1, color=colors[idx % len(colors)], marker=markers[idx % len(markers)],
            linewidth=1.8, markersize=5, label=label,
        )

    axes[0].set_title("GT coverage recall")
    axes[1].set_title("Box+label grounding F1")
    for ax in axes:
        ax.set_xlabel("IoG threshold")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.08, 0.72)
        ax.grid(True, color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Rate")
    fig.suptitle("Multi-system grounding sensitivity", fontsize=15, weight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "203-image OSHA-aligned benchmark; full 0-1 y-axis; category-aware matches use one-to-one greedy IoG matching",
        ha="center",
        fontsize=9,
        color="#6F768A",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.12, 1, 0.90])
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def notes_text(selected: list[str], roles: dict[str, str]) -> str:
    selected_lines = "\n".join(
        f"- `{system}`: {DISPLAY_SYSTEMS.get(system, system)}; {roles.get(system, '')}"
        for system in selected
    )
    return f"""# Multi-System Grounding Sensitivity Notes

## Scope

All values are recomputed offline from cached local JSONL predictions on the
same 203-image universe and local COCO annotations used by the main suite.

## Selected Systems

{selected_lines}

The best single-pass system by identification F1 and the best single-pass
system by GT coverage are selected from `aggregate_metrics.csv`. When either
duplicates a named system, its role is recorded rather than duplicating rows.

## Definitions

- IoG = intersection area divided by ground-truth box area.
- IoP = intersection area divided by predicted box area.
- IoU = intersection area divided by union area.
- GT coverage recall is category-agnostic and counts a GT box covered by at
  least one prediction at the selected overlap threshold.
- Pred-to-GT coverage counts predicted boxes overlapping at least one GT box.
- GT-to-pred containment reports the share of predictions with IoP at or above
  the row threshold.
- Category-aware precision and recall use one-to-one greedy matching by
  descending overlap, with canonical category equality required.
- Box+label grounding F1 is the harmonic mean of category-aware grounding
  precision and recall.
- Mean IoU and mean IoP are calculated for predictions whose best
  category-agnostic match passes the row's overlap-family threshold.

## Comparability Caveat

GROVE boxes come from GroundingDINO with DETR/OpenCLIP fallback, whereas
single-pass baseline boxes are VLM-native. The sensitivity analysis compares
observed localization behavior under common metrics; it does not imply
identical box-generation mechanisms.
"""


def run_grounding_sensitivity(
    root: Path,
    results_dir: Path,
    output_dir: Path,
) -> None:
    args = SimpleNamespace(
        root=str(root),
        image_dir="All_Images",
        gt_coco="final_evaluation_package_2026-05-01/ground_truth/_annotations.coco.json",
        paper_grove_jsonl="final_evaluation_package_2026-05-01/results/archived_raw_runs/exp01_modular_qwen_results/exp01_run1.jsonl",
        trace_grove_jsonl="sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl",
        qwen_direct_jsonl="sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen35_9b/exp_e2e_qwen35_9b_run1.jsonl",
    )
    image_names = load_image_names(root / args.image_dir, root / args.gt_coco)
    _gt_cats, gt_boxes, _gt_rows = load_ground_truth(root / args.gt_coco, image_names)
    systems = build_systems(args)
    predictions, _manifest = collect_predictions(systems, image_names)
    aggregate_rows = read_aggregate(results_dir)
    selected, roles = select_systems(aggregate_rows, predictions)
    long_rows = build_long_rows(selected, roles, predictions, gt_boxes, image_names)
    iog_rows, iou_rows, iop_rows, box_rows = build_wide_tables(selected, roles, long_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "grounding_sensitivity_detailed.csv", long_rows)
    write_csv(output_dir / "iog_sensitivity.csv", iog_rows)
    write_csv(output_dir / "iou_sensitivity.csv", iou_rows)
    write_csv(output_dir / "iop_sensitivity.csv", iop_rows)
    write_csv(output_dir / "box_label_sensitivity.csv", box_rows)

    iog_columns = ["System", "IoG@0.3", "IoG@0.4", "IoG@0.5", "IoG@0.6", "IoG@0.7", "No-box Rate"]
    iou_columns = ["System", "IoU@0.3", "IoU@0.4", "IoU@0.5", "Mean IoU", "Mean Tightness", "No-box Rate"]
    iop_columns = ["System", "IoP@0.3", "IoP@0.5", "Pred Containment@0.3", "Mean Tightness", "No-box Rate"]
    box_columns = [
        "System",
        "Box+Label F1@IoG0.3",
        "Box+Label F1@IoG0.5",
        "Box+Label F1@IoU0.3",
        "Box+Label F1@IoU0.5",
    ]
    write_markdown_table(output_dir / "iog_sensitivity.md", iog_rows, iog_columns, max_rows=20)
    write_markdown_table(output_dir / "iou_sensitivity.md", iou_rows, iou_columns, max_rows=20)
    write_markdown_table(output_dir / "iop_sensitivity.md", iop_rows, iop_columns, max_rows=20)
    write_markdown_table(output_dir / "box_label_sensitivity.md", box_rows, box_columns, max_rows=20)
    write_latex_table(
        output_dir / "iog_sensitivity.tex",
        iog_rows,
        iog_columns,
        "Grounding sensitivity under IoG thresholds.",
        "tab:iog_sensitivity",
    )
    write_latex_table(
        output_dir / "iou_sensitivity.tex",
        iou_rows,
        iou_columns,
        "Grounding sensitivity under stricter IoU thresholds.",
        "tab:iou_sensitivity",
    )
    write_latex_table(
        output_dir / "iop_sensitivity.tex",
        iop_rows,
        iop_columns,
        "Grounding sensitivity under IoP tightness thresholds.",
        "tab:iop_sensitivity",
    )
    write_latex_table(
        output_dir / "box_label_sensitivity.tex",
        box_rows,
        box_columns,
        "Category-aware box and label grounding sensitivity.",
        "tab:box_label_sensitivity",
    )
    plot_multisystem(
        output_dir / "threshold_sensitivity_multisystem.png",
        output_dir / "threshold_sensitivity_multisystem.pdf",
        long_rows,
        selected,
    )
    (output_dir / "grounding_sensitivity_notes.md").write_text(
        notes_text(selected, roles),
        encoding="utf-8",
    )
    print(f"Multi-system grounding sensitivity complete: {output_dir}")
    print(f"Systems: {', '.join(selected)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute multi-system grounding sensitivity from cached local predictions.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--results-dir", default="ai_open_results/full_suite_2026-06-08")
    parser.add_argument("--output-dir", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root).resolve()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (root / results_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir / "grounding_sensitivity_multisystem"
    run_grounding_sensitivity(root, results_dir, output_dir)


if __name__ == "__main__":
    main()
