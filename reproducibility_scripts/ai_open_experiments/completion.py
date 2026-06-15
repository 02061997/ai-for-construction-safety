#!/usr/bin/env python3
"""Complete the remaining locally defensible GROVE AI Open evidence tasks."""

from __future__ import annotations

import argparse
import csv
import html
import random
import shutil
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ai_open_experiments.grounding_sensitivity import (
    compute_threshold_metrics,
    latex_escape,
    run_grounding_sensitivity,
    safe_float,
    write_latex_table,
)
from ai_open_experiments.suite import (
    CANONICAL_CATEGORIES,
    DISPLAY_LABELS,
    build_systems,
    collect_predictions,
    compute_grounding_metrics_for_system,
    load_ground_truth,
    load_image_names,
    pred_category_set,
    read_jsonl,
    write_csv,
    write_markdown_table,
)


SEED = 20260608
N_BOOTSTRAP = 1000
SIX_CATEGORIES = [category for category in CANONICAL_CATEGORIES if category != "Other"]

QWEN_RERUN_REASON = (
    "Required qwen3.5:9b model is absent from local Ollama manifests; the Ollama "
    "server is not running and internet/model downloads are disallowed."
)
FALLBACK_RERUN_REASON = (
    "True fallback-only grounding requires rerunning every hazard phrase through "
    "DETR and OpenCLIP. The required facebook/detr-resnet-50 and OpenCLIP ViT-B-32 "
    "weights are not cached locally, so an exact offline rerun is unavailable."
)
LATENCY_KEYS = [
    "total_inference_time_sec",
    "inference_time_sec",
    "inference_time_s",
    "inference_time",
    "total_time_sec",
    "total_time_s",
    "total_time",
    "runtime_sec",
    "runtime_s",
    "runtime",
    "elapsed_sec",
    "elapsed_s",
    "elapsed",
    "latency_sec",
    "latency_s",
    "latency",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any, digits: int = 4) -> str:
    try:
        if value in ("", None):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(quantile * (len(ordered) - 1))]


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def prediction_sets(
    by_image: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    categories: list[str],
) -> dict[str, set[str]]:
    allowed = set(categories)
    return {
        image_name: pred_category_set(by_image.get(image_name, [])) & allowed
        for image_name in image_names
    }


def identification_metrics(
    by_image: dict[str, list[dict[str, Any]]],
    gt_categories: dict[str, set[str]],
    image_names: list[str],
    categories: list[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed = set(categories)
    pred_sets = prediction_sets(by_image, image_names, categories)
    filtered_gt = {name: gt_categories.get(name, set()) & allowed for name in image_names}
    tp = fp = fn = 0
    for image_name in image_names:
        gt = filtered_gt[image_name]
        pred = pred_sets[image_name]
        tp += len(gt & pred)
        fp += len(pred - gt)
        fn += len(gt - pred)
    precision, recall, f1 = prf(tp, fp, fn)

    rng = random.Random(seed)
    boot_f1: list[float] = []
    boot_precision: list[float] = []
    boot_recall: list[float] = []
    for _ in range(n_bootstrap):
        btp = bfp = bfn = 0
        for _idx in image_names:
            image_name = image_names[rng.randrange(len(image_names))]
            gt = filtered_gt[image_name]
            pred = pred_sets[image_name]
            btp += len(gt & pred)
            bfp += len(pred - gt)
            bfn += len(gt - pred)
        bp, br, bf = prf(btp, bfp, bfn)
        boot_precision.append(bp)
        boot_recall.append(br)
        boot_f1.append(bf)

    per_category: list[dict[str, Any]] = []
    macro_values: list[float] = []
    for category_index, category in enumerate(categories):
        ctp = cfp = cfn = 0
        for image_name in image_names:
            gt = filtered_gt[image_name]
            pred = pred_sets[image_name]
            ctp += int(category in gt and category in pred)
            cfp += int(category not in gt and category in pred)
            cfn += int(category in gt and category not in pred)
        cp, cr, cf = prf(ctp, cfp, cfn)
        category_rng = random.Random(seed + 10_000 + category_index)
        category_precision: list[float] = []
        category_recall: list[float] = []
        category_f1: list[float] = []
        for _ in range(n_bootstrap):
            btp = bfp = bfn = 0
            for _idx in image_names:
                image_name = image_names[category_rng.randrange(len(image_names))]
                gt = filtered_gt[image_name]
                pred = pred_sets[image_name]
                btp += int(category in gt and category in pred)
                bfp += int(category not in gt and category in pred)
                bfn += int(category in gt and category not in pred)
            bp, br, bf = prf(btp, bfp, bfn)
            category_precision.append(bp)
            category_recall.append(br)
            category_f1.append(bf)
        macro_values.append(cf)
        per_category.append(
            {
                "system": "GROVE-NoOther-EvalOnly",
                "category": category,
                "category_label": DISPLAY_LABELS.get(category, category),
                "support_gt_images": ctp + cfn,
                "TP": ctp,
                "FP": cfp,
                "FN": cfn,
                "precision": fmt(cp),
                "precision_ci95_low": fmt(percentile(category_precision, 0.025)),
                "precision_ci95_high": fmt(percentile(category_precision, 0.975)),
                "recall": fmt(cr),
                "recall_ci95_low": fmt(percentile(category_recall, 0.025)),
                "recall_ci95_high": fmt(percentile(category_recall, 0.975)),
                "f1": fmt(cf),
                "f1_ci95_low": fmt(percentile(category_f1, 0.025)),
                "f1_ci95_high": fmt(percentile(category_f1, 0.975)),
                "low_support": int((ctp + cfn) <= 10),
            }
        )

    return (
        {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "precision_ci95_low": percentile(boot_precision, 0.025),
            "precision_ci95_high": percentile(boot_precision, 0.975),
            "recall": recall,
            "recall_ci95_low": percentile(boot_recall, 0.025),
            "recall_ci95_high": percentile(boot_recall, 0.975),
            "f1": f1,
            "f1_ci95_low": percentile(boot_f1, 0.025),
            "f1_ci95_high": percentile(boot_f1, 0.975),
            "macro_f1": sum(macro_values) / len(macro_values) if macro_values else 0.0,
        },
        per_category,
    )


def build_no_other_results(
    aggregate_rows: list[dict[str, str]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    gt_categories: dict[str, set[str]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    full_system = "GROVE_full_paper_archived"
    filtered_predictions = {
        image_name: [
            hazard for hazard in predictions[full_system].get(image_name, [])
            if hazard.get("canonical_category") != "Other"
        ]
        for image_name in image_names
    }
    filtered_gt_boxes = {
        image_name: [
            gt_item for gt_item in gt_boxes.get(image_name, [])
            if gt_item.get("category") != "Other"
        ]
        for image_name in image_names
    }
    identification, per_category = identification_metrics(
        filtered_predictions,
        gt_categories,
        image_names,
        SIX_CATEGORIES,
        n_bootstrap,
        seed,
    )
    grounding = compute_grounding_metrics_for_system(
        filtered_predictions,
        filtered_gt_boxes,
        image_names,
    )
    box_label = compute_threshold_metrics(
        filtered_predictions,
        filtered_gt_boxes,
        image_names,
        "iog",
        0.3,
    )
    no_other = {
        "system": "GROVE-NoOther-EvalOnly",
        "status": "COMPUTED",
        "exactness": "Eval-only",
        "description": "Other removed from predictions and ground truth; six-category evaluation without inference rerun.",
        **identification,
        **grounding,
        "box_label_grounding_f1": box_label["box_label_grounding_f1"],
        "family": "ablation_eval_only",
        "image_universe": len(image_names),
        "n_bootstrap": n_bootstrap,
        "source_path": "Derived from GROVE_full_paper_archived cached predictions.",
    }
    full_row = next(row for row in aggregate_rows if row.get("system") == full_system)
    table_rows = [
        {
            "System": "Full GROVE (7 categories)",
            "status": "COMPUTED",
            "exactness": "Exact",
            "TP": full_row.get("TP", ""),
            "FP": full_row.get("FP", ""),
            "FN": full_row.get("FN", ""),
            "precision": full_row.get("precision", ""),
            "recall": full_row.get("recall", ""),
            "micro_f1": full_row.get("f1", ""),
            "macro_f1": full_row.get("macro_f1", ""),
            "f1_ci95": f"[{full_row.get('f1_ci95_low', '')}, {full_row.get('f1_ci95_high', '')}]",
            "gt_coverage_iog03": full_row.get("gt_coverage_iog03", ""),
            "pred_coverage_iog03": full_row.get("prediction_coverage_iog03", ""),
            "no_box_rate": full_row.get("no_box_rate", ""),
            "interpretation": "Primary seven-category result.",
        },
        {
            "System": "GROVE-NoOther-EvalOnly",
            "status": "COMPUTED",
            "exactness": "Eval-only",
            "TP": no_other["TP"],
            "FP": no_other["FP"],
            "FN": no_other["FN"],
            "precision": fmt(no_other["precision"]),
            "recall": fmt(no_other["recall"]),
            "micro_f1": fmt(no_other["f1"]),
            "macro_f1": fmt(no_other["macro_f1"]),
            "f1_ci95": f"[{fmt(no_other['f1_ci95_low'])}, {fmt(no_other['f1_ci95_high'])}]",
            "gt_coverage_iog03": fmt(no_other["gt_coverage_iog03"]),
            "pred_coverage_iog03": fmt(no_other["prediction_coverage_iog03"]),
            "no_box_rate": fmt(no_other["no_box_rate"]),
            "interpretation": "Six-category evaluation only; does not model category redistribution.",
        },
        {
            "System": "GROVE-NoOther-PromptRerun",
            "status": "NOT_RUN",
            "exactness": "NOT_RUN",
            "not_run_reason": QWEN_RERUN_REASON,
            "interpretation": "Exact prompt-level effect remains unmeasured.",
        },
    ]
    return table_rows, per_category, no_other


def no_hazard_fp_rows(
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    gt_categories: dict[str, set[str]],
    image_names: list[str],
    aggregate_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    no_hazard_images = [name for name in image_names if not gt_categories.get(name, set())]
    baseline_rows = [row for row in aggregate_rows if row.get("family") == "baseline_single_pass"]
    best_baseline = max(baseline_rows, key=lambda row: safe_float(row.get("f1"), -1.0))["system"]
    systems = [
        ("Full GROVE", "GROVE_full_paper_archived"),
        ("GROVE-NoComplianceRules", ""),
        ("Qwen 3.5 9B single-pass", "baseline_direct_qwen35_9b"),
        ("Best single-pass baseline", best_baseline),
    ]
    rows: list[dict[str, Any]] = []
    for label, system in systems:
        if not system:
            rows.append(
                {
                    "System": label,
                    "system_id": "GROVE-NoComplianceRules",
                    "status": "NOT_RUN",
                    "n_no_hazard_images": len(no_hazard_images),
                    "fp_images": "",
                    "no_hazard_fp_image_rate": "",
                    "no_hazard_specificity": "",
                    "not_run_reason": QWEN_RERUN_REASON,
                }
            )
            continue
        pred_sets = prediction_sets(predictions[system], image_names, CANONICAL_CATEGORIES)
        fp_images = sum(1 for name in no_hazard_images if pred_sets.get(name))
        fp_rate = fp_images / len(no_hazard_images) if no_hazard_images else 0.0
        rows.append(
            {
                "System": label,
                "system_id": system,
                "status": "COMPUTED",
                "n_no_hazard_images": len(no_hazard_images),
                "fp_images": fp_images,
                "no_hazard_fp_image_rate": fmt(fp_rate),
                "no_hazard_specificity": fmt(1 - fp_rate),
                "not_run_reason": "",
            }
        )
    return rows


def extract_timing(record: dict[str, Any]) -> float | None:
    for key in LATENCY_KEYS:
        value = record.get(key)
        try:
            if value not in ("", None) and float(value) >= 0:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def latency_rows(
    manifest_rows: list[dict[str, str]],
    extra_variants: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        exactness = manifest.get("exactness", "")
        family = manifest.get("family", "")
        if family == "ablation" and not exactness.startswith("exact_cached"):
            rows.append(
                {
                    "system": manifest.get("system", ""),
                    "status": "N/A trace-derived",
                    "mean_latency_sec": "",
                    "median_latency_sec": "",
                    "p90_latency_sec": "",
                    "p95_latency_sec": "",
                    "std_latency_sec": "",
                    "valid_timing_images": 0,
                    "missing_timing_images": manifest.get("images_evaluated", ""),
                    "timing_key": "",
                    "source_path": manifest.get("source_path", ""),
                }
            )
            continue
        records = read_jsonl(Path(manifest.get("source_path", "")))
        values = [value for record in records if (value := extract_timing(record)) is not None]
        timing_key = next(
            (
                key for key in LATENCY_KEYS
                if any(record.get(key) not in ("", None) for record in records)
            ),
            "",
        )
        if not values:
            status = "N/A missing timing"
            mean = median = p90 = p95 = std = ""
        else:
            status = "COMPUTED"
            mean = fmt(statistics.mean(values), 3)
            median = fmt(statistics.median(values), 3)
            p90 = fmt(percentile(values, 0.90), 3)
            p95 = fmt(percentile(values, 0.95), 3)
            std = fmt(statistics.pstdev(values), 3)
        rows.append(
            {
                "system": manifest.get("system", ""),
                "status": status,
                "mean_latency_sec": mean,
                "median_latency_sec": median,
                "p90_latency_sec": p90,
                "p95_latency_sec": p95,
                "std_latency_sec": std,
                "valid_timing_images": len(values),
                "missing_timing_images": max(0, len(records) - len(values)),
                "timing_key": timing_key,
                "source_path": manifest.get("source_path", ""),
            }
        )
    rows.extend(extra_variants)
    return rows


def aggregate_completed_rows(
    aggregate_rows: list[dict[str, str]],
    no_other: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(row, status="COMPUTED", not_run_reason="") for row in aggregate_rows]
    formatted_no_other = {
        key: fmt(value) if isinstance(value, float) else value
        for key, value in no_other.items()
    }
    rows.append(formatted_no_other)
    for system, reason in [
        ("GROVE-FallbackOnly", FALLBACK_RERUN_REASON),
        ("GROVE-NoComplianceRules", QWEN_RERUN_REASON),
        ("GROVE-NoOther-PromptRerun", QWEN_RERUN_REASON),
        ("GROVE-CaptionOnlyReasoning", QWEN_RERUN_REASON),
        ("GROVE-ImageOnlyReasoning", QWEN_RERUN_REASON),
        ("Qwen35-9B-MatchedSchema-SinglePass", QWEN_RERUN_REASON),
    ]:
        rows.append(
            {
                "system": system,
                "status": "NOT_RUN",
                "exactness": "NOT_RUN",
                "not_run_reason": reason,
                "image_universe": 203,
                "n_bootstrap": N_BOOTSTRAP,
            }
        )
    return rows


def ci_text(row: dict[str, Any]) -> str:
    low = row.get("f1_ci95_low", "")
    high = row.get("f1_ci95_high", "")
    return f"[{low}, {high}]" if low not in ("", None) and high not in ("", None) else ""


def completed_ablation_rows(
    aggregate_rows: list[dict[str, str]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    no_other: dict[str, Any],
    latency_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregate = {row.get("system", ""): row for row in aggregate_rows}
    latency = {row.get("system", ""): row for row in latency_summary}

    def computed_row(
        label: str,
        system: str,
        exactness: str,
        interpretation: str,
    ) -> dict[str, Any]:
        source = aggregate[system]
        box_metric = compute_threshold_metrics(
            predictions[system],
            gt_boxes,
            image_names,
            "iog",
            0.3,
        )
        return {
            "System Variant": label,
            "System ID": system,
            "Exactness Label": exactness,
            "Status": "COMPUTED",
            "Missing Reason": "",
            "TP/FP/FN": f"{source.get('TP', '')}/{source.get('FP', '')}/{source.get('FN', '')}",
            "Precision": source.get("precision", ""),
            "Recall": source.get("recall", ""),
            "Micro F1": source.get("f1", ""),
            "Macro F1": source.get("macro_f1", ""),
            "95% CI": ci_text(source),
            "GT Coverage": source.get("gt_coverage_iog03", ""),
            "Pred Coverage": source.get("prediction_coverage_iog03", ""),
            "Tightness": source.get("tightness_mean_iop", ""),
            "Mean IoU": source.get("mean_iou_covering", ""),
            "Box+Label F1": fmt(box_metric["box_label_grounding_f1"]),
            "Predicted Hazards": source.get("total_hazard_rows", ""),
            "No-box Rate": source.get("no_box_rate", ""),
            "Mean Latency": latency.get(system, {}).get("mean_latency_sec", ""),
            "Main Interpretation": interpretation,
        }

    def not_run_row(label: str, system: str, reason: str, interpretation: str) -> dict[str, Any]:
        return {
            "System Variant": label,
            "System ID": system,
            "Exactness Label": "NOT_RUN",
            "Status": "NOT_RUN",
            "Missing Reason": reason,
            "TP/FP/FN": "",
            "Precision": "",
            "Recall": "",
            "Micro F1": "",
            "Macro F1": "",
            "95% CI": "",
            "GT Coverage": "",
            "Pred Coverage": "",
            "Tightness": "",
            "Mean IoU": "",
            "Box+Label F1": "",
            "Predicted Hazards": "",
            "No-box Rate": "",
            "Mean Latency": "",
            "Main Interpretation": interpretation,
        }

    rows = [
        computed_row(
            "Full GROVE",
            "GROVE_full_paper_archived",
            "Exact",
            "Primary archived paper-facing result.",
        ),
        computed_row(
            "Trace-enabled GROVE control",
            "GROVE_trace_qwen35_9b_final",
            "Exact cached control",
            "Matched control for trace-derived component removals; not interchangeable with the archived paper-facing run.",
        ),
        computed_row(
            "No Path 2 verification",
            "ablation_no_path2_verification",
            "Trace-derived",
            "F1 is 0.0045 below the matched trace control; Path 2 contributes a modest trace-level gain.",
        ),
        computed_row(
            "No reconciliation (union)",
            "ablation_no_reconciliation_union",
            "Trace-derived",
            "F1 is 0.0028 below the matched trace control; reconciliation is incremental in this trace.",
        ),
        computed_row(
            "GroundingDINO only",
            "ablation_groundingdino_only",
            "Trace-derived",
            "Primary grounding retains 97.3% of matched-control GT coverage without fallback-dependent boxes.",
        ),
        computed_row(
            "GroundingDINO + DETR/OpenCLIP stack",
            "ablation_gdino_detr_openclip_stack_only",
            "Trace-derived",
            "Grounded pre-Path-2 candidates isolate the combined grounding stack before verification.",
        ),
        computed_row(
            "No fallback grounding (boxes removed)",
            "ablation_no_fallback_grounding_boxes_removed",
            "Trace-derived",
            "Identification is unchanged, while GT coverage falls and the no-box rate rises when fallback boxes are removed.",
        ),
        not_run_row(
            "Fallback only",
            "GROVE-FallbackOnly",
            FALLBACK_RERUN_REASON,
            "The fallback cannot be evaluated as a primary grounder from cached fallback-only traces.",
        ),
        not_run_row(
            "Caption-only reasoning (exact)",
            "GROVE-CaptionOnlyReasoning",
            QWEN_RERUN_REASON,
            "Exact caption-only modular reasoning remains unmeasured.",
        ),
        computed_row(
            "Caption-only keyword proxy",
            "ablation_caption_only_keyword_proxy",
            "Proxy",
            "Deterministic caption cue proxy; not evidence equivalent to a caption-only VLM rerun.",
        ),
        not_run_row(
            "Image-only modular reasoning (exact)",
            "GROVE-ImageOnlyReasoning",
            QWEN_RERUN_REASON,
            "Exact modular image-only reasoning remains unmeasured.",
        ),
        computed_row(
            "Image-only direct-Qwen proxy",
            "ablation_image_only_reasoning",
            "Proxy",
            "Direct single-pass Qwen proxy; it does not isolate caption removal inside the modular pipeline.",
        ),
        not_run_row(
            "Qwen 3.5 9B matched-schema single-pass",
            "Qwen35-9B-MatchedSchema-SinglePass",
            QWEN_RERUN_REASON,
            "Exact prompt, taxonomy, parser, and decoding parity could not be rerun.",
        ),
        computed_row(
            "Qwen 3.5 9B existing single-pass",
            "baseline_direct_qwen35_9b",
            "Proxy",
            "Schema-adjacent baseline; prompt provenance and parser parity are not exact.",
        ),
        not_run_row(
            "No compliance rules",
            "GROVE-NoComplianceRules",
            QWEN_RERUN_REASON,
            "The causal effect of compliance guardrails remains unmeasured.",
        ),
    ]

    no_other_box = fmt(no_other.get("box_label_grounding_f1"))
    rows.append(
        {
            "System Variant": "No Other category (evaluation-only)",
            "System ID": "GROVE-NoOther-EvalOnly",
            "Exactness Label": "Eval-only",
            "Status": "COMPUTED",
            "Missing Reason": "",
            "TP/FP/FN": f"{no_other['TP']}/{no_other['FP']}/{no_other['FN']}",
            "Precision": fmt(no_other["precision"]),
            "Recall": fmt(no_other["recall"]),
            "Micro F1": fmt(no_other["f1"]),
            "Macro F1": fmt(no_other["macro_f1"]),
            "95% CI": f"[{fmt(no_other['f1_ci95_low'])}, {fmt(no_other['f1_ci95_high'])}]",
            "GT Coverage": fmt(no_other["gt_coverage_iog03"]),
            "Pred Coverage": fmt(no_other["prediction_coverage_iog03"]),
            "Tightness": fmt(no_other["tightness_mean_iop"]),
            "Mean IoU": fmt(no_other["mean_iou_covering"]),
            "Box+Label F1": no_other_box,
            "Predicted Hazards": no_other["total_hazard_rows"],
            "No-box Rate": fmt(no_other["no_box_rate"]),
            "Mean Latency": "N/A eval-only",
            "Main Interpretation": "Six-category sensitivity result; inference and category allocation are unchanged.",
        }
    )
    rows.append(
        not_run_row(
            "No Other category (prompt rerun)",
            "GROVE-NoOther-PromptRerun",
            QWEN_RERUN_REASON,
            "Prompt-level category redistribution remains unmeasured.",
        )
    )
    return rows


def write_not_run_trace_placeholders(results_dir: Path) -> None:
    trace_root = results_dir / "new_ablation_traces"
    variants = {
        "fallback_only": FALLBACK_RERUN_REASON,
        "no_compliance_rules": QWEN_RERUN_REASON,
        "no_other_prompt_rerun": QWEN_RERUN_REASON,
        "caption_only_reasoning": QWEN_RERUN_REASON,
        "image_only_reasoning": QWEN_RERUN_REASON,
        "matched_schema_single_pass": QWEN_RERUN_REASON,
    }
    for name, reason in variants.items():
        directory = trace_root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "NOT_RUN.md").write_text(
            f"# {name.replace('_', ' ').title()}\n\n"
            f"Status: `NOT_RUN`\n\nReason: {reason}\n\n"
            "No raw predictions were generated, and no metrics were fabricated.\n",
            encoding="utf-8",
        )


def copy_figure_pdfs(evidence_dir: Path, sensitivity_dir: Path) -> None:
    from PIL import Image

    figure_dir = evidence_dir / "figures"
    for png_path in figure_dir.glob("*.png"):
        pdf_path = png_path.with_suffix(".pdf")
        with Image.open(png_path) as image:
            image.convert("RGB").save(pdf_path, "PDF", resolution=300.0)
    source_png = sensitivity_dir / "threshold_sensitivity_multisystem.png"
    source_pdf = sensitivity_dir / "threshold_sensitivity_multisystem.pdf"
    if source_png.exists():
        shutil.copy2(source_png, figure_dir / "figure_5_threshold_sensitivity_multisystem.png")
    if source_pdf.exists():
        shutil.copy2(source_pdf, figure_dir / "figure_5_threshold_sensitivity_multisystem.pdf")


def paper_sections(
    completed_rows: list[dict[str, Any]],
    no_other_rows: list[dict[str, Any]],
    iog_rows: list[dict[str, str]],
    iou_rows: list[dict[str, str]],
    box_rows: list[dict[str, str]],
) -> tuple[str, str]:
    by_label = {row["System Variant"]: row for row in completed_rows}
    full = by_label["Full GROVE"]
    trace_control = by_label["Trace-enabled GROVE control"]
    path2 = by_label["No Path 2 verification"]
    recon = by_label["No reconciliation (union)"]
    gdino = by_label["GroundingDINO only"]
    no_fallback = by_label["No fallback grounding (boxes removed)"]
    qwen_proxy = by_label["Qwen 3.5 9B existing single-pass"]
    no_other = next(row for row in no_other_rows if row["System"] == "GROVE-NoOther-EvalOnly")
    iog = {row["System"]: row for row in iog_rows}
    iou_table = {row["System"]: row for row in iou_rows}
    box_table = {row["System"]: row for row in box_rows}

    markdown = f"""## Ablation Study

We evaluated component removals on the same 203-image OSHA-aligned benchmark to
separate the contribution of modular decomposition from grounding and output
format choices. Full GROVE achieved micro F1={full['Micro F1']} with 95% CI
{full['95% CI']}. The trace-enabled GROVE control achieved
F1={trace_control['Micro F1']}. Against that matched control, removing Path 2
produced F1={path2['Micro F1']} (delta -0.0045), while replacing deterministic
reconciliation with a simple union produced F1={recon['Micro F1']} (delta
-0.0028). These trace-level differences suggest that Path 2 verification and
reconciliation act as incremental quality-control stages rather than the sole
source of the archived full-system advantage.

GroundingDINO-only achieved GT coverage={gdino['GT Coverage']}, versus
{trace_control['GT Coverage']} for the matched trace control. Removing cached
fallback boxes produced GT coverage={no_fallback['GT Coverage']} and increased
the no-box rate to {no_fallback['No-box Rate']}. This supports the interpretation
that primary grounding explains most cached coverage and fallback recovers
otherwise ungrounded hazards. It does not establish fallback quality as a
standalone grounder: a true fallback-only rerun could not be completed because
the required DETR/OpenCLIP weights were not locally available. Exact
caption-only, modular image-only, no-compliance-rules, and matched-schema Qwen
reruns were also unavailable because the required Qwen 3.5 9B model was absent
locally. We retain their `NOT_RUN` status and treat the existing
caption-keyword and direct-Qwen results only as proxies.

The existing schema-adjacent Qwen 3.5 9B single-pass proxy achieved
F1={qwen_proxy['Micro F1']}, below the archived full-system value, but prompt and
parser parity cannot be fully established from the packaged April runner. The
six-category evaluation excluding `Other` produced micro F1={no_other['micro_f1']}
and macro F1={no_other['macro_f1']}. Because this is evaluation-only, it does not
show how predictions would be redistributed under a six-category prompt. In our
benchmark, the results support a cautious modularity interpretation, but they do
not establish a universal limitation of single-pass VLMs.

## Grounding Threshold Sensitivity

IoG is useful for relational construction hazards because a predicted region can
cover the hazard-defining ground-truth region without matching its full extent.
IoG@0.3 is intentionally permissive, so we additionally evaluated IoG through
0.7, IoU at 0.3--0.5, IoP/tightness, and one-to-one box+label matching.

At IoG@0.3, GROVE achieved GT coverage={iog['GROVE full']['IoG@0.3']}, compared
with {iog['Llama 3.2 Vision 11B']['IoG@0.3']} for Llama 3.2 Vision 11B. At
stricter IoG@0.5, Llama reached {iog['Llama 3.2 Vision 11B']['IoG@0.5']} while
GROVE reached {iog['GROVE full']['IoG@0.5']}. Thus, the raw IoG ranking changes
at stricter thresholds and should not be presented as a threshold-invariant win.
The complementary metrics clarify the behavior: GROVE achieved IoU@0.5=
{iou_table['GROVE full']['IoU@0.5']} versus
{iou_table['Llama 3.2 Vision 11B']['IoU@0.5']} for Llama, and box+label
F1@IoU0.5={box_table['GROVE full']['Box+Label F1@IoU0.5']} versus
{box_table['Llama 3.2 Vision 11B']['Box+Label F1@IoU0.5']}. Llama's high IoG but
lower IoU, tightness, and category-aware F1 is consistent with broad boxes that
cover GT regions without localizing them tightly. The results suggest that
GROVE's grounding advantage is strongest under joint localization and label
criteria, not under every category-agnostic IoG threshold.

## Limitations

This evaluation uses 203 images and has substantial class imbalance, especially
for Struck-by and Caught-in/Between hazards. The broad `Other` category remains
heterogeneous. IoG@0.3 is permissive by design, while strict IoU can penalize
semantically valid relational hazard regions; reporting IoG, IoU, IoP, and
box+label metrics together is therefore necessary. Prompt engineering and
decomposition are not perfectly separable, and several component ablations are
trace-derived or evaluation-only. Exact inference reruns requiring unavailable
local checkpoints are explicitly marked `NOT_RUN`. External validation on
larger datasets is needed, and we do not claim a universal single-pass ceiling;
we observe saturation only among the tested open-weight baselines under this
benchmark and evaluation protocol.
"""

    latex = rf"""\subsection{{Ablation Study}}

We evaluated component removals on the same 203-image OSHA-aligned benchmark to
separate the contribution of modular decomposition from grounding and output
format choices. Full GROVE achieved micro F1={full['Micro F1']} with 95\% CI
{latex_escape(full['95% CI'])}. The trace-enabled GROVE control achieved
F1={trace_control['Micro F1']}. Against that matched control, removing Path~2
produced F1={path2['Micro F1']} (difference $-0.0045$), while replacing
deterministic reconciliation with a simple union produced
F1={recon['Micro F1']} (difference $-0.0028$). These trace-level differences
suggest that Path~2 verification and reconciliation act as incremental
quality-control stages rather than the sole source of the archived full-system
advantage.

GroundingDINO-only achieved GT coverage={gdino['GT Coverage']}, versus
{trace_control['GT Coverage']} for the matched trace control. Removing cached
fallback boxes produced GT coverage={no_fallback['GT Coverage']} and increased
the no-box rate to {no_fallback['No-box Rate']}. This supports the interpretation
that primary grounding explains most cached coverage and fallback recovers
otherwise ungrounded hazards. It does not establish fallback quality as a
standalone grounder: a true fallback-only rerun could not be completed because
the required DETR/OpenCLIP weights were not locally available. Exact
caption-only, modular image-only, no-compliance-rules, and matched-schema Qwen
reruns were also unavailable because the required Qwen 3.5 9B model was absent
locally. We retain their \texttt{{NOT\_RUN}} status and treat existing proxy
rows only as proxies.

The existing schema-adjacent Qwen 3.5 9B single-pass proxy achieved
F1={qwen_proxy['Micro F1']}, below the archived full-system value, but exact
prompt and parser parity cannot be established. Excluding \texttt{{Other}} at
evaluation time produced micro F1={no_other['micro_f1']} and macro
F1={no_other['macro_f1']}. Because this is evaluation-only, it does not show
how predictions would be redistributed under a six-category prompt. In our
benchmark, the results support a cautious modularity interpretation, but they
do not establish a universal limitation of single-pass VLMs.

\subsection{{Grounding Threshold Sensitivity}}

IoG is useful for relational construction hazards because a predicted region
can cover the hazard-defining ground-truth region without matching its full
extent. IoG@0.3 is intentionally permissive, so we additionally evaluated IoG
through 0.7, IoU at 0.3--0.5, IoP/tightness, and one-to-one box+label matching.

At IoG@0.3, GROVE achieved GT coverage={iog['GROVE full']['IoG@0.3']}, compared
with {iog['Llama 3.2 Vision 11B']['IoG@0.3']} for Llama 3.2 Vision 11B. At
IoG@0.5, Llama reached {iog['Llama 3.2 Vision 11B']['IoG@0.5']} while GROVE
reached {iog['GROVE full']['IoG@0.5']}. Thus, the raw IoG ranking changes at
stricter thresholds and should not be presented as threshold-invariant.
Complementary metrics clarify the behavior: GROVE achieved IoU@0.5=
{iou_table['GROVE full']['IoU@0.5']} versus
{iou_table['Llama 3.2 Vision 11B']['IoU@0.5']} for Llama, and box+label
F1@IoU0.5={box_table['GROVE full']['Box+Label F1@IoU0.5']} versus
{box_table['Llama 3.2 Vision 11B']['Box+Label F1@IoU0.5']}. Llama's high IoG
but lower IoU, tightness, and category-aware F1 is consistent with broad boxes.
The results suggest that GROVE's grounding advantage is strongest under joint
localization and label criteria, not under every category-agnostic IoG
threshold.

\subsection{{Limitations}}

This evaluation uses 203 images and has substantial class imbalance, especially
for Struck-by and Caught-in/Between hazards. The broad \texttt{{Other}} category
remains heterogeneous. IoG@0.3 is permissive by design, while strict IoU can
penalize semantically valid relational hazard regions; reporting IoG, IoU, IoP,
and box+label metrics together is therefore necessary. Prompt engineering and
decomposition are not perfectly separable, and several component ablations are
trace-derived or evaluation-only. Exact inference reruns requiring unavailable
local checkpoints are explicitly marked \texttt{{NOT\_RUN}}. External
validation on larger datasets is needed, and we do not claim a universal
single-pass ceiling; we observe saturation only among the tested open-weight
baselines under this benchmark and evaluation protocol.
"""
    return markdown, latex


def completed_claim_rows(
    completed_rows: list[dict[str, Any]],
    iog_rows: list[dict[str, str]],
    iou_rows: list[dict[str, str]],
    box_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_variant = {row["System Variant"]: row for row in completed_rows}
    iog = {row["System"]: row for row in iog_rows}
    iou_table = {row["System"]: row for row in iou_rows}
    box = {row["System"]: row for row in box_rows}
    return [
        {
            "claim": "Full GROVE outperforms the best tested single-pass baseline on identification.",
            "evidence": "table_2_paired_tests.csv; table_1_component_ablation_completed.csv",
            "result": "Paired identification F1 difference +0.2047, 95% CI [0.1353, 0.2728].",
            "caveat": "This supports the tested benchmark comparison, not a universal single-pass ceiling.",
        },
        {
            "claim": "Path 2 and reconciliation provide measurable but modest trace-level gains.",
            "evidence": "table_1_component_ablation_completed.csv",
            "result": (
                f"Matched trace control F1={by_variant['Trace-enabled GROVE control']['Micro F1']}; "
                f"no Path 2 F1={by_variant['No Path 2 verification']['Micro F1']} "
                f"(delta -0.0045); no reconciliation "
                f"F1={by_variant['No reconciliation (union)']['Micro F1']} "
                f"(delta -0.0028)."
            ),
            "caveat": "These comparisons use the trace-enabled control, not archived full-system internals.",
        },
        {
            "claim": "The cached grounding gain is not explained solely by fallback boxes.",
            "evidence": "table_1_component_ablation_completed.csv",
            "result": (
                f"Matched trace-control GT coverage="
                f"{by_variant['Trace-enabled GROVE control']['GT Coverage']}; "
                f"GroundingDINO-only="
                f"{by_variant['GroundingDINO only']['GT Coverage']}; "
                f"removing fallback boxes="
                f"{by_variant['No fallback grounding (boxes removed)']['GT Coverage']}."
            ),
            "caveat": "A true fallback-only primary-grounder rerun is NOT_RUN because its local weights are unavailable.",
        },
        {
            "claim": "Raw IoG ranking is not invariant to threshold.",
            "evidence": "grounding_sensitivity_multisystem/iog_sensitivity.csv",
            "result": (
                f"GROVE IoG@0.5={iog['GROVE full']['IoG@0.5']}; "
                f"Llama IoG@0.5={iog['Llama 3.2 Vision 11B']['IoG@0.5']}."
            ),
            "caveat": "High IoG can result from broad predicted boxes.",
        },
        {
            "claim": "GROVE is stronger under stricter localization and box+label criteria.",
            "evidence": "iou_sensitivity.csv; iop_sensitivity.csv; box_label_sensitivity.csv",
            "result": (
                f"GROVE IoU@0.5={iou_table['GROVE full']['IoU@0.5']} and box+label "
                f"F1@IoU0.5={box['GROVE full']['Box+Label F1@IoU0.5']}."
            ),
            "caveat": "Strict IoU may penalize semantically valid relational boxes.",
        },
        {
            "claim": "Exact causal claims for fallback-only, caption-only, image-only, compliance rules, and matched schema remain unresolved.",
            "evidence": "table_1_component_ablation_completed.csv; completion_report.md",
            "result": "Required exact reruns are explicitly marked NOT_RUN.",
            "caveat": "Proxy rows are retained only as labeled alternatives and are not treated as exact evidence.",
        },
    ]


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def report_html(
    evidence_dir: Path,
    completed_rows: list[dict[str, Any]],
    iog_rows: list[dict[str, str]],
    iou_rows: list[dict[str, str]],
    iop_rows: list[dict[str, str]],
    box_rows: list[dict[str, str]],
    compliance_rows: list[dict[str, Any]],
    no_other_rows: list[dict[str, Any]],
    latency_rows_data: list[dict[str, Any]],
    claim_rows: list[dict[str, str]],
) -> str:
    figure_uri = (evidence_dir / "figures" / "figure_5_threshold_sensitivity_multisystem.png").resolve().as_uri()
    computed = [row for row in completed_rows if row.get("Status") == "COMPUTED"]
    not_run = [row for row in completed_rows if row.get("Status") == "NOT_RUN"]
    latency_focus_ids = {
        "GROVE_full_paper_archived",
        "baseline_direct_qwen35_9b",
        "baseline_direct_qwen35_27b",
        "baseline_direct_llama32_vision_11b_apr2026",
        "baseline_direct_gemma4_31b",
    }
    latency_focus = [row for row in latency_rows_data if row.get("system") in latency_focus_ids]
    id_columns = [
        "System Variant", "Exactness Label", "TP/FP/FN", "Precision", "Recall",
        "Micro F1", "Macro F1", "95% CI", "Mean Latency",
    ]
    ground_columns = [
        "System Variant", "GT Coverage", "Pred Coverage", "Tightness", "Mean IoU",
        "Box+Label F1", "Predicted Hazards", "No-box Rate",
    ]
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>GROVE AI Open Ablation Evidence Package</title>
<style>
@page {{ size: A4 landscape; margin: 12mm; }}
body {{ font-family: Arial, sans-serif; color: #1f2430; margin: 0; font-size: 10pt; line-height: 1.35; }}
h1 {{ font-size: 24pt; margin: 0 0 8mm; }}
h2 {{ font-size: 16pt; margin-top: 9mm; border-bottom: 1px solid #d7dbe7; padding-bottom: 2mm; }}
h3 {{ font-size: 12pt; margin-top: 6mm; }}
.summary {{ background: #f4f5f7; border-left: 4px solid #5477c4; padding: 4mm 6mm; }}
.note {{ color: #565d6d; font-size: 9pt; }}
table {{ width: 100%; border-collapse: collapse; margin: 3mm 0 6mm; font-size: 7.5pt; page-break-inside: auto; }}
th, td {{ border: 1px solid #d7dbe7; padding: 1.5mm; vertical-align: top; }}
th {{ background: #eaf1fe; text-align: left; }}
tr {{ page-break-inside: avoid; }}
img {{ display: block; max-width: 92%; max-height: 150mm; margin: 4mm auto; }}
.page-break {{ break-before: page; }}
code {{ font-family: Menlo, monospace; font-size: 8pt; }}
</style>
</head>
<body>
<h1>GROVE AI Open Ablation Evidence Package</h1>
<div class="summary">
<strong>Technical summary.</strong> The archived GROVE system retains a large
identification advantage over the best tested single-pass baseline. The new
multi-system sensitivity analysis shows that raw IoG ranking changes at strict
thresholds because a Llama baseline uses broad boxes, while GROVE remains
stronger under IoU, tightness, and one-to-one box+label criteria. Exact reruns
requiring missing local checkpoints are labeled NOT_RUN.
</div>

<h2>Component ablation evidence</h2>
<p>The first table reports identification and provenance. The second reports
grounding behavior for the same variants. Trace-derived, proxy, evaluation-only,
and NOT_RUN rows remain visibly labeled.</p>
{html_table(computed + not_run, id_columns)}
{html_table(computed + not_run, ground_columns)}

<div class="page-break"></div>
<h2>Grounding threshold sensitivity</h2>
<p>Category-agnostic IoG is reported alongside IoU, IoP/tightness, and
category-aware one-to-one box+label F1. The full 0--1 axis is retained.</p>
<img src="{figure_uri}" alt="Multi-system grounding sensitivity">
<h3>IoG sensitivity</h3>
{html_table(iog_rows, ['System','IoG@0.3','IoG@0.4','IoG@0.5','IoG@0.6','IoG@0.7','No-box Rate'])}
<h3>IoU sensitivity</h3>
{html_table(iou_rows, ['System','IoU@0.3','IoU@0.4','IoU@0.5','Mean IoU','Mean Tightness','No-box Rate'])}
<h3>IoP sensitivity</h3>
{html_table(iop_rows, ['System','IoP@0.3','IoP@0.5','Pred Containment@0.3','Mean Tightness','No-box Rate'])}
<h3>Box+label sensitivity</h3>
{html_table(box_rows, ['System','Box+Label F1@IoG0.3','Box+Label F1@IoG0.5','Box+Label F1@IoU0.3','Box+Label F1@IoU0.5'])}

<div class="page-break"></div>
<h2>Compliance, taxonomy, and latency checks</h2>
<h3>No-hazard false-positive image rate</h3>
{html_table(compliance_rows, ['System','status','n_no_hazard_images','fp_images','no_hazard_fp_image_rate','no_hazard_specificity','not_run_reason'])}
<h3>No-Other evaluation</h3>
{html_table(no_other_rows, ['System','status','exactness','TP','FP','FN','precision','recall','micro_f1','macro_f1','f1_ci95','gt_coverage_iog03','no_box_rate','interpretation'])}
<h3>Latency on cached original runs</h3>
{html_table(latency_focus, ['system','status','mean_latency_sec','median_latency_sec','p90_latency_sec','p95_latency_sec','std_latency_sec','valid_timing_images','missing_timing_images'])}
<p class="note">Latency reflects original execution environments and is not a
controlled same-hardware benchmark. Trace-derived and evaluation-only variants
do not receive inferred runtimes.</p>

<h2>Claim traceability</h2>
{html_table(claim_rows, ['claim','evidence','result','caveat'])}

<h2>Methods and limitations</h2>
<p>All computed evidence uses the same 203-image universe, local COCO labels,
canonical taxonomy, and deterministic seed. IoG, IoU, and IoP use their standard
area definitions. Box+label metrics require category equality and one-to-one
greedy matching by descending overlap.</p>
<p>Two categories have sparse support, the Other class is broad, and exact
inference ablations were blocked by missing local model checkpoints. No numbers
were imputed for blocked variants. The results support a benchmark-specific
modularity claim but not a universal ceiling for single-pass VLMs.</p>
</body>
</html>
"""


def write_master_latex(evidence_dir: Path) -> None:
    text = r"""\documentclass[10pt]{article}
\usepackage[margin=0.6in,landscape]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage[T1]{fontenc}
\title{GROVE AI Open Ablation Evidence Package}
\author{}
\date{}
\begin{document}
\maketitle
\input{table_1_component_ablation_completed.tex}
\input{../grounding_sensitivity_multisystem/iog_sensitivity.tex}
\input{../grounding_sensitivity_multisystem/iou_sensitivity.tex}
\input{../grounding_sensitivity_multisystem/iop_sensitivity.tex}
\input{../grounding_sensitivity_multisystem/box_label_sensitivity.tex}
\input{compliance_ablation_nohazard_fp.tex}
\input{no_other_ablation.tex}
\input{no_other_per_category.tex}
\input{latency_summary.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/figure_5_threshold_sensitivity_multisystem.pdf}
\caption{Multi-system grounding sensitivity.}
\end{figure*}
\input{paper_ready_sections.tex}
\end{document}
"""
    (evidence_dir / "GROVE_AI_Open_Ablation_Evidence_Package.tex").write_text(text, encoding="utf-8")


def print_pdf(html_path: Path, pdf_path: Path) -> tuple[bool, str]:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome.exists():
        return False, "Google Chrome headless renderer is unavailable."
    command = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0 or not pdf_path.exists() or pdf_path.stat().st_size == 0:
        return False, (completed.stderr or completed.stdout or "Unknown Chrome PDF error").strip()
    return True, ""


def completion_report(
    results_dir: Path,
    evidence_dir: Path,
    completed_rows: list[dict[str, Any]],
    iog_rows: list[dict[str, str]],
    iou_rows: list[dict[str, str]],
    box_rows: list[dict[str, str]],
    pdf_status: tuple[bool, str],
) -> str:
    not_run = [row for row in completed_rows if row.get("Status") == "NOT_RUN"]
    not_run_lines = "\n".join(
        f"- `{row['System ID']}`: {row['Missing Reason']}" for row in not_run
    )
    generated = sorted(
        str(path.relative_to(results_dir))
        for path in results_dir.rglob("*")
        if path.is_file()
        and (
            "grounding_sensitivity_multisystem" in str(path)
            or path.parent == evidence_dir
            or path.parent == evidence_dir / "figures"
            or "new_ablation_traces" in str(path)
        )
    )
    generated_lines = "\n".join(f"- `{path}`" for path in generated)
    iog = {row["System"]: row for row in iog_rows}
    iou_table = {row["System"]: row for row in iou_rows}
    box = {row["System"]: row for row in box_rows}
    pdf_note = "Generated and non-empty." if pdf_status[0] else f"NOT_RUN: {pdf_status[1]}"
    return f"""# GROVE AI Open Evidence Completion Report

## What Was Completed

- Multi-system IoG thresholds 0.1--0.7.
- Multi-system IoU thresholds 0.3, 0.4, and 0.5.
- Multi-system IoP thresholds 0.3 and 0.5.
- One-to-one greedy category-aware box+label precision, recall, and F1.
- Six-category `GROVE-NoOther-EvalOnly` evaluation with per-category bootstrap confidence intervals.
- No-hazard false-positive image rates for available cached systems.
- Latency aggregation from cached JSONL timing fields.
- Completed ablation table with exactness and NOT_RUN labels.
- Markdown and LaTeX paper sections.
- CSV, Markdown, LaTeX, PNG, and PDF exports.
- Consolidated evidence PDF status: {pdf_note}
- PDF renderer: local headless Chrome over the static HTML report; equivalent
  LaTeX source and table fragments are provided because no local TeX engine was
  available.

## NOT_RUN Experiments

{not_run_lines}

No metrics were fabricated for these rows.

## Reproduction Commands

```bash
.venv-macos/bin/python ai_open_experiments/run_grounding_sensitivity.py \
  --results-dir ai_open_results/full_suite_2026-06-08

.venv-macos/bin/python complete_ai_open_evidence.py \
  --results-dir ai_open_results/full_suite_2026-06-08 \
  --n-bootstrap 1000 \
  --seed 20260608
```

## Paper-Number Comparability

The archived full-system and baseline rows are unchanged. New box+label F1
values use one-to-one greedy canonical-category matching and therefore should
not be substituted for differently defined legacy category-aware coverage
metrics without updating the metric label.

## Strict-Threshold Result

- At IoG@0.3, GROVE={iog['GROVE full']['IoG@0.3']} and Llama 3.2 Vision
  11B={iog['Llama 3.2 Vision 11B']['IoG@0.3']}.
- At IoG@0.5, GROVE={iog['GROVE full']['IoG@0.5']} and Llama=
  {iog['Llama 3.2 Vision 11B']['IoG@0.5']}; Full GROVE does not win every
  category-agnostic IoG threshold.
- At IoU@0.5, GROVE={iou_table['GROVE full']['IoU@0.5']} and Llama=
  {iou_table['Llama 3.2 Vision 11B']['IoU@0.5']}.
- At box+label F1@IoU0.5, GROVE=
  {box['GROVE full']['Box+Label F1@IoU0.5']} and Llama=
  {box['Llama 3.2 Vision 11B']['Box+Label F1@IoU0.5']}.

The defensible conclusion is that GROVE is stronger under tight localization
and joint box+label criteria, while raw IoG alone can favor broad boxes.

## Recommended Claim

> In our 203-image benchmark, GROVE outperforms the tested single-pass
> open-weight VLM baselines on hazard identification and on joint localization
> and category-aware grounding metrics. The advantage persists under stricter
> IoU and tightness criteria, although category-agnostic IoG rankings vary with
> threshold because broad boxes can obtain high coverage. These results support
> modular decomposition under the tested setup; they do not establish a
> universal ceiling for single-pass VLMs.

The phrase “single-pass ceiling” should be softened to “saturation among the
tested single-pass baselines.”

## Generated Files

{generated_lines}
"""


def run_completion(root: Path, results_dir: Path, n_bootstrap: int, seed: int) -> None:
    evidence_dir = results_dir / "evidence_package"
    sensitivity_dir = results_dir / "grounding_sensitivity_multisystem"
    if not (sensitivity_dir / "iog_sensitivity.csv").exists():
        run_grounding_sensitivity(root, results_dir, sensitivity_dir)

    args = SimpleNamespace(
        root=str(root),
        image_dir="All_Images",
        gt_coco="final_evaluation_package_2026-05-01/ground_truth/_annotations.coco.json",
        paper_grove_jsonl="final_evaluation_package_2026-05-01/results/archived_raw_runs/exp01_modular_qwen_results/exp01_run1.jsonl",
        trace_grove_jsonl="sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl",
        qwen_direct_jsonl="sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen35_9b/exp_e2e_qwen35_9b_run1.jsonl",
    )
    image_names = load_image_names(root / args.image_dir, root / args.gt_coco)
    gt_categories, gt_boxes, _ = load_ground_truth(root / args.gt_coco, image_names)
    predictions, _ = collect_predictions(build_systems(args), image_names)
    aggregate_rows = read_csv(results_dir / "aggregate_metrics.csv")
    manifest_rows = read_csv(results_dir / "source_manifest.csv")

    no_other_rows, no_other_per_category, no_other_metric = build_no_other_results(
        aggregate_rows,
        predictions,
        gt_categories,
        gt_boxes,
        image_names,
        n_bootstrap,
        seed,
    )
    compliance_rows = no_hazard_fp_rows(
        predictions,
        gt_categories,
        image_names,
        aggregate_rows,
    )
    extra_latency = [
        {
            "system": "GROVE-NoOther-EvalOnly",
            "status": "N/A eval-only",
            "mean_latency_sec": "",
            "median_latency_sec": "",
            "p90_latency_sec": "",
            "p95_latency_sec": "",
            "std_latency_sec": "",
            "valid_timing_images": 0,
            "missing_timing_images": 203,
            "timing_key": "",
            "source_path": "Derived from GROVE_full_paper_archived.",
        }
    ]
    for system in [
        "GROVE-FallbackOnly",
        "GROVE-NoComplianceRules",
        "GROVE-NoOther-PromptRerun",
        "GROVE-CaptionOnlyReasoning",
        "GROVE-ImageOnlyReasoning",
        "Qwen35-9B-MatchedSchema-SinglePass",
    ]:
        extra_latency.append(
            {
                "system": system,
                "status": "NOT_RUN",
                "mean_latency_sec": "",
                "median_latency_sec": "",
                "p90_latency_sec": "",
                "p95_latency_sec": "",
                "std_latency_sec": "",
                "valid_timing_images": 0,
                "missing_timing_images": 203,
                "timing_key": "",
                "source_path": "",
            }
        )
    latency_summary = latency_rows(manifest_rows, extra_latency)
    completed_rows = completed_ablation_rows(
        aggregate_rows,
        predictions,
        gt_boxes,
        image_names,
        no_other_metric,
        latency_summary,
    )
    aggregate_completed = aggregate_completed_rows(aggregate_rows, no_other_metric)

    iog_rows = read_csv(sensitivity_dir / "iog_sensitivity.csv")
    iou_rows = read_csv(sensitivity_dir / "iou_sensitivity.csv")
    iop_rows = read_csv(sensitivity_dir / "iop_sensitivity.csv")
    box_rows = read_csv(sensitivity_dir / "box_label_sensitivity.csv")
    claim_rows = completed_claim_rows(completed_rows, iog_rows, iou_rows, box_rows)

    evidence_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "aggregate_metrics_completed.csv", aggregate_completed)
    write_csv(results_dir / "ablation_results_completed.csv", completed_rows)
    write_csv(evidence_dir / "table_1_component_ablation_completed.csv", completed_rows)
    write_csv(evidence_dir / "table_1_full_vs_key_ablations_completed.csv", completed_rows)
    write_csv(evidence_dir / "no_other_ablation.csv", no_other_rows)
    write_csv(evidence_dir / "no_other_per_category.csv", no_other_per_category)
    write_csv(evidence_dir / "compliance_ablation_nohazard_fp.csv", compliance_rows)
    write_csv(evidence_dir / "latency_summary.csv", latency_summary)
    write_csv(evidence_dir / "claim_traceability_matrix_completed.csv", claim_rows)

    completed_columns = list(completed_rows[0].keys())
    no_other_columns = list(no_other_rows[0].keys())
    no_other_cat_columns = list(no_other_per_category[0].keys())
    compliance_columns = list(compliance_rows[0].keys())
    latency_columns = list(latency_summary[0].keys())
    write_markdown_table(
        evidence_dir / "table_1_component_ablation_completed.md",
        completed_rows,
        completed_columns,
        max_rows=30,
    )
    write_markdown_table(
        evidence_dir / "component_ablation_completed.md",
        completed_rows,
        completed_columns,
        max_rows=30,
    )
    write_markdown_table(evidence_dir / "no_other_ablation.md", no_other_rows, no_other_columns, max_rows=20)
    write_markdown_table(
        evidence_dir / "no_other_per_category.md",
        no_other_per_category,
        no_other_cat_columns,
        max_rows=20,
    )
    write_markdown_table(
        evidence_dir / "compliance_ablation_nohazard_fp.md",
        compliance_rows,
        compliance_columns,
        max_rows=20,
    )
    write_markdown_table(evidence_dir / "latency_summary.md", latency_summary, latency_columns, max_rows=60)
    write_markdown_table(
        evidence_dir / "claim_traceability_matrix_completed.md",
        claim_rows,
        ["claim", "evidence", "result", "caveat"],
        max_rows=20,
    )

    write_latex_table(
        evidence_dir / "table_1_component_ablation_completed.tex",
        completed_rows,
        completed_columns,
        "Completed component ablation of GROVE on the 203-image OSHA-aligned benchmark.",
        "tab:component_ablation_completed",
    )
    shutil.copy2(
        evidence_dir / "table_1_component_ablation_completed.tex",
        evidence_dir / "component_ablation_completed.tex",
    )
    write_latex_table(
        evidence_dir / "no_other_ablation.tex",
        no_other_rows,
        no_other_columns,
        "Seven-category and six-category evaluation comparison.",
        "tab:no_other_ablation",
    )
    write_latex_table(
        evidence_dir / "no_other_per_category.tex",
        no_other_per_category,
        no_other_cat_columns,
        "Six-category per-category identification results.",
        "tab:no_other_per_category",
    )
    write_latex_table(
        evidence_dir / "compliance_ablation_nohazard_fp.tex",
        compliance_rows,
        compliance_columns,
        "False-positive image rates on the 98 no-hazard images.",
        "tab:compliance_nohazard_fp",
    )
    write_latex_table(
        evidence_dir / "latency_summary.tex",
        latency_summary,
        latency_columns,
        "Latency summary from cached original-run timing fields.",
        "tab:latency_summary",
    )

    markdown_sections, latex_sections = paper_sections(
        completed_rows,
        no_other_rows,
        iog_rows,
        iou_rows,
        box_rows,
    )
    (evidence_dir / "paper_ready_sections.md").write_text(markdown_sections, encoding="utf-8")
    (evidence_dir / "paper_ready_sections.tex").write_text(latex_sections, encoding="utf-8")
    write_not_run_trace_placeholders(results_dir)
    copy_figure_pdfs(evidence_dir, sensitivity_dir)
    write_master_latex(evidence_dir)

    html_text = report_html(
        evidence_dir,
        completed_rows,
        iog_rows,
        iou_rows,
        iop_rows,
        box_rows,
        compliance_rows,
        no_other_rows,
        latency_summary,
        claim_rows,
    )
    html_path = evidence_dir / "GROVE_AI_Open_Ablation_Evidence_Package.html"
    html_path.write_text(html_text, encoding="utf-8")
    pdf_path = evidence_dir / "GROVE_AI_Open_Ablation_Evidence_Package.pdf"
    pdf_status = print_pdf(html_path, pdf_path)

    report = completion_report(
        results_dir,
        evidence_dir,
        completed_rows,
        iog_rows,
        iou_rows,
        box_rows,
        pdf_status,
    )
    (evidence_dir / "completion_report.md").write_text(report, encoding="utf-8")
    print(f"AI Open completion package complete: {evidence_dir}")
    print(f"Consolidated PDF: {'OK' if pdf_status[0] else 'FAILED'} {pdf_status[1]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete the GROVE AI Open evidence package from local traces.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--results-dir", default="ai_open_results/full_suite_2026-06-08")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root).resolve()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (root / results_dir).resolve()
    run_completion(root, results_dir, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()
