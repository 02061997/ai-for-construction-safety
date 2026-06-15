#!/usr/bin/env python3
"""Offline, reproducible experiment suite for GROVE.

The suite evaluates cached local JSONL predictions against local COCO labels.
It does not download models or use the network. When a requested ablation needs
trace fields that are unavailable in the archived paper run, the suite uses the
trace-enabled local modular run and records the provenance/exactness in every
summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


CANONICAL_CATEGORIES = [
    "Fall_Hazard",
    "StruckBy_Hazard",
    "Electrocution_Hazard",
    "CaughtInBetween_hazard",
    "PPE_Violation",
    "Housekeeping_Storage",
    "Other",
]

DISPLAY_LABELS = {
    "Fall_Hazard": "Fall Hazard",
    "StruckBy_Hazard": "Struck-By Hazard",
    "Electrocution_Hazard": "Electrocution Hazard",
    "CaughtInBetween_hazard": "Caught-In/Between Hazard",
    "PPE_Violation": "PPE Violation",
    "Housekeeping_Storage": "Housekeeping/Storage",
    "Other": "Other",
}

GT_CATEGORY_MAP = {
    "Electrical Hazard": "Electrocution_Hazard",
    "Equipment and Machine Safety": "CaughtInBetween_hazard",
    "Fall from Height": "Fall_Hazard",
    "Falling Hazard": "Fall_Hazard",
    "Falling Object": "StruckBy_Hazard",
    "Falling Object Hazard": "StruckBy_Hazard",
    "Housekeeping": "Housekeeping_Storage",
    "Slip and Trip": "Housekeeping_Storage",
    "Slip and Trip Hazard": "Housekeeping_Storage",
    "PPE Violation": "PPE_Violation",
    "PPE Deficiency": "PPE_Violation",
    "Signage and Communication": "Other",
    "Hazards": "Other",
    "Other": "Other",
}

CAPTION_CUE_KEYWORDS = {
    "Fall_Hazard": [
        "fall",
        "ladder",
        "scaffold",
        "roof",
        "elevated",
        "edge",
        "guardrail",
        "height",
        "platform",
        "stair",
        "opening",
        "harness",
        "formwork",
        "plank",
    ],
    "StruckBy_Hazard": [
        "overhead",
        "suspended",
        "falling",
        "crane",
        "load",
        "hoist",
        "pipes",
        "object",
        "tagline",
    ],
    "Electrocution_Hazard": [
        "electric",
        "electrical",
        "wire",
        "wiring",
        "power",
        "cable",
        "cord",
        "panel",
        "voltage",
        "energized",
        "switchgear",
    ],
    "CaughtInBetween_hazard": [
        "machine",
        "machinery",
        "rotating",
        "moving part",
        "trench",
        "excavation",
        "caught",
        "pinch",
        "gear",
        "forklift",
        "excavator",
    ],
    "PPE_Violation": [
        "ppe",
        "hard hat",
        "helmet",
        "vest",
        "glove",
        "goggles",
        "glasses",
        "face shield",
        "high-visibility",
        "high visibility",
        "respirator",
        "protective",
    ],
    "Housekeeping_Storage": [
        "housekeeping",
        "debris",
        "clutter",
        "storage",
        "scattered",
        "trip",
        "slip",
        "material",
        "tools",
        "walkway",
        "obstruct",
        "loose",
    ],
    "Other": [
        "sign",
        "signage",
        "warning",
        "barrier",
        "label",
        "communication",
        "emergency",
        "restricted",
        "fence",
        "barricade",
    ],
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}


def normalize_image_name(name: str) -> str:
    s = os.path.basename(str(name).strip())
    s = re.sub(r"\.rf\.[a-zA-Z0-9]{6,}\.[a-zA-Z0-9]+$", "", s)
    s = re.sub(r"_(png|jpg|jpeg|webp|avif)$", r".\1", s, flags=re.IGNORECASE)
    base, ext = os.path.splitext(s)
    return base + ext.lower() if ext else s


def image_sort_key(name: str) -> tuple[int, str]:
    base = os.path.splitext(os.path.basename(str(name)))[0]
    try:
        return (int(base), str(name))
    except ValueError:
        return (10**9, str(name))


def map_gt_category(raw_cat: str) -> str:
    return GT_CATEGORY_MAP.get(str(raw_cat).strip(), "Other")


def map_pred_category(raw_cat: str) -> str:
    if not raw_cat:
        return "UNMAPPED"
    s = str(raw_cat).lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[/_-]+", " ", s)
    s = re.sub(r"\s+", " ", s)

    direct = {
        "fall hazard": "Fall_Hazard",
        "fall from height": "Fall_Hazard",
        "fall protection": "Fall_Hazard",
        "ppe violation": "PPE_Violation",
        "ppe deficiency": "PPE_Violation",
        "ppe": "PPE_Violation",
        "personal protective equipment": "PPE_Violation",
        "personal protective equipment ppe": "PPE_Violation",
        "electrocution hazard": "Electrocution_Hazard",
        "electrical hazard": "Electrocution_Hazard",
        "electrical safety": "Electrocution_Hazard",
        "housekeeping storage": "Housekeeping_Storage",
        "housekeeping": "Housekeeping_Storage",
        "material storage": "Housekeeping_Storage",
        "slip and trip hazard": "Housekeeping_Storage",
        "caughtinbetween hazard": "CaughtInBetween_hazard",
        "caught in between hazard": "CaughtInBetween_hazard",
        "caught in between": "CaughtInBetween_hazard",
        "equipment and machine safety": "CaughtInBetween_hazard",
        "equipment": "CaughtInBetween_hazard",
        "equipment safety": "CaughtInBetween_hazard",
        "unsafe equipment use": "CaughtInBetween_hazard",
        "machine guarding": "CaughtInBetween_hazard",
        "struckby hazard": "StruckBy_Hazard",
        "struck by hazard": "StruckBy_Hazard",
        "struck by": "StruckBy_Hazard",
        "falling object hazard": "StruckBy_Hazard",
        "falling object": "StruckBy_Hazard",
        "other": "Other",
        "signage and communication": "Other",
        "safety signage": "Other",
        "safety culture training": "Other",
        "safety signs and labels": "Other",
        "general safety": "Other",
    }
    if s in direct:
        return direct[s]

    keyword_groups = [
        (
            "PPE_Violation",
            [
                "ppe",
                "personal protective",
                "hard hat",
                "helmet",
                "safety glasses",
                "gloves",
                "high visibility",
                "hi vis",
                "reflective vest",
                "safety vest",
                "goggles",
                "face shield",
                "respirator",
                "safety boots",
                "missing ppe",
            ],
        ),
        (
            "Fall_Hazard",
            [
                "fall from height",
                "fall hazard",
                "fall protection",
                "guardrail",
                "unprotected edge",
                "scaffold",
                "ladder",
                "elevated work",
                "roof",
                "working at height",
                "fall arrest",
                "leading edge",
                "floor opening",
            ],
        ),
        (
            "Electrocution_Hazard",
            [
                "electr",
                "wiring",
                "exposed wire",
                "power line",
                "energized",
                "lockout",
                "tagout",
                "arc flash",
                "voltage",
            ],
        ),
        (
            "Housekeeping_Storage",
            [
                "housekeeping",
                "clutter",
                "debris",
                "obstruct",
                "blocked",
                "tripping",
                "slip",
                "trip",
                "walkway",
                "aisle",
                "storage",
                "stacking",
                "material storage",
                "disorganiz",
                "untidy",
                "messy",
            ],
        ),
        (
            "CaughtInBetween_hazard",
            [
                "caught",
                "pinch",
                "crush",
                "machine guard",
                "rotating",
                "moving part",
                "conveyor",
                "equipment guard",
                "nip point",
                "equipment and machine",
                "forklift",
                "excavator",
            ],
        ),
        (
            "StruckBy_Hazard",
            [
                "struck",
                "falling object",
                "overhead",
                "dropped",
                "flying object",
                "projectile",
                "suspended load",
            ],
        ),
        ("Other", ["signage", "sign ", "warning", "label", "barrier", "safety culture"]),
    ]
    for canonical, keywords in keyword_groups:
        if any(keyword in s for keyword in keywords):
            return canonical
    return "UNMAPPED"


def coco_to_xyxy(bbox: list[Any]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def parse_box(raw_box: Any) -> list[float] | None:
    if isinstance(raw_box, list) and len(raw_box) == 4:
        try:
            box = [float(v) for v in raw_box]
        except (TypeError, ValueError):
            return None
        if box[2] > box[0] and box[3] > box[1]:
            return box
    return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iog(gt_box: list[float], pred_box: list[float]) -> float:
    return intersection_area(gt_box, pred_box) / (box_area(gt_box) + 1e-9)


def iop(gt_box: list[float], pred_box: list[float]) -> float:
    return intersection_area(gt_box, pred_box) / (box_area(pred_box) + 1e-9)


def iou(gt_box: list[float], pred_box: list[float]) -> float:
    inter = intersection_area(gt_box, pred_box)
    union = box_area(gt_box) + box_area(pred_box) - inter
    return inter / (union + 1e-9)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None, max_rows: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    shown = rows[:max_rows]
    with path.open("w", encoding="utf-8") as handle:
        if not rows:
            handle.write("_No rows._\n")
            return
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in shown:
            vals = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
            handle.write("| " + " | ".join(vals) + " |\n")
        if len(rows) > max_rows:
            handle.write(f"\n_Showing {max_rows} of {len(rows)} rows._\n")


def load_image_names(image_dir: Path, coco_path: Path) -> list[str]:
    image_names: set[str] = set()
    if image_dir.exists():
        for child in image_dir.iterdir():
            if child.suffix.lower() in IMAGE_EXTS:
                image_names.add(normalize_image_name(child.name))
    if coco_path.exists():
        with coco_path.open(encoding="utf-8") as handle:
            coco = json.load(handle)
        for img in coco.get("images", []):
            image_names.add(normalize_image_name(img.get("file_name", "")))
    return sorted(image_names, key=image_sort_key)


def load_ground_truth(coco_path: Path, image_names: list[str]) -> tuple[dict[str, set[str]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    gt_cats = {name: set() for name in image_names}
    gt_boxes = {name: [] for name in image_names}
    rows: list[dict[str, Any]] = []
    if not coco_path.exists():
        return gt_cats, gt_boxes, rows

    with coco_path.open(encoding="utf-8") as handle:
        coco = json.load(handle)

    cat_by_id = {cat["id"]: cat.get("name", "") for cat in coco.get("categories", [])}
    img_by_id = {
        img["id"]: normalize_image_name(img.get("file_name", ""))
        for img in coco.get("images", [])
    }
    raw_img_by_id = {img["id"]: img.get("file_name", "") for img in coco.get("images", [])}

    for ann in coco.get("annotations", []):
        image_name = img_by_id.get(ann.get("image_id"), "")
        if image_name not in gt_cats:
            gt_cats[image_name] = set()
            gt_boxes[image_name] = []
        raw_cat = cat_by_id.get(ann.get("category_id"), "Other")
        cat = map_gt_category(raw_cat)
        box = coco_to_xyxy(ann.get("bbox", [0, 0, 0, 0]))
        gt_cats[image_name].add(cat)
        gt_boxes[image_name].append(
            {
                "bbox_xyxy": box,
                "category": cat,
                "raw_category": raw_cat,
                "annotation_id": ann.get("id", ""),
            }
        )
        rows.append(
            {
                "image_name": image_name,
                "coco_file_name": raw_img_by_id.get(ann.get("image_id"), ""),
                "annotation_id": ann.get("id", ""),
                "raw_category": raw_cat,
                "canonical_category": cat,
                "bbox_xyxy": json.dumps(box),
                "bbox_xywh": json.dumps(ann.get("bbox", [])),
                "area": ann.get("area", ""),
            }
        )
    return gt_cats, gt_boxes, rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_records_by_image(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for record in read_jsonl(path):
        name = normalize_image_name(record.get("image_name", ""))
        if name:
            records[name] = record
    return records


def parse_hazard_json(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    text = str(raw).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


def is_real_hazard(hazard: dict[str, Any]) -> bool:
    h_text = str(hazard.get("hazard", "")).upper().replace(" ", "_")
    c_text = str(hazard.get("category", "")).upper().replace(" ", "_")
    return h_text not in {"NO_HAZARD", "COMPLIANT", "NONE"} and c_text not in {"NO_HAZARD", "COMPLIANT"}


def source_is_fallback(source: Any) -> bool:
    source_text = str(source or "").lower()
    return "detr" in source_text or "clip" in source_text


def source_is_gdino(source: Any) -> bool:
    return "groundingdino" in str(source or "").lower()


def parse_openclip_similarity(source: Any) -> float | None:
    match = re.search(r"sim\s*=\s*([0-9.]+)", str(source or ""), flags=re.IGNORECASE)
    return safe_float(match.group(1)) if match else None


def hazard_category(hazard: dict[str, Any]) -> str:
    cat = hazard.get("canonical_category") or map_pred_category(hazard.get("category") or hazard.get("hazard") or "")
    return cat if cat in CANONICAL_CATEGORIES else "UNMAPPED"


def normalize_hazard(hazard: dict[str, Any], source_record: dict[str, Any], hazard_index: int) -> dict[str, Any] | None:
    if not is_real_hazard(hazard):
        return None
    cat = hazard_category(hazard)
    if cat == "UNMAPPED":
        return None
    box = parse_box(hazard.get("bbox_xyxy"))
    source = hazard.get("grounding_source") or hazard.get("path1_box_source") or ""
    score = safe_float(hazard.get("bbox_confidence"))
    if score is None and source_is_fallback(source):
        score = parse_openclip_similarity(source)
    if score is None:
        score = 0.5 if box else 0.0
    out = dict(hazard)
    out["canonical_category"] = cat
    out["bbox_xyxy"] = box
    out["bbox_confidence"] = score
    out["grounding_source"] = source
    out["_hazard_index"] = hazard_index
    out["_image_name"] = normalize_image_name(source_record.get("image_name", ""))
    out["_hazard_key"] = hazard_key(out)
    return out


def hazard_key(hazard: dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", str(hazard.get("hazard", "")).lower()).strip()
    return f"{hazard_category(hazard)}::{text[:140]}"


def dedupe_hazards(hazards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hazard in hazards:
        key = hazard_key(hazard)
        if key in seen:
            continue
        seen.add(key)
        out.append(hazard)
    return out


def hazards_from_field(record: dict[str, Any], field: str) -> list[dict[str, Any]]:
    hazards = []
    for idx, hazard in enumerate(parse_hazard_json(record.get(field))):
        norm = normalize_hazard(hazard, record, idx)
        if norm is not None:
            hazards.append(norm)
    return hazards


def clone_without_box(hazard: dict[str, Any]) -> dict[str, Any]:
    h = dict(hazard)
    h["bbox_xyxy"] = None
    h["bbox_confidence"] = 0.0
    h["grounding_source"] = ""
    h["path1_box_source"] = ""
    return h


def transform_final(record: dict[str, Any]) -> list[dict[str, Any]]:
    return hazards_from_field(record, "hazards_json")


def transform_pre_reasoning(record: dict[str, Any]) -> list[dict[str, Any]]:
    return hazards_from_field(record, "pre_reasoning_hazards_json")


def transform_path2_only(record: dict[str, Any]) -> list[dict[str, Any]]:
    hazards = []
    for hazard in transform_final(record):
        decision = str(hazard.get("path2_reasoning_decision") or "").upper()
        if decision == "REJECT":
            continue
        hazards.append(clone_without_box(hazard))
    return hazards


def transform_final_no_boxes(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [clone_without_box(h) for h in transform_final(record)]


def transform_no_reconciliation_union(record: dict[str, Any]) -> list[dict[str, Any]]:
    pre = transform_pre_reasoning(record)
    final = transform_final(record)
    return dedupe_hazards(pre + final)


def transform_gdino_only(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [h for h in transform_pre_reasoning(record) if h.get("bbox_xyxy") and source_is_gdino(h.get("grounding_source"))]


def transform_grounding_stack_only(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [h for h in transform_pre_reasoning(record) if h.get("bbox_xyxy")]


def transform_no_fallback_drop(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for hazard in transform_final(record):
        if source_is_fallback(hazard.get("grounding_source")):
            continue
        out.append(hazard)
    return out


def transform_no_fallback_boxes_removed(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for hazard in transform_final(record):
        if source_is_fallback(hazard.get("grounding_source")):
            out.append(clone_without_box(hazard))
        else:
            out.append(hazard)
    return out


def transform_no_path2_stage_correction(record: dict[str, Any]) -> list[dict[str, Any]]:
    pre_by_key = {hazard_key(h): h for h in transform_pre_reasoning(record)}
    out = []
    for hazard in transform_final(record):
        h = dict(hazard)
        pre = pre_by_key.get(hazard_key(hazard))
        if pre:
            for key in ["category", "risk_level", "canonical_category"]:
                if pre.get(key):
                    h[key] = pre.get(key)
        out.append(h)
    return out


def transform_grounded_final_only(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [h for h in transform_final(record) if h.get("bbox_xyxy")]


def transform_caption_keyword(record: dict[str, Any]) -> list[dict[str, Any]]:
    caption = str(record.get("caption_text") or "").lower()
    hazards = []
    for cat in CANONICAL_CATEGORIES:
        if any(keyword in caption for keyword in CAPTION_CUE_KEYWORDS[cat]):
            hazards.append(
                normalize_hazard(
                    {
                        "hazard": f"Caption cue for {DISPLAY_LABELS[cat]}",
                        "category": DISPLAY_LABELS[cat],
                        "canonical_category": cat,
                        "risk_level": "UNKNOWN",
                        "mitigation": "",
                        "bbox_xyxy": None,
                        "grounding_source": "",
                        "bbox_confidence": 0.0,
                    },
                    record,
                    len(hazards),
                )
            )
    return [h for h in hazards if h is not None]


def make_system(
    name: str,
    path: Path,
    transform: Callable[[dict[str, Any]], list[dict[str, Any]]],
    family: str,
    description: str,
    exactness: str,
    control: str = "",
) -> dict[str, Any]:
    return {
        "system": name,
        "source_path": str(path),
        "transform": transform,
        "family": family,
        "description": description,
        "exactness": exactness,
        "control_system": control,
    }


def discover_direct_baselines(root: Path) -> list[Path]:
    paths: list[Path] = []
    for base in [
        root / "sitecortex_results_2026-04-24" / "raw_results",
        root / "code_files",
        root / "final_evaluation_package_2026-05-01" / "results" / "archived_raw_runs",
    ]:
        if base.exists():
            paths.extend(base.glob("**/exp_e2e*_run1.jsonl"))
    seen: set[str] = set()
    out: list[Path] = []
    for path in sorted(paths, key=direct_source_sort_key):
        exp = jsonl_experiment_id(path)
        if exp in seen:
            continue
        seen.add(exp)
        out.append(path)
    return out


def direct_source_sort_key(path: Path) -> tuple[int, str]:
    text = str(path)
    if "sitecortex_results_2026-04-24/raw_results" in text:
        priority = 0
    elif "/code_files/" in text:
        priority = 1
    else:
        priority = 2
    return (priority, text)


def jsonl_experiment_id(path: Path) -> str:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("experiment_id"):
                        return str(record["experiment_id"])
                    break
    except Exception:
        pass
    return re.sub(r"_run\d+$", "", path.stem)


def build_systems(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.root).resolve()
    paper_path = root / args.paper_grove_jsonl
    trace_path = root / args.trace_grove_jsonl
    qwen_direct = root / args.qwen_direct_jsonl

    systems = [
        make_system(
            "GROVE_full_paper_archived",
            paper_path,
            transform_final,
            "grove_main",
            "Archived paper-facing GROVE Qwen 3.5 9B final reconciled outputs.",
            "exact_cached_full_system",
        ),
        make_system(
            "GROVE_trace_qwen35_9b_final",
            trace_path,
            transform_final,
            "grove_trace_control",
            "Trace-enabled current modular Qwen 3.5 9B final outputs used as ablation control.",
            "exact_cached_trace_full_system",
        ),
    ]

    for path in discover_direct_baselines(root):
        exp = jsonl_experiment_id(path)
        systems.append(
            make_system(
                f"baseline_direct_{exp.replace('exp_e2e_', '')}",
                path,
                transform_final,
                "baseline_single_pass",
                "Single-pass image reasoning baseline from local cached JSONL.",
                "exact_cached_baseline",
            )
        )

    systems.extend(
        [
            make_system(
                "ablation_path1_only_no_path2",
                trace_path,
                transform_pre_reasoning,
                "ablation",
                "Path 1 only: caption plus hazard reasoning plus grounding; no Path 2 verification.",
                "exact_from_trace_pre_reasoning",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_path2_only_no_grounding",
                trace_path,
                transform_path2_only,
                "ablation",
                "Path 2 only lower-bound: retained Path 2 KEEP/REVISE hazards with all boxes stripped.",
                "posthoc_lower_bound_trace_retained_only",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_no_reconciliation_union",
                trace_path,
                transform_no_reconciliation_union,
                "ablation",
                "No reconciliation: simple union of Path 1 pre-reasoning candidates and final retained hazards.",
                "posthoc_union_rule",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_groundingdino_only",
                trace_path,
                transform_gdino_only,
                "ablation",
                "GroundingDINO-only: Path 1 hazards retained only when the primary source produced a box.",
                "posthoc_source_filter",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_gdino_detr_openclip_stack_only",
                trace_path,
                transform_grounding_stack_only,
                "ablation",
                "GroundingDINO plus DETR/OpenCLIP fallback only: grounded Path 1 candidates before Path 2.",
                "exact_from_trace_grounded_pre_reasoning",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_qwen_image_only_no_grounding",
                qwen_direct,
                transform_final_no_boxes,
                "ablation",
                "Qwen-only image reasoning without grounding: direct Qwen 3.5 9B predictions with boxes removed.",
                "exact_cached_direct_identification_posthoc_no_boxes",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_caption_only_keyword_proxy",
                trace_path,
                transform_caption_keyword,
                "ablation",
                "Caption-only deterministic fallback: taxonomy keyword cues in cached captions, no image/model rerun.",
                "fallback_caption_keyword_proxy",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_image_only_reasoning",
                qwen_direct,
                transform_final,
                "ablation",
                "Image-only single-pass Qwen 3.5 9B reasoning baseline.",
                "exact_cached_direct",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_no_fallback_grounding_drop",
                trace_path,
                transform_no_fallback_drop,
                "ablation",
                "No fallback grounding: final hazards using DETR/OpenCLIP fallback are dropped.",
                "posthoc_source_filter",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_no_fallback_grounding_boxes_removed",
                trace_path,
                transform_no_fallback_boxes_removed,
                "ablation",
                "No fallback grounding: fallback boxes removed while hazard text/category decisions remain.",
                "posthoc_box_filter",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_no_path2_verification",
                trace_path,
                transform_pre_reasoning,
                "ablation",
                "No Path 2 verification: alias of Path 1 pre-reasoning hazards.",
                "exact_from_trace_pre_reasoning",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_no_path2_stage_correction",
                trace_path,
                transform_no_path2_stage_correction,
                "ablation",
                "No stage-specific correction from Path 2: final keep/drop decisions with pre-Path-2 category/risk labels restored when matched.",
                "posthoc_text_match",
                "GROVE_trace_qwen35_9b_final",
            ),
            make_system(
                "ablation_grounded_final_only",
                trace_path,
                transform_grounded_final_only,
                "ablation",
                "Grounded final hazards only: final hazards without a valid box are removed.",
                "posthoc_grounding_filter",
                "GROVE_trace_qwen35_9b_final",
            ),
        ]
    )
    return [system for system in systems if Path(system["source_path"]).exists()]


def collect_predictions(
    systems: list[dict[str, Any]],
    image_names: list[str],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    manifest: list[dict[str, Any]] = []
    for system in systems:
        records = load_records_by_image(Path(system["source_path"]))
        by_image: dict[str, list[dict[str, Any]]] = {}
        missing = 0
        for image_name in image_names:
            record = records.get(image_name)
            if record is None:
                by_image[image_name] = []
                missing += 1
            else:
                by_image[image_name] = system["transform"](record)
        predictions[system["system"]] = by_image
        manifest.append(
            {
                "system": system["system"],
                "family": system["family"],
                "description": system["description"],
                "exactness": system["exactness"],
                "control_system": system.get("control_system", ""),
                "source_path": system["source_path"],
                "records_in_source": len(records),
                "images_evaluated": len(image_names),
                "missing_prediction_images": missing,
            }
        )
    return predictions, manifest


def pred_category_set(hazards: list[dict[str, Any]]) -> set[str]:
    return {hazard["canonical_category"] for hazard in hazards if hazard.get("canonical_category") in CANONICAL_CATEGORIES}


def evaluate_identification(
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    gt_cats: dict[str, set[str]],
    image_names: list[str],
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    per_category_rows: list[dict[str, Any]] = []
    image_category_rows: list[dict[str, Any]] = []

    for system, by_image in predictions.items():
        pred_sets = {img: pred_category_set(by_image.get(img, [])) for img in image_names}
        tp = fp = fn = 0
        exact_match = 0
        for img in image_names:
            gt = gt_cats.get(img, set())
            pred = pred_sets.get(img, set())
            local_tp = len(gt & pred)
            local_fp = len(pred - gt)
            local_fn = len(gt - pred)
            tp += local_tp
            fp += local_fp
            fn += local_fn
            if gt == pred:
                exact_match += 1
            per_image_rows.append(
                {
                    "system": system,
                    "image_name": img,
                    "gt_categories": "|".join(sorted(gt)) if gt else "NO_HAZARD",
                    "pred_categories": "|".join(sorted(pred)) if pred else "NO_HAZARD",
                    "tp_categories": "|".join(sorted(gt & pred)),
                    "fp_categories": "|".join(sorted(pred - gt)),
                    "fn_categories": "|".join(sorted(gt - pred)),
                    "n_gt_categories": len(gt),
                    "n_pred_categories": len(pred),
                    "n_pred_hazards": len(by_image.get(img, [])),
                    "n_pred_boxes": sum(1 for hazard in by_image.get(img, []) if hazard.get("bbox_xyxy")),
                    "pred_hazards_json": json.dumps(strip_internal_keys(by_image.get(img, [])), ensure_ascii=False),
                }
            )
            for cat in CANONICAL_CATEGORIES:
                gt_flag = int(cat in gt)
                pred_flag = int(cat in pred)
                if gt_flag and pred_flag:
                    outcome = "TP"
                elif pred_flag and not gt_flag:
                    outcome = "FP"
                elif gt_flag and not pred_flag:
                    outcome = "FN"
                else:
                    outcome = "TN"
                image_category_rows.append(
                    {
                        "system": system,
                        "image_name": img,
                        "category": cat,
                        "gt_present": gt_flag,
                        "pred_present": pred_flag,
                        "outcome": outcome,
                    }
                )

        precision, recall, f1 = prf(tp, fp, fn)
        ci = bootstrap_identification(pred_sets, gt_cats, image_names, n_boot, seed)
        macro_vals = []
        for cat in CANONICAL_CATEGORIES:
            ctp = cfp = cfn = 0
            for img in image_names:
                gt = gt_cats.get(img, set())
                pred = pred_sets.get(img, set())
                ctp += int(cat in gt and cat in pred)
                cfp += int(cat not in gt and cat in pred)
                cfn += int(cat in gt and cat not in pred)
            cp, cr, cf1 = prf(ctp, cfp, cfn)
            support = ctp + cfn
            macro_vals.append((cp, cr, cf1))
            per_category_rows.append(
                {
                    "system": system,
                    "category": cat,
                    "support_gt_images": support,
                    "support_pred_images": ctp + cfp,
                    "low_support": int(support < 10),
                    "TP": ctp,
                    "FP": cfp,
                    "FN": cfn,
                    "precision": fmt(cp),
                    "recall": fmt(cr),
                    "f1": fmt(cf1),
                }
            )
        aggregate_rows.append(
            {
                "system": system,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": fmt(precision),
                "precision_ci95_low": fmt(ci["precision"][0]),
                "precision_ci95_high": fmt(ci["precision"][2]),
                "recall": fmt(recall),
                "recall_ci95_low": fmt(ci["recall"][0]),
                "recall_ci95_high": fmt(ci["recall"][2]),
                "f1": fmt(f1),
                "f1_ci95_low": fmt(ci["f1"][0]),
                "f1_ci95_high": fmt(ci["f1"][2]),
                "macro_precision": fmt(sum(v[0] for v in macro_vals) / len(macro_vals)),
                "macro_recall": fmt(sum(v[1] for v in macro_vals) / len(macro_vals)),
                "macro_f1": fmt(sum(v[2] for v in macro_vals) / len(macro_vals)),
                "exact_image_category_match_rate": fmt(exact_match / len(image_names) if image_names else 0.0),
                "image_universe": len(image_names),
                "n_bootstrap": n_boot,
            }
        )
    return aggregate_rows, per_image_rows, per_category_rows, image_category_rows


def strip_internal_keys(hazards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for hazard in hazards:
        out.append({k: v for k, v in hazard.items() if not k.startswith("_")})
    return out


def bootstrap_identification(
    pred_sets: dict[str, set[str]],
    gt_cats: dict[str, set[str]],
    image_names: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    rng = random.Random(seed)
    values = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        tp = fp = fn = 0
        for _ in image_names:
            img = image_names[rng.randrange(len(image_names))]
            gt = gt_cats.get(img, set())
            pred = pred_sets.get(img, set())
            tp += len(gt & pred)
            fp += len(pred - gt)
            fn += len(gt - pred)
        p, r, f1 = prf(tp, fp, fn)
        values["precision"].append(p)
        values["recall"].append(r)
        values["f1"].append(f1)
    return {key: percentile_triplet(vals) for key, vals in values.items()}


def percentile_triplet(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    vals = sorted(values)
    return (
        vals[int(0.025 * (len(vals) - 1))],
        vals[int(0.500 * (len(vals) - 1))],
        vals[int(0.975 * (len(vals) - 1))],
    )


def hazard_rows(
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    image_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system, by_image in predictions.items():
        for img in image_names:
            for idx, hazard in enumerate(by_image.get(img, [])):
                box = hazard.get("bbox_xyxy")
                rows.append(
                    {
                        "system": system,
                        "image_name": img,
                        "hazard_index": idx,
                        "hazard_text": hazard.get("hazard", ""),
                        "canonical_category": hazard.get("canonical_category", ""),
                        "risk_level": hazard.get("risk_level", ""),
                        "bbox_xyxy": json.dumps(box) if box else "",
                        "bbox_confidence": fmt(float(hazard.get("bbox_confidence") or 0.0)),
                        "grounding_source": hazard.get("grounding_source", ""),
                        "bbox_area_pct": hazard.get("bbox_area_pct", ""),
                        "path2_reasoning_decision": hazard.get("path2_reasoning_decision", ""),
                        "path_comparison_decision": hazard.get("path_comparison_decision", ""),
                        "grounding_phrase_used": hazard.get("grounding_phrase_used", ""),
                    }
                )
    return rows


def compute_grounding_metrics_for_system(
    by_image: dict[str, list[dict[str, Any]]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    iog_threshold: float = 0.3,
    include_ap: bool = True,
) -> dict[str, Any]:
    total_gt = gt_cov = ca_gt_cov = iou03 = iou05 = 0
    total_pred = pred_cov = pred_contained = ca_pred_cov = 0
    total_hazards = no_box = 0
    iop_vals = []
    iou_vals = []
    ca_iou_vals = []
    cat_support = Counter()
    cat_cov = Counter()

    for img in image_names:
        gt = gt_boxes.get(img, [])
        hazards = by_image.get(img, [])
        preds = []
        for hazard in hazards:
            total_hazards += 1
            box = hazard.get("bbox_xyxy")
            if box:
                preds.append(
                    {
                        "bbox_xyxy": box,
                        "category": hazard.get("canonical_category", ""),
                        "score": safe_float(hazard.get("bbox_confidence")) or 0.0,
                    }
                )
            else:
                no_box += 1

        total_gt += len(gt)
        total_pred += len(preds)
        for gt_item in gt:
            gt_box = gt_item["bbox_xyxy"]
            gt_cat = gt_item["category"]
            cat_support[gt_cat] += 1
            all_iog = [(iog(gt_box, pred["bbox_xyxy"]), pred) for pred in preds]
            all_iou = [(iou(gt_box, pred["bbox_xyxy"]), pred) for pred in preds]
            if all_iog and max(score for score, _ in all_iog) >= iog_threshold:
                gt_cov += 1
                cat_cov[gt_cat] += 1
            if all_iou and max(score for score, _ in all_iou) >= 0.3:
                iou03 += 1
            if all_iou and max(score for score, _ in all_iou) >= 0.5:
                iou05 += 1
            cat_scores = [iog(gt_box, pred["bbox_xyxy"]) for pred in preds if pred["category"] == gt_cat]
            if cat_scores and max(cat_scores) >= iog_threshold:
                ca_gt_cov += 1

        for pred in preds:
            pred_box = pred["bbox_xyxy"]
            all_iog = [(iog(gt_item["bbox_xyxy"], pred_box), gt_item) for gt_item in gt]
            all_iop = [(iop(gt_item["bbox_xyxy"], pred_box), gt_item) for gt_item in gt]
            all_iou = [(iou(gt_item["bbox_xyxy"], pred_box), gt_item) for gt_item in gt]
            if all_iog and max(score for score, _ in all_iog) >= iog_threshold:
                pred_cov += 1
                best_gt = max(all_iog, key=lambda x: x[0])[1]
                iop_vals.append(iop(best_gt["bbox_xyxy"], pred_box))
                iou_vals.append(iou(best_gt["bbox_xyxy"], pred_box))
            if all_iop and max(score for score, _ in all_iop) >= iog_threshold:
                pred_contained += 1
            ca_scores = [
                (iog(gt_item["bbox_xyxy"], pred_box), iou(gt_item["bbox_xyxy"], pred_box))
                for gt_item in gt
                if gt_item["category"] == pred["category"]
            ]
            if ca_scores and max(score for score, _ in ca_scores) >= iog_threshold:
                ca_pred_cov += 1
                ca_iou_vals.append(max(iou_score for _, iou_score in ca_scores))

    ap_values = average_precision_by_category(by_image, gt_boxes, image_names, iou_threshold=0.5) if include_ap else {}
    metrics = {
        "ground_iog_threshold": iog_threshold,
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "total_hazard_rows": total_hazards,
        "no_box_hazard_rows": no_box,
        "no_box_rate": no_box / total_hazards if total_hazards else 0.0,
        "gt_coverage_iog03": gt_cov / total_gt if total_gt else 0.0,
        "prediction_coverage_iog03": pred_cov / total_pred if total_pred else 0.0,
        "category_aware_grounding_success_iog03": ca_gt_cov / total_gt if total_gt else 0.0,
        "category_aware_prediction_coverage_iog03": ca_pred_cov / total_pred if total_pred else 0.0,
        "iou_at_0_3": iou03 / total_gt if total_gt else 0.0,
        "iou_at_0_5": iou05 / total_gt if total_gt else 0.0,
        "tightness_mean_iop": sum(iop_vals) / len(iop_vals) if iop_vals else 0.0,
        "mean_iou_covering": sum(iou_vals) / len(iou_vals) if iou_vals else 0.0,
        "gt_to_prediction_containment_iop03": pred_contained / total_pred if total_pred else 0.0,
        "map_like_iou50_category_aware": sum(ap_values.values()) / len(ap_values) if ap_values else 0.0,
    }
    for cat in CANONICAL_CATEGORIES:
        metrics[f"gt_coverage_iog03_{cat}"] = cat_cov[cat] / cat_support[cat] if cat_support[cat] else 0.0
        metrics[f"support_gt_boxes_{cat}"] = cat_support[cat]
        metrics[f"low_support_{cat}"] = int(cat_support[cat] < 10)
    return metrics


def average_precision_by_category(
    by_image: dict[str, list[dict[str, Any]]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    iou_threshold: float,
) -> dict[str, float]:
    ap_by_cat = {}
    for cat in CANONICAL_CATEGORIES:
        gt_for_cat: dict[str, list[dict[str, Any]]] = {
            img: [gt for gt in gt_boxes.get(img, []) if gt["category"] == cat]
            for img in image_names
        }
        n_gt = sum(len(v) for v in gt_for_cat.values())
        if n_gt == 0:
            continue
        preds = []
        for img in image_names:
            for order, hazard in enumerate(by_image.get(img, [])):
                if hazard.get("canonical_category") != cat or not hazard.get("bbox_xyxy"):
                    continue
                preds.append(
                    (
                        safe_float(hazard.get("bbox_confidence")) or 0.5,
                        -order,
                        img,
                        hazard["bbox_xyxy"],
                    )
                )
        preds.sort(reverse=True)
        matched: dict[str, set[int]] = defaultdict(set)
        tp = []
        fp = []
        for _, _, img, pred_box in preds:
            candidates = gt_for_cat.get(img, [])
            best_idx = -1
            best_iou = 0.0
            for idx, gt in enumerate(candidates):
                if idx in matched[img]:
                    continue
                score = iou(gt["bbox_xyxy"], pred_box)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                matched[img].add(best_idx)
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        if not preds:
            ap_by_cat[cat] = 0.0
            continue
        cum_tp = 0
        cum_fp = 0
        precisions = []
        recalls = []
        for t, f in zip(tp, fp):
            cum_tp += t
            cum_fp += f
            precisions.append(cum_tp / (cum_tp + cum_fp))
            recalls.append(cum_tp / n_gt)
        ap = 0.0
        for threshold in [i / 100 for i in range(0, 101)]:
            eligible = [p for p, r in zip(precisions, recalls) if r >= threshold]
            ap += max(eligible) if eligible else 0.0
        ap_by_cat[cat] = ap / 101
    return ap_by_cat


def evaluate_grounding(
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    per_category = []
    ci_metrics = [
        "gt_coverage_iog03",
        "prediction_coverage_iog03",
        "category_aware_grounding_success_iog03",
        "iou_at_0_3",
        "iou_at_0_5",
        "mean_iou_covering",
        "gt_to_prediction_containment_iop03",
    ]
    for system, by_image in predictions.items():
        metrics = compute_grounding_metrics_for_system(by_image, gt_boxes, image_names)
        ci = bootstrap_grounding(by_image, gt_boxes, image_names, n_boot, seed, ci_metrics)
        row = {"system": system}
        for key, value in metrics.items():
            if isinstance(value, float):
                row[key] = fmt(value)
            else:
                row[key] = value
        for metric_name, triplet in ci.items():
            row[f"{metric_name}_ci95_low"] = fmt(triplet[0])
            row[f"{metric_name}_ci95_high"] = fmt(triplet[2])
        rows.append(row)
        for cat in CANONICAL_CATEGORIES:
            per_category.append(
                {
                    "system": system,
                    "category": cat,
                    "support_gt_boxes": metrics[f"support_gt_boxes_{cat}"],
                    "low_support": metrics[f"low_support_{cat}"],
                    "gt_coverage_iog03": fmt(metrics[f"gt_coverage_iog03_{cat}"]),
                }
            )
    return rows, per_category


def bootstrap_grounding(
    by_image: dict[str, list[dict[str, Any]]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    n_boot: int,
    seed: int,
    metrics: list[str],
) -> dict[str, tuple[float, float, float]]:
    rng = random.Random(seed)
    vals = {metric: [] for metric in metrics}
    for _ in range(n_boot):
        sample = [image_names[rng.randrange(len(image_names))] for _ in image_names]
        result = compute_grounding_metrics_for_system(by_image, gt_boxes, sample, include_ap=False)
        for metric in metrics:
            vals[metric].append(float(result.get(metric, 0.0)))
    return {metric: percentile_triplet(values) for metric, values in vals.items()}


def merge_aggregate_metrics(
    id_rows: list[dict[str, Any]],
    grounding_rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grounding_by_system = {row["system"]: row for row in grounding_rows}
    manifest_by_system = {row["system"]: row for row in manifest}
    rows = []
    for row in id_rows:
        system = row["system"]
        out = dict(row)
        g = grounding_by_system.get(system, {})
        for key, value in g.items():
            if key != "system" and not key.startswith("gt_coverage_iog03_") and not key.startswith("support_") and not key.startswith("low_support_"):
                out[key] = value
        meta = manifest_by_system.get(system, {})
        out["family"] = meta.get("family", "")
        out["exactness"] = meta.get("exactness", "")
        out["description"] = meta.get("description", "")
        out["source_path"] = meta.get("source_path", "")
        rows.append(out)
    return rows


def build_ablation_results(aggregate_rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_system = {row["system"]: row for row in aggregate_rows}
    rows = []
    for meta in manifest:
        if meta["family"] != "ablation":
            continue
        control = meta.get("control_system") or "GROVE_trace_qwen35_9b_final"
        row = by_system.get(meta["system"], {})
        base = by_system.get(control, {})
        f1 = safe_float(row.get("f1")) or 0.0
        base_f1 = safe_float(base.get("f1")) or 0.0
        ground = safe_float(row.get("gt_coverage_iog03")) or 0.0
        base_ground = safe_float(base.get("gt_coverage_iog03")) or 0.0
        rows.append(
            {
                "ablation": meta["system"],
                "control_system": control,
                "exactness": meta["exactness"],
                "description": meta["description"],
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "f1": row.get("f1", ""),
                "delta_f1_vs_control": fmt(f1 - base_f1),
                "gt_coverage_iog03": row.get("gt_coverage_iog03", ""),
                "delta_gt_coverage_iog03_vs_control": fmt(ground - base_ground),
                "no_box_rate": row.get("no_box_rate", ""),
                "source_path": meta["source_path"],
            }
        )
    return rows


def paired_bootstrap_comparison(
    full_system: str,
    comparator: str,
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    gt_cats: dict[str, set[str]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    id_pred_full = {img: pred_category_set(predictions[full_system].get(img, [])) for img in image_names}
    id_pred_comp = {img: pred_category_set(predictions[comparator].get(img, [])) for img in image_names}
    metric_values = {
        "precision": [],
        "recall": [],
        "f1": [],
        "gt_coverage_iog03": [],
        "iou_at_0_3": [],
        "iou_at_0_5": [],
        "mean_iou_covering": [],
        "category_aware_grounding_success_iog03": [],
    }
    for _ in range(n_boot):
        sample = [image_names[rng.randrange(len(image_names))] for _ in image_names]
        p1, r1, f1 = identification_metrics_for_sample(id_pred_full, gt_cats, sample)
        p2, r2, f2 = identification_metrics_for_sample(id_pred_comp, gt_cats, sample)
        metric_values["precision"].append(p1 - p2)
        metric_values["recall"].append(r1 - r2)
        metric_values["f1"].append(f1 - f2)
        g1 = compute_grounding_metrics_for_system(predictions[full_system], gt_boxes, sample, include_ap=False)
        g2 = compute_grounding_metrics_for_system(predictions[comparator], gt_boxes, sample, include_ap=False)
        for metric in [
            "gt_coverage_iog03",
            "iou_at_0_3",
            "iou_at_0_5",
            "mean_iou_covering",
            "category_aware_grounding_success_iog03",
        ]:
            metric_values[metric].append(float(g1.get(metric, 0.0)) - float(g2.get(metric, 0.0)))

    for metric, diffs in metric_values.items():
        ci = percentile_triplet(diffs)
        le_zero = sum(1 for d in diffs if d <= 0) / len(diffs)
        ge_zero = sum(1 for d in diffs if d >= 0) / len(diffs)
        rows.append(
            {
                "reference_system": full_system,
                "comparator_system": comparator,
                "metric": metric,
                "paired_diff_reference_minus_comparator": fmt(sum(diffs) / len(diffs)),
                "diff_ci95_low": fmt(ci[0]),
                "diff_ci95_high": fmt(ci[2]),
                "paired_bootstrap_p_two_sided": fmt(min(1.0, 2 * min(le_zero, ge_zero))),
                "n_bootstrap": n_boot,
            }
        )

    b = c = 0
    for img in image_names:
        gt = gt_cats.get(img, set())
        for cat in CANONICAL_CATEGORIES:
            full_correct = (cat in gt) == (cat in id_pred_full.get(img, set()))
            comp_correct = (cat in gt) == (cat in id_pred_comp.get(img, set()))
            b += int(full_correct and not comp_correct)
            c += int(comp_correct and not full_correct)
    n = b + c
    chi = ((abs(b - c) - 1) ** 2 / n) if n else 0.0
    p_mcnemar = math.erfc(math.sqrt(chi / 2)) if n else 1.0
    rows.append(
        {
            "reference_system": full_system,
            "comparator_system": comparator,
            "metric": "image_category_correctness_mcnemar",
            "paired_diff_reference_minus_comparator": b - c,
            "diff_ci95_low": "",
            "diff_ci95_high": "",
            "paired_bootstrap_p_two_sided": "",
            "mcnemar_b_reference_only_correct": b,
            "mcnemar_c_comparator_only_correct": c,
            "mcnemar_chi2_continuity": fmt(chi),
            "mcnemar_p_approx": fmt(p_mcnemar),
            "n_image_category_units": len(image_names) * len(CANONICAL_CATEGORIES),
        }
    )
    return rows


def identification_metrics_for_sample(
    pred_sets: dict[str, set[str]],
    gt_cats: dict[str, set[str]],
    sample: list[str],
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for img in sample:
        gt = gt_cats.get(img, set())
        pred = pred_sets.get(img, set())
        tp += len(gt & pred)
        fp += len(pred - gt)
        fn += len(gt - pred)
    return prf(tp, fp, fn)


def build_statistical_comparisons(
    aggregate_rows: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    gt_cats: dict[str, set[str]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    full_system = "GROVE_full_paper_archived"
    if full_system not in predictions:
        return []
    manifest_by_system = {m["system"]: m for m in manifest}
    baseline_rows = [
        row for row in aggregate_rows
        if manifest_by_system.get(row["system"], {}).get("family") == "baseline_single_pass"
    ]
    if not baseline_rows:
        return []
    best = max(baseline_rows, key=lambda row: safe_float(row.get("f1")) or -1)
    comparators = [best["system"]]
    rows = []
    for comparator in comparators:
        rows.extend(
            paired_bootstrap_comparison(
                full_system,
                comparator,
                predictions,
                gt_cats,
                gt_boxes,
                image_names,
                n_boot,
                seed,
            )
        )
    for row in rows:
        row["comparison_role"] = "GROVE_vs_best_single_pass_baseline"
    return rows


def filter_hazards_for_threshold(
    hazards: list[dict[str, Any]],
    threshold_type: str,
    threshold: float,
) -> list[dict[str, Any]]:
    out = []
    for hazard in hazards:
        h = dict(hazard)
        box = h.get("bbox_xyxy")
        if threshold_type == "groundingdino_confidence" and box and source_is_gdino(h.get("grounding_source")):
            if (safe_float(h.get("bbox_confidence")) or 0.0) < threshold:
                h = clone_without_box(h)
        elif threshold_type == "openclip_similarity" and box and source_is_fallback(h.get("grounding_source")):
            sim = parse_openclip_similarity(h.get("grounding_source"))
            if sim is None or sim < threshold:
                h = clone_without_box(h)
        elif threshold_type == "oversized_box_cutoff_pct" and box:
            area_pct = safe_float(h.get("bbox_area_pct"))
            if area_pct is not None and area_pct > threshold:
                h = clone_without_box(h)
        out.append(h)
    return out


def build_threshold_sensitivity(
    base_system: str,
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    gt_boxes: dict[str, list[dict[str, Any]]],
    image_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    if base_system not in predictions:
        return rows
    base = predictions[base_system]
    threshold_grid = {
        "groundingdino_confidence": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "openclip_similarity": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "oversized_box_cutoff_pct": [35, 50, 65, 85, 100],
    }
    for threshold_type, thresholds in threshold_grid.items():
        for threshold in thresholds:
            filtered = {
                img: filter_hazards_for_threshold(base.get(img, []), threshold_type, float(threshold))
                for img in image_names
            }
            metrics = compute_grounding_metrics_for_system(filtered, gt_boxes, image_names)
            rows.append(
                {
                    "system": base_system,
                    "threshold_type": threshold_type,
                    "threshold": threshold,
                    "gt_coverage_iog03": fmt(metrics["gt_coverage_iog03"]),
                    "prediction_coverage_iog03": fmt(metrics["prediction_coverage_iog03"]),
                    "category_aware_grounding_success_iog03": fmt(metrics["category_aware_grounding_success_iog03"]),
                    "iou_at_0_3": fmt(metrics["iou_at_0_3"]),
                    "iou_at_0_5": fmt(metrics["iou_at_0_5"]),
                    "mean_iou_covering": fmt(metrics["mean_iou_covering"]),
                    "no_box_rate": fmt(metrics["no_box_rate"]),
                }
            )
    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50]:
        metrics = compute_grounding_metrics_for_system(base, gt_boxes, image_names, iog_threshold=threshold)
        rows.append(
            {
                "system": base_system,
                "threshold_type": "iog_success_threshold",
                "threshold": threshold,
                "gt_coverage_iog03": fmt(metrics["gt_coverage_iog03"]),
                "prediction_coverage_iog03": fmt(metrics["prediction_coverage_iog03"]),
                "category_aware_grounding_success_iog03": fmt(metrics["category_aware_grounding_success_iog03"]),
                "iou_at_0_3": fmt(metrics["iou_at_0_3"]),
                "iou_at_0_5": fmt(metrics["iou_at_0_5"]),
                "mean_iou_covering": fmt(metrics["mean_iou_covering"]),
                "no_box_rate": fmt(metrics["no_box_rate"]),
            }
        )
    return rows


def caption_has_cue(caption: str, category: str) -> bool:
    text = (caption or "").lower()
    return any(keyword in text for keyword in CAPTION_CUE_KEYWORDS[category])


def build_failure_attribution(
    system: str,
    predictions: dict[str, dict[str, list[dict[str, Any]]]],
    records_path: Path,
    gt_cats: dict[str, set[str]],
    image_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records = load_records_by_image(records_path)
    detail = []
    if system not in predictions:
        return [], [], []
    for img in image_names:
        gt = gt_cats.get(img, set())
        hazards = predictions[system].get(img, [])
        pred_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for hazard in hazards:
            pred_by_cat[hazard.get("canonical_category", "")].append(hazard)
        pred = set(pred_by_cat)
        caption = records.get(img, {}).get("caption_text", "")
        for cat in sorted(gt - pred):
            if caption_has_cue(caption, cat):
                failure_type = "reasoning_omission"
                short = "caption cued category but hazard response omitted it"
            else:
                failure_type = "caption_omission"
                short = "caption lacked a direct cue for the missed category"
            detail.append(
                {
                    "system": system,
                    "failure_type": failure_type,
                    "image_name": img,
                    "event_type": "FN",
                    "category": cat,
                    "gt_categories": "|".join(sorted(gt)),
                    "pred_categories": "|".join(sorted(pred)),
                    "grounding_sources": "",
                    "hazard_text": "",
                    "short_evidence": short,
                }
            )
        for cat in sorted(pred - gt):
            cat_hazards = pred_by_cat[cat]
            sources = [str(h.get("grounding_source") or "") for h in cat_hazards]
            if any(source_is_fallback(src) for src in sources):
                failure_type = "fallback_grounding_failure"
                short = "fallback source on false-positive category"
            elif any(not h.get("grounding_source") or not h.get("bbox_xyxy") for h in cat_hazards):
                failure_type = "primary_grounding_failure"
                short = "hazard category emitted without a valid final grounded box"
            else:
                failure_type = "reasoning_hallucination"
                short = "hazard reasoner emitted category absent from GT"
            detail.append(
                {
                    "system": system,
                    "failure_type": failure_type,
                    "image_name": img,
                    "event_type": "FP",
                    "category": cat,
                    "gt_categories": "|".join(sorted(gt)),
                    "pred_categories": "|".join(sorted(pred)),
                    "grounding_sources": "|".join(sorted(set(sources))),
                    "hazard_text": " | ".join(str(h.get("hazard", "")) for h in cat_hazards),
                    "short_evidence": short,
                }
            )
    counts = Counter(row["failure_type"] for row in detail)
    total = sum(counts.values())
    summary = [
        {
            "system": system,
            "failure_type": failure,
            "count": count,
            "percent_of_all_f1_errors": fmt(count / total if total else 0.0),
        }
        for failure, count in sorted(counts.items())
    ]
    confusion = []
    for (event, failure), count in sorted(Counter((row["event_type"], row["failure_type"]) for row in detail).items()):
        confusion.append({"system": system, "event_type": event, "failure_type": failure, "count": count})
    return summary, detail, confusion


def build_failure_agreement_support(
    out_dir: Path,
    detail_rows: list[dict[str, Any]],
    double_annotation_path: Path | None,
    iaa_pairs_path: Path,
) -> list[dict[str, Any]]:
    template_rows = []
    for row in detail_rows:
        template = dict(row)
        template["annotator_1_failure_type"] = ""
        template["annotator_2_failure_type"] = ""
        template_rows.append(template)
    write_csv(out_dir / "failure_attribution_double_annotation_template.csv", template_rows)

    rows = []
    if double_annotation_path and double_annotation_path.exists():
        with double_annotation_path.open(newline="", encoding="utf-8") as handle:
            ann_rows = list(csv.DictReader(handle))
        labels1 = [row.get("annotator_1_failure_type", "") for row in ann_rows]
        labels2 = [row.get("annotator_2_failure_type", "") for row in ann_rows]
        rows.append(cohen_kappa_summary(labels1, labels2, "failure_attribution_double_annotation"))
    else:
        rows.append(
            {
                "agreement_source": "failure_attribution_double_annotation",
                "status": "template_created_no_double_annotation_file",
                "n_items": len(template_rows),
                "observed_agreement": "",
                "cohen_kappa": "",
            }
        )

    if iaa_pairs_path.exists():
        with iaa_pairs_path.open(newline="", encoding="utf-8") as handle:
            pairs = list(csv.DictReader(handle))
        if pairs:
            cat_match = [row.get("cat_match") == "True" for row in pairs]
            rows.append(
                {
                    "agreement_source": "existing_gt_box_iaa_matched_pairs",
                    "status": "computed_from_existing_file",
                    "n_items": len(pairs),
                    "observed_agreement": fmt(sum(cat_match) / len(cat_match)),
                    "cohen_kappa": "",
                    "mean_iou": fmt(sum(float(row["iou"]) for row in pairs) / len(pairs)),
                    "mean_iog_a1_ref": fmt(sum(float(row["iog_a1_ref"]) for row in pairs) / len(pairs)),
                    "mean_iop_a1_ref": fmt(sum(float(row["iop_a1_ref"]) for row in pairs) / len(pairs)),
                    "source_path": str(iaa_pairs_path),
                }
            )
    return rows


def cohen_kappa_summary(labels1: list[str], labels2: list[str], source: str) -> dict[str, Any]:
    pairs = [(a, b) for a, b in zip(labels1, labels2) if a or b]
    if not pairs:
        return {
            "agreement_source": source,
            "status": "no_labeled_pairs",
            "n_items": 0,
            "observed_agreement": "",
            "cohen_kappa": "",
        }
    labels = sorted(set(a for a, _ in pairs) | set(b for _, b in pairs))
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    p1 = Counter(a for a, _ in pairs)
    p2 = Counter(b for _, b in pairs)
    expected = sum((p1[label] / n) * (p2[label] / n) for label in labels)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 0.0
    return {
        "agreement_source": source,
        "status": "computed_from_double_annotation_file",
        "n_items": n,
        "observed_agreement": fmt(observed),
        "cohen_kappa": fmt(kappa),
    }


def make_plots(
    out_dir: Path,
    aggregate_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting skipped: {exc}")
        return []

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []

    selected = [
        row for row in aggregate_rows
        if row.get("system") in {"GROVE_full_paper_archived", "GROVE_trace_qwen35_9b_final"}
        or row.get("family") == "baseline_single_pass"
    ]
    selected = sorted(selected, key=lambda r: safe_float(r.get("f1")) or 0.0)[-12:]
    if selected:
        path = plot_dir / "identification_f1_by_system.png"
        horizontal_bar_plot(
            path,
            [row["system"] for row in selected],
            [safe_float(row.get("f1")) or 0.0 for row in selected],
            "Identification F1 by system",
            "Micro-F1 over the local 203-image evaluation set; 95% CIs are in aggregate_metrics.csv.",
            "F1",
            plt,
        )
        written.append(path)

    grounding_selected = sorted(selected, key=lambda r: safe_float(r.get("gt_coverage_iog03")) or 0.0)[-12:]
    if grounding_selected:
        path = plot_dir / "grounding_gt_coverage_by_system.png"
        horizontal_bar_plot(
            path,
            [row["system"] for row in grounding_selected],
            [safe_float(row.get("gt_coverage_iog03")) or 0.0 for row in grounding_selected],
            "Ground-truth box coverage by system",
            "GT coverage uses IoG >= 0.3; stricter IoU metrics are in grounding_metrics.csv.",
            "GT coverage",
            plt,
        )
        written.append(path)

    if ablation_rows:
        path = plot_dir / "ablation_delta_f1.png"
        rows = sorted(ablation_rows, key=lambda r: safe_float(r.get("delta_f1_vs_control")) or 0.0)
        horizontal_bar_plot(
            path,
            [row["ablation"] for row in rows],
            [safe_float(row.get("delta_f1_vs_control")) or 0.0 for row in rows],
            "Ablation effect on identification F1",
            "Delta versus GROVE_trace_qwen35_9b_final; exactness flags are in ablation_results.csv.",
            "Delta F1",
            plt,
            signed=True,
        )
        written.append(path)

    sens = [row for row in threshold_rows if row["threshold_type"] in {"groundingdino_confidence", "openclip_similarity", "iog_success_threshold"}]
    if sens:
        path = plot_dir / "threshold_sensitivity_gt_coverage.png"
        line_plot_threshold(path, sens, plt)
        written.append(path)

    return written


def horizontal_bar_plot(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    subtitle: str,
    xlabel: str,
    plt: Any,
    signed: bool = False,
) -> None:
    colors = ["#A3BEFA" if v >= 0 else "#F0986E" for v in values] if signed else ["#A3BEFA"] * len(values)
    fig_h = max(4.5, 0.38 * len(labels) + 1.6)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor="#FCFCFD")
    ax.set_facecolor("#FFFFFF")
    ax.barh(range(len(labels)), values, color=colors, edgecolor="#2E4780", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([wrap_label(label, 34) for label in labels], fontsize=8, color="#1F2430")
    ax.set_xlabel(xlabel, color="#1F2430")
    ax.grid(axis="x", color="#E6E8F0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    if signed:
        ax.axvline(0, color="#464C55", linewidth=1.0)
    for idx, value in enumerate(values):
        ha = "left" if value >= 0 else "right"
        offset = 0.006 if value >= 0 else -0.006
        ax.text(value + offset, idx, f"{value:.3f}", va="center", ha=ha, fontsize=8, color="#1F2430")
    add_header(fig, title, subtitle)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def line_plot_threshold(path: Path, rows: list[dict[str, Any]], plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#FCFCFD")
    ax.set_facecolor("#FFFFFF")
    colors = {
        "groundingdino_confidence": "#5477C4",
        "openclip_similarity": "#CC6F47",
        "iog_success_threshold": "#71B436",
    }
    for threshold_type, color in colors.items():
        part = sorted([row for row in rows if row["threshold_type"] == threshold_type], key=lambda r: float(r["threshold"]))
        if not part:
            continue
        ax.plot(
            [float(row["threshold"]) for row in part],
            [safe_float(row["gt_coverage_iog03"]) or 0.0 for row in part],
            marker="o",
            linewidth=1.5,
            color=color,
            label=threshold_type.replace("_", " "),
        )
    ax.set_xlabel("Threshold", color="#1F2430")
    ax.set_ylabel("GT coverage", color="#1F2430")
    ax.grid(True, color="#E6E8F0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    add_header(
        fig,
        "Threshold sensitivity for GROVE grounding",
        "GT coverage under primary/fallback confidence filters and IoG success thresholds.",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def wrap_label(label: str, width: int) -> str:
    if len(label) <= width:
        return label
    chunks = []
    current = ""
    for part in label.split("_"):
        token = part if not current else "_" + part
        if len(current) + len(token) > width:
            chunks.append(current)
            current = part
        else:
            current += token
    if current:
        chunks.append(current)
    return "\n".join(chunks)


def add_header(fig: Any, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.975, title, ha="left", va="top", fontsize=13, color="#1F2430", weight="bold")
    fig.text(0.02, 0.925, subtitle, ha="left", va="top", fontsize=9, color="#6F768A")


def write_result_readme(
    out_dir: Path,
    args: argparse.Namespace,
    manifest: list[dict[str, Any]],
    plot_paths: list[Path],
) -> None:
    lines = [
        "# GROVE AI Open Offline Experiment Suite",
        "",
        "This directory was generated from local images, local COCO annotations, and cached local JSONL prediction files. No internet access or model download is required for the evaluation path.",
        "",
        "## Rerun",
        "",
        "```bash",
        f"{args.python_hint} run_ai_open_experiments.py --output-dir {out_dir} --n-bootstrap {args.n_bootstrap}",
        "```",
        "",
        "Use `--help` to override source JSONL files, image directory, ground truth, bootstrap count, or failure double-annotation input.",
        "",
        "## Evaluation Set",
        "",
        f"- Image directory: `{Path(args.root) / args.image_dir}`",
        f"- Ground truth COCO: `{Path(args.root) / args.gt_coco}`",
        "- Images absent from the COCO annotation file are treated as `NO_HAZARD`, matching the paper-era evaluator behavior.",
        "",
        "## Key Outputs",
        "",
        "- `per_image_predictions.csv`: image-level GT, predictions, final hazard JSON, TP/FP/FN category sets for every system.",
        "- `hazard_predictions.csv`: hazard-level category and grounding predictions for every system.",
        "- `per_category_predictions.csv`: image-category binary matrix for paired tests and audits.",
        "- `aggregate_metrics.csv`: identification, grounding, confidence intervals, and provenance fields.",
        "- `ablation_results.csv`: modular ablations with deltas versus the trace-enabled GROVE control.",
        "- `statistical_comparisons.csv`: paired bootstrap and McNemar-style comparison against the best single-pass baseline.",
        "- `failure_attribution_results.csv` and `failure_attribution_detail.csv`: deterministic failure attribution for the paper-facing GROVE row.",
        "- `threshold_sensitivity.csv`: GroundingDINO, OpenCLIP, IoG, and oversized-box sensitivity.",
        "- `plots/`: static PNG summaries.",
        "",
        "## Ablation Rules",
        "",
        "- `ablation_path1_only_no_path2`: uses `pre_reasoning_hazards_json` from the trace-enabled modular run.",
        "- `ablation_path2_only_no_grounding`: uses retained Path 2 KEEP/REVISE final hazards and strips all boxes. This is a lower-bound fallback because rejected/dropped Path 2 candidates are not fully recoverable from cached traces.",
        "- `ablation_no_reconciliation_union`: simple union of Path 1 candidates and final retained hazards.",
        "- `ablation_groundingdino_only`: retains only Path 1 candidates with a GroundingDINO box.",
        "- `ablation_gdino_detr_openclip_stack_only`: retains all grounded Path 1 candidates before Path 2.",
        "- `ablation_caption_only_keyword_proxy`: deterministic taxonomy keyword proxy over cached captions, used because no caption-only VLM rerun/checkpoint is available offline.",
        "- `ablation_no_fallback_grounding_drop`: drops final hazards grounded only by DETR/OpenCLIP fallback.",
        "- `ablation_no_fallback_grounding_boxes_removed`: removes fallback boxes but keeps hazard text/category decisions.",
        "- `ablation_no_path2_stage_correction`: keeps final inclusion decisions while restoring matched pre-Path-2 category/risk labels.",
        "",
        "## Source Manifest",
        "",
    ]
    for row in manifest:
        lines.append(f"- `{row['system']}`: {row['exactness']} from `{row['source_path']}`")
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for path in plot_paths:
            lines.append(f"- `{path.relative_to(out_dir)}`")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the offline GROVE AI Open experiment suite.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--image-dir", default="All_Images", help="Local image directory, relative to root.")
    parser.add_argument(
        "--gt-coco",
        default="final_evaluation_package_2026-05-01/ground_truth/_annotations.coco.json",
        help="COCO ground truth annotations, relative to root.",
    )
    parser.add_argument(
        "--paper-grove-jsonl",
        default="final_evaluation_package_2026-05-01/results/archived_raw_runs/exp01_modular_qwen_results/exp01_run1.jsonl",
        help="Paper-facing full GROVE cached JSONL.",
    )
    parser.add_argument(
        "--trace-grove-jsonl",
        default="sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl",
        help="Trace-enabled modular GROVE JSONL for ablations.",
    )
    parser.add_argument(
        "--qwen-direct-jsonl",
        default="sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen35_9b/exp_e2e_qwen35_9b_run1.jsonl",
        help="Direct Qwen image-only baseline JSONL.",
    )
    parser.add_argument("--output-dir", default="ai_open_results/full_suite_2026-06-08")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--failure-double-annotations", default="", help="Optional CSV with two failure-attribution annotator columns.")
    parser.add_argument("--python-hint", default=".venv-macos/bin/python")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_names = load_image_names(root / args.image_dir, root / args.gt_coco)
    gt_cats, gt_boxes, gt_rows = load_ground_truth(root / args.gt_coco, image_names)
    systems = build_systems(args)
    predictions, manifest = collect_predictions(systems, image_names)

    write_csv(out_dir / "source_manifest.csv", manifest)
    write_csv(out_dir / "ground_truth_annotations.csv", gt_rows)
    gt_matrix_rows = []
    for img in image_names:
        row = {
            "image_name": img,
            "has_hazard": int(bool(gt_cats.get(img))),
            "n_gt_boxes": len(gt_boxes.get(img, [])),
            "categories": "|".join(sorted(gt_cats.get(img, set()))) if gt_cats.get(img) else "NO_HAZARD",
        }
        for cat in CANONICAL_CATEGORIES:
            row[cat] = int(cat in gt_cats.get(img, set()))
        gt_matrix_rows.append(row)
    write_csv(out_dir / "ground_truth_image_category_matrix.csv", gt_matrix_rows)

    id_rows, per_image_rows, per_cat_id_rows, image_cat_rows = evaluate_identification(
        predictions,
        gt_cats,
        image_names,
        args.n_bootstrap,
        args.seed,
    )
    hazard_pred_rows = hazard_rows(predictions, image_names)
    grounding_rows, grounding_cat_rows = evaluate_grounding(
        predictions,
        gt_boxes,
        image_names,
        args.n_bootstrap,
        args.seed,
    )
    aggregate_rows = merge_aggregate_metrics(id_rows, grounding_rows, manifest)
    ablation_rows = build_ablation_results(aggregate_rows, manifest)
    stat_rows = build_statistical_comparisons(
        aggregate_rows,
        manifest,
        predictions,
        gt_cats,
        gt_boxes,
        image_names,
        args.n_bootstrap,
        args.seed,
    )
    threshold_rows = build_threshold_sensitivity("GROVE_full_paper_archived", predictions, gt_boxes, image_names)
    failure_summary, failure_detail, failure_confusion = build_failure_attribution(
        "GROVE_full_paper_archived",
        predictions,
        root / args.paper_grove_jsonl,
        gt_cats,
        image_names,
    )
    agreement_rows = build_failure_agreement_support(
        out_dir,
        failure_detail,
        Path(args.failure_double_annotations).resolve() if args.failure_double_annotations else None,
        root / "iaa" / "iaa_matched_pairs.csv",
    )

    write_csv(out_dir / "per_image_predictions.csv", per_image_rows)
    write_csv(out_dir / "hazard_predictions.csv", hazard_pred_rows)
    write_csv(out_dir / "per_category_predictions.csv", image_cat_rows)
    write_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(out_dir / "per_category_identification_metrics.csv", per_cat_id_rows)
    write_csv(out_dir / "grounding_metrics.csv", grounding_rows)
    write_csv(out_dir / "grounding_per_category_metrics.csv", grounding_cat_rows)
    write_csv(out_dir / "ablation_results.csv", ablation_rows)
    write_csv(out_dir / "statistical_comparisons.csv", stat_rows)
    write_csv(out_dir / "threshold_sensitivity.csv", threshold_rows)
    write_csv(out_dir / "failure_attribution_results.csv", failure_summary)
    write_csv(out_dir / "failure_attribution_detail.csv", failure_detail)
    write_csv(out_dir / "failure_confusion_breakdown.csv", failure_confusion)
    write_csv(out_dir / "failure_attribution_agreement.csv", agreement_rows)

    write_markdown_table(
        out_dir / "aggregate_metrics.md",
        sorted(aggregate_rows, key=lambda r: safe_float(r.get("f1")) or 0.0, reverse=True),
        ["system", "family", "precision", "recall", "f1", "f1_ci95_low", "f1_ci95_high", "gt_coverage_iog03", "iou_at_0_5", "no_box_rate", "exactness"],
    )
    write_markdown_table(
        out_dir / "ablation_results.md",
        ablation_rows,
        ["ablation", "f1", "delta_f1_vs_control", "gt_coverage_iog03", "delta_gt_coverage_iog03_vs_control", "no_box_rate", "exactness"],
    )
    write_markdown_table(
        out_dir / "statistical_comparisons.md",
        stat_rows,
        ["reference_system", "comparator_system", "metric", "paired_diff_reference_minus_comparator", "diff_ci95_low", "diff_ci95_high", "paired_bootstrap_p_two_sided", "mcnemar_p_approx"],
    )
    plot_paths = make_plots(out_dir, aggregate_rows, ablation_rows, threshold_rows)
    write_result_readme(out_dir, args, manifest, plot_paths)

    print(f"AI Open experiment suite complete: {out_dir}")
    print(f"Images evaluated: {len(image_names)}")
    print(f"Systems evaluated: {len(systems)}")
    print("Key files:")
    for name in [
        "per_image_predictions.csv",
        "hazard_predictions.csv",
        "aggregate_metrics.csv",
        "ablation_results.csv",
        "statistical_comparisons.csv",
        "failure_attribution_results.csv",
        "threshold_sensitivity.csv",
        "README.md",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
