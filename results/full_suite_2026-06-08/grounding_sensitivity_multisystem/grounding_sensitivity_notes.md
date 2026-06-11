# Multi-System Grounding Sensitivity Notes

## Scope

All values are recomputed offline from cached local JSONL predictions on the
same 203-image universe and local COCO annotations used by the main suite.

## Selected Systems

- `GROVE_full_paper_archived`: GROVE full; Named comparison
- `baseline_direct_qwen35_9b`: Qwen 3.5 9B single-pass; Named comparison
- `baseline_direct_qwen35_27b`: Qwen 3.5 27B single-pass; Named comparison; Best single-pass by identification F1
- `baseline_direct_llama32_vision_11b_apr2026`: Llama 3.2 Vision 11B; Named comparison; Best single-pass by GT coverage
- `baseline_direct_gemma4_31b`: Gemma 4 31B; Named comparison

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
