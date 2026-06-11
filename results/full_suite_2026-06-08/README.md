# GROVE AI Open Offline Experiment Suite

This directory was generated from local images, local COCO annotations, and cached local JSONL prediction files. No internet access or model download is required for the evaluation path.

## Rerun

```bash
.venv-macos/bin/python run_ai_open_experiments.py --output-dir <LOCAL_EVIDENCE_ROOT>/ai_open_results/full_suite_2026-06-08 --n-bootstrap 1000
```

Use `--help` to override source JSONL files, image directory, ground truth, bootstrap count, or failure double-annotation input.

## Evaluation Set

- Image directory: `All_Images`
- Ground truth COCO: `final_evaluation_package_2026-05-01/ground_truth/_annotations.coco.json`
- Images absent from the COCO annotation file are treated as `NO_HAZARD`, matching the paper-era evaluator behavior.

## Key Outputs

- `per_image_predictions.csv`: image-level GT, predictions, final hazard JSON, TP/FP/FN category sets for every system.
- `hazard_predictions.csv`: hazard-level category and grounding predictions for every system.
- `per_category_predictions.csv`: image-category binary matrix for paired tests and audits.
- `aggregate_metrics.csv`: identification, grounding, confidence intervals, and provenance fields.
- `ablation_results.csv`: modular ablations with deltas versus the trace-enabled GROVE control.
- `statistical_comparisons.csv`: paired bootstrap and McNemar-style comparison against the best single-pass baseline.
- `failure_attribution_results.csv` and `failure_attribution_detail.csv`: deterministic failure attribution for the paper-facing GROVE row.
- `threshold_sensitivity.csv`: GroundingDINO, OpenCLIP, IoG, and oversized-box sensitivity.
- `plots/`: static PNG summaries.

## Ablation Rules

- `ablation_path1_only_no_path2`: uses `pre_reasoning_hazards_json` from the trace-enabled modular run.
- `ablation_path2_only_no_grounding`: uses retained Path 2 KEEP/REVISE final hazards and strips all boxes. This is a lower-bound fallback because rejected/dropped Path 2 candidates are not fully recoverable from cached traces.
- `ablation_no_reconciliation_union`: simple union of Path 1 candidates and final retained hazards.
- `ablation_groundingdino_only`: retains only Path 1 candidates with a GroundingDINO box.
- `ablation_gdino_detr_openclip_stack_only`: retains all grounded Path 1 candidates before Path 2.
- `ablation_caption_only_keyword_proxy`: deterministic taxonomy keyword proxy over cached captions, used because no caption-only VLM rerun/checkpoint is available offline.
- `ablation_no_fallback_grounding_drop`: drops final hazards grounded only by DETR/OpenCLIP fallback.
- `ablation_no_fallback_grounding_boxes_removed`: removes fallback boxes but keeps hazard text/category decisions.
- `ablation_no_path2_stage_correction`: keeps final inclusion decisions while restoring matched pre-Path-2 category/risk labels.

## Source Manifest

- `GROVE_full_paper_archived`: exact_cached_full_system from `<LOCAL_EVIDENCE_ROOT>/final_evaluation_package_2026-05-01/results/archived_raw_runs/exp01_modular_qwen_results/exp01_run1.jsonl`
- `GROVE_trace_qwen35_9b_final`: exact_cached_trace_full_system from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `baseline_direct_gemma4_26b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_gemma4_26b/exp_e2e_gemma4_26b_run1.jsonl`
- `baseline_direct_gemma4_31b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_gemma4_31b/exp_e2e_gemma4_31b_run1.jsonl`
- `baseline_direct_gemma4_e2b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_gemma4_e2b/exp_e2e_gemma4_e2b_run1.jsonl`
- `baseline_direct_gemma4_e4b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_gemma4_e4b/exp_e2e_gemma4_e4b_run1.jsonl`
- `baseline_direct_llama32_vision_11b_apr2026`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_llama32_vision_11b_apr2026/exp_e2e_llama32_vision_11b_apr2026_run1.jsonl`
- `baseline_direct_qwen35_27b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen35_27b/exp_e2e_qwen35_27b_run1.jsonl`
- `baseline_direct_qwen35_2b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen35_2b/exp_e2e_qwen35_2b_run1.jsonl`
- `baseline_direct_qwen35_4b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen35_4b/exp_e2e_qwen35_4b_run1.jsonl`
- `baseline_direct_qwen35_9b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen35_9b/exp_e2e_qwen35_9b_run1.jsonl`
- `baseline_direct_qwen36_35b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen36_35b/exp_e2e_qwen36_35b_run1.jsonl`
- `baseline_direct_gemma3_12b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/code_files/exp_e2e_gemma3_12b/results/exp_e2e_gemma3_12b_run1.jsonl`
- `baseline_direct_gemma3_4b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/code_files/exp_e2e_gemma3_4b/results/exp_e2e_gemma3_4b_run1.jsonl`
- `baseline_direct_internvl3_5_8b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/code_files/exp_e2e_internvl3_5_8b/results/exp_e2e_internvl3_5_8b_run1.jsonl`
- `baseline_direct_llama32_vision_11b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/code_files/exp_e2e_llama32_vision_11b/results/exp_e2e_llama32_vision_11b_run1.jsonl`
- `baseline_direct_qwen_35b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/code_files/exp_e2e_qwen_35b/results/exp_e2e_qwen_35b_run1.jsonl`
- `baseline_direct_qwen_4b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/code_files/exp_e2e_qwen_4b/results/exp_e2e_qwen_4b_run1.jsonl`
- `baseline_direct_qwen_9b`: exact_cached_baseline from `<LOCAL_EVIDENCE_ROOT>/code_files/exp_e2e_qwen_9b/results/exp_e2e_qwen_9b_run1.jsonl`
- `ablation_path1_only_no_path2`: exact_from_trace_pre_reasoning from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_path2_only_no_grounding`: posthoc_lower_bound_trace_retained_only from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_no_reconciliation_union`: posthoc_union_rule from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_groundingdino_only`: posthoc_source_filter from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_gdino_detr_openclip_stack_only`: exact_from_trace_grounded_pre_reasoning from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_qwen_image_only_no_grounding`: exact_cached_direct_identification_posthoc_no_boxes from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen35_9b/exp_e2e_qwen35_9b_run1.jsonl`
- `ablation_caption_only_keyword_proxy`: fallback_caption_keyword_proxy from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_image_only_reasoning`: exact_cached_direct from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_e2e_qwen35_9b/exp_e2e_qwen35_9b_run1.jsonl`
- `ablation_no_fallback_grounding_drop`: posthoc_source_filter from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_no_fallback_grounding_boxes_removed`: posthoc_box_filter from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_no_path2_verification`: exact_from_trace_pre_reasoning from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_no_path2_stage_correction`: posthoc_text_match from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`
- `ablation_grounded_final_only`: posthoc_grounding_filter from `<LOCAL_EVIDENCE_ROOT>/sitecortex_results_2026-04-24/raw_results/exp_modular_qwen35_9b/exp_modular_qwen35_9b_run1.jsonl`

## Plots

- `plots/identification_f1_by_system.png`
- `plots/grounding_gt_coverage_by_system.png`
- `plots/ablation_delta_f1.png`
- `plots/threshold_sensitivity_gt_coverage.png`
