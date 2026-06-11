# GROVE AI Open Evidence Completion Report

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
- Consolidated evidence PDF status: Generated and non-empty.
- PDF renderer: local headless Chrome over the static HTML report; equivalent
  LaTeX source and table fragments are provided because no local TeX engine was
  available.

## NOT_RUN Experiments

- `GROVE-FallbackOnly`: True fallback-only grounding requires rerunning every hazard phrase through DETR and OpenCLIP. The required facebook/detr-resnet-50 and OpenCLIP ViT-B-32 weights are not cached locally, so an exact offline rerun is unavailable.
- `GROVE-CaptionOnlyReasoning`: Required qwen3.5:9b model is absent from local Ollama manifests; the Ollama server is not running and internet/model downloads are disallowed.
- `GROVE-ImageOnlyReasoning`: Required qwen3.5:9b model is absent from local Ollama manifests; the Ollama server is not running and internet/model downloads are disallowed.
- `Qwen35-9B-MatchedSchema-SinglePass`: Required qwen3.5:9b model is absent from local Ollama manifests; the Ollama server is not running and internet/model downloads are disallowed.
- `GROVE-NoComplianceRules`: Required qwen3.5:9b model is absent from local Ollama manifests; the Ollama server is not running and internet/model downloads are disallowed.
- `GROVE-NoOther-PromptRerun`: Required qwen3.5:9b model is absent from local Ollama manifests; the Ollama server is not running and internet/model downloads are disallowed.

No metrics were fabricated for these rows.

## Reproduction Commands

```bash
.venv-macos/bin/python ai_open_experiments/run_grounding_sensitivity.py   --results-dir ai_open_results/full_suite_2026-06-08

.venv-macos/bin/python complete_ai_open_evidence.py   --results-dir ai_open_results/full_suite_2026-06-08   --n-bootstrap 1000   --seed 20260608
```

## Paper-Number Comparability

The archived full-system and baseline rows are unchanged. New box+label F1
values use one-to-one greedy canonical-category matching and therefore should
not be substituted for differently defined legacy category-aware coverage
metrics without updating the metric label.

## Strict-Threshold Result

- At IoG@0.3, GROVE=0.7449 and Llama 3.2 Vision
  11B=0.7409.
- At IoG@0.5, GROVE=0.6275 and Llama=
  0.7247; Full GROVE does not win every
  category-agnostic IoG threshold.
- At IoU@0.5, GROVE=0.5101 and Llama=
  0.1984.
- At box+label F1@IoU0.5, GROVE=
  0.3027 and Llama=
  0.0246.

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

- `evidence_package/GROVE_AI_Open_Ablation_Evidence_Package.html`
- `evidence_package/GROVE_AI_Open_Ablation_Evidence_Package.pdf`
- `evidence_package/GROVE_AI_Open_Ablation_Evidence_Package.tex`
- `evidence_package/README.md`
- `evidence_package/captions.md`
- `evidence_package/claim_traceability_matrix.csv`
- `evidence_package/claim_traceability_matrix.md`
- `evidence_package/claim_traceability_matrix_completed.csv`
- `evidence_package/claim_traceability_matrix_completed.md`
- `evidence_package/completion_plan.md`
- `evidence_package/completion_report.md`
- `evidence_package/compliance_ablation_nohazard_fp.csv`
- `evidence_package/compliance_ablation_nohazard_fp.md`
- `evidence_package/compliance_ablation_nohazard_fp.tex`
- `evidence_package/component_ablation_completed.md`
- `evidence_package/component_ablation_completed.tex`
- `evidence_package/evidence_index.csv`
- `evidence_package/figures/figure_1_modular_ablation_map.pdf`
- `evidence_package/figures/figure_1_modular_ablation_map.png`
- `evidence_package/figures/figure_2_category_metric_gain_heatmap.pdf`
- `evidence_package/figures/figure_2_category_metric_gain_heatmap.png`
- `evidence_package/figures/figure_3_no_box_grounding_tradeoff.pdf`
- `evidence_package/figures/figure_3_no_box_grounding_tradeoff.png`
- `evidence_package/figures/figure_4_threshold_stability.pdf`
- `evidence_package/figures/figure_4_threshold_stability.png`
- `evidence_package/figures/figure_5_threshold_sensitivity_multisystem.pdf`
- `evidence_package/figures/figure_5_threshold_sensitivity_multisystem.png`
- `evidence_package/latency_summary.csv`
- `evidence_package/latency_summary.md`
- `evidence_package/latency_summary.tex`
- `evidence_package/no_other_ablation.csv`
- `evidence_package/no_other_ablation.md`
- `evidence_package/no_other_ablation.tex`
- `evidence_package/no_other_per_category.csv`
- `evidence_package/no_other_per_category.md`
- `evidence_package/no_other_per_category.tex`
- `evidence_package/paper_ready_sections.md`
- `evidence_package/paper_ready_sections.tex`
- `evidence_package/table_1_component_ablation_completed.csv`
- `evidence_package/table_1_component_ablation_completed.md`
- `evidence_package/table_1_component_ablation_completed.tex`
- `evidence_package/table_1_full_vs_key_ablations.csv`
- `evidence_package/table_1_full_vs_key_ablations.md`
- `evidence_package/table_1_full_vs_key_ablations_completed.csv`
- `evidence_package/table_2_paired_tests.csv`
- `evidence_package/table_2_paired_tests.md`
- `evidence_package/table_3_threshold_sensitivity.csv`
- `evidence_package/table_3_threshold_sensitivity.md`
- `evidence_package/table_4_per_category_performance.csv`
- `evidence_package/table_4_per_category_performance.md`
- `evidence_package/table_5_failure_attribution.csv`
- `evidence_package/table_5_failure_attribution.md`
- `grounding_sensitivity_multisystem/box_label_sensitivity.csv`
- `grounding_sensitivity_multisystem/box_label_sensitivity.md`
- `grounding_sensitivity_multisystem/box_label_sensitivity.tex`
- `grounding_sensitivity_multisystem/grounding_sensitivity_detailed.csv`
- `grounding_sensitivity_multisystem/grounding_sensitivity_notes.md`
- `grounding_sensitivity_multisystem/iog_sensitivity.csv`
- `grounding_sensitivity_multisystem/iog_sensitivity.md`
- `grounding_sensitivity_multisystem/iog_sensitivity.tex`
- `grounding_sensitivity_multisystem/iop_sensitivity.csv`
- `grounding_sensitivity_multisystem/iop_sensitivity.md`
- `grounding_sensitivity_multisystem/iop_sensitivity.tex`
- `grounding_sensitivity_multisystem/iou_sensitivity.csv`
- `grounding_sensitivity_multisystem/iou_sensitivity.md`
- `grounding_sensitivity_multisystem/iou_sensitivity.tex`
- `grounding_sensitivity_multisystem/threshold_sensitivity_multisystem.pdf`
- `grounding_sensitivity_multisystem/threshold_sensitivity_multisystem.png`
- `new_ablation_traces/caption_only_reasoning/NOT_RUN.md`
- `new_ablation_traces/fallback_only/NOT_RUN.md`
- `new_ablation_traces/image_only_reasoning/NOT_RUN.md`
- `new_ablation_traces/matched_schema_single_pass/NOT_RUN.md`
- `new_ablation_traces/no_compliance_rules/NOT_RUN.md`
- `new_ablation_traces/no_other_prompt_rerun/NOT_RUN.md`
