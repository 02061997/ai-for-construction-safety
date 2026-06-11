# GROVE AI Open Evidence Completion Plan

## Scope

This completion pass extends the existing offline 203-image evaluation without
modifying or overwriting the archived paper-facing predictions. New outputs are
derived from the same local image universe, COCO annotations, canonical category
mapping, cached JSONL traces, and deterministic seed (`20260608`).

## Computable From Existing Traces

1. Multi-system IoG sensitivity at thresholds 0.1 through 0.7.
2. Multi-system IoU sensitivity at thresholds 0.3, 0.4, and 0.5.
3. Multi-system IoP sensitivity at thresholds 0.3 and 0.5.
4. One-to-one greedy box+label precision, recall, and F1.
5. Six-category `GROVE-NoOther-EvalOnly` identification and grounding metrics.
6. Six-category per-category metrics and bootstrap confidence intervals.
7. No-hazard false-positive image rates for cached full and baseline systems.
8. Latency aggregation from cached JSONL timing fields.
9. Completed CSV, Markdown, LaTeX, PNG, and PDF evidence artifacts.
10. Paper-ready ablation, sensitivity, and limitations text.

## Requires Inference Reruns

The following variants cannot be derived exactly from existing traces:

1. True fallback-only grounding for every hazard phrase.
2. `GROVE-NoComplianceRules`.
3. `GROVE-NoOther-PromptRerun`.
4. Exact `GROVE-CaptionOnlyReasoning`.
5. Exact modular `GROVE-ImageOnlyReasoning`.
6. Exact `Qwen35-9B-MatchedSchema-SinglePass`.

## Local Execution Audit

- Ollama executable exists, but the Ollama server is not running.
- The required `qwen3.5:9b` model manifest is not present locally.
- The only local Ollama model manifest found is `InternVL3_5:8b`.
- GroundingDINO weights are configured to a missing Windows path.
- No local cached DETR/OpenCLIP model checkpoints were found.
- Therefore, the six exact reruns above will be emitted as `NOT_RUN` rows
  rather than replaced with fabricated or silently substituted results.
- Existing caption-keyword and direct-Qwen rows remain available as explicitly
  labeled proxies, but they are not treated as exact reruns.

## Commands

```bash
.venv-macos/bin/python ai_open_experiments/run_grounding_sensitivity.py \
  --results-dir ai_open_results/full_suite_2026-06-08

.venv-macos/bin/python complete_ai_open_evidence.py \
  --results-dir ai_open_results/full_suite_2026-06-08 \
  --n-bootstrap 1000 \
  --seed 20260608
```

## Expected Outputs

- `grounding_sensitivity_multisystem/*.csv`
- `grounding_sensitivity_multisystem/*.md`
- `grounding_sensitivity_multisystem/*.tex`
- `grounding_sensitivity_multisystem/threshold_sensitivity_multisystem.png`
- `grounding_sensitivity_multisystem/threshold_sensitivity_multisystem.pdf`
- `evidence_package/table_1_component_ablation_completed.*`
- `evidence_package/no_other_ablation.*`
- `evidence_package/no_other_per_category.*`
- `evidence_package/compliance_ablation_nohazard_fp.*`
- `evidence_package/latency_summary.*`
- `evidence_package/paper_ready_sections.*`
- `evidence_package/GROVE_AI_Open_Ablation_Evidence_Package.pdf`
- `evidence_package/completion_report.md`

## Risks And Assumptions

- Cached single-pass boxes are VLM-native, while GROVE boxes are produced by
  the external grounding stack; threshold comparisons therefore evaluate
  observed localization behavior, not identical box-generation mechanisms.
- The archived full GROVE output lacks all internal stage traces. Component
  ablations requiring internals use the trace-enabled Qwen run and are labeled
  `Trace-derived`.
- `GROVE-NoOther-EvalOnly` is an evaluation-universe ablation and does not show
  how the model would redistribute predictions if `Other` were removed from the
  prompt.
- Latency from cached runs reflects the original execution environments and is
  descriptive, not a controlled same-hardware benchmark.
- A LaTeX compiler is not available locally. LaTeX source tables will still be
  generated; the consolidated evidence PDF will be rendered directly from the
  same tables and figures using Matplotlib's PDF backend.
