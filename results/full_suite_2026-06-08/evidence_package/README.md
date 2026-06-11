# GROVE AI Open Evidence Package

This directory contains reviewer-facing evidence derived from the completed local experiment suite in:

`<LOCAL_EVIDENCE_ROOT>/ai_open_results/full_suite_2026-06-08`

The completion scripts read local cached traces, recompute derived metrics with
fixed seeds, render publication-ready figures, and write CSV, Markdown, LaTeX,
HTML, and PDF artifacts. No internet access or model downloads are used.

## Reproduce

From the repository root:

```bash
.venv-macos/bin/python run_ai_open_experiments.py --output-dir <LOCAL_EVIDENCE_ROOT>/ai_open_results/full_suite_2026-06-08 --n-bootstrap 1000 --seed 20260608
.venv-macos/bin/python generate_ai_open_evidence_package.py --results-dir <LOCAL_EVIDENCE_ROOT>/ai_open_results/full_suite_2026-06-08
.venv-macos/bin/python ai_open_experiments/run_grounding_sensitivity.py --results-dir <LOCAL_EVIDENCE_ROOT>/ai_open_results/full_suite_2026-06-08
.venv-macos/bin/python complete_ai_open_evidence.py --results-dir <LOCAL_EVIDENCE_ROOT>/ai_open_results/full_suite_2026-06-08 --n-bootstrap 1000 --seed 20260608
```

The best single-pass baseline used for paired comparisons is `baseline_direct_qwen35_27b`.

## Main Artifacts

- `table_1_full_vs_key_ablations.csv`: full system, best baseline, and key ablations.
- `table_2_paired_tests.csv`: paired bootstrap and McNemar-style tests.
- `table_3_threshold_sensitivity.csv`: threshold stress tests.
- `table_4_per_category_performance.csv`: support counts, low-support flags, per-category CIs, and category grounding.
- `table_5_failure_attribution.csv`: failure attribution counts and percentages.
- `claim_traceability_matrix.csv`: manuscript claims mapped to exact tables and figures.
- `table_1_component_ablation_completed.csv`: final exact/trace/proxy/eval-only/NOT_RUN component table.
- `claim_traceability_matrix_completed.csv`: updated claim-to-evidence mapping.
- `latency_summary.csv`: cached original-run timing aggregation.
- `GROVE_AI_Open_Ablation_Evidence_Package.pdf`: consolidated seven-page evidence report.
- `paper_ready_sections.tex`: cautious manuscript-ready ablation, sensitivity, and limitations text.
- `captions.md`: concise table and figure captions.
- `figures/`: publication-ready PNG figures.

## Important Caveats

- The archived paper-facing GROVE run is treated as the primary system.
- Internal ablations that require stage traces use the trace-enabled GROVE run and report deltas against that trace control.
- The archived full-system row and trace-enabled control are both shown; trace-derived removals must be compared with the trace control.
- The Path 2-only row is a lower-bound alternative because rejected/dropped Path 2 candidates are not fully recoverable from cached traces.
- The caption-only row is a deterministic keyword proxy over cached captions, not a fresh VLM inference run.
- Threshold and oversized-box analyses are post hoc stress tests over cached predictions.
- Exact fallback-only, caption-only, modular image-only, matched-schema single-pass, no-compliance, and no-Other prompt reruns are `NOT_RUN` because required local checkpoints are unavailable.
- Latencies reflect original execution environments and are not a controlled same-hardware benchmark.
- The PDF is rendered locally from the static HTML report with headless Chrome;
  equivalent LaTeX source and table fragments are included because no TeX
  compiler is installed in the offline environment.

## Output Directory

`<LOCAL_EVIDENCE_ROOT>/ai_open_results/full_suite_2026-06-08/evidence_package`
