# GROVE AI Open Co-Author Handoff

Prepared: June 11, 2026

This package contains the complete paper-facing experiment outputs generated
for:

> GROVE: Grounded OSHA Violation Evaluator for Construction Hazard Detection
> and Spatial Grounding

The package is intended for integrating the new evaluation evidence into the
AI Open manuscript. It contains all evaluated result CSVs, publication-ready
tables and figures, paper-ready text, claim guidance, and the scripts used to
generate the evidence.

## Start Here

1. Open `COAUTHOR_PROMPT.md` and use it together with the current manuscript.
2. Read `results/full_suite_2026-06-08/evidence_package/claims_analysis/strongest_defensible_claims.html`.
3. Read `results/full_suite_2026-06-08/evidence_package/completion_report.md`.
4. Use `results/full_suite_2026-06-08/evidence_package/paper_ready_sections.tex`
   as a drafting base, adapting it to the manuscript's notation and structure.
5. Copy final numbers from CSV or LaTeX tables, not from screenshots.

## Most Important Files

- `evidence_package/table_1_component_ablation_completed.csv`
  - Full system, trace control, key component removals, proxies, evaluation-only
    analyses, and explicit `NOT_RUN` experiments.
- `evidence_package/table_2_paired_tests.csv`
  - Paired bootstrap and McNemar-style comparisons against the best baseline.
- `evidence_package/table_4_per_category_performance.csv`
  - Category support, confidence intervals, low-support flags, and grounding.
- `grounding_sensitivity_multisystem/`
  - Multi-system IoG, IoU, IoP, and box-plus-label threshold sensitivity.
- `evidence_package/table_5_failure_attribution.csv`
  - Stage-level failure counts and percentages.
- `evidence_package/compliance_ablation_nohazard_fp.csv`
  - No-hazard false-positive image rates for available systems.
- `evidence_package/no_other_ablation.csv`
  - Seven-category versus six-category evaluation-only sensitivity.
- `evidence_package/latency_summary.csv`
  - Timing aggregated from original cached runs.
- `evidence_package/figures/`
  - Publication-ready PNG and PDF figures.
- `evidence_package/captions.md`
  - Captions ready to adapt for the manuscript.
- `evidence_package/claims_analysis/defensible_claims.csv`
  - Claim strength, required boundary, and controlling evidence.

## Headline Results

- Full GROVE identification: precision `0.7261`, recall `0.7952`, F1
  `0.7591`, 95% CI `[0.7014, 0.8077]`.
- Best single-pass identification baseline, Qwen 3.5 27B: F1 `0.5558`.
- Paired F1 difference: `+0.2047`, 95% CI `[0.1353, 0.2728]`,
  `p < 0.001`.
- GROVE IoU@0.5: `0.5101`; Qwen 3.5 27B IoU@0.5: `0.2267`.
- Paired IoU@0.5 difference: `+0.2838`, 95% CI `[0.2016, 0.3658]`.
- GROVE box-plus-label F1@IoU0.5: `0.3027`.
- GroundingDINO-only retains `97.3%` of matched trace-control GT coverage.
- Removing fallback boxes raises the no-box rate from `0.0120` to `0.1507`.
- Path 2 contributes `+0.0045` F1 and reconciliation contributes `+0.0028`
  F1 relative to the matched trace-enabled control.

## Required Caution

Do not claim:

- a universal single-pass VLM performance ceiling;
- that fallback-only grounding was validated as a primary grounder;
- that compliance rules causally reduced false positives;
- that exact caption-only or modular image-only ablations were completed;
- that matched output schema was ruled out as an explanation;
- strong category-level conclusions for Struck-by (`n=3`) or
  Caught-in/Between (`n=10`).

Exact fallback-only, caption-only, modular image-only, matched-schema
single-pass, no-compliance-rules, and no-Other prompt reruns are explicitly
`NOT_RUN` because the required local model checkpoints were unavailable.
No values were fabricated for these rows.

## Suggested Paper Placement

Main paper:

1. Completed component-ablation table, shortened to the most important rows.
2. Paired statistical comparison table.
3. Multi-system grounding-threshold sensitivity figure.
4. Supported-category performance table or compact figure.
5. Ablation Study, Grounding Threshold Sensitivity, and Limitations text.

Supplement:

1. Full 17-row ablation table with provenance labels.
2. Complete IoG, IoU, IoP, and box-plus-label tables.
3. Failure attribution.
4. No-hazard false-positive analysis.
5. No-Other evaluation-only analysis.
6. Latency table, noting that runs were not controlled on identical hardware.
7. Complete claim traceability matrix.

## Package Boundary

This handoff includes the complete generated results directory and all
reproducibility scripts. It does not include the approximately 180 MB of source
images and archived raw model runs. Those remain in the original research
workspace and are not needed for manuscript integration.
