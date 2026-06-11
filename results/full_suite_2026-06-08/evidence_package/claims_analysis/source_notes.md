# Claims Analysis Source Notes

## Reporting Contract

- Audience: technical research reviewers and paper authors.
- Delivery mode: static HTML.
- Comparison universe: 203 local images.
- Bootstrap seed and count: inherited from the completed suite, seed 20260608 and 1,000 resamples.

## Controlling Sources

- `../table_1_component_ablation_completed.csv`
- `../table_2_paired_tests.csv`
- `../table_4_per_category_performance.csv`
- `../compliance_ablation_nohazard_fp.csv`
- `../no_other_ablation.csv`
- `../../grounding_sensitivity_multisystem/iog_sensitivity.csv`
- `../../grounding_sensitivity_multisystem/iou_sensitivity.csv`
- `../../grounding_sensitivity_multisystem/box_label_sensitivity.csv`

## Chart Map

1. `paired_differences.png`
   - Family: uncertainty and benchmark; experiment-lift interval plot.
   - Question: Which paired GROVE gains remain positive under bootstrap uncertainty?
   - Claim: Identification and strict grounding gains exclude zero.
2. `category_evidence.png`
   - Family: comparison and ranking; horizontal bars.
   - Question: Where are category gains supported by enough examples?
   - Claim: Supported-category gains are broad; two categories are underpowered.
3. `threshold_robustness.png`
   - Family: highlighted multi-series trend.
   - Question: Does the grounding conclusion depend on overlap metric and threshold?
   - Claim: Raw IoG ranking changes, while strict IoU favors GROVE.

## Validation Notes

- The report distinguishes archived full-system comparisons from trace-control ablations.
- No `NOT_RUN` experiment is assigned a metric.
- No-compliance, fallback-only, exact caption-only, modular image-only, matched-schema, and prompt-level no-Other claims remain unresolved.
- Latency is omitted from the claims memo because original runs were not controlled on identical hardware.
- The HTML report was visually inspected after rendering.
