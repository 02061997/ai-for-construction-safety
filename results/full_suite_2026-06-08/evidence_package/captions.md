# Captions for AI Open Evidence Package

Table 1. Full-system and ablation comparison. Identification and grounding metrics are computed on the same 203 local-image evaluation set. Ablation deltas are relative to the trace-enabled GROVE control when exact trace fields are required; proxy or lower-bound rows are labeled in the exactness column.

Table 2. Paired statistical comparison between the archived GROVE full system and the best single-pass baseline (baseline_direct_qwen35_27b). Identification and grounding differences use paired bootstrap resampling over images; the image-category correctness row reports a McNemar-style paired test.

Table 3. Threshold sensitivity for the archived GROVE full system. Rows vary one threshold family at a time and report changes in GT coverage, prediction coverage, IoU, and no-box rate relative to that threshold family's default.

Table 4. Per-category identification and grounding evidence. Support counts identify categories with limited evidence, and low-support rows are flagged when either image support or box support is at or below 10. Confidence intervals are bootstrap 95% intervals over images.

Table 5. Stage-level failure attribution for the archived GROVE full system. Counts are percentages of identification F1 errors and separate caption, reasoning, primary grounding, and fallback grounding failures.

Figure 1. Modular GROVE pipeline and ablation map. The diagram shows the main captioning, reasoning, grounding, verification, correction, and reconciliation modules, with callouts marking which component each key ablation removes or replaces.

Figure 2. Category-level gain heatmap. Deltas compare archived GROVE with the best single-pass baseline; absolute GROVE F1 and GT coverage show where gains are supported by strong within-system performance. Asterisks mark sparse categories.

Figure 3. Grounding trade-off across systems. Bars show GT coverage at IoG >= 0.3 and no-box rate for the full system, baseline, trace control, and key grounding ablations; diamonds show identification F1. This figure checks whether stronger grounding comes from missing boxes, oversized boxes, or fallback-heavy behavior.

Figure 4. Threshold stability and calibration proxy. Because cached predictions do not expose full probabilistic calibration curves for every stage, this defensible alternative varies available threshold families and tracks GT coverage and no-box rate.

Figure 5. Multi-system grounding sensitivity. GT coverage and one-to-one
category-aware box+label F1 are swept across IoG thresholds for GROVE and four
single-pass baselines. Raw IoG can favor broad boxes at strict thresholds;
companion IoU, IoP, and box+label tables are required for interpretation.

Table 6. Completed component ablation. The archived paper-facing GROVE result is
reported separately from the trace-enabled control used for causal component
comparisons. Exact cached, trace-derived, proxy, evaluation-only, and
`NOT_RUN` variants are explicitly labeled.

Table 7. Grounding sensitivity under IoG thresholds. Values report GT coverage
recall for the common 203-image universe; no-box rate is included to expose
coverage obtained by emitting more boxes.

Table 8. Grounding sensitivity under stricter IoU thresholds. Mean IoU and
tightness distinguish close localization from broad-box coverage.

Table 9. Box+label grounding sensitivity. A prediction is correct only when its
canonical category matches and it is greedily matched one-to-one with a ground
truth instance at the stated overlap threshold.
