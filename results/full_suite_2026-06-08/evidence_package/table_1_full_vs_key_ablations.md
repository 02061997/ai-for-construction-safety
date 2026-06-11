| label | role | precision | recall | f1 | f1_ci95 | gt_coverage_iog03 | iou_at_0_5 | no_box_rate | exactness |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GROVE full system | Primary system | 0.7261 | 0.7952 | 0.7591 | [0.7014, 0.8077] | 0.7449 | 0.5101 | 0.0036 | exact_cached_full_system |
| GROVE trace control | Ablation control | 0.4080 | 0.6762 | 0.5090 | [0.4481, 0.5629] | 0.7530 | 0.4980 | 0.0120 | exact_cached_trace_full_system |
| Best single-pass baseline | Fairness baseline | 0.5142 | 0.6048 | 0.5558 | [0.4958, 0.6065] | 0.4696 | 0.2267 | 0.3220 | exact_cached_baseline |
| Path 1 only | Ablation | 0.4040 | 0.6714 | 0.5045 | [0.4457, 0.5574] | 0.7530 | 0.4980 | 0.0120 | exact_from_trace_pre_reasoning |
| Path 2 only | Ablation | 0.4444 | 0.6476 | 0.5271 | [0.4665, 0.5821] | 0.0000 | 0.0000 | 1.0000 | posthoc_lower_bound_trace_retained_only |
| No reconciliation | Ablation | 0.4046 | 0.6762 | 0.5062 | [0.4457, 0.5593] | 0.7530 | 0.4980 | 0.0119 | posthoc_union_rule |
| GroundingDINO only | Ablation | 0.4323 | 0.6238 | 0.5107 | [0.4524, 0.5635] | 0.7328 | 0.4858 | 0.0000 | posthoc_source_filter |
| GDINO + DETR/OpenCLIP stack | Ablation | 0.4087 | 0.6714 | 0.5081 | [0.4492, 0.5619] | 0.7530 | 0.4980 | 0.0000 | exact_from_trace_grounded_pre_reasoning |
| Qwen image-only, no grounding | Ablation | 0.3835 | 0.6190 | 0.4736 | [0.4219, 0.5229] | 0.0000 | 0.0000 | 1.0000 | exact_cached_direct_identification_posthoc_no_boxes |
| Caption-only keyword proxy | Ablation | 0.1637 | 0.9762 | 0.2804 | [0.2392, 0.3163] | 0.0000 | 0.0000 | 1.0000 | fallback_caption_keyword_proxy |
| Image-only reasoning | Ablation | 0.3835 | 0.6190 | 0.4736 | [0.4219, 0.5229] | 0.3887 | 0.1417 | 0.3928 | exact_cached_direct |
| No fallback, drop hazards | Ablation | 0.4314 | 0.6286 | 0.5116 | [0.4497, 0.5637] | 0.7328 | 0.4858 | 0.0139 | posthoc_source_filter |
| No fallback, strip boxes | Ablation | 0.4080 | 0.6762 | 0.5090 | [0.4481, 0.5629] | 0.7328 | 0.4858 | 0.1507 | posthoc_box_filter |
| No Path 2 verification | Ablation | 0.4040 | 0.6714 | 0.5045 | [0.4457, 0.5574] | 0.7530 | 0.4980 | 0.0120 | exact_from_trace_pre_reasoning |
| No Path 2 stage correction | Ablation | 0.4080 | 0.6762 | 0.5090 | [0.4481, 0.5629] | 0.7530 | 0.4980 | 0.0120 | posthoc_text_match |
| Grounded final only | Ablation | 0.4128 | 0.6762 | 0.5126 | [0.4517, 0.5674] | 0.7530 | 0.4980 | 0.0000 | posthoc_grounding_filter |
