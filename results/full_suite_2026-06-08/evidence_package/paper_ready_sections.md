## Ablation Study

We evaluated component removals on the same 203-image OSHA-aligned benchmark to
separate the contribution of modular decomposition from grounding and output
format choices. Full GROVE achieved micro F1=0.7591 with 95% CI
[0.7014, 0.8077]. The trace-enabled GROVE control achieved
F1=0.5090. Against that matched control, removing Path 2
produced F1=0.5045 (delta -0.0045), while replacing deterministic
reconciliation with a simple union produced F1=0.5062 (delta
-0.0028). These trace-level differences suggest that Path 2 verification and
reconciliation act as incremental quality-control stages rather than the sole
source of the archived full-system advantage.

GroundingDINO-only achieved GT coverage=0.7328, versus
0.7530 for the matched trace control. Removing cached
fallback boxes produced GT coverage=0.7328 and increased
the no-box rate to 0.1507. This supports the interpretation
that primary grounding explains most cached coverage and fallback recovers
otherwise ungrounded hazards. It does not establish fallback quality as a
standalone grounder: a true fallback-only rerun could not be completed because
the required DETR/OpenCLIP weights were not locally available. Exact
caption-only, modular image-only, no-compliance-rules, and matched-schema Qwen
reruns were also unavailable because the required Qwen 3.5 9B model was absent
locally. We retain their `NOT_RUN` status and treat the existing
caption-keyword and direct-Qwen results only as proxies.

The existing schema-adjacent Qwen 3.5 9B single-pass proxy achieved
F1=0.4736, below the archived full-system value, but prompt and
parser parity cannot be fully established from the packaged April runner. The
six-category evaluation excluding `Other` produced micro F1=0.7841
and macro F1=0.5714. Because this is evaluation-only, it does not
show how predictions would be redistributed under a six-category prompt. In our
benchmark, the results support a cautious modularity interpretation, but they do
not establish a universal limitation of single-pass VLMs.

## Grounding Threshold Sensitivity

IoG is useful for relational construction hazards because a predicted region can
cover the hazard-defining ground-truth region without matching its full extent.
IoG@0.3 is intentionally permissive, so we additionally evaluated IoG through
0.7, IoU at 0.3--0.5, IoP/tightness, and one-to-one box+label matching.

At IoG@0.3, GROVE achieved GT coverage=0.7449, compared
with 0.7409 for Llama 3.2 Vision 11B. At
stricter IoG@0.5, Llama reached 0.7247 while
GROVE reached 0.6275. Thus, the raw IoG ranking changes
at stricter thresholds and should not be presented as a threshold-invariant win.
The complementary metrics clarify the behavior: GROVE achieved IoU@0.5=
0.5101 versus
0.1984 for Llama, and box+label
F1@IoU0.5=0.3027 versus
0.0246. Llama's high IoG but
lower IoU, tightness, and category-aware F1 is consistent with broad boxes that
cover GT regions without localizing them tightly. The results suggest that
GROVE's grounding advantage is strongest under joint localization and label
criteria, not under every category-agnostic IoG threshold.

## Limitations

This evaluation uses 203 images and has substantial class imbalance, especially
for Struck-by and Caught-in/Between hazards. The broad `Other` category remains
heterogeneous. IoG@0.3 is permissive by design, while strict IoU can penalize
semantically valid relational hazard regions; reporting IoG, IoU, IoP, and
box+label metrics together is therefore necessary. Prompt engineering and
decomposition are not perfectly separable, and several component ablations are
trace-derived or evaluation-only. Exact inference reruns requiring unavailable
local checkpoints are explicitly marked `NOT_RUN`. External validation on
larger datasets is needed, and we do not claim a universal single-pass ceiling;
we observe saturation only among the tested open-weight baselines under this
benchmark and evaluation protocol.
