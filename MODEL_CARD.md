# Model Card

## Scope

This repository is an evaluation and evidence companion for GROVE, a grounded
construction-hazard analysis system. It is not a deployable safety product.

## Evidence

The public package contains aggregate metrics, ablations, bootstrap intervals,
threshold analyses, and failure attribution derived from permitted cached
outputs. It does not contain the restricted source-image dataset or model
weights.

## Intended use

- Reproduce evidence tables and claim-level traceability.
- Inspect evaluation methodology and failure modes.
- Support research review of grounded vision-language evaluation.

## Out-of-scope use

- Automated compliance decisions.
- Worker monitoring or disciplinary action.
- Safety-critical deployment without qualified human review.

## Limitations

Performance is dataset- and threshold-dependent. Cached-output reproduction
does not verify model-serving behavior, hardware latency, or generalization to
new sites. Proxy and post-hoc ablations are labeled in the evidence package.
