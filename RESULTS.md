# Results

Verified locally on June 15, 2026 with:

```bash
uv sync --frozen
make test
make reproduce-smoke
make reproduce-results
```

The committed evidence package reports the archived paper-facing GROVE run at
precision 0.7261, recall 0.7952, and F1 0.7591 on 203 evaluated examples.

These values are reproduced from the included cached evidence, not from a new
model-inference run. The repository's claim report distinguishes computed,
cached, post-hoc, proxy, and unavailable evidence. Local source paths have been
sanitized as `<LOCAL_EVIDENCE_ROOT>`.

Run:

```bash
make reproduce-results
```

Then inspect `results/full_suite_2026-06-08/evidence_package/` and the generated
claim report. Restricted source images, copyrighted manuscripts, and model
weights are not distributed.
