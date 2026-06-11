.PHONY: test reproduce-smoke reproduce-results
test:
	PYTHONPATH=reproducibility_scripts uv run python -m compileall -q reproducibility_scripts
reproduce-smoke:
	PYTHONPATH=reproducibility_scripts uv run python reproducibility_scripts/generate_grove_claims_report.py --help
reproduce-results:
	PYTHONPATH=reproducibility_scripts uv run python reproducibility_scripts/generate_grove_claims_report.py --results-dir results/full_suite_2026-06-08
