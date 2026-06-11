#!/usr/bin/env python3
"""Top-level runner for the GROVE AI Open experiment suite.

This wrapper keeps the command easy to cite:

    python run_ai_open_experiments.py --output-dir ai_open_results/full_suite_2026-06-08
"""

from ai_open_experiments.suite import main


if __name__ == "__main__":
    main()
