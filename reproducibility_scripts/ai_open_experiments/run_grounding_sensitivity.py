#!/usr/bin/env python3
"""Required entry point for the GROVE multi-system grounding sensitivity run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_open_experiments.grounding_sensitivity import main


if __name__ == "__main__":
    main()
