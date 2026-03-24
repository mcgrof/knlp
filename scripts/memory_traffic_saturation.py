#!/usr/bin/env python3
"""Standalone public runner for Memory-Traffic Saturation in Autoregressive Decode.

This is the public reproduction entrypoint for the cross-GPU decode
characterization that grew out of BPA. It wraps the underlying BPA paper dataset
framework so users can reproduce the standalone result without needing to learn
its internal script layout first.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


TARGET = Path(__file__).resolve().parent / "paper" / "bpa_paper" / "run_dataset.py"


if __name__ == "__main__":
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
