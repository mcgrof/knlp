# SPDX-License-Identifier: GPL-2.0
"""Offline, repo-rooted test setup for the Jacobian-KV screen."""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (REPO, os.path.dirname(__file__)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
