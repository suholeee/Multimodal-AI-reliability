"""Shared repository path setup for runnable scripts."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
RESULTS_DIR = REPO_ROOT / "results"
ARCHIVE_DIR = REPO_ROOT / "archive"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
