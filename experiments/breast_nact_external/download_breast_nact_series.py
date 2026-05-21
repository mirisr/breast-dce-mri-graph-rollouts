#!/usr/bin/env python3
"""Download Breast-MRI-NACT-Pilot series from a focused manifest."""
from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.qin_transfer.download_qin_series import main as download_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(download_main())
