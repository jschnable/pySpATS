#!/usr/bin/env bash
set -euo pipefail
python3 -m pytest -q tests/test_r_parity.py tests/test_rewrite.py
