"""Run the frozen R-reference and independent mathematical checks."""

from pathlib import Path
import pytest

if __name__ == "__main__":
    root = Path(__file__).parent
    raise SystemExit(
        pytest.main(
            ["-q", str(root / "test_r_parity.py"), str(root / "test_rewrite.py")]
        )
    )
