from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

WINDOWS_GLOO_REGRESSION = sys.platform == "win32" and torch.__version__.startswith(("2.8", "2.9"))


@pytest.mark.skipif(
    WINDOWS_GLOO_REGRESSION,
    reason="PyTorch 2.8-2.9 Windows wheels have no usable Gloo transport",
)
def test_two_process_cpu_gloo_engine(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    marker = tmp_path / "rank-zero.txt"
    subprocess.run(
        [
            sys.executable,
            str(root / "tests" / "ddp_engine_smoke.py"),
            str(marker),
        ],
        cwd=root,
        check=True,
        timeout=120,
    )
    assert marker.read_text(encoding="utf-8") == "ok\n"
