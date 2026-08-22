from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
