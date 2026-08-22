from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


def _isolated_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _artifact_requirement(package: dict[str, Any]) -> str:
    wheels = package.get("wheels", [])
    artifact: dict[str, Any]
    if isinstance(wheels, list) and len(wheels) == 1 and isinstance(wheels[0], dict):
        artifact = wheels[0]
    elif not wheels and isinstance(package.get("sdist"), dict):
        artifact = package["sdist"]
    else:
        raise ValueError(f"package {package.get('name')} must have exactly one locked artifact")
    hashes = artifact.get("hashes")
    if not isinstance(hashes, dict) or not isinstance(hashes.get("sha256"), str):
        raise ValueError(f"package {package.get('name')} has no SHA-256")
    name = package.get("name")
    url = artifact.get("url")
    if not isinstance(name, str) or not isinstance(url, str):
        raise ValueError("locked package name and URL must be strings")
    return f"{name} @ {url}#sha256={hashes['sha256']}"


def _install_requirements(requirements: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix="singing-vocoders-lock-") as temporary_dir:
        requirements_path = Path(temporary_dir) / "requirements.lock"
        requirements_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--isolated",
                "--ignore-installed",
                "--require-hashes",
                "--no-deps",
                "--no-build-isolation",
                "--requirement",
                str(requirements_path),
            ],
            check=True,
            env=_isolated_environment(),
        )


def install(lock_path: Path, project_root: Path) -> None:
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("the win-64 lock requires CPython 3.11 on Windows")
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = lock.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("Python lock contains no packages")
    requirements = [_artifact_requirement(package) for package in packages]
    bootstrap = [
        _artifact_requirement(package)
        for package in packages
        if package.get("name") == "setuptools"
    ]
    if len(bootstrap) != 1:
        raise ValueError("Python lock must contain exactly one setuptools artifact")
    _install_requirements(bootstrap)
    _install_requirements(requirements)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--isolated",
            "--no-deps",
            "--no-build-isolation",
            "--editable",
            str(project_root),
        ],
        check=True,
        env=_isolated_environment(),
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
        env=_isolated_environment(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lock", type=Path)
    parser.add_argument("project_root", type=Path)
    arguments = parser.parse_args()
    install(arguments.lock.resolve(), arguments.project_root.resolve())


if __name__ == "__main__":
    main()
