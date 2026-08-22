import os
import shutil
import subprocess
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from click.testing import CliRunner

from singing_vocoders import __version__
from singing_vocoders.cli import main
from singing_vocoders.recipes import list_recipes


def test_version_command() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_publishes_all_workflows_and_only_vocoder_entrypoint() -> None:
    runner = CliRunner()
    help_result = runner.invoke(main, ["--help"])
    assert help_result.exit_code == 0
    for command in ("recipes", "config", "preprocess", "train", "validate", "checkpoint", "export"):
        assert command in help_result.output
    recipes_result = runner.invoke(main, ["recipes", "list"])
    assert recipes_result.exit_code == 0
    assert recipes_result.output.splitlines() == list(list_recipes())
    scripts = {
        item.name
        for item in entry_points(group="console_scripts")
        if item.dist is not None and item.dist.name == "singing-vocoders"
    }
    assert scripts == {"vocoder"}


@pytest.mark.skipif(os.name != "nt", reason="PowerShell environment safety regression")
@pytest.mark.parametrize(("create_exit_code", "expect_remove"), [(1, False), (0, True)])
def test_environment_script_only_removes_after_successful_conda_creation(
    tmp_path: Path,
    create_exit_code: int,
    expect_remove: bool,
) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")
    calls = tmp_path / "calls.txt"
    fake_conda = tmp_path / "conda.cmd"
    fake_conda.write_text(
        "\r\n".join(
            (
                "@echo off",
                f'echo %*>>"{calls}"',
                'if "%1 %2"=="env list" (echo {"envs":[]} & exit /b 0)',
                f'if "%1"=="create" exit /b {create_exit_code}',
                'if "%1"=="run" exit /b 1',
                'if "%1 %2"=="env remove" exit /b 0',
                "exit /b 1",
            )
        ),
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "create_env.ps1"

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-CondaPath",
            str(fake_conda),
            "-EnvironmentName",
            "singing-vocoders-script-test",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    invocations = calls.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("create ") for line in invocations)
    assert any(line.startswith("env remove ") for line in invocations) is expect_remove
