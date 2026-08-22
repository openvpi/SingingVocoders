from importlib.metadata import entry_points

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
