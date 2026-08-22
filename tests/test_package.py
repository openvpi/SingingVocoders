from click.testing import CliRunner

from singing_vocoders import __version__
from singing_vocoders.cli import main


def test_version_command() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
