from __future__ import annotations

import click

from singing_vocoders import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__)
def main() -> None:
    """Train, validate, inspect, and export singing voice vocoders."""
