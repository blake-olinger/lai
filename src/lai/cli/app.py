"""Root Typer CLI application wiring all subcommands and command groups."""

from __future__ import annotations

from typing import Annotated

import typer

from lai.cli.apps_cmd import apps_app
from lai.cli.init_cmd import init_command
from lai.cli.keys_cmd import keys_app
from lai.cli.models_cmd import models_app
from lai.cli.serve_cmd import serve_command
from lai.cli.tunnel_cmd import tunnel_app
from lai.domain.version import PACKAGE_VERSION

app = typer.Typer(
    name="lai",
    no_args_is_help=True,
    help="Local AI platform CLI for initialization, keys, apps, models, serving, and tunnels.",
)


def _version_callback(show_version: bool) -> None:
    """Print the installed LAI version and exit cleanly when --version is passed."""
    if show_version:
        typer.echo(f"lai {PACKAGE_VERSION}")
        raise typer.Exit


@app.callback()
def root_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the LAI version and exit.",
        ),
    ] = False,
) -> None:
    """Root CLI callback exposing global options such as --version."""
    del version


app.command("init")(init_command)
app.command("serve")(serve_command)
app.add_typer(apps_app, name="apps")
app.add_typer(keys_app, name="keys")
app.add_typer(models_app, name="models")
app.add_typer(tunnel_app, name="tunnel")


def main() -> None:
    """CLI entrypoint."""
    app()
