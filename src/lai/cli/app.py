"""Root Typer CLI application wiring all subcommands and command groups."""

from __future__ import annotations

import typer

from lai.cli.init_cmd import init_command
from lai.cli.keys_cmd import keys_app
from lai.cli.models_cmd import models_app
from lai.cli.serve_cmd import serve_command
from lai.cli.tunnel_cmd import tunnel_app

app = typer.Typer(
    name="lai",
    no_args_is_help=True,
    help="Local AI platform CLI for initialization, keys, models, serving, and tunnels.",
)

app.command("init")(init_command)
app.command("serve")(serve_command)
app.add_typer(keys_app, name="keys")
app.add_typer(models_app, name="models")
app.add_typer(tunnel_app, name="tunnel")


def main() -> None:
    """CLI entrypoint."""
    app()
