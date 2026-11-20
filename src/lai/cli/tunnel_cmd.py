"""CLI command rendering Cloudflare Named Tunnel configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from lai.cli._common import chmod_posix, resolve_settings

_POSIX_FILE_MODE = 0o600

tunnel_app = typer.Typer(
    name="tunnel",
    no_args_is_help=True,
    help="Cloudflare Named Tunnel helpers.",
)


def _default_credentials_file(tunnel: str) -> str:
    """Resolve absolute default cloudflared credentials path under the user's home directory."""
    return str(Path.home() / ".cloudflared" / f"{tunnel}.json")


def render_cloudflared_config(
    *,
    tunnel: str,
    hostname: str,
    origin_url: str,
    credentials_file: Path | None = None,
) -> str:
    """Render a cloudflared YAML configuration routing hostname to the local LAI server."""
    creds_str = (
        str(credentials_file) if credentials_file is not None else _default_credentials_file(tunnel)
    )
    payload: dict[str, object] = {
        "tunnel": tunnel,
        "credentials-file": creds_str,
        "ingress": [
            {
                "hostname": hostname,
                "service": origin_url,
                "originRequest": {
                    "noTLSVerify": False,
                    "connectTimeout": "30s",
                    "keepAliveTimeout": "90s",
                },
            },
            {"service": "http_status:404"},
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False)


@tunnel_app.command("render-config")
def render_config_command(  # noqa: PLR0913, PLR0917
    hostname: Annotated[
        str,
        typer.Option("--hostname", "-H", help="Public Cloudflare hostname (e.g. ai.example.com)."),
    ],
    tunnel: Annotated[
        str,
        typer.Option("--tunnel", "-t", help="Cloudflare Tunnel UUID or name."),
    ] = "lai-local",
    origin_url: Annotated[
        str | None,
        typer.Option("--origin-url", help="Local origin service URL (default http://host:port)."),
    ] = None,
    credentials_file: Annotated[
        Path | None,
        typer.Option("--credentials-file", help="Path to cloudflared tunnel credentials JSON."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional path to write rendered YAML."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional LAI YAML config path."),
    ] = None,
) -> None:
    """Render a cloudflared Named Tunnel ingress configuration for LAI."""
    settings = resolve_settings(config)
    resolved_origin = origin_url or f"http://{settings.host}:{settings.port}"
    rendered = render_cloudflared_config(
        tunnel=tunnel,
        hostname=hostname,
        origin_url=resolved_origin,
        credentials_file=credentials_file,
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        chmod_posix(output, _POSIX_FILE_MODE)
        typer.echo(f"Wrote cloudflared config to: {output}")
    else:
        typer.echo(rendered, nl=False)
