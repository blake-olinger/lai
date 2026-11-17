<#
.SYNOPSIS
    Launch the Local AI Platform (LAI) server on Windows.
.DESCRIPTION
    Synchronizes locked dependencies (unless -SkipSync is set), optionally sets
    LAI_CONFIG_FILE, starts Gemma 4 (:8081), ShieldGemma-2B (:8082), and FastAPI
    (:8400), and can optionally supervise a Cloudflare Named Tunnel.
#>
[CmdletBinding()]
param(
    [string]$ConfigPath = "",
    [string]$HostAddress = "",
    [int]$Port = 0,
    [string]$Profile = "",
    [switch]$NoSupervise,
    [switch]$SkipSync,
    [string]$CloudflaredConfig = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot

$TunnelProcess = $null
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is required on PATH. Install from https://docs.astral.sh/uv/."
    }

    if (-not $SkipSync) {
        uv sync --locked
        if ($LASTEXITCODE -ne 0) {
            throw "uv sync --locked failed with exit code $LASTEXITCODE"
        }
    }

    if ($ConfigPath -ne "") {
        $env:LAI_CONFIG_FILE = (Resolve-Path $ConfigPath).Path
    }

    if ($CloudflaredConfig -ne "") {
        if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
            throw "cloudflared not found on PATH."
        }
        $ResolvedTunnelCfg = (Resolve-Path $CloudflaredConfig).Path
        $TunnelProcess = Start-Process -FilePath "cloudflared" -ArgumentList @("tunnel", "--config", $ResolvedTunnelCfg, "run") -PassThru
    }

    $ServeArgs = @("run", "python", "-m", "lai.cli", "serve")
    if ($ConfigPath -ne "") {
        $ServeArgs += @("--config", $env:LAI_CONFIG_FILE)
    }
    if ($HostAddress -ne "") {
        $ServeArgs += @("--host", $HostAddress)
    }
    if ($Port -gt 0) {
        $ServeArgs += @("--port", "$Port")
    }
    if ($Profile -ne "") {
        $ServeArgs += @("--profile", $Profile)
    }
    if ($NoSupervise) {
        $ServeArgs += "--no-supervise"
    }

    & uv @ServeArgs
    exit $LASTEXITCODE
}
finally {
    if ($null -ne $TunnelProcess -and -not $TunnelProcess.HasExited) {
        Stop-Process -Id $TunnelProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
