# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-19

### Added

- Local multi-tenant FastAPI server and Typer CLI (`init`, `keys`, `apps`, `models`, `serve`,
  `tunnel`) with `--version` and single-source-of-truth SemVer synchronization.
- Isolated project workspaces with HTTP-only streaming file ingestion, hybrid FTS5 + LanceDB
  retrieval, and adaptive context routing (`direct`, `retrieval`, `map_reduce`).
- Supervised local CUDA `llama.cpp` (Windows & Linux) and Ollama inference backends with
  streaming reasoning support and CPU-only ShieldGemma-2B pre-flight prompt guardrails.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints, SSE keepalives for
  Cloudflare Named Tunnels, and per-user quotas with fair scheduling.
- Google OIDC multi-audience JWT authentication with configurable clock-skew leeway, secret-free
  public SPA/desktop client app binding, in-place client app updates (`PATCH /v1/client-apps/{id}`
  and `lai apps update`), and guarded dual-plane tool execution with human approvals.
- 24-hour rotating structured JSONL logs with secret/PII scrubbing, error-level alert routing,
  configurable outbound SMTP email & webhook alerting (`POST /v1/system/alerting/test`),
  privacy-preserving SQLite hourly endpoint/token usage analytics (`GET /v1/system/analytics`),
  isolated zero-restart per-subsystem runtime hot-swapping (`/v1/system/config/*` and scoped
  `POST /v1/system/config/reload`), CORS preflight diagnostics, and interactive Scalar OpenAPI 3.1
  documentation.

[Unreleased]: https://github.com/blake-olinger/lai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/blake-olinger/lai/releases/tag/v0.1.0
