# Contributing to LAI

Thank you for contributing to Local AI Platform (`lai`). All pull requests are expected to pass
the repository's architectural, typing, coverage, mutation, and schema-synchronization gates.

## Prerequisites & Local Setup

- **Python**: `3.13` (`>=3.13,<3.14`)
- **Package & environment manager**: [`uv`](https://docs.astral.sh/uv/)
- **Task runner (optional)**: [`mise`](https://mise.jdx.dev/) using [`mise.toml`](mise.toml)

Install Python 3.13 and locked dependencies:

```powershell
uv python install 3.13
uv sync --locked
```

Or with `mise`:

```powershell
mise run setup
```

## 6-Layer Architecture

`import-linter` enforces strict layering and subsystem independence in
[`pyproject.toml`](pyproject.toml):

1. **Surface** (`lai.cli`, `lai.api`) — FastAPI HTTP/SSE server and Typer CLI.
2. **Orchestration** (`lai.orchestrator`) — Guardrail pre-flight gate, context routing, and tool loop.
3. **Independent Subsystems** (`lai.scheduler`, `lai.tools`, `lai.guardrails`, `lai.inference`,
   `lai.ingestion`, `lai.db`, `lai.security`, `lai.config`) — Layer-3 modules **must not** import
   one another directly.
4. **Observability** (`lai.observability`) — Structured 24h rotating JSONL logging, scrubbing, alerts.
5. **Ports** (`lai.ports`) — PEP 544 `Protocol` contracts.
6. **Domain** (`lai.domain`) — Pure immutable Pydantic v2 models and `LaiError` hierarchy.

Run boundary checks anytime:

```powershell
uv run python scripts/verify.py boundaries
```

## Code Quality, Budgets & Gates

- **350-line file size budget**: Every Python file under `src/`, `tests/`, and `scripts/` must stay
  `<= 350` lines (`uv run python scripts/check_file_sizes.py`). Split cohesive helpers into focused
  private modules (for example `_helpers.py`, `_validators.py`) instead of growing monoliths.
- **Formatting & Linting (`ruff`)**: Strict lint rules (including McCabe complexity `<= 10`,
  `ANN`, `PTH`, `T20`, `TC`, and Bandit `S`) and formatting checks:
  ```powershell
  uv run python -m ruff format src tests scripts
  uv run python scripts/verify.py lint
  ```
- **Strict Typechecking (`basedpyright` + `mypy`)**: Both typecheckers run in strict mode across
  `src`, `tests`, and `scripts`:
  ```powershell
  uv run python scripts/verify.py typecheck
  ```
- **90% Branch Coverage**: Unit and integration tests enforce `--cov=lai --cov-branch --cov-fail-under=90`:
  ```powershell
  uv run python scripts/verify.py test --cov-fail-under 90
  ```
- **85% Mutation Testing Gate (`pytest-gremlins`)**: Security-critical modules must achieve `>= 85%`
  mutant kill rate on the `critical` tier (and `full` tier on releases):
  ```powershell
  uv run python scripts/verify.py mutate --tier critical
  ```

## OpenAPI Schema & Documentation Sync

Whenever FastAPI routes, headers, or request/response DTOs change, regenerate both `openapi.json`
and `docs/api-reference.md`:

```powershell
uv run python scripts/export_openapi.py
```

Verify that both checked-in artifacts match the live FastAPI schema (run automatically by CI):

```powershell
uv run python scripts/export_openapi.py --check
```

To build and inspect the static Scalar OpenAPI documentation site locally:

```powershell
uv run python scripts/build_pages_docs.py --output-dir site
```

## Pull Request Checklist

Before opening a pull request, run the full verification suite:

```powershell
uv run python scripts/verify.py all --tier critical --cov-fail-under 90
```

- Keep commits focused and free of local artifacts (`config/config.yaml`, `data/`, `models/`,
  `bin/`, `site/`, `deploy/cloudflared/config.yml`).
- Update unit/integration tests and regenerate `openapi.json` + `docs/api-reference.md` in the
  same PR whenever API behavior changes.
