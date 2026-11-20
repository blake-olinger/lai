# Contributing to LAI

Thank you for contributing to Local AI Platform (`lai`). All pull requests are expected to pass
the repository's architectural, typing, security, coverage, mutation, and schema-synchronization
gates inside the shared Linux validation container.

## Prerequisites & Local Setup (Ubuntu / WSL2 Ubuntu)

- **Linux / WSL2 Ubuntu 24.04+** with native **Docker** + **Buildx**
- **[`mise`](https://mise.jdx.dev/)** reading [`mise.toml`](mise.toml) — the single source of truth
  for Python (`3.13`), `uv`, `gitleaks`, and every verification task DAG.

Bootstrap your workstation and activate container-backed git hooks:

```bash
mise trust && mise install
mise run setup
mise run hooks   # runs: git config core.hooksPath .githooks
```

Build the shared Ubuntu 24.04 CI container once (cached automatically across runs):

```bash
bash scripts/run-in-container.sh --build-only
```

## 6-Layer Architecture

`import-linter` enforces strict layering and subsystem independence in
[`pyproject.toml`](pyproject.toml):

1. **Surface** (`lai.cli`, `lai.api`) — FastAPI HTTP/SSE server and Typer CLI.
2. **Orchestration** (`lai.orchestrator`) — Guardrail ingress/egress gates, context routing, and tool loop.
3. **Independent Subsystems** (`lai.scheduler`, `lai.tools`, `lai.guardrails`, `lai.inference`,
   `lai.ingestion`, `lai.db`, `lai.security`, `lai.config`, `lai.updates`) — Layer-3 modules
   **must not** import one another directly.
4. **Observability** (`lai.observability`) — Structured 24h rotating JSONL logging, scrubbing,
   POSIX `PR_SET_PDEATHSIG` process helpers, SHA-256 integrity verification, and alert routing.
5. **Ports** (`lai.ports`) — PEP 544 `Protocol` contracts.
6. **Domain** (`lai.domain`) — Pure immutable Pydantic v2 models and `LaiError` hierarchy.

Run boundary checks anytime:

```bash
bash scripts/run-in-container.sh boundaries
```

## Code Quality, Budgets & Container-Backed Gates

Every local check, git hook, and GitHub Actions workflow runs the **exact same `mise` tasks**
inside `docker/ci.Dockerfile`:

- **350-line file size budget**: Every Python file under `src/`, `tests/`, and `scripts/` must stay
  `<= 350` lines (`scripts/check_file_sizes.py`). Split cohesive helpers into focused private
  modules instead of growing monoliths.
- **Formatting & Linting (`ruff`)**: Strict lint rules (McCabe complexity `<= 10`, `ANN`, `PTH`,
  `T20`, `TC`, and Bandit `S`):
  ```bash
  mise run fmt
  bash scripts/run-in-container.sh lint
  ```
- **Strict Typechecking (`basedpyright` + `mypy`)**:
  ```bash
  bash scripts/run-in-container.sh typecheck
  ```
- **90% Branch Coverage**:
  ```bash
  bash scripts/run-in-container.sh test
  ```
- **85% Mutation Testing Gate (`pytest-gremlins`)**: Security-critical modules (including
  `integrity.py`, `_egress.py`, `updates/sources.py`, `updates/coordinator.py`, `_validators.py`,
  `deps.py`, `oidc_provider.py`, and `guardrails/*`) must achieve `>= 85%` mutant kill rate:
  ```bash
  bash scripts/run-in-container.sh mutate       # critical tier (PR gate)
  bash scripts/run-in-container.sh mutate-full  # full tier (scheduled/release gate)
  ```
- **Security Suite (`gitleaks`, Ruff Bandit `S`, isolated hash-locked `pip-audit`)**:
  ```bash
  bash scripts/run-in-container.sh security
  ```

### Local Git Hooks (`.githooks/`)

After running `mise run hooks`:

- **`pre-commit`** runs `bash scripts/run-in-container.sh pre-commit` (`lint`, `secrets:staged`,
  `typecheck`, `boundaries`, `schema`).
- **`pre-push`** runs `bash scripts/run-in-container.sh pre-push` (`pre-commit`, `test`, `build`,
  `security`, `docs`, `mutate`).

## OpenAPI Schema & Documentation Sync

Whenever FastAPI routes, headers, or request/response DTOs change, regenerate `openapi.json` and
`docs/api-reference.md` inside the container:

```bash
bash scripts/run-in-container.sh openapi
bash scripts/run-in-container.sh schema
```

Build and inspect the static Scalar OpenAPI documentation site:

```bash
bash scripts/run-in-container.sh docs
```

## Dependency Automation (Renovate) & Releases

- **Renovate (`renovate.json`)** manages Python dependencies (`pyproject.toml` and
  `.github/tools/pip-audit/pyproject.toml`), `mise.toml`, `docker/ci.Dockerfile`, GitHub Actions,
  and custom version pins (Scalar CDN, `ggml-org/llama.cpp`, `jdx/mise`) with a mandatory **7-day
  release cooldown** (`minimumReleaseAge: "7 days"` matching `uv` `exclude-newer = "7 days"`).
  Checksum-coupled custom pins disable automerge so SHA-256 digests are reviewed explicitly.
- **Versioning & Releases**: `pyproject.toml` is the single source of truth for LAI's 3-part SemVer
  version. Bump versions in lockstep with:
  ```bash
  bash scripts/run-in-container.sh bump 0.2.0
  ```
  Tagging a release (`git tag v0.2.0 && git push origin v0.2.0`) triggers `.github/workflows/release.yml`,
  which runs full verification inside `docker/ci.Dockerfile` and publishes wheel + sdist +
  `SHA256SUMS.txt` consumed by `lai upgrade`.

## Pull Request Checklist

Before opening a pull request, run the full containerized pre-push gate:

```bash
bash scripts/run-in-container.sh pre-push
```

- Keep commits atomic, linear, LF-normalized, and free of local artifacts (`config/config.yaml`,
  `data/`, `models/`, `bin/`, `site/`, `dist/`, `deploy/cloudflared/config.yml`).
- Update unit/integration tests and regenerate `openapi.json` + `docs/api-reference.md` in the same
  PR whenever API behavior changes.
- Update public guides under `docs/` (`docs/deployment.md`, `docs/client-apps-and-projects.md`)
  when CLI subcommands, lifecycle workflows, or configuration options change.
