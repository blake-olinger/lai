#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Already inside the shared Linux validation container: execute directly.
if [[ "${LAI_IN_CONTAINER:-0}" == "1" ]]; then
  case "${1:-}" in
    --build-only)
      echo "Already running inside LAI validation container; nothing to build."
      exit 0
      ;;
    --shell)
      exec bash
      ;;
    "")
      exec mise run ci
      ;;
    *)
      exec mise run "$@"
      ;;
  esac
fi

# When invoked from Windows Git Bash (MINGW/MSYS/CYGWIN), delegate into WSL2 Ubuntu
# so Docker Engine runs natively on Linux with proper WSL paths and named volumes.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    wsl_distro="${LAI_WSL_DISTRO:-Ubuntu}"
    win_repo_root="$(cygpath -m "${REPO_ROOT}")"
    wsl_repo_root="$(MSYS_NO_PATHCONV=1 wsl.exe -d "${wsl_distro}" --exec wslpath -u "${win_repo_root}" | tr -d '\r')"
    exec env MSYS_NO_PATHCONV=1 wsl.exe -d "${wsl_distro}" --exec bash -lc \
      'cd "$1" && shift && exec bash scripts/run-in-container.sh "$@"' \
      _ "${wsl_repo_root}" "$@"
    ;;
esac

BUILD_ONLY=0
FORCE_REBUILD=0
OPEN_SHELL=0
TASK_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --rebuild)
      FORCE_REBUILD=1
      shift
      ;;
    --shell)
      OPEN_SHELL=1
      shift
      ;;
    --)
      shift
      TASK_ARGS+=("$@")
      break
      ;;
    *)
      TASK_ARGS+=("$1")
      shift
      ;;
  esac
done

hash_inputs=(
  "${REPO_ROOT}/docker/ci.Dockerfile"
  "${REPO_ROOT}/mise.toml"
  "${REPO_ROOT}/uv.lock"
  "${REPO_ROOT}/.github/tools/pip-audit/uv.lock"
)

for input_file in "${hash_inputs[@]}"; do
  if [[ ! -f "${input_file}" ]]; then
    echo "ERROR: Missing required container hash input: ${input_file}" >&2
    exit 1
  fi
done

IMAGE_SHA="$(cat "${hash_inputs[@]}" | sha256sum | cut -c1-12)"
IMAGE_TAG="lai-ci:${IMAGE_SHA}"

if [[ "${FORCE_REBUILD}" == "1" ]] || ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  echo "Building LAI validation image ${IMAGE_TAG}..." >&2
  docker buildx build \
    --load \
    -f "${REPO_ROOT}/docker/ci.Dockerfile" \
    -t "${IMAGE_TAG}" \
    "${REPO_ROOT}"
fi

if [[ "${BUILD_ONLY}" == "1" ]]; then
  echo "${IMAGE_TAG}"
  exit 0
fi

if [[ "${OPEN_SHELL}" == "0" && "${#TASK_ARGS[@]}" -eq 0 ]]; then
  TASK_ARGS=("ci")
fi

DOCKER_TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY_ARGS+=(-it)
elif [[ "${OPEN_SHELL}" == "1" ]]; then
  DOCKER_TTY_ARGS+=(-i)
fi

DOCKER_MOUNT_ARGS=(
  -v "${REPO_ROOT}:/workspace"
  -v "lai-uv-cache:/cache/uv"
  -v "lai-mypy-cache:/workspace/.mypy_cache"
  -v "lai-ruff-cache:/workspace/.ruff_cache"
  -v "lai-pytest-cache:/workspace/.pytest_cache"
  -v "lai-import-linter-cache:/workspace/.import_linter_cache"
)

DOCKER_ENV_ARGS=(
  -e "LAI_IN_CONTAINER=1"
  -e "UV_PROJECT_ENVIRONMENT=/opt/lai-venv"
  -e "UV_CACHE_DIR=/cache/uv"
  -e "UV_LINK_MODE=copy"
)

# Support git worktrees (including Windows-created worktrees whose .git file has a C:/... path).
if [[ -f "${REPO_ROOT}/.git" ]]; then
  raw_gitdir="$(sed -n 's/^gitdir:[[:space:]]*//p' "${REPO_ROOT}/.git" | tr -d '\r' | head -n 1)"
  if [[ -n "${raw_gitdir}" ]]; then
    if [[ "${raw_gitdir}" =~ ^[A-Za-z]:[/\\] ]] && command -v wslpath >/dev/null 2>&1; then
      resolved_gitdir="$(wslpath -u "${raw_gitdir}")"
    elif [[ "${raw_gitdir}" = /* ]]; then
      resolved_gitdir="${raw_gitdir}"
    else
      resolved_gitdir="$(cd -- "${REPO_ROOT}/${raw_gitdir}" 2>/dev/null && pwd || true)"
    fi
    if [[ -n "${resolved_gitdir}" && -d "${resolved_gitdir}" ]]; then
      if [[ -f "${resolved_gitdir}/commondir" ]]; then
        rel_common="$(tr -d '\r\n' < "${resolved_gitdir}/commondir")"
        git_common_dir="$(cd -- "${resolved_gitdir}/${rel_common}" && pwd)"
        wt_rel="${resolved_gitdir#"${git_common_dir}/"}"
        DOCKER_MOUNT_ARGS+=(-v "${git_common_dir}:/git-common")
        DOCKER_ENV_ARGS+=(
          -e "GIT_DIR=/git-common/${wt_rel}"
          -e "GIT_WORK_TREE=/workspace"
        )
      else
        DOCKER_MOUNT_ARGS+=(-v "${resolved_gitdir}:/git-dir")
        DOCKER_ENV_ARGS+=(
          -e "GIT_DIR=/git-dir"
          -e "GIT_WORK_TREE=/workspace"
        )
      fi
    fi
  fi
fi

for var_name in LAI_COV_FAIL_UNDER LAI_MIN_MUTATION_SCORE LAI_MUTANT_TIMEOUT; do
  if [[ -n "${!var_name:-}" ]]; then
    DOCKER_ENV_ARGS+=(-e "${var_name}=${!var_name}")
  fi
done

CONTAINER_BOOTSTRAP='
set -euo pipefail
git config --global --add safe.directory /workspace
if [[ -n "${GIT_DIR:-}" ]]; then
  git config --global --add safe.directory /git-common
  git config --global --add safe.directory "${GIT_DIR}"
fi
uv pip install --python /opt/lai-venv/bin/python --no-deps -e /workspace >/dev/null
export UV_NO_SYNC=1
'

if [[ "${OPEN_SHELL}" == "1" ]]; then
  exec docker run --rm --init \
    "${DOCKER_TTY_ARGS[@]}" \
    "${DOCKER_MOUNT_ARGS[@]}" \
    "${DOCKER_ENV_ARGS[@]}" \
    -w /workspace \
    "${IMAGE_TAG}" \
    bash -lc "${CONTAINER_BOOTSTRAP}"'exec bash'
fi

exec docker run --rm --init \
  "${DOCKER_TTY_ARGS[@]}" \
  "${DOCKER_MOUNT_ARGS[@]}" \
  "${DOCKER_ENV_ARGS[@]}" \
  -w /workspace \
  "${IMAGE_TAG}" \
  bash -lc "${CONTAINER_BOOTSTRAP}"'exec mise run "$@"' _ "${TASK_ARGS[@]}"
