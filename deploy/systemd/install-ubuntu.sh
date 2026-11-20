#!/usr/bin/env bash
# Idempotent Ubuntu LTS FHS bootstrap script for Local AI Platform (LAI).
set -euo pipefail
umask 022

# renovate: datasource=github-releases depName=jdx/mise
MISE_VERSION="2026.11.7"
MISE_SHA256_AMD64="3198a5ed5716ca8c17ecfc54521e2769d6cacbf2d13f24f0c2f9706620ec0024"
MISE_SHA256_ARM64="b8de6c8e7aa85752a6a2368df32393552a2d8947b6f4072ecd29a1ad972e5c40"

# renovate: datasource=github-releases depName=astral-sh/uv
UV_VERSION="0.13.0"
UV_SHA256_AMD64="1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154"
UV_SHA256_ARM64="3ccfb6af6e242433eb552f7d9676abd5412c6595c497e990d9c8cb7b5bd4d2c3"
PYTHON_VERSION="3.13"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOOTSTRAP_WHEEL=""
BOOTSTRAP_SHA256=""
BOOTSTRAP_REQUIREMENTS=""
BOOTSTRAP_REQUIREMENTS_SHA256=""
BOOTSTRAP_VERSION=""
GITHUB_REPO="blake-olinger/lai"
_CLEANUP_PATHS=()
_INCOMPLETE_SLOT=""

_cleanup_on_exit() {
  local path
  if [[ -n "${_INCOMPLETE_SLOT}" ]]; then
    rm -rf "${_INCOMPLETE_SLOT}"
  fi
  for path in "${_CLEANUP_PATHS[@]}"; do
    rm -rf "${path}"
  done
}
trap _cleanup_on_exit EXIT

usage() {
  cat <<'EOF'
Usage: sudo bash install-ubuntu.sh [OPTIONS]

Idempotently provisions the `lai` service user, FHS directories, pinned `mise`
and `uv` binaries under /usr/local/bin, managed Python 3.13 under
/opt/lai/python, systemd units, and optionally bootstraps the initial release
slot under /opt/lai/releases/v<version> with an atomic /opt/lai/current symlink.

Options:
  --version <ver>                Download & verify release v<ver> from GitHub Releases
  --repo <owner/repo>            GitHub repository for --version (default: blake-olinger/lai)
  --wheel <path>                 Stage local wheel for initial slot bootstrap
  --sha256 <hex>                 Required 64-hex SHA-256 digest for --wheel
  --requirements <path>          Required hashed requirements-lock.txt for --wheel
  --requirements-sha256 <hex>    Optional 64-hex SHA-256 digest for --requirements
                                 (auto-discovers <requirements>.sha256 or SHA256SUMS.txt)
  -h, --help                     Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      BOOTSTRAP_VERSION="${2:?--version requires a value}"
      shift 2
      ;;
    --repo)
      GITHUB_REPO="${2:?--repo requires a value}"
      shift 2
      ;;
    --wheel)
      BOOTSTRAP_WHEEL="${2:?--wheel requires a value}"
      shift 2
      ;;
    --sha256)
      BOOTSTRAP_SHA256="${2:?--sha256 requires a value}"
      shift 2
      ;;
    --requirements)
      BOOTSTRAP_REQUIREMENTS="${2:?--requirements requires a value}"
      shift 2
      ;;
    --requirements-sha256)
      BOOTSTRAP_REQUIREMENTS_SHA256="${2:?--requirements-sha256 requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${BOOTSTRAP_WHEEL}" && -n "${BOOTSTRAP_VERSION}" ]]; then
  echo "ERROR: Specify either --wheel or --version, not both." >&2
  exit 2
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: install-ubuntu.sh must be run as root (e.g. sudo bash $0)." >&2
  exit 1
fi

ensure_service_user() {
  if ! getent group lai >/dev/null 2>&1; then
    groupadd --system lai
  fi
  if ! id -u lai >/dev/null 2>&1; then
    useradd \
      --system \
      --gid lai \
      --no-create-home \
      --home-dir /var/lib/lai \
      --shell /usr/sbin/nologin \
      --comment "Local AI Platform service user" \
      lai
  fi
}

ensure_fhs_directories() {
  install -d -m 0755 -o root -g root \
    /opt/lai \
    /opt/lai/releases \
    /opt/lai/uv-cache \
    /opt/lai/python
  install -d -m 0750 -o root -g lai /etc/lai
  install -d -m 0750 -o lai -g lai \
    /var/lib/lai \
    /var/lib/lai/data \
    /var/lib/lai/models \
    /var/lib/lai/bin \
    /var/log/lai
  install -d -m 0700 -o lai -g lai /var/lib/lai/backups

  if [[ ! -f /etc/lai/lai.env ]]; then
    install -m 0640 -o root -g lai /dev/null /etc/lai/lai.env
    cat <<'EOF' > /etc/lai/lai.env
# Secret environment overrides for LAI (root:lai 0640; never commit secrets to YAML).
# Example:
# LAI_LOGGING__ALERT_SMTP__PASSWORD=replace-me
# LAI_LOGGING__ALERT_WEBHOOK_URL=https://hooks.example.com/services/...
EOF
    chmod 0640 /etc/lai/lai.env
    chown root:lai /etc/lai/lai.env
  fi
}

install_pinned_mise() {
  if command -v mise >/dev/null 2>&1; then
    local installed_version
    installed_version="$(mise --version | awk '{print $1}')"
    if [[ "${installed_version}" == "${MISE_VERSION}" ]]; then
      echo "mise ${MISE_VERSION} is already installed."
      return 0
    fi
  fi

  local arch mise_arch mise_sha tmp_dir archive
  arch="$(dpkg --print-architecture)"
  case "${arch}" in
    amd64)
      mise_arch="x64"
      mise_sha="${MISE_SHA256_AMD64}"
      ;;
    arm64)
      mise_arch="arm64"
      mise_sha="${MISE_SHA256_ARM64}"
      ;;
    *)
      echo "ERROR: Unsupported architecture for pinned mise binary: ${arch}" >&2
      exit 1
      ;;
  esac

  tmp_dir="$(mktemp -d)"
  _CLEANUP_PATHS+=("${tmp_dir}")
  archive="${tmp_dir}/mise-v${MISE_VERSION}-linux-${mise_arch}.tar.gz"
  curl -fsSL \
    "https://github.com/jdx/mise/releases/download/v${MISE_VERSION}/mise-v${MISE_VERSION}-linux-${mise_arch}.tar.gz" \
    -o "${archive}"
  echo "${mise_sha}  ${archive}" | sha256sum -c -
  tar -xzf "${archive}" -C "${tmp_dir}" mise/bin/mise
  install -m 0755 -o root -g root "${tmp_dir}/mise/bin/mise" /usr/local/bin/mise
  echo "Installed mise ${MISE_VERSION} to /usr/local/bin/mise."
}

install_pinned_uv_and_python() {
  local need_install="1"
  if [[ -x /usr/local/bin/uv ]]; then
    local installed_uv
    installed_uv="$(/usr/local/bin/uv --version | awk '{print $2}')"
    if [[ "${installed_uv}" == "${UV_VERSION}" ]]; then
      echo "uv ${UV_VERSION} is already installed at /usr/local/bin/uv."
      need_install="0"
    fi
  fi

  if [[ "${need_install}" == "1" ]]; then
    local arch uv_arch uv_sha tmp_dir archive uv_bin
    arch="$(dpkg --print-architecture)"
    case "${arch}" in
      amd64)
        uv_arch="x86_64"
        uv_sha="${UV_SHA256_AMD64}"
        ;;
      arm64)
        uv_arch="aarch64"
        uv_sha="${UV_SHA256_ARM64}"
        ;;
      *)
        echo "ERROR: Unsupported architecture for pinned uv binary: ${arch}" >&2
        exit 1
        ;;
    esac

    tmp_dir="$(mktemp -d)"
    _CLEANUP_PATHS+=("${tmp_dir}")
    archive="${tmp_dir}/uv-${uv_arch}-unknown-linux-gnu.tar.gz"
    curl -fsSL \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${uv_arch}-unknown-linux-gnu.tar.gz" \
      -o "${archive}"
    echo "${uv_sha}  ${archive}" | sha256sum -c -
    tar -xzf "${archive}" -C "${tmp_dir}"
    uv_bin="${tmp_dir}/uv-${uv_arch}-unknown-linux-gnu/uv"
    if [[ ! -f "${uv_bin}" ]]; then
      uv_bin="${tmp_dir}/uv"
    fi
    install -m 0755 -o root -g root "${uv_bin}" /usr/local/bin/uv
    echo "Installed uv ${UV_VERSION} to /usr/local/bin/uv."
  fi

  umask 022
  UV_PYTHON_INSTALL_DIR="/opt/lai/python" \
  UV_CACHE_DIR="/opt/lai/uv-cache" \
  UV_LINK_MODE="copy" \
    /usr/local/bin/uv python install --no-bin "${PYTHON_VERSION}"
  chmod -R a+rX /opt/lai/python
  echo "Ensured managed Python ${PYTHON_VERSION} under /opt/lai/python."
}

lookup_sha256_in_manifest() {
  local manifest_file="$1"
  local target_name="$2"
  awk -v name="${target_name}" '
    /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
    {
      fname = $2
      sub(/^\*/, "", fname)
      sub(/^.*\//, "", fname)
      if (fname == name) {
        print tolower($1)
        exit 0
      }
    }
  ' "${manifest_file}"
}

install_verified_slot() {
  local version="$1"
  local wheel_file="$2"
  local req_file="$3"
  local slot_dir="/opt/lai/releases/v${version}"
  local staging_link="/opt/lai/.current.staging.$$"

  if [[ -L /opt/lai/current && "$(readlink -f /opt/lai/current)" == "${slot_dir}" ]]; then
    echo "Release slot ${slot_dir} is already active at /opt/lai/current; keeping active slot intact."
    return 0
  fi

  _CLEANUP_PATHS+=("${staging_link}")
  rm -rf "${slot_dir}"
  _INCOMPLETE_SLOT="${slot_dir}"
  install -d -m 0755 -o root -g root "${slot_dir}"

  umask 022
  UV_PYTHON_INSTALL_DIR="/opt/lai/python" \
  UV_CACHE_DIR="/opt/lai/uv-cache" \
  UV_LINK_MODE="copy" \
    /usr/local/bin/uv venv --python "${PYTHON_VERSION}" "${slot_dir}/.venv"
  UV_PYTHON_INSTALL_DIR="/opt/lai/python" \
  UV_CACHE_DIR="/opt/lai/uv-cache" \
  UV_LINK_MODE="copy" \
    /usr/local/bin/uv pip install --python "${slot_dir}/.venv/bin/python" \
      --require-hashes -r "${req_file}"
  UV_PYTHON_INSTALL_DIR="/opt/lai/python" \
  UV_CACHE_DIR="/opt/lai/uv-cache" \
  UV_LINK_MODE="copy" \
    /usr/local/bin/uv pip install --python "${slot_dir}/.venv/bin/python" \
      --no-deps "${wheel_file}"
  chmod -R a+rX "${slot_dir}"

  ln -s "${slot_dir}" "${staging_link}"
  mv -Tf "${staging_link}" /opt/lai/current
  _INCOMPLETE_SLOT=""
  echo "Activated release slot ${slot_dir} -> /opt/lai/current."
}

bootstrap_initial_slot() {
  if [[ -n "${BOOTSTRAP_WHEEL}" ]]; then
    if [[ -z "${BOOTSTRAP_SHA256}" || -z "${BOOTSTRAP_REQUIREMENTS}" ]]; then
      echo "ERROR: --wheel requires both --sha256 <hex> and --requirements <path>." >&2
      exit 2
    fi
    if [[ ! -f "${BOOTSTRAP_WHEEL}" ]]; then
      echo "ERROR: Wheel file not found: ${BOOTSTRAP_WHEEL}" >&2
      exit 1
    fi
    if [[ ! -f "${BOOTSTRAP_REQUIREMENTS}" ]]; then
      echo "ERROR: Requirements lockfile not found: ${BOOTSTRAP_REQUIREMENTS}" >&2
      exit 1
    fi

    local wheel_base wheel_ver req_base req_sha
    wheel_base="$(basename "${BOOTSTRAP_WHEEL}")"
    if [[ ! "${wheel_base}" =~ ^lai-([0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9._+]*)-.+\.whl$ ]]; then
      echo "ERROR: Invalid wheel filename: ${wheel_base}" >&2
      exit 1
    fi
    wheel_ver="${BASH_REMATCH[1]}"
    echo "${BOOTSTRAP_SHA256,,}  ${BOOTSTRAP_WHEEL}" | sha256sum -c -

    req_base="$(basename "${BOOTSTRAP_REQUIREMENTS}")"
    req_sha="${BOOTSTRAP_REQUIREMENTS_SHA256,,}"
    if [[ -z "${req_sha}" && -f "${BOOTSTRAP_REQUIREMENTS}.sha256" ]]; then
      req_sha="$(awk '{print tolower($1); exit}' "${BOOTSTRAP_REQUIREMENTS}.sha256")"
    fi
    if [[ -z "${req_sha}" && -f "$(dirname "${BOOTSTRAP_REQUIREMENTS}")/SHA256SUMS.txt" ]]; then
      req_sha="$(lookup_sha256_in_manifest "$(dirname "${BOOTSTRAP_REQUIREMENTS}")/SHA256SUMS.txt" "${req_base}")"
    fi
    if [[ -z "${req_sha}" && -f "$(dirname "${BOOTSTRAP_WHEEL}")/SHA256SUMS.txt" ]]; then
      req_sha="$(lookup_sha256_in_manifest "$(dirname "${BOOTSTRAP_WHEEL}")/SHA256SUMS.txt" "${req_base}")"
    fi
    if [[ ! "${req_sha}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "ERROR: Missing SHA-256 digest for ${req_base}; provide --requirements-sha256, ${req_base}.sha256, or SHA256SUMS.txt." >&2
      exit 1
    fi
    echo "${req_sha}  ${BOOTSTRAP_REQUIREMENTS}" | sha256sum -c -
    install_verified_slot "${wheel_ver}" "${BOOTSTRAP_WHEEL}" "${BOOTSTRAP_REQUIREMENTS}"
    return 0
  fi

  if [[ -n "${BOOTSTRAP_VERSION}" ]]; then
    local clean_ver base_url tmp_dir sums_file wheel_name wheel_file req_file wheel_sha req_sha
    clean_ver="${BOOTSTRAP_VERSION#v}"
    clean_ver="${clean_ver#V}"
    base_url="https://github.com/${GITHUB_REPO}/releases/download/v${clean_ver}"
    wheel_name="lai-${clean_ver}-py3-none-any.whl"

    tmp_dir="$(mktemp -d)"
    _CLEANUP_PATHS+=("${tmp_dir}")
    sums_file="${tmp_dir}/SHA256SUMS.txt"
    wheel_file="${tmp_dir}/${wheel_name}"
    req_file="${tmp_dir}/requirements-lock.txt"

    curl -fsSL "${base_url}/SHA256SUMS.txt" -o "${sums_file}"
    curl -fsSL "${base_url}/${wheel_name}" -o "${wheel_file}"
    curl -fsSL "${base_url}/requirements-lock.txt" -o "${req_file}"

    wheel_sha="$(lookup_sha256_in_manifest "${sums_file}" "${wheel_name}")"
    req_sha="$(lookup_sha256_in_manifest "${sums_file}" "requirements-lock.txt")"
    if [[ ! "${wheel_sha}" =~ ^[0-9a-f]{64}$ || ! "${req_sha}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "ERROR: SHA256SUMS.txt missing checksum for ${wheel_name} or requirements-lock.txt." >&2
      exit 1
    fi
    echo "${wheel_sha}  ${wheel_file}" | sha256sum -c -
    echo "${req_sha}  ${req_file}" | sha256sum -c -
    install_verified_slot "${clean_ver}" "${wheel_file}" "${req_file}"
  fi
}

install_systemd_units() {
  install -m 0644 -o root -g root "${SCRIPT_DIR}/lai.service" /etc/systemd/system/lai.service
  install -m 0644 -o root -g root "${SCRIPT_DIR}/lai-upgrade.service" /etc/systemd/system/lai-upgrade.service
  install -m 0644 -o root -g root "${SCRIPT_DIR}/lai-upgrade.timer" /etc/systemd/system/lai-upgrade.timer

  if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
    systemctl daemon-reload
  fi
}

ensure_service_user
ensure_fhs_directories
install_pinned_mise
install_pinned_uv_and_python
bootstrap_initial_slot
install_systemd_units

cat <<'EOF'
LAI Ubuntu FHS bootstrap complete.

Next steps:
  1. Copy an environment config overlay to /etc/lai/config.yaml (chmod 0640, chown root:lai).
  2. Populate secrets in /etc/lai/lai.env (chmod 0640, chown root:lai).
  3. If you did not pass --version or --wheel, bootstrap the initial verified release slot:
       sudo bash deploy/systemd/install-ubuntu.sh --version <version>
     or for air-gapped deploys:
       sudo bash deploy/systemd/install-ubuntu.sh --wheel /tmp/lai-<ver>-py3-none-any.whl \
         --sha256 <64-hex> --requirements /tmp/requirements-lock.txt
  4. Initialize storage and download verified models/binaries as `lai`:
       sudo -u lai /opt/lai/current/.venv/bin/lai init --config /etc/lai/config.yaml
       sudo -u lai /opt/lai/current/.venv/bin/lai models setup --config /etc/lai/config.yaml
  5. Enable and start LAI:
       sudo systemctl enable --now lai.service
  6. (Optional) Enable automated weekly maintenance upgrades after setting updates.enabled=true:
       sudo systemctl enable --now lai-upgrade.timer
EOF
