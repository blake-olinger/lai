#!/usr/bin/env bash
# Idempotent Ubuntu LTS FHS bootstrap script for Local AI Platform (LAI).
set -euo pipefail

# renovate: datasource=github-releases depName=jdx/mise
MISE_VERSION="2026.11.7"
MISE_SHA256_AMD64="3198a5ed5716ca8c17ecfc54521e2769d6cacbf2d13f24f0c2f9706620ec0024"
MISE_SHA256_ARM64="b8de6c8e7aa85752a6a2368df32393552a2d8947b6f4072ecd29a1ad972e5c40"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

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
  install -d -m 0755 -o root -g root /opt/lai /opt/lai/releases
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
  trap 'rm -rf "${tmp_dir}"' RETURN
  archive="${tmp_dir}/mise-v${MISE_VERSION}-linux-${mise_arch}.tar.gz"
  curl -fsSL \
    "https://github.com/jdx/mise/releases/download/v${MISE_VERSION}/mise-v${MISE_VERSION}-linux-${mise_arch}.tar.gz" \
    -o "${archive}"
  echo "${mise_sha}  ${archive}" | sha256sum -c -
  tar -xzf "${archive}" -C "${tmp_dir}" mise/bin/mise
  install -m 0755 -o root -g root "${tmp_dir}/mise/bin/mise" /usr/local/bin/mise
  echo "Installed mise ${MISE_VERSION} to /usr/local/bin/mise."
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
install_systemd_units

cat <<'EOF'
LAI Ubuntu FHS bootstrap complete.

Next steps:
  1. Copy an environment config overlay to /etc/lai/config.yaml (chmod 0640, chown root:lai).
  2. Populate secrets in /etc/lai/lai.env (chmod 0640, chown root:lai).
  3. Install a verified release wheel into /opt/lai/releases/v<version>/.venv and point /opt/lai/current to it.
  4. Enable and start LAI:
       sudo systemctl enable --now lai.service
  5. (Optional) Enable automated weekly maintenance upgrades after setting updates.enabled=true:
       sudo systemctl enable --now lai-upgrade.timer
EOF
