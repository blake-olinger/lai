# syntax=docker/dockerfile:1.7
# Shared Ubuntu 24.04 Linux validation container for local git hooks and GitHub Actions.
FROM ubuntu:24.04@sha256:73b999393d07f2bf65a40373778a44dc9a6d54ad7718d29247d8ff9413c4d6a8

ARG TARGETARCH
# renovate: datasource=github-releases depName=jdx/mise
ARG MISE_VERSION="2026.11.7"
ARG MISE_SHA256_AMD64="3198a5ed5716ca8c17ecfc54521e2769d6cacbf2d13f24f0c2f9706620ec0024"
ARG MISE_SHA256_ARM64="b8de6c8e7aa85752a6a2368df32393552a2d8947b6f4072ecd29a1ad972e5c40"

ENV DEBIAN_FRONTEND="noninteractive" \
    MISE_DATA_DIR="/opt/mise" \
    MISE_CONFIG_DIR="/etc/mise" \
    MISE_CACHE_DIR="/opt/mise-cache" \
    MISE_STATE_DIR="/opt/mise-state" \
    MISE_TRUSTED_CONFIG_PATHS="/workspace" \
    MISE_YES="1" \
    PATH="/opt/mise/shims:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    UV_PROJECT_ENVIRONMENT="/opt/lai-venv" \
    UV_LINK_MODE="copy" \
    LAI_IN_CONTAINER="1"

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gzip \
        libgomp1 \
        tar \
        xz-utils \
        zstd \
    && rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH:-amd64}" in \
        amd64) mise_arch="x64"; mise_sha="${MISE_SHA256_AMD64}" ;; \
        arm64) mise_arch="arm64"; mise_sha="${MISE_SHA256_ARM64}" ;; \
        *) echo "Unsupported TARGETARCH: ${TARGETARCH:-unknown}" >&2; exit 1 ;; \
    esac \
    && archive="/tmp/mise-v${MISE_VERSION}-linux-${mise_arch}.tar.gz" \
    && curl -fsSL "https://github.com/jdx/mise/releases/download/v${MISE_VERSION}/mise-v${MISE_VERSION}-linux-${mise_arch}.tar.gz" -o "${archive}" \
    && echo "${mise_sha}  ${archive}" | sha256sum -c - \
    && tar -xzf "${archive}" -C /tmp mise/bin/mise \
    && install -m 0755 /tmp/mise/bin/mise /usr/local/bin/mise \
    && rm -rf /tmp/mise "${archive}" \
    && git config --system --add safe.directory /workspace \
    && git config --global --add safe.directory /workspace

WORKDIR /workspace

COPY mise.toml .python-version ./

RUN mise trust /workspace/mise.toml \
    && mise install \
    && mise reshim

COPY pyproject.toml uv.lock ./
COPY .github/tools/pip-audit/pyproject.toml .github/tools/pip-audit/uv.lock ./.github/tools/pip-audit/

RUN --mount=type=cache,target=/root/.cache/uv \
    mise exec -- uv sync --locked --no-install-project \
    && UV_PROJECT_ENVIRONMENT="/opt/pip-audit-venv" mise exec -- uv sync --project .github/tools/pip-audit --locked
