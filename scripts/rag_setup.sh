#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Chat with RAG — One-command setup (macOS/Linux)
# -----------------------------------------------------------------------------
# What this script does:
#   1) Validates basic prerequisites (git, python3, make, docker, compose)
#   2) Ensures a local .env exists (copies from .env.example if present)
#   3) Prompts for OPENAI_API_KEY if missing and writes it into .env (local only)
#   4) Starts infrastructure + app via `make start`
#   5) Creates a Python venv (./venv), installs deps, and seeds sample data
#   6) (Optional) Runs an OpenAI API smoke test (auth check) via scripts/api_smoke_test.py
#
# How to run:
#   From the repo root:
#     bash scripts/rag_setup.sh
#
# Notes:
#   - This script is intentionally explicit (no curl|bash).
#   - The API key is stored in .env. Treat it like a password. Do not commit it.
#   - If your environment already exports OPENAI_API_KEY, your app may prefer it
#     over .env (depending on your config). This script only writes .env.
# -----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "❌ Missing required command: $1" >&2
    exit 1
  }
}

# Basic prerequisites
need git
need python3
need make
need docker

# Compose can be either `docker compose` (v2) or `docker-compose` (v1)
if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  docker compose version >/dev/null 2>&1 || {
    echo "❌ Missing Docker Compose. Install Docker Desktop (recommended) or docker-compose." >&2
    exit 1
  }
  COMPOSE_CMD="docker compose"
fi

# Docker daemon may not be running yet (especially on macOS).
# `make start` will attempt to start Docker Desktop and wait for the daemon.
if ! docker info >/dev/null 2>&1; then
  echo "⚠️  Docker daemon is not running yet. Continuing..." >&2
  echo "    Note: 'make start' will attempt to start Docker Desktop (macOS) and wait for it." >&2
fi

cd "$REPO_ROOT"

echo "📍 Repo root: $REPO_ROOT"

# -----------------------------------------------------------------------------
# 1) Prepare .env
# -----------------------------------------------------------------------------

echo "🧩 Preparing .env ..."
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "✅ Created .env from .env.example"
  else
    : > .env
    echo "✅ Created empty .env"
  fi
else
  echo "ℹ️  .env already exists"
fi

# If OPENAI_API_KEY is missing or empty in .env, prompt and write it.
# (We don't print it back to the console.)
if ! grep -qE '^OPENAI_API_KEY=.+$' .env || grep -qE '^OPENAI_API_KEY=$' .env; then
  echo
  echo "🔑 OPENAI_API_KEY not found in .env"

  while true; do
    read -r -s -p "Enter your OpenAI API key (input hidden) and press Enter: " OPENAI_API_KEY
    echo

    # Re-prompt on empty input
    if [ -z "${OPENAI_API_KEY}" ]; then
      echo "⚠️  API key cannot be empty. Please paste your OpenAI API key." >&2
      continue
    fi

    # Reject whitespace (most common paste mistake)
    if [[ "$OPENAI_API_KEY" =~ [[:space:]] ]]; then
      echo "❌ Invalid OPENAI_API_KEY (contains spaces). Please paste the full key." >&2
      continue
    fi

    break
  done
else
  echo "✅ OPENAI_API_KEY already present in .env (not prompting)"
fi

# -----------------------------------------------------------------------------
# 2) Start services (Qdrant + app)
# -----------------------------------------------------------------------------

echo
echo "🐳 Starting services via make ..."
make start

echo
echo "✅ Services started"

# -----------------------------------------------------------------------------
# 3) Python environment + dependencies + seed
# -----------------------------------------------------------------------------

echo
echo "🐍 Setting up Python environment ..."

if [ ! -d venv ]; then
  python3 -m venv venv
  echo "✅ Created venv/"
else
  echo "ℹ️  venv/ already exists (reusing)"
fi

# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "🌱 Seeding sample data ..."
make seed

deactivate

# -----------------------------------------------------------------------------
# 4) Optional OpenAI API smoke test (non-fatal)
# -----------------------------------------------------------------------------

if [ "${SKIP_OPENAI_SMOKE_TEST:-0}" = "1" ]; then
  echo
  echo "🧪 Skipping OpenAI API smoke test (SKIP_OPENAI_SMOKE_TEST=1)"
else
  echo
  echo "🧪 Running OpenAI API smoke test (auth check) ..."
  if python3 scripts/api_smoke_test.py; then
    echo "✅ Smoke test passed"
  else
    echo "⚠️  Smoke test failed. Setup is still complete, but your API key, budget, or network may need attention." >&2
    echo "    Next steps:" >&2
    echo "      1) Update OPENAI_API_KEY in .env (and verify OpenAI usage limits/budget)" >&2
    echo "      2) Re-run: make start" >&2
    echo "      3) Re-test: python3 scripts/api_smoke_test.py" >&2
  fi
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------

echo
echo "🎉 Setup complete!"
echo "   Open: http://localhost:8000"

# Quick sanity check (non-fatal): show compose tool detected
echo "   Compose detected: $COMPOSE_CMD"