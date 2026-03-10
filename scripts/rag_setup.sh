#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Chat with RAG — One-command setup (macOS/Linux)
# -----------------------------------------------------------------------------
# What this script does:
#   1) Validates basic prerequisites (git, python3, make, docker, compose)
#   2) Ensures a local .env exists (copies from .env.example if present)
#   3) Starts infrastructure + app via `make start`
#   4) Creates a Python venv (./venv), installs deps, and seeds sample data
#   5) (Optional) Runs API smoke tests (auth check) via scripts/api_smoke_test_*.py
#
# How to run:
#   From the repo root:
#     bash scripts/rag_setup.sh
#
# Notes:
#   - This script is intentionally explicit (no curl|bash).
#   - After setup completes, add your API keys to .env (OPENAI_API_KEY and/or GEMINI_API_KEY)
#   - Treat API keys like passwords. Do not commit them.
#   - If your environment already exports OPENAI_API_KEY or GEMINI_API_KEY, your app may prefer them
#     over .env (depending on your config).
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

# Note: API keys should be added to .env after setup completes

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
# 4) Optional API smoke tests (non-fatal)
# -----------------------------------------------------------------------------

if [ "${SKIP_API_SMOKE_TESTS:-0}" = "1" ]; then
  echo
  echo "🧪 Skipping API smoke tests (SKIP_API_SMOKE_TESTS=1)"
else
  echo
  echo "🧪 Running API smoke tests (auth checks) ..."
  
  # Test OpenAI if key is present
  if grep -qE '^OPENAI_API_KEY=.+$' .env || grep -qE '^OPENAI_API_KEY=$' .env; then
    echo "🧪 Testing OpenAI API ..."
    if python3 scripts/api_smoke_test_openai.py; then
      echo "✅ OpenAI smoke test passed"
    else
      echo "⚠️  OpenAI smoke test failed. Your API key, budget, or network may need attention." >&2
      echo "    Next steps:" >&2
      echo "      1) Update OPENAI_API_KEY in .env (and verify OpenAI usage limits/budget)" >&2
      echo "      2) Re-run: python3 scripts/api_smoke_test_openai.py" >&2
    fi
  else
    echo "ℹ️  No OpenAI API key found - skipping OpenAI smoke test"
  fi
  
  # Test Gemini if key is present
  if grep -qE '^GEMINI_API_KEY=.+$' .env || grep -qE '^GEMINI_API_KEY=$' .env; then
    echo "🧪 Testing Gemini API ..."
    if python3 scripts/api_smoke_test_gemini.py; then
      echo "✅ Gemini smoke test passed"
    else
      echo "⚠️  Gemini smoke test failed. Your API key, quota, or network may need attention." >&2
      echo "    Next steps:" >&2
      echo "      1) Update GEMINI_API_KEY in .env (and verify Google AI Studio quotas)" >&2
      echo "      2) Re-run: python3 scripts/api_smoke_test_gemini.py" >&2
    fi
  else
    echo "ℹ️  No Gemini API key found - skipping Gemini smoke test"
  fi
  
  # Overall status
  echo
  echo "🎯 API smoke tests completed"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------

echo
echo "🎉 Setup complete!"
echo "   Open: http://localhost:8000"
echo
echo "📝 Next step: Add your API keys to .env:"
echo "   OPENAI_API_KEY=your_openai_key_here"
echo "   GEMINI_API_KEY=your_gemini_key_here"

# Quick sanity check (non-fatal): show compose tool detected
echo "   Compose detected: $COMPOSE_CMD"
