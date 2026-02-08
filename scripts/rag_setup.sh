#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Chat with RAG — One-command setup (macOS/Linux)
# -----------------------------------------------------------------------------
# What this script does:
#   1) Validates basic prerequisites (git, python3, make, docker, compose)
#   2) Ensures a local .env exists (copies from .env.example if present)
#   3) Prompts for OPENAI_API_KEY and/or GEMINI_API_KEY if missing and writes them into .env (local only)
#   4) Starts infrastructure + app via `make start`
#   5) Creates a Python venv (./venv), installs deps, and seeds sample data
#   6) (Optional) Runs API smoke tests (auth check) via scripts/api_smoke_test_*.py
#
# How to run:
#   From the repo root:
#     bash scripts/rag_setup.sh
#
# Notes:
#   - This script is intentionally explicit (no curl|bash).
#   - The API keys are stored in .env. Treat them like passwords. Do not commit them.
#   - If your environment already exports OPENAI_API_KEY or GEMINI_API_KEY, your app may prefer them
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

# If API keys are missing or empty in .env, prompt and write them.
# (We don't print them back to console.)
if ! grep -qE '^OPENAI_API_KEY=.+$' .env || ! grep -qE '^GEMINI_API_KEY=.+$' .env || ! grep -qE '^OPENAI_API_KEY=$' .env || ! grep -qE '^GEMINI_API_KEY=$' .env; then
  echo
  echo "🔑 No API keys found in .env"
  echo

  # Prompt for OpenAI API Key
  while true; do
    read -r -s -p "Enter your OpenAI API key (or press Enter to skip): " OPENAI_API_KEY
    echo

    # Accept empty input (skip OpenAI)
    if [ -z "${OPENAI_API_KEY}" ]; then
      echo "ℹ️  Skipping OpenAI API key"
      break
    fi

    # Reject whitespace (most common paste mistake)
    if [[ "$OPENAI_API_KEY" =~ [[:space:]] ]]; then
      echo "❌ Invalid OPENAI_API_KEY (contains spaces). Please paste the full key." >&2
      continue
    fi

    break
  done

  echo
  
  # Prompt for Gemini API Key
  while true; do
    read -r -s -p "Enter your Gemini API key (or press Enter to skip): " GEMINI_API_KEY
    echo

    # Accept empty input (skip Gemini)
    if [ -z "${GEMINI_API_KEY}" ]; then
      echo "ℹ️  Skipping Gemini API key"
      break
    fi

    # Reject whitespace (most common paste mistake)
    if [[ "$GEMINI_API_KEY" =~ [[:space:]] ]]; then
      echo "❌ Invalid GEMINI_API_KEY (contains spaces). Please paste the full key." >&2
      continue
    fi

    break
  done

  echo

  # Require at least one API key
  if [ -z "${OPENAI_API_KEY}" ] && [ -z "${GEMINI_API_KEY}" ]; then
    echo "❌ At least one API key (OpenAI or Gemini) is required to continue." >&2
    exit 1
  fi

  # Remove any existing API key lines, then append new ones.
  # macOS/BSD sed differs from GNU sed, so we use grep+mv for portability.
  grep -v '^OPENAI_API_KEY=' .env > .env.tmp || true
  grep -v '^GEMINI_API_KEY=' .env.tmp > .env.tmp2 || true
  mv .env.tmp2 .env
  
  # Append OpenAI key if provided
  if [ -n "${OPENAI_API_KEY}" ]; then
    printf "OPENAI_API_KEY=%s\n" "$OPENAI_API_KEY" >> .env
    echo "✅ Saved OPENAI_API_KEY to .env"
  fi
  
  # Append Gemini key if provided
  if [ -n "${GEMINI_API_KEY}" ]; then
    printf "GEMINI_API_KEY=%s\n" "$GEMINI_API_KEY" >> .env
    echo "✅ Saved GEMINI_API_KEY to .env"
  fi
  
  # Clean up temp files
  rm -f .env.tmp .env.tmp2
else
  echo "✅ API keys already present in .env (not prompting)"
fi

set -a
source .env
set +a

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

# Quick sanity check (non-fatal): show compose tool detected
echo "   Compose detected: $COMPOSE_CMD"
