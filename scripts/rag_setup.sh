

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

# Helpful info (non-fatal)
if ! docker info >/dev/null 2>&1; then
  echo "⚠️  Docker daemon does not appear to be running." >&2
  echo "    If you're on macOS, start Docker Desktop and re-run this script." >&2
  exit 1
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
  read -r -s -p "Enter your OpenAI API key (input hidden): " OPENAI_API_KEY
  echo

  if [ -z "${OPENAI_API_KEY}" ]; then
    echo "❌ OPENAI_API_KEY is required." >&2
    exit 1
  fi

  # Remove any existing OPENAI_API_KEY line (empty or old), then append.
  # macOS/BSD sed differs from GNU sed, so we use grep+mv for portability.
  grep -v '^OPENAI_API_KEY=' .env > .env.tmp || true
  mv .env.tmp .env
  printf "OPENAI_API_KEY=%s\n" "$OPENAI_API_KEY" >> .env

  echo "✅ Saved OPENAI_API_KEY to .env"
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
# Done
# -----------------------------------------------------------------------------

echo
echo "🎉 Setup complete!"
echo "   Open: http://localhost:8000"

# Quick sanity check (non-fatal): show compose tool detected
echo "   Compose detected: $COMPOSE_CMD"