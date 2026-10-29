#!/usr/bin/env bash
# One-command setup for InterviewTrack AI (Demo Mode needs nothing else).
#
#   ./scripts/setup.sh
#   streamlit run app.py      # then click "Start Demo Mode"
#
# Idempotent: safe to re-run. Creates a local venv in .venv.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.11}"
echo "==> InterviewTrack AI setup"
echo "==> Python: $($PYTHON --version)"

if [ ! -d .venv ]; then
  echo "==> Creating virtualenv (.venv)"
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --quiet --upgrade pip

echo "==> Installing core requirements"
python -m pip install --quiet -r requirements-core.txt

if [ "${INSTALL_NEURAL:-0}" = "1" ]; then
  echo "==> Installing optional neural dependencies"
  python -m pip install --quiet -r requirements.txt
fi

export EMBEDDINGS_ENABLED="${EMBEDDINGS_ENABLED:-false}"

echo "==> Running tests (offline, no model download)"
python -m pytest -q || { echo "!! tests failed"; exit 1; }

cat <<'DONE'

==> Setup complete.

   Demo Mode (no credentials needed):
       source .venv/bin/activate
       streamlit run app.py        # click "Start Demo Mode"

   Or headless:
       python scripts/demo.py                  # scenario reconstruction
       python scripts/evaluate_cross_inbox.py  # 3 mailbox profiles

   Optional Gmail Mode:
       python -m interviewtrack.auth           # one-time, read-only
DONE
