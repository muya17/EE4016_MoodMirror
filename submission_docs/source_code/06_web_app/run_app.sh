#!/usr/bin/env bash
set -euo pipefail

# Always run from repository root (directory of this script).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Create a local virtual environment if it does not exist yet.
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

# Activate environment and install dependencies.
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Launch Streamlit app using current integration settings.
python -m streamlit run demo_app.py --server.headless true --server.port 8502
