#!/usr/bin/env bash
# Shared setup used by install_MAC.sh and install_Linux.sh. Run those instead of this.
# Usage: bash scripts/setup_env.sh <python-executable>
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${1:-python3}"

echo "==> Using $($PY --version) at $(command -v "$PY")"
$PY - <<'EOF'
import sys
if sys.version_info < (3, 9):
    sys.exit("Python >= 3.9 is required (3.11 or 3.12 recommended).")
EOF

if [ ! -d .venv ]; then
  echo "==> Creating virtual environment in .venv/"
  $PY -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing packages from requirements.txt"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt

echo "==> Registering Jupyter kernel 'Python (cv-repro)'"
python -m ipykernel install --user --name cv-repro --display-name "Python (cv-repro)" >/dev/null

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "==> Git setup for the shared notebook"
  # 1. strip outputs/execution counts from notebooks when committing (diffs = code only)
  nbstripout --install || true
  # keep cell ids stable (otherwise nbstripout renumbers them and every commit touches every cell)
  git config filter.nbstripout.clean "\"$PWD/.venv/bin/python\" -m nbstripout --keep-id"
  # 2. notebook-aware diff/merge (cell by cell) as a safety net for real conflicts.
  #    Absolute paths so it also works when the venv is not activated.
  nbdime config-git --enable >/dev/null 2>&1 || true
  git config merge.jupyternotebook.driver "$PWD/.venv/bin/git-nbmergedriver merge %O %A %B %L %P"
  git config diff.jupyternotebook.command "$PWD/.venv/bin/git-nbdiffdriver diff"
  git checkout -- .gitattributes 2>/dev/null || true   # nbdime appends to it; the committed one is complete
else
  echo "NOTE: not inside a git clone, so git notebook tools were not configured."
  echo "      After cloning the repo, run this install script again inside the clone."
fi

echo "==> Smoke test: status board, read from notebooks/project.ipynb"
python code/tests.py | tail -1
echo "==> Smoke test: full pipeline on the toy scene"
python code/main.py --toy --skip-ablation > /dev/null && echo "pipeline OK"

cat <<'EOF'

Setup complete.
  Activate the environment in every new terminal:   source .venv/bin/activate
  Open the project notebook (ALL work happens here): jupyter lab notebooks/project.ipynb
                                                    (choose kernel "Python (cv-repro)")
  Status from a terminal (reads the SAVED notebook): python3 code/tests.py
  Real data (a few hundred MB):                    bash data/download.sh
EOF
