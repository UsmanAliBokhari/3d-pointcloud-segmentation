#!/usr/bin/env bash
# Single command that reproduces all results (course requirement).   Usage:  bash run.sh
# Extra arguments are passed on, e.g.  bash run.sh --max-points 2000000
set -e
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No .venv found -> run  bash install_MAC.sh  or  bash install_Linux.sh  first."
  exit 1
fi

if [ ! -f data/GT_Madame1_2.ply ] || [ ! -f data/GT_Madame1_3.ply ]; then
  bash data/download.sh
fi

python3 code/export_code.py            # plain-Python copy of the notebook functions
python3 code/main.py --config config.yaml "$@"   # runs the functions from notebooks/project.ipynb
echo "Done. Metrics in results/metrics.csv, figures in results/figures/"
