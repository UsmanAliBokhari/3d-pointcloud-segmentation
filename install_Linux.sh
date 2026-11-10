#!/usr/bin/env bash
# One-command setup on Linux (Ubuntu/Debian/Fedora/Arch...).   Usage:  bash install_Linux.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    PY="$c"; break
  fi
done

if [ -z "$PY" ]; then
  cat <<'EOF'
No Python >= 3.9 found. Install it, then rerun:
  Ubuntu/Debian:  sudo apt update && sudo apt install python3 python3-venv python3-pip git
  Fedora:         sudo dnf install python3 python3-pip git
  Arch:           sudo pacman -S python python-pip git
EOF
  exit 1
fi

if ! "$PY" -c 'import venv, ensurepip' >/dev/null 2>&1; then
  echo "The 'venv' module is missing. On Ubuntu/Debian run:  sudo apt install python3-venv"
  exit 1
fi

bash scripts/setup_env.sh "$PY"
