#!/usr/bin/env bash
# One-command setup on macOS (Intel or Apple Silicon).   Usage:  bash install_MAC.sh
set -euo pipefail
cd "$(dirname "$0")"

# Pick the newest suitable Python: Homebrew 3.12/3.11 first, then whatever python3 is.
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    PY="$c"; break
  fi
done

if [ -z "$PY" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "==> No Python >= 3.9 found, installing python@3.12 with Homebrew"
    brew install python@3.12
    PY="$(brew --prefix)/bin/python3.12"
  else
    cat <<'EOF'
No Python >= 3.9 found and Homebrew is not installed. Do ONE of these, then rerun:
  a) Install Homebrew:   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  b) Install Python 3.12 from https://www.python.org/downloads/macos/
EOF
    exit 1
  fi
fi

if ! xcode-select -p >/dev/null 2>&1; then
  echo "NOTE: Command Line Tools missing (needed for git). Run:  xcode-select --install"
fi

bash scripts/setup_env.sh "$PY"
