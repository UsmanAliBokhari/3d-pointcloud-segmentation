#!/usr/bin/env bash
# Downloads and unpacks the Paris-rue-Madame dataset into data/.   Usage:  bash data/download.sh
set -euo pipefail
cd "$(dirname "$0")"

URL="http://cmm.mines-paristech.fr/Projects/TerraMobilita/rueMadame_database.zip"
PAGE="https://people.cmm.minesparis.psl.eu/users/serna/rueMadameDataset.html"

if [ -f GT_Madame1_2.ply ] && [ -f GT_Madame1_3.ply ]; then
  echo "Dataset already present."; exit 0
fi

echo "==> Downloading Paris-rue-Madame (this can take a few minutes)"
if ! curl -L --fail -o rueMadame_database.zip "$URL"; then
  cat <<EOF
Automatic download failed. Download the zip manually from:
  $PAGE
put it in data/ as rueMadame_database.zip and run this script again.
EOF
  exit 1
fi

echo "==> Unpacking"
unzip -o -j rueMadame_database.zip '*.ply' -d .
rm -f rueMadame_database.zip
ls -lh GT_Madame1_2.ply GT_Madame1_3.ply
echo "==> Done. Licence: CC BY-NC-ND 3.0, see data/README-data.txt"
