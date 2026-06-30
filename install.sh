#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Nexova Skills — self-installer (Option 3)
# Copies skills into a target project's .cursor/skills/ folder.
#
# Usage (run from anywhere):
#   bash install.sh                 # installs ALL skills into ./.cursor/skills
#   bash install.sh /path/to/project   # install into a specific project
#   bash install.sh . demand-forecasting inventory-optimization
#                                   # install only the named skills
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/skills"

# First arg (if it isn't a skill name) = target project root. Default: current dir.
TARGET_ROOT="."
if [ "${1:-}" != "" ] && [ -d "$SRC/${1:-}" ]; then
  : # first arg is a skill name, keep TARGET_ROOT=.
elif [ "${1:-}" != "" ]; then
  TARGET_ROOT="$1"; shift
fi

DEST="$TARGET_ROOT/.cursor/skills"
mkdir -p "$DEST"

if [ "$#" -gt 0 ]; then
  # Selective install
  for name in "$@"; do
    if [ -d "$SRC/$name" ]; then
      cp -r "$SRC/$name" "$DEST/$name"
      echo "  ✓ $name"
    else
      echo "  ✗ not found: $name"
    fi
  done
else
  # Install everything
  cp -r "$SRC/"* "$DEST/"
  echo "  ✓ installed $(ls "$SRC" | wc -l | tr -d ' ') skills"
fi

echo ""
echo "Done → $DEST"
echo "Restart Cursor, then invoke a skill with /skill-name (e.g. /demand-forecasting)."
echo "Commit .cursor/skills/ to git so your team and agents get them too."
