#!/usr/bin/env bash
# Clone and pin HeST into external/ (gitignored). Upstream is MIT (copyright line
# still the PyPA template — gate B0). We use it as a library and never patch the
# clone; defects go upstream as PRs. Dependencies HeST needs at import time:
#   pip install qetpy numba      (its setup.py also lists detprocess, whose 'aplus'
#                                 dependency does not build on modern Python — skip it)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF="${HEST_REF:-8ffd23e82dedbfbf531bb3ac58e789a81fd6e764}"   # pinned: 2026-03-08, 89 commits
mkdir -p "$HERE/external"
if [ ! -d "$HERE/external/HeST/.git" ]; then
  git clone https://github.com/spice-herald/HeST.git "$HERE/external/HeST"
fi
cd "$HERE/external/HeST"
git fetch --all --tags --quiet || true
git checkout --quiet "$REF" 2>/dev/null || echo "WARNING: could not check out $REF (shallow clone?)"
git rev-parse HEAD > "$HERE/external/hest_commit.txt"
echo "HeST at $(cat "$HERE/external/hest_commit.txt")"
