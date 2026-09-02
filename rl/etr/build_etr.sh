#!/usr/bin/env bash
# Build Extreme Tux Racer 0.8.4 with the headless --rl bridge.
#
#   rl/etr/build_etr.sh [SRC_DIR]
#
# SRC_DIR defaults to ~/devel/extremetuxracer-0.8.4. If it does not
# exist the Debian source package is fetched there with apt-get source
# (needs deb-src entries). The patches in rl/etr/patches/ are applied
# in order (each skipped when the tree already carries it), then autoreconf,
# configure with the Debian data paths and make. The binary lands at
# SRC_DIR/src/etr; export ETR_RL_BIN to point the Python side at it.
#
# Build dependencies (Debian): build-essential autoconf automake
# pkgconf libsfml-dev (>= 3.0) libglu1-mesa-dev. Runtime data comes
# from the distribution package (extremetuxracer-data at
# /usr/share/games/etr), which the configure prefix below matches.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-$HOME/devel/extremetuxracer-0.8.4}"
PATCH="$HERE/patches/0001-Add-a-headless-rl-bridge-for-reinforcement-learning.patch"
JOBS="${JOBS:-$(nproc)}"

if [ ! -d "$SRC" ]; then
  echo "fetching extremetuxracer source into $(dirname "$SRC")"
  mkdir -p "$(dirname "$SRC")"
  (cd "$(dirname "$SRC")" && apt-get source extremetuxracer=0.8.4-2)
fi

cd "$SRC"
if [ ! -d .git ]; then
  git init -q
  git add -A
  git -c user.name=build -c user.email=build@localhost commit -q -m "Import Extreme Tux Racer 0.8.4"
fi

for patch in "$HERE"/patches/*.patch; do
  if git apply --check --reverse "$patch" >/dev/null 2>&1; then
    echo "already applied: $(basename "$patch")"
    continue
  fi
  echo "applying $(basename "$patch")"
  git am -q "$patch" || { echo "patch failed: $patch" >&2; exit 1; }
done

autoreconf -fi >/dev/null
./configure --prefix=/usr --bindir=/usr/games --datadir=/usr/share/games >/dev/null
make -j"$JOBS" >/dev/null
echo "built: $SRC/src/etr"
echo "export ETR_RL_BIN=$SRC/src/etr"
