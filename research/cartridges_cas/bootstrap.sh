#!/bin/bash
# Bootstrap the CAS harness: clone the pinned HazyResearch cartridges package,
# install it, apply the knlp patches (compiled FlexAttention on CUDA + teacher
# top-k flatten edge-case fix), and drop the CAS scripts in place. Idempotent.
#
# Env:
#   CART_ROOT   where to place the cartridges checkout (default /root/cartridges)
#   PYTHON      interpreter with a CUDA torch already installed (default python)
set -eu

CART_PIN="8cb6823"   # HazyResearch/cartridges commit this harness was built on
CART_ROOT="${CART_ROOT:-/root/cartridges}"
PYTHON="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$CART_ROOT/.git" ] && [ ! -f "$CART_ROOT/pyproject.toml" ]; then
  if [ -n "${CART_TARBALL:-}" ]; then
    # supply-chain fallback: upstream rewrote history and the pinned commit
    # is no longer fetchable (observed 2026-08-26); unpack a git-archive of
    # the pin instead (kept on prune at /data/cartridges, `git archive $CART_PIN`)
    echo "[bootstrap] unpacking pinned tree from $CART_TARBALL"
    mkdir -p "$CART_ROOT"
    tar xzf "$CART_TARBALL" -C "$CART_ROOT"
  else
    echo "[bootstrap] cloning HazyResearch/cartridges @ $CART_PIN"
    git clone https://github.com/HazyResearch/cartridges "$CART_ROOT"
    git -C "$CART_ROOT" checkout "$CART_PIN" || {
      echo "[bootstrap] pinned commit $CART_PIN is gone upstream; set" >&2
      echo "[bootstrap] CART_TARBALL to a git-archive of the pin and rerun" >&2
      exit 1
    }
  fi
fi

echo "[bootstrap] installing deps (torch left untouched)"
"$PYTHON" -m pip install --no-input openai datasets "transformers>=4.49.0,<=4.55" \
  numpy einops tqdm wandb pydrantic tiktoken peft evaluate matplotlib markdown pyarrow ninja
"$PYTHON" -m pip install --no-input --no-deps -e "$CART_ROOT"

echo "[bootstrap] applying knlp patches"
CART_ROOT="$CART_ROOT" "$PYTHON" "$HERE/scripts/apply_pod_patches.py" --cart-root "$CART_ROOT" || true

echo "[bootstrap] placing CAS scripts"
cp -f "$HERE"/scripts/cas_*.py "$CART_ROOT/"
cp -f "$HERE"/scripts/synth_pod.py "$CART_ROOT/"
cp -f "$HERE"/scripts/cas_joint.py "$CART_ROOT/cartridges/initialization/"

echo "[bootstrap] done. CART_ROOT=$CART_ROOT (pin $CART_PIN)"
