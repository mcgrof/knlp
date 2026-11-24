#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
out=$(realpath -m "$1")
patch=$(realpath tools/reproduce/diskgraph/diskgraph-read-intent.patch)
pin=acaa797a277982c2ac4737a89a9c577eeafdde06
mkdir -p "$out"
case ${DISKGRAPH_WHEEL:-} in
    '') fingerprint=$(sha256sum "$patch" | cut -d' ' -f1) ;;
    *) fingerprint=$(sha256sum "$DISKGRAPH_WHEEL" | cut -d' ' -f1) ;;
esac
receipt="$out/runtime"
if test -f "$receipt" && test "$(cat "$receipt")" = "$fingerprint" && test -x "$out/venv/bin/python"; then
    exit
fi
if [[ -n ${DISKGRAPH_WHEEL:-} ]]; then
    wheel=$(realpath "$DISKGRAPH_WHEEL")
else
    src="$out/source-$fingerprint"
    if ! test -f "$src/.patched"; then
        mkdir -p "$src"
        git -C "$src" init -q
        git -C "$src" fetch --depth=1 https://github.com/NVIDIA/kumo-engine.git "$pin"
        test "$(git -C "$src" rev-parse FETCH_HEAD)" = "$pin"
        git -C "$src" archive "$pin" packages/diskgraph | tar -x -C "$src"
        git -C "$src" apply --directory=packages/diskgraph "$patch"
        touch "$src/.patched"
    fi
    mkdir -p "$src/wheels"
    (cd "$src/packages/diskgraph" && uv build --wheel --out-dir "$src/wheels")
    wheels=("$src"/wheels/diskgraph-*.whl)
    test "${#wheels[@]}" = 1
    wheel=${wheels[0]}
fi
if ! test -x "$out/venv/bin/python"; then uv venv --python 3.12 "$out/venv"; fi
uv pip install --python "$out/venv/bin/python" --no-deps --reinstall "$wheel"
uv pip install --python "$out/venv/bin/python" 'numpy==2.2.6' 'pyarrow==18.1.0'
"$out/venv/bin/python" -c 'import diskgraph, numpy, pyarrow'
printf '%s\n' "$fingerprint" >"$receipt"
