#!/bin/sh
# build_mlx.sh — clone/check out and build the MLX static library the engine links against.
#   sh Tools/build_mlx.sh            clone if missing, check out the pin, build
#   sh Tools/build_mlx.sh --status   report pin vs checkout, no mutation
# The pinned commit lives in Tools/mlx.pin (single source of truth — bump deliberately).
# The checkout lives as a sibling of this repo (../mlx). The build is machine-specific
# (libmlx.a bakes in an absolute metallib path) — rebuild per machine, never sync.
# A dirty checkout away from the pin is never touched: stash or commit first.

set -eu
cd "$(dirname "$0")/.."            # repo root
CODE="$(cd .. && pwd)"             # the code/ folder holding all siblings
MLX="$CODE/mlx"

PIN="$(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' Tools/mlx.pin | head -1 | tr -d '[:space:]')"
[ -n "$PIN" ] || { echo "error empty Tools/mlx.pin" >&2; exit 1; }

if [ ! -d "$MLX" ]; then
    git -C "$CODE" clone https://github.com/ml-explore/mlx.git
fi

HEAD_NOW="$(git -C "$MLX" rev-parse HEAD)"
DIRTY="$(git -C "$MLX" status --porcelain)"

if [ "${1:-}" = "--status" ]; then
    echo "pin    $PIN"
    echo "HEAD   $HEAD_NOW ($(git -C "$MLX" describe --tags --always))"
    [ -n "$DIRTY" ] && echo "state  dirty" || echo "state  clean"
    [ "$HEAD_NOW" = "$PIN" ] && echo "match  yes" || echo "match  no"
    exit 0
fi

if [ "$HEAD_NOW" != "$PIN" ]; then
    if [ -n "$DIRTY" ]; then
        echo "error ../mlx is dirty and not at the pin — stash or commit there first" >&2
        exit 1
    fi
    git -C "$MLX" cat-file -e "$PIN^{commit}" 2>/dev/null || git -C "$MLX" fetch origin
    git -C "$MLX" checkout --detach "$PIN"
elif [ -n "$DIRTY" ]; then
    echo "warn  ../mlx has uncommitted changes on top of the pin — building as-is"
fi

# A build tree configured at a different source path (checkout moved) poisons both the
# CMake cache and the absolute metallib path baked into libmlx.a — start clean.
CACHE="$MLX/build/CMakeCache.txt"
if [ -f "$CACHE" ] && [ "$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$CACHE")" != "$MLX" ]; then
    echo "warn  stale CMake cache (checkout was moved) — wiping ../mlx/build for a clean build"
    rm -rf "$MLX/build"
fi

cmake -S "$MLX" -B "$MLX/build" -DBUILD_SHARED_LIBS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF
cmake --build "$MLX/build" --target mlx

LIB="$MLX/build/libmlx.a"
METALLIB="$MLX/build/mlx/backend/metal/kernels/mlx.metallib"
[ -f "$LIB" ]      || { echo "error missing $LIB" >&2; exit 1; }
[ -f "$METALLIB" ] || { echo "error missing $METALLIB" >&2; exit 1; }

echo "built mlx $(git -C "$MLX" describe --tags --always) -> $LIB"
echo "      $METALLIB"
