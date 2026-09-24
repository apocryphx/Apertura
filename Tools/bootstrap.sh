#!/bin/sh
# bootstrap.sh — materialize the working layout around a fresh Apertura clone.
# Run from the repo root on a new machine (or after a re-clone):
#   sh Tools/bootstrap.sh
#
# ObjCTokenizer is a git submodule at External/ObjCTokenizer, initialized here if
# the clone wasn't made with --recurse-submodules. The in-repo Apertura.xcworkspace
# references it; there is no outer workspace to generate anymore.
# MLX is NOT handled here: it lives as a sibling ../mlx, pinned via Tools/mlx.pin,
# and libmlx.a must be rebuilt per machine (its metallib path bakes in absolutely) —
# run `sh Tools/build_mlx.sh` (see SYNC.md).

set -eu
cd "$(dirname "$0")/.."           # repo root

if [ ! -f External/ObjCTokenizer/ObjCTokenizer.xcodeproj/project.pbxproj ]; then
    git submodule update --init External/ObjCTokenizer
    echo "ok    ObjCTokenizer submodule initialized"
else
    echo "ok    ObjCTokenizer submodule present"
fi

MODELS="/Volumes/Macintosh HD/Users/apocryphx/Models"
if [ ! -d "$MODELS/.git" ]; then
    echo "note  persona repo not present at $MODELS —"
    echo "      git clone git@github.com:apocryphx/isolde-scripture.git \"\$MODELS\""
    echo "      (clone into an empty Models dir, or init+pull if model bundles already live there)"
else
    echo "ok    persona repo present"
fi

echo "done. Open Apertura.xcworkspace in Xcode. Engine dep: sh Tools/build_mlx.sh. Model bundles: see SYNC.md."
