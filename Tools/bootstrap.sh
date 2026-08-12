#!/bin/sh
# bootstrap.sh — materialize the working layout around a fresh Apertura clone.
# Run from the repo root on a new machine (or after a re-clone):
#   sh Tools/bootstrap.sh
#
# The layout it guarantees, all siblings of this repo in the same code/ folder:
#   code/Apertura.xcworkspace   <- written here if missing (Xcode never sees a partial file)
#   code/Apertura               <- this repo
#   code/ObjCTokenizer          <- cloned if missing (required to build)
# The MLX fork is NOT cloned here: it needs the pinned branch and a libmlx.a
# build whose metallib path bakes in absolutely — see SYNC.md.

set -eu
cd "$(dirname "$0")/.."           # repo root
CODE="$(cd .. && pwd)"            # the code/ folder holding all siblings

WS="$CODE/Apertura.xcworkspace"
if [ ! -d "$WS" ]; then
    mkdir "$WS"
    cat > "$WS/contents.xcworkspacedata" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<Workspace
   version = "1.0">
   <FileRef
      location = "group:ObjCTokenizer/ObjCTokenizer.xcodeproj">
   </FileRef>
   <FileRef
      location = "group:Apertura/Apertura.xcodeproj">
   </FileRef>
</Workspace>
EOF
    echo "wrote $WS"
else
    echo "ok    $WS exists"
fi

if [ ! -d "$CODE/ObjCTokenizer" ]; then
    git -C "$CODE" clone https://github.com/apocryphx/ObjCTokenizer.git
else
    echo "ok    ObjCTokenizer sibling exists"
fi

MODELS="/Volumes/Macintosh HD/Users/apocryphx/Models"
if [ ! -d "$MODELS/.git" ]; then
    echo "note  persona repo not present at $MODELS —"
    echo "      git clone git@github.com:apocryphx/isolde-scripture.git \"\$MODELS\""
    echo "      (clone into an empty Models dir, or init+pull if model bundles already live there)"
else
    echo "ok    persona repo present"
fi

echo "done. Open $WS in Xcode. Model bundles + engine deps: see SYNC.md."
