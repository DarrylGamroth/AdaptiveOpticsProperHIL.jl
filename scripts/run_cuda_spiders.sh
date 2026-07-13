#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-spiders}
REMOTE_HOME=${REMOTE_HOME:-$(ssh "$REMOTE" 'printf %s "$HOME"')}
REMOTE_BASE=${REMOTE_BASE:-$REMOTE_HOME/workspaces/codex/hil_cuda_work}
REMOTE_ROOT=${REMOTE_ROOT:-$REMOTE_BASE/AdaptiveOpticsProperHIL.jl}
LOCAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOCAL_WORKSPACE=$(cd "$LOCAL_ROOT/.." && pwd)

ssh "$REMOTE" "mkdir -p '$REMOTE_BASE'"
rsync -az --delete \
  --exclude .git \
  --exclude Manifest.toml \
  --exclude results \
  "$LOCAL_WORKSPACE/AdaptiveOpticsSim.jl/" "$REMOTE:$REMOTE_BASE/AdaptiveOpticsSim.jl/"
rsync -az --delete \
  --exclude .git \
  --exclude Manifest.toml \
  --exclude 'env/*/Manifest.toml' \
  --exclude results \
  "$LOCAL_ROOT/" "$REMOTE:$REMOTE_ROOT/"
ssh "$REMOTE" "cd $REMOTE_ROOT && julia --project=env/cuda -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()' && julia --project=env/cuda scripts/validate_cuda.jl && julia --project=env/cuda scripts/benchmark_cuda.jl \"${1:-20}\" \"${2:-5.0}\" \"${3:-128}\""
