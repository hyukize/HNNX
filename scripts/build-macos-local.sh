#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="0.1.16"
ARTIFACT="$ROOT_DIR/dist/HNNX-${VERSION}-arm64.dmg"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "HNNX macOS builds must run on macOS." >&2
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "HNNX currently supports local macOS packaging on Apple Silicon (arm64) only." >&2
    exit 1
fi

for command in node npm codesign; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done

cd "$ROOT_DIR"

echo "HNNX local macOS build"
echo "Node: $(node --version)"
echo "npm:  $(npm --version)"

if [[ "${1:-}" == "--skip-install" ]]; then
    if [[ ! -d node_modules ]]; then
        echo "node_modules is missing; rerun without --skip-install." >&2
        exit 1
    fi
else
    npm ci
fi

npm run build:mac-hnnx

if [[ ! -f "$ARTIFACT" ]]; then
    echo "Expected build artifact was not created: $ARTIFACT" >&2
    exit 1
fi

echo
echo "Build complete: $ARTIFACT"
shasum -a 256 "$ARTIFACT"
echo
echo "Open the DMG and copy HNNX to Applications."
echo "This locally built app is ad-hoc signed, not Apple Developer ID signed or notarized."
