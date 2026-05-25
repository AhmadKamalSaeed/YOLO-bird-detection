#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these lines, everything else is automatic.
# ---------------------------------------------------------------------------
VIDEO="/Users/ahmadkamal/Documents/Spoor/birds_2.mp4"
OUTPUT_DIR="/Users/ahmadkamal/Documents/Spoor/output"
SHOW_PREVIEW=false   # set to true to open the live cv2 preview window
SAVE_VIDEO=true      # set to false for CSV-only (faster, no .mp4 written)

# ---------------------------------------------------------------------------
# Resolve paths and activate the venv.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV" ]]; then
    echo "Error: venv not found at $VENV"
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

if [[ ! -f "$VIDEO" ]]; then
    echo "Error: video not found at $VIDEO"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Video:        $VIDEO"
echo "Output dir:   $OUTPUT_DIR"
echo "Show preview: $SHOW_PREVIEW"
echo "Save video:   $SAVE_VIDEO"
echo ""

EXTRA_FLAGS=()
[[ "$SHOW_PREVIEW" == true ]]  && EXTRA_FLAGS+=(--show-preview)
[[ "$SAVE_VIDEO"   == false ]] && EXTRA_FLAGS+=(--no-save-video)

"$VENV" "$SCRIPT_DIR/main.py" \
    --video "$VIDEO" \
    --output-dir "$OUTPUT_DIR" \
    ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
