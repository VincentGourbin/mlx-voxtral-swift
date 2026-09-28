#!/bin/bash
# =============================================================================
# Voxtral Core ML Encoder Conversion Script
# =============================================================================
# This script converts the Voxtral audio encoder to Core ML format for use
# with the Apple Neural Engine (ANE) or GPU.
#
# Prerequisites:
#   - Python 3.11-3.13 as `python3` (wheels of the pinned torch 2.7.0,
#     coremltools 9.0 and numpy 2.3.5; see requirements.txt)
#   - ~10GB disk space for intermediate files
#   - Voxtral model weights (downloaded automatically, pinned revision below)
#
# Output:
#   - VoxtralEncoderFull.mlmodelc (~1.2GB) - Ready to use with VoxtralApp
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
MODEL_DIR="${SCRIPT_DIR}/voxtral-mini-3b"
# Pinned Hub revision of mistralai/Voxtral-Mini-3B-2507 (file listing identical to
# `main` on 2026-09-28; repo last updated 2025-07-28).
MODEL_REVISION="3060fe34b35ba5d44202ce9ff3c097642914f8f3"

echo "============================================================"
echo "VOXTRAL CORE ML ENCODER CONVERSION"
echo "============================================================"

# Step 1: Create virtual environment if needed
if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
    echo ""
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv "${SCRIPT_DIR}/.venv"
fi

# Activate virtual environment
source "${SCRIPT_DIR}/.venv/bin/activate"

# Step 2: Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install -q -r "${SCRIPT_DIR}/requirements.txt"

# Step 3: Download model if needed
if [ ! -d "${MODEL_DIR}" ]; then
    echo ""
    echo "[3/5] Downloading Voxtral model from HuggingFace..."
    # `hf` comes from the pinned huggingface_hub in requirements.txt
    # (huggingface_hub 2.0 no longer ships `huggingface-cli`).
    hf download mistralai/Voxtral-Mini-3B-2507 --revision "${MODEL_REVISION}" --local-dir "${MODEL_DIR}"
else
    echo ""
    echo "[3/5] Model already downloaded at ${MODEL_DIR}"
fi

# Step 4: Extract encoder weights
mkdir -p "${OUTPUT_DIR}"
WEIGHTS_FILE="${OUTPUT_DIR}/voxtral_encoder.pt"

if [ ! -f "${WEIGHTS_FILE}" ]; then
    echo ""
    echo "[4/5] Extracting encoder weights..."
    cd "${SCRIPT_DIR}"
    python convert_weights.py --model-path "${MODEL_DIR}" --variant mini --output "${WEIGHTS_FILE}"
else
    echo ""
    echo "[4/5] Weights already extracted at ${WEIGHTS_FILE}"
fi

# Step 5: Convert to Core ML
# The ANE model always includes the multimodal projector
# (ANEVoxtralEncoderWithProjector); --variant sets its output width
# (mini = 3072). This script converts Mini only; for Small see README.md.
echo ""
echo "[5/5] Converting to Core ML..."
cd "${SCRIPT_DIR}"
python convert_to_coreml_ane.py \
    --weights "${WEIGHTS_FILE}" \
    --variant mini \
    --output "${OUTPUT_DIR}/VoxtralEncoderFull.mlpackage"

# Step 6: Compile to mlmodelc (faster loading)
echo ""
echo "[6/6] Compiling Core ML model..."
xcrun coremlcompiler compile "${OUTPUT_DIR}/VoxtralEncoderFull.mlpackage" "${OUTPUT_DIR}/"

echo ""
echo "============================================================"
echo "CONVERSION COMPLETE!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - ${OUTPUT_DIR}/VoxtralEncoderFull.mlmodelc (use this)"
echo "  - ${OUTPUT_DIR}/VoxtralEncoderFull.mlpackage"
echo ""
echo "To use with VoxtralApp:"
echo "  1. Copy VoxtralEncoderFull.mlmodelc to Sources/VoxtralApp/Resources/"
echo "  2. Rebuild with xcodebuild, not swift build (Metal shaders; see CLAUDE.md):"
echo "     xcodebuild -scheme VoxtralApp -configuration Release -derivedDataPath .build/xcode -destination 'platform=macOS' build"
echo ""
echo "Or set environment variable:"
echo "  export VOXTRAL_RESOURCES_PATH=${OUTPUT_DIR}"
echo ""
