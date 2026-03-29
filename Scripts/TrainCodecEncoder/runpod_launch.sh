#!/bin/bash
# ================================================================
# Voxtral Codec Encoder Training — RunPod Launch Script
# Template: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# Recommended GPU: B200 (180GB VRAM) or A100 (40-80GB)
# ================================================================
set -e

echo "============================================"
echo "Voxtral Codec Encoder Training"
echo "============================================"

# ---- Config ----
REPO_URL="https://github.com/VincentGourbin/mlx-voxtral-swift.git"
BRANCH="feat/tts-voxtral"
WORK_DIR="/workspace/voxtral-codec-training"
MODEL_REPO="mistralai/Voxtral-4B-TTS-2603"
MODEL_DIR="${WORK_DIR}/model"
OUTPUT_DIR="${WORK_DIR}/checkpoints"

# Dataset options:
#   "mls_french"    → French only (~1K hours, fast for demo)
#   "multilingual"  → All 9 Voxtral languages (MLS + FLEURS)
#   "librispeech"   → English only
DATASET="mls_french"

# Detect GPU and auto-tune batch size
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "$GPU_MEM" ]; then
    echo "No GPU detected! Use --device cpu"
    exit 1
fi
echo "GPU Memory: ${GPU_MEM} MiB"

if [ "$GPU_MEM" -gt 100000 ]; then
    # B200 / H100-80G: go big
    BATCH_SIZE=64
    MAX_AUDIO_SEC=10.0
    WHISPER_MODEL="medium"
    NUM_WORKERS=16
elif [ "$GPU_MEM" -gt 60000 ]; then
    # A100-80G
    BATCH_SIZE=32
    MAX_AUDIO_SEC=10.0
    WHISPER_MODEL="small"
    NUM_WORKERS=8
elif [ "$GPU_MEM" -gt 30000 ]; then
    # A100-40G / A6000
    BATCH_SIZE=16
    MAX_AUDIO_SEC=8.0
    WHISPER_MODEL="small"
    NUM_WORKERS=8
else
    # 3090 / 4090
    BATCH_SIZE=8
    MAX_AUDIO_SEC=5.0
    WHISPER_MODEL="base"
    NUM_WORKERS=4
fi

MAX_STEPS=100000
DTYPE="bfloat16"

# ---- Setup ----
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

echo ""
echo "[1/5] Cloning repo..."
if [ ! -d "mlx-voxtral-swift" ]; then
    git clone --branch ${BRANCH} --depth 1 ${REPO_URL}
fi
cd mlx-voxtral-swift/Scripts/TrainCodecEncoder

echo ""
echo "[2/5] Installing dependencies..."
apt-get update -qq && apt-get install -y -qq libsndfile1 ffmpeg > /dev/null 2>&1 || true
pip uninstall -y torchcodec > /dev/null 2>&1 || true
pip install -q safetensors soundfile librosa pesq pystoi tqdm wandb openai-whisper "datasets<3.6" huggingface_hub

echo ""
echo "[3/5] Downloading model weights..."
if [ ! -f "${MODEL_DIR}/consolidated.safetensors" ]; then
    pip install -q huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_REPO}', local_dir='${MODEL_DIR}',
                  allow_patterns=['consolidated.safetensors', 'params.json'])
print('Model downloaded!')
"
fi

echo ""
echo "[4/5] System info"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

echo ""
echo "[5/5] Launching training..."
echo "  GPU Memory:    ${GPU_MEM} MiB"
echo "  Batch size:    ${BATCH_SIZE}"
echo "  Dataset:       ${DATASET} (9 languages)"
echo "  Whisper:       ${WHISPER_MODEL}"
echo "  Max audio:     ${MAX_AUDIO_SEC}s"
echo "  Max steps:     ${MAX_STEPS}"
echo "  Dtype:         ${DTYPE}"
echo "  Output:        ${OUTPUT_DIR}"
echo ""

python train.py \
    --tts_model_path ${MODEL_DIR} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --device cuda \
    --dtype ${DTYPE} \
    --batch_size ${BATCH_SIZE} \
    --max_steps ${MAX_STEPS} \
    --max_audio_seconds ${MAX_AUDIO_SEC} \
    --whisper_model ${WHISPER_MODEL} \
    --num_workers ${NUM_WORKERS}
