#!/bin/bash
# ================================================================
# Voxtral Codec Encoder Training — RunPod Launch Script
# Template: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# ================================================================
set -e

echo "============================================"
echo "Voxtral Codec Encoder Training Setup"
echo "============================================"

# ---- Config ----
REPO_URL="https://github.com/VincentGourbin/mlx-voxtral-swift.git"
BRANCH="feat/tts-voxtral"
WORK_DIR="/workspace/voxtral-codec-training"
MODEL_REPO="mistralai/Voxtral-4B-TTS-2603"
MODEL_DIR="${WORK_DIR}/model"
OUTPUT_DIR="${WORK_DIR}/checkpoints"
DATASET_SPLIT="train-clean-360"  # Larger split for real training (104h)

# Training params — adjust based on your GPU
BATCH_SIZE=16          # A100-80G: 16-32, A100-40G: 8-16, 3090: 4-8
MAX_STEPS=100000
MAX_AUDIO_SEC=10.0
WHISPER_MODEL="small"  # "base" is faster, "small" is better quality
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
pip install -q safetensors soundfile librosa pesq pystoi tqdm wandb openai-whisper
# torch is already in the RunPod image

echo ""
echo "[3/5] Downloading TTS model (decoder weights)..."
if [ ! -d "${MODEL_DIR}" ]; then
    pip install -q huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_REPO}', local_dir='${MODEL_DIR}',
                  allow_patterns=['consolidated.safetensors', 'params.json'])
print('Model downloaded!')
"
fi
echo "Model: $(ls -lh ${MODEL_DIR}/consolidated.safetensors 2>/dev/null | awk '{print $5}' || echo 'using sharded')"

echo ""
echo "[4/5] GPU Info"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)}')"

echo ""
echo "[5/5] Launching training..."
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Dataset: LibriSpeech ${DATASET_SPLIT}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python train.py \
    --tts_model_path ${MODEL_DIR} \
    --dataset librispeech \
    --librispeech_split ${DATASET_SPLIT} \
    --output_dir ${OUTPUT_DIR} \
    --device cuda \
    --dtype ${DTYPE} \
    --batch_size ${BATCH_SIZE} \
    --max_steps ${MAX_STEPS} \
    --max_audio_seconds ${MAX_AUDIO_SEC} \
    --whisper_model ${WHISPER_MODEL} \
    --compile \
    --use_wandb \
    --num_workers 8

echo ""
echo "============================================"
echo "Training complete!"
echo "Encoder: ${OUTPUT_DIR}/encoder_final.safetensors"
echo "Swift:   ${OUTPUT_DIR}/encoder_for_swift.safetensors"
echo "============================================"
