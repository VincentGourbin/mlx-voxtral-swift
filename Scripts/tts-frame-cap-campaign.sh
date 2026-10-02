#!/bin/zsh
# K-14 — TTS frame-cap campaign: 12 texts (EN/FR, 4 to 202 words) × 3 seeds × 3 packs, one `bench tts` line each
# (frames, text_tokens, frame_cap, truncated). Usage: Scripts/tts-frame-cap-campaign.sh <tag> [packs…]
# Lines land in .local-runs/bench.noindex/k14-<tag>/bench.jsonl; the distribution frames/token is read from them.
set -u
cd "$(dirname "$0")/.."
CLI=.build/xcode/Build/Products/Release/VoxtralCLI
TAG=${1:?tag}; shift
PACKS=("$@")
(( ${#PACKS} )) || PACKS=(tts-4b-4bit tts-4b-6bit tts-4b-mlx)
OUT=.local-runs/bench.noindex/k14-$TAG
for pack in $PACKS; do
  for f in Scripts/tts-frame-cap-texts/*.txt; do
    name=$(basename $f .txt)
    [[ $name == fr* ]] && voice=fr_female || voice=neutral_female
    for seed in 1 2 3; do
      $CLI bench tts --model $pack --voice $voice --text-file $f --seed $seed --passes 1 --warmup 0 --cooldown 0 \
        --tag "$TAG-$name-s$seed" --beacon --out $OUT 2>&1 | grep -aE "^(REFUSED)|Error" || true
    done
  done
done
echo DONE
