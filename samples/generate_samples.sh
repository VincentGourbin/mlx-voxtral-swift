#!/bin/bash
# Generate TTS demo audio samples for the repository
# Run this script from the project root after building with xcodebuild:
#   xcodebuild -scheme VoxtralCLI -configuration Debug -derivedDataPath .build/xcode \
#     -destination 'platform=macOS' -skipPackagePluginValidation build
#   bash samples/generate_samples.sh

set -e

# Use xcodebuild binary (has proper Metal/GPU access)
CLI="./.build/xcode/Build/Products/Debug/VoxtralCLI"
if [ ! -f "$CLI" ]; then
    echo "CLI not found at $CLI"
    echo "Build first with: xcodebuild -scheme VoxtralCLI -configuration Debug -derivedDataPath .build/xcode -destination 'platform=macOS' -skipPackagePluginValidation build"
    exit 1
fi
SAMPLES_DIR="samples"

echo "=== Generating TTS demo samples ==="
echo ""

# English sample
echo "[1/2] Generating English sample..."
$CLI tts \
    "Trained on a large speech dataset, Voxtral TTS is built for global application. It supports state-of-the-art performance in 9 languages: English, French, German, Spanish, Dutch, Portuguese, Italian, Hindi, and Arabic. The model was trained to adapt to a custom voice with a reference as little as 3s and capture not just the voice but also nuances like subtle accent, inflections, intonations and even disfluencies similar to those expressed in the reference. We offer some preset voice options in the API but it is simple to extend to your in-house voice library customizing it to the use-case, localize it to the language and accent, keep it neutral or more emotive, casual or formal, more natural and conversational or robotic. The model also demonstrates zero-shot cross-lingual voice adaptation even though it is not explicitly trained for it. For example, the model can generate English speech with a French voice prompt and English text. The resulting speech sounds natural while adopting the accent of the provided voice prompt. In this example, the generated speech has a natural French-accented English. This makes the model useful for building cascaded speech-to-speech translation systems." \
    -v neutral_female \
    -o "$SAMPLES_DIR/en_voxtral_tts_demo.wav"

echo ""

# French sample
echo "[2/2] Generating French sample..."
$CLI tts \
    "Entraîné sur un large jeu de données vocales, Voxtral TTS est conçu pour une utilisation mondiale. Il offre des performances à l'état de l'art dans 9 langues : anglais, français, allemand, espagnol, néerlandais, portugais, italien, hindi et arabe. Le modèle a été entraîné pour s'adapter à une voix personnalisée à partir d'une référence d'à peine 3 secondes, et pour capturer non seulement la voix, mais aussi les nuances comme l'accent subtil, les inflexions, les intonations et même les hésitations similaires à celles exprimées dans la référence. Nous proposons des voix prédéfinies dans l'API, mais il est simple de l'étendre à votre propre bibliothèque de voix en la personnalisant selon le cas d'usage, en la localisant selon la langue et l'accent, en la gardant neutre ou plus émotive, décontractée ou formelle, plus naturelle et conversationnelle ou robotique. Le modèle démontre également une adaptation vocale cross-linguale en zéro-shot, bien qu'il n'ait pas été explicitement entraîné pour cela. Par exemple, le modèle peut générer de la parole en anglais avec une invite vocale française et un texte anglais. La parole résultante sonne naturellement tout en adoptant l'accent de l'invite vocale fournie. Dans cet exemple, la parole générée a un anglais naturellement accentué à la française. Cela rend le modèle utile pour construire des systèmes de traduction parole-à-parole en cascade." \
    -v fr_female \
    -o "$SAMPLES_DIR/fr_voxtral_tts_demo.wav"

echo ""
echo "=== Done! Samples saved to $SAMPLES_DIR/ ==="
ls -lh "$SAMPLES_DIR"/*.wav
