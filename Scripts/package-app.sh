#!/bin/bash
# Packages VoxtralApp as a runnable Voxtral.app (K-28).
#
# Built with xcodebuild in Release: `swift build` does not compile MLX's Metal shaders completely (issue #11).
# The SwiftPM resource bundles (MLX `default.metallib`, Hub, Crypto) are copied into Contents/Resources, where
# `Bundle.module` and MLX look them up. The Core ML encoder is not bundled: the app downloads it (ASK-27).
#
# Usage: Scripts/package-app.sh [output .app path]   (default: .build/Voxtral.app)

set -euo pipefail
cd "$(dirname "$0")/.."

DERIVED_DATA=.build/xcode-app   # dedicated: no stale bundle from other schemes or older builds
PRODUCTS="$DERIVED_DATA/Build/Products/Release"
APP="${1:-.build/Voxtral.app}"

xcodebuild -scheme VoxtralApp -configuration Release -derivedDataPath "$DERIVED_DATA" \
    -destination 'platform=macOS' -onlyUsePackageVersionsFromResolvedFile build

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PRODUCTS/VoxtralApp" "$APP/Contents/MacOS/VoxtralApp"
cp Sources/VoxtralApp/Resources/Info.plist "$APP/Contents/Info.plist"
for bundle in "$PRODUCTS"/*.bundle; do
    cp -R "$bundle" "$APP/Contents/Resources/"
done
printf 'APPL????' > "$APP/Contents/PkgInfo"
codesign --force --deep --sign - "$APP"   # ad hoc: runs on this Mac; distribution needs a Developer ID

echo "Packaged $APP ($(du -sh "$APP" | cut -f1)); run: open \"$APP\""
