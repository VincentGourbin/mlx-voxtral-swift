#!/bin/bash

# Script to create a proper macOS .app bundle from the VoxtralApp executable

set -e

APP_NAME="Voxtral"
BUNDLE_DIR="$APP_NAME.app"
# Find the executable - check both possible locations
if [ -f ".build/arm64-apple-macosx/debug/VoxtralApp" ]; then
    EXECUTABLE_PATH=".build/arm64-apple-macosx/debug/VoxtralApp"
elif [ -f ".build/debug/VoxtralApp" ]; then
    EXECUTABLE_PATH=".build/debug/VoxtralApp"
else
    echo "Error: VoxtralApp executable not found. Run 'swift build --product VoxtralApp' first."
    exit 1
fi

echo "Creating $BUNDLE_DIR..."

# Clean up existing bundle
rm -rf "$BUNDLE_DIR"

# Create bundle structure
mkdir -p "$BUNDLE_DIR/Contents/MacOS"
mkdir -p "$BUNDLE_DIR/Contents/Resources"

# Copy executable
cp "$EXECUTABLE_PATH" "$BUNDLE_DIR/Contents/MacOS/$APP_NAME"

# Create Info.plist
cat > "$BUNDLE_DIR/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>Voxtral</string>
    <key>CFBundleIdentifier</key>
    <string>com.voxtral.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Voxtral</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
EOF

# Create PkgInfo
echo -n "APPL????" > "$BUNDLE_DIR/Contents/PkgInfo"

echo "Done! App bundle created at: $BUNDLE_DIR"
echo ""
echo "To run: open $BUNDLE_DIR"
echo "Or double-click Voxtral.app in Finder"
