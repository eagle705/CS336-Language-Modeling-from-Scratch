#!/bin/bash
# Build MLPractice as a proper macOS .app bundle
set -e

echo "Building MLPractice..."
swift build -c release 2>&1

APP_NAME="ML Practice"
APP_DIR=".build/${APP_NAME}.app"
CONTENTS="${APP_DIR}/Contents"
MACOS="${CONTENTS}/MacOS"

# Create .app bundle structure
rm -rf "${APP_DIR}"
mkdir -p "${MACOS}"
mkdir -p "${CONTENTS}/Resources"

# Copy binary
cp .build/release/MLPractice "${MACOS}/MLPractice"

# Copy Info.plist
cp Sources/Resources/Info.plist "${CONTENTS}/Info.plist"

echo ""
echo "Build complete: ${APP_DIR}"
echo ""
echo "Run with:"
echo "  open '.build/${APP_NAME}.app'"
echo ""
echo "Or install to /Applications:"
echo "  cp -r '.build/${APP_NAME}.app' /Applications/"
