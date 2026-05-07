#!/bin/bash
set -euo pipefail

APP_NAME="ML Popup Quiz"
BINARY_NAME="MLPopupQuiz"
APP_DIR=".build/${APP_NAME}.app"
CONTENTS="${APP_DIR}/Contents"
MACOS="${CONTENTS}/MacOS"
RESOURCES="${CONTENTS}/Resources"

echo "Building ${APP_NAME}..."
swift build -c release

rm -rf "${APP_DIR}"
mkdir -p "${MACOS}" "${RESOURCES}"

cp ".build/release/${BINARY_NAME}" "${MACOS}/${BINARY_NAME}"
cp "Sources/Resources/Info.plist" "${CONTENTS}/Info.plist"
cp "Sources/Resources/quiz_bank.json" "${RESOURCES}/quiz_bank.json"
if [ -f "Sources/Resources/AppIcon.icns" ]; then
    cp "Sources/Resources/AppIcon.icns" "${RESOURCES}/AppIcon.icns"
fi

RESOURCE_BUNDLE=".build/release/MLPopupQuiz_MLPopupQuiz.bundle"
if [ -d "${RESOURCE_BUNDLE}" ]; then
    cp -R "${RESOURCE_BUNDLE}" "${RESOURCES}/"
fi

codesign --force --sign - "${APP_DIR}" >/dev/null 2>&1 || true

echo "Build complete: ${APP_DIR}"
echo "Run with: open '${APP_DIR}'"
