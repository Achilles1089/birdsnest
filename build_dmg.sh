#!/usr/bin/env bash
#
# build_dmg.sh — Build Bird's Nest macOS .app and .dmg
#
# Usage: bash build_dmg.sh
#
set -euo pipefail

VERSION=$(cat VERSION)
APP_NAME="Bird's Nest"
DMG_NAME="BirdsNest-${VERSION}"

echo "🪹 Building Bird's Nest v${VERSION}"
echo "─────────────────────────────────────"

# ── Step 1: Activate venv if it exists ──
if [ -f ".venv/bin/activate" ]; then
    echo "🐍 Activating virtual environment..."
    source .venv/bin/activate
fi

# ── Step 2: Check dependencies ──
echo "📋 Checking build dependencies..."
command -v python3 >/dev/null || { echo "❌ python3 not found"; exit 1; }
command -v pyinstaller >/dev/null || { echo "⚠️  Installing PyInstaller..."; pip3 install pyinstaller; }

if ! command -v create-dmg >/dev/null; then
    echo "⚠️  create-dmg not found. Installing via Homebrew..."
    brew install create-dmg
fi

# ── Step 3: Ensure icon exists ──
if [ ! -f "assets/icon.icns" ]; then
    echo "⚠️  No icon found. Generating from emoji..."
    bash build_icon.sh
fi

# ── Step 4: Clean previous builds ──
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/

# ── Step 5: Run PyInstaller ──
echo "📦 Running PyInstaller (this takes a few minutes)..."
pyinstaller birdsnest.spec --noconfirm

if [ ! -d "dist/${APP_NAME}.app" ]; then
    echo "❌ PyInstaller failed — no .app created"
    exit 1
fi

echo "✅ .app bundle created: dist/${APP_NAME}.app"
APP_SIZE=$(du -sh "dist/${APP_NAME}.app" | awk '{print $1}')
echo "   Size: ${APP_SIZE}"

# ── Step 5: Create DMG ──
echo "💿 Creating DMG..."
rm -f "dist/${DMG_NAME}.dmg"

create-dmg \
    --volname "${APP_NAME}" \
    --volicon "assets/icon.icns" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "${APP_NAME}.app" 150 190 \
    --app-drop-link 450 190 \
    --hide-extension "${APP_NAME}.app" \
    "dist/${DMG_NAME}.dmg" \
    "dist/${APP_NAME}.app"

if [ -f "dist/${DMG_NAME}.dmg" ]; then
    DMG_SIZE=$(du -sh "dist/${DMG_NAME}.dmg" | awk '{print $1}')
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Done!"
    echo "   .app: dist/${APP_NAME}.app (${APP_SIZE})"
    echo "   .dmg: dist/${DMG_NAME}.dmg (${DMG_SIZE})"
    echo ""
    echo "To release:"
    echo "   gh release create v${VERSION} dist/${DMG_NAME}.dmg --title 'v${VERSION}'"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "❌ DMG creation failed"
    exit 1
fi
