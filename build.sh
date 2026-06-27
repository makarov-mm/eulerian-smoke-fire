#!/usr/bin/env bash
#
# build.sh — compile the Metal shaders and Swift sources into a runnable
# macOS .app bundle, with no Xcode project.
#
# Usage:
#   ./build.sh        build only
#   ./build.sh run    build, then launch
#
# Requirements: Xcode (full install, not just Command Line Tools) so that the
# Metal toolchain (`metal`, `metallib`) is available via xcrun. On recent Xcode
# you may need to install the "Metal Toolchain" component once:
#   xcodebuild -downloadComponent MetalToolchain
#
set -euo pipefail

APP="Smoke3D"
BUNDLE="${APP}.app"
SDK="macosx"

# Source layout — put Fluid3D.metal, FluidSimulator.swift, main.swift here.
SRC_DIR="Sources"
METAL_SRC="${SRC_DIR}/Fluid3D.metal"
SWIFT_SRCS=("${SRC_DIR}/FluidSimulator.swift" "${SRC_DIR}/main.swift")

CONTENTS="${BUNDLE}/Contents"
MACOS_DIR="${CONTENTS}/MacOS"
RES_DIR="${CONTENTS}/Resources"

echo "==> Laying out ${BUNDLE}"
rm -rf "${BUNDLE}"
mkdir -p "${MACOS_DIR}" "${RES_DIR}"

# 1. Metal -> AIR -> metallib.
#    Name it default.metallib so MTLDevice.makeDefaultLibrary() finds it.
echo "==> Compiling Metal shaders"
xcrun -sdk "${SDK}" metal -O -c "${METAL_SRC}" -o "/tmp/${APP}.air"
xcrun -sdk "${SDK}" metallib "/tmp/${APP}.air" -o "${RES_DIR}/default.metallib"

# 2. Swift -> executable.
#    -target pins the host arch and a minimum macOS the running system can
#    satisfy. Without it, swiftc embeds the SDK's (very recent) deployment
#    target and LaunchServices refuses to start the app.
DEPLOY_TARGET="$(uname -m)-apple-macos13.0"
echo "==> Compiling Swift (target ${DEPLOY_TARGET})"
xcrun -sdk "${SDK}" swiftc -O \
    -target "${DEPLOY_TARGET}" \
    "${SWIFT_SRCS[@]}" \
    -o "${MACOS_DIR}/${APP}" \
    -framework Metal -framework MetalKit -framework Cocoa

# 3. Info.plist.
echo "==> Writing Info.plist"
cat > "${CONTENTS}/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>${APP}</string>
    <key>CFBundleExecutable</key><string>${APP}</string>
    <key>CFBundleIdentifier</key><string>com.nightmarez.${APP}</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
</dict>
</plist>
PLIST

echo "==> Built ${BUNDLE}"

if [[ "${1:-}" == "run" ]]; then
    echo "==> Launching"
    open "${BUNDLE}"
fi