#!/usr/bin/env bash
# Smoke-test a ClipShow macOS DMG release artifact.
#
# Usage:
#   scripts/smoke-test-macos.sh [TAG] [VARIANT]
#
# Examples:
#   scripts/smoke-test-macos.sh              # latest release, lite variant
#   scripts/smoke-test-macos.sh v0.4.0 full  # specific tag, full variant
#
# Requires: gh (GitHub CLI), ffmpeg

set -euo pipefail

TAG="${1:-$(gh release view --json tagName -q .tagName)}"
VARIANT="${2:-lite}"
WORKDIR="$(mktemp -d)"
DMG_PATTERN="ClipShow-*-macos-${VARIANT}.dmg"
MOUNT_POINT="${WORKDIR}/mnt"
TEST_VIDEO="${WORKDIR}/test_input.mp4"
TEST_OUTPUT="${WORKDIR}/test_output.mp4"
PASS=0
FAIL=0

cleanup() {
    # Unmount if still mounted
    if mount | grep -q "${MOUNT_POINT}" 2>/dev/null; then
        hdiutil detach "${MOUNT_POINT}" -quiet 2>/dev/null || true
    fi
    rm -rf "${WORKDIR}"
}
trap cleanup EXIT

log()  { printf "  %s\n" "$*"; }
pass() { log "PASS: $*"; PASS=$((PASS + 1)); }
fail() { log "FAIL: $*"; FAIL=$((FAIL + 1)); }

echo "=== ClipShow macOS Smoke Test ==="
echo "  Tag:     ${TAG}"
echo "  Variant: ${VARIANT}"
echo "  Workdir: ${WORKDIR}"
echo ""

# --- Step 1: Download DMG ---
log "Downloading DMG from release ${TAG}..."
gh release download "${TAG}" --pattern "${DMG_PATTERN}" --dir "${WORKDIR}"
DMG_FILE="$(ls "${WORKDIR}"/*.dmg 2>/dev/null | head -1)"
if [[ -z "${DMG_FILE}" ]]; then
    fail "No DMG file found matching ${DMG_PATTERN}"
    echo ""
    echo "Results: ${PASS} passed, ${FAIL} failed"
    exit 1
fi
pass "Downloaded $(basename "${DMG_FILE}") ($(du -h "${DMG_FILE}" | cut -f1))"

# --- Step 2: Mount DMG ---
log "Mounting DMG..."
mkdir -p "${MOUNT_POINT}"
hdiutil attach "${DMG_FILE}" -mountpoint "${MOUNT_POINT}" -nobrowse -quiet
APP_PATH="${MOUNT_POINT}/ClipShow.app"
if [[ -d "${APP_PATH}" ]]; then
    pass "DMG mounted, ClipShow.app found"
else
    fail "ClipShow.app not found in DMG"
    echo ""
    echo "Results: ${PASS} passed, ${FAIL} failed"
    exit 1
fi

# --- Step 3: Generate synthetic test video ---
log "Generating test video..."
ffmpeg -y -loglevel error \
    -f lavfi -i "color=c=black:s=320x240:d=1:r=30,format=yuv420p" \
    -f lavfi -i "color=c=white:s=320x240:d=1:r=30,format=yuv420p" \
    -f lavfi -i "anullsrc=r=44100:cl=mono" \
    -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0[v]" \
    -map "[v]" -map 2:a -t 2 -c:v libx264 -c:a aac -shortest \
    "${TEST_VIDEO}"
pass "Test video generated"

# --- Step 4: Run ClipShow in headless auto mode ---
log "Running ClipShow --headless --auto..."
CLIPSHOW_BIN="${APP_PATH}/Contents/MacOS/ClipShow"
if [[ ! -x "${CLIPSHOW_BIN}" ]]; then
    fail "ClipShow binary not found or not executable at ${CLIPSHOW_BIN}"
    echo ""
    echo "Results: ${PASS} passed, ${FAIL} failed"
    exit 1
fi

set +e
QT_QPA_PLATFORM=offscreen "${CLIPSHOW_BIN}" \
    --headless --output "${TEST_OUTPUT}" "${TEST_VIDEO}" 2>&1
EXIT_CODE=$?
set -e

if [[ ${EXIT_CODE} -eq 0 ]]; then
    pass "ClipShow exited with code 0"
else
    fail "ClipShow exited with code ${EXIT_CODE}"
fi

# --- Step 5: Verify output ---
if [[ -f "${TEST_OUTPUT}" ]]; then
    OUTPUT_SIZE=$(stat -f%z "${TEST_OUTPUT}" 2>/dev/null || stat -c%s "${TEST_OUTPUT}" 2>/dev/null)
    if [[ ${OUTPUT_SIZE} -gt 0 ]]; then
        pass "Output file exists ($(du -h "${TEST_OUTPUT}" | cut -f1))"
    else
        fail "Output file is empty"
    fi
else
    fail "Output file not created"
fi

# --- Results ---
echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
if [[ ${FAIL} -gt 0 ]]; then
    exit 1
fi
exit 0
