#!/usr/bin/env bash
# gdrive_download.sh
# Downloads a file from Google Drive, extracts it, and removes the archive.
# Compatible with macOS (BSD) and Linux (GNU).
#
# Usage:
#   ./gdrive_download.sh <google_drive_url> [output_filename]
#
# Examples:
#   ./gdrive_download.sh "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
#   ./gdrive_download.sh "https://drive.google.com/file/d/FILE_ID/view?usp=sharing" myarchive.tar.gz

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 <google_drive_url> [output_filename]"
    exit 1
}

require() {
    command -v "$1" &>/dev/null || { echo "Error: '$1' is required but not installed."; exit 1; }
}

# ── argument handling ─────────────────────────────────────────────────────────
[[ $# -lt 1 ]] && usage

DRIVE_URL="$1"
OUTPUT_FILE="${2:-downloaded_archive.tar.gz}"

# ── dependency checks ─────────────────────────────────────────────────────────
require curl
require tar

# ── extract File ID ───────────────────────────────────────────────────────────
# Use grep -oE (POSIX extended regex, supported on both BSD and GNU grep)
# \K (lookbehind shorthand) is Perl-only, so we use a capture group + sed instead
FILE_ID=$(echo "$DRIVE_URL" | grep -oE '/file/d/[a-zA-Z0-9_-]+' | sed 's|/file/d/||' || true)

if [[ -z "$FILE_ID" ]]; then
    # Try the ?id= or &id= format
    FILE_ID=$(echo "$DRIVE_URL" | grep -oE '[?&]id=[a-zA-Z0-9_-]+' | sed 's/^[?&]id=//' || true)
fi

if [[ -z "$FILE_ID" ]]; then
    echo "Error: Could not extract a File ID from the provided URL."
    echo "       Make sure the URL is a valid Google Drive sharing link."
    exit 1
fi

echo ">>> File ID: ${FILE_ID}"

# ── download ──────────────────────────────────────────────────────────────────
echo ">>> Downloading → '${OUTPUT_FILE}' …"

COOKIE_JAR="/tmp/gdrive_cookies_$$.txt"
RESPONSE_FILE="/tmp/gdrive_response_$$.html"
BASE_URL="https://drive.usercontent.google.com/download"
DOWNLOAD_URL="${BASE_URL}?id=${FILE_ID}&export=download&authuser=0"

# Step 1: Make initial request, capture response to look for confirmation form
curl -c "$COOKIE_JAR" -L "$DOWNLOAD_URL" --silent -o "$RESPONSE_FILE"

# Step 2: Check if we got a confirmation page (large file virus scan warning)
# Use grep -oE + sed instead of grep -oP (no Perl regex on macOS)
CONFIRM_TOKEN=$(grep -oE 'name="uuid" value="[^"]+' "$RESPONSE_FILE" 2>/dev/null \
                | sed 's/name="uuid" value="//' || true)

if [[ -n "$CONFIRM_TOKEN" ]]; then
    echo "    (Large file detected – bypassing confirmation…)"
    curl -Lb "$COOKIE_JAR" \
        "${BASE_URL}?id=${FILE_ID}&export=download&authuser=0&confirm=t&uuid=${CONFIRM_TOKEN}" \
        -o "$OUTPUT_FILE"
else
    # Check if the response is HTML or the actual file
    # Use `file -b` without --mime-type (BSD compatible; returns human-readable type)
    CONTENT_TYPE=$(file -b "$RESPONSE_FILE")
    if echo "$CONTENT_TYPE" | grep -qiE 'html|text'; then
        # Still HTML — try with confirm=t directly
        echo "    (Trying direct download with confirm bypass…)"
        curl -Lb "$COOKIE_JAR" \
            "${DOWNLOAD_URL}&confirm=t" \
            -o "$OUTPUT_FILE"
    else
        # Response was the actual file
        mv "$RESPONSE_FILE" "$OUTPUT_FILE"
    fi
fi

rm -f "$COOKIE_JAR" "$RESPONSE_FILE"
echo "    Download complete."

# ── verify the file is actually a tar.gz ──────────────────────────────────────
# Use `file -b` without --mime-type for BSD compatibility
FILE_TYPE=$(file -b "$OUTPUT_FILE")
if ! echo "$FILE_TYPE" | grep -qiE 'gzip|tar|compress'; then
    echo ""
    echo "Error: Downloaded file does not appear to be a gzip/tar archive."
    echo "       Detected type: $FILE_TYPE"
    echo "       First 300 bytes:"
    head -c 300 "$OUTPUT_FILE"
    echo ""
    echo "       Possible causes:"
    echo "         - File is not shared as 'Anyone with the link can view'"
    echo "         - The Google Drive link has expired"
    rm -f "$OUTPUT_FILE"
    exit 1
fi

# ── extract ───────────────────────────────────────────────────────────────────
echo ">>> Extracting '${OUTPUT_FILE}' …"
tar -xzf "$OUTPUT_FILE"
echo "    Extraction complete."

# ── clean up ──────────────────────────────────────────────────────────────────
echo ">>> Removing archive '${OUTPUT_FILE}' …"
rm -f "$OUTPUT_FILE"
echo "    Done."