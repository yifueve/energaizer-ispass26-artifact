#!/usr/bin/env bash
# gdrive_download.sh
# Downloads a file from Google Drive, extracts it, and removes the archive.
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
FILE_ID=$(echo "$DRIVE_URL" | grep -oP '/file/d/\K[a-zA-Z0-9_-]+' || \
          echo "$DRIVE_URL" | grep -oP '[?&]id=\K[a-zA-Z0-9_-]+' || true)

if [[ -z "$FILE_ID" ]]; then
    echo "Error: Could not extract a File ID from the provided URL."
    echo "       Make sure the URL is a valid Google Drive sharing link."
    exit 1
fi

echo ">>> File ID: ${FILE_ID}"

# ── download ──────────────────────────────────────────────────────────────────
echo ">>> Downloading → '${OUTPUT_FILE}' …"

COOKIE_JAR="/tmp/gdrive_cookies_$$.txt"
BASE_URL="https://drive.usercontent.google.com/download"
DOWNLOAD_URL="${BASE_URL}?id=${FILE_ID}&export=download&authuser=0"

# Step 1: Make initial request, capture response to look for confirmation form
curl -c "$COOKIE_JAR" -L "$DOWNLOAD_URL" --silent -o /tmp/gdrive_response_$$.html

# Step 2: Check if we got a confirmation page (large file virus scan warning)
CONFIRM_TOKEN=$(grep -oP '(?<=name="uuid" value=")[^"]+' /tmp/gdrive_response_$$.html 2>/dev/null || true)

if [[ -n "$CONFIRM_TOKEN" ]]; then
    echo "    (Large file detected – bypassing confirmation…)"
    curl -Lb "$COOKIE_JAR" \
        "${BASE_URL}?id=${FILE_ID}&export=download&authuser=0&confirm=t&uuid=${CONFIRM_TOKEN}" \
        -o "$OUTPUT_FILE"
else
    # Check if the response itself is already the file (small files download directly)
    CONTENT_TYPE=$(file --mime-type -b /tmp/gdrive_response_$$.html)
    if echo "$CONTENT_TYPE" | grep -qE 'html|text'; then
        # Still HTML — try with confirm=t directly
        echo "    (Trying direct download with confirm bypass…)"
        curl -Lb "$COOKIE_JAR" \
            "${DOWNLOAD_URL}&confirm=t" \
            -o "$OUTPUT_FILE"
    else
        # Response was the actual file
        mv /tmp/gdrive_response_$$.html "$OUTPUT_FILE"
    fi
fi

rm -f "$COOKIE_JAR" /tmp/gdrive_response_$$.html
echo "    Download complete."

# ── verify the file is actually a tar.gz ──────────────────────────────────────
FILE_TYPE=$(file --mime-type -b "$OUTPUT_FILE")
if ! echo "$FILE_TYPE" | grep -qE 'gzip|tar|x-compress|octet-stream'; then
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