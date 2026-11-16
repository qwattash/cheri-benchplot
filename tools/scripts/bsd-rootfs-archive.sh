#!/usr/bin/env sh

#
# Archive an existing rootfs from cheribuild into an archive with
# the destination system permissions from the METALOG.
#

set -eu

usage() {
    printf 'Usage: %s <rootfs-dir> [output-archive]\n' "$(basename "$0")" >&2
    exit 1
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    usage
fi

if ! command -v bsdtar >/dev/null 2>&1; then
    printf 'Error: bsdtar is required (e.g., install libarchive-tools on Debian/Ubuntu).\n' >&2
    exit 1
fi

ROOTFS=$1
if [ ! -d "$ROOTFS" ]; then
    printf 'Error: rootfs directory does not exist: %s\n' "$ROOTFS" >&2
    exit 1
fi

if [ ! -f "$ROOTFS/METALOG" ]; then
    printf 'Error: METALOG not found in %s\n' "$ROOTFS" >&2
    exit 1
fi

ROOTFS_NAME=$(basename "$ROOTFS")
DEST=${2:-"$HOME/cheribsd-${ROOTFS_NAME}.tar.zst"}

# Resolve destination to an absolute path before changing into $ROOTFS
DEST_DIR=$(dirname "$DEST")
mkdir -p "$DEST_DIR"
ABS_DEST="$(cd "$DEST_DIR" && pwd)/$(basename "$DEST")"

printf 'Archiving %s -> %s ...\n' "$ROOTFS" "$ABS_DEST"
(
    cd "$ROOTFS"
    bsdtar -a -cf "$ABS_DEST" @METALOG
)
