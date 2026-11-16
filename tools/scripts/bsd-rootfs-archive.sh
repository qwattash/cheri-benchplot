#!/usr/bin/env sh

#
# Archive an existing rootfs from cheribuild into an archive with
# the destination system permissions from the METALOG.
#
set -euo pipefail

usage() {
    printf 'Usage: %s [-e extra-files-dir] [-E] <rootfs-dir> [output-dir]\n' "$(basename "$0")" >&2
    printf '  -e dir    Overlay directory to include as root:wheel (default: ~/cheri/extra-files)\n' >&2
    printf '  -E        Do not include any extra-files overlay\n' >&2
    exit 1
}

EXTRA_DIR=""
NO_EXTRA=0

while getopts "e:Eh" opt; do
    case "$opt" in
        e)
            EXTRA_DIR=$OPTARG
            ;;
        E)
            NO_EXTRA=1
            ;;
        h|\?)
            usage
            ;;
    esac
done
shift $((OPTIND - 1))

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

if [ "$NO_EXTRA" -eq 1 ]; then
    EXTRA_DIR=""
elif [ -z "$EXTRA_DIR" ] && [ -d "$HOME/cheri/extra-files" ]; then
    EXTRA_DIR="$HOME/cheri/extra-files"
elif [ -n "$EXTRA_DIR" ] && [ ! -d "$EXTRA_DIR" ]; then
    printf 'Error: extra-files directory does not exist: %s\n' "$EXTRA_DIR" >&2
    exit 1
fi

ROOTFS_NAME=$(basename "$ROOTFS")
OUT_DIR=${2:-"$HOME"}

mkdir -p "$OUT_DIR"
ABS_DEST="$(cd "$OUT_DIR" && pwd)/cheribsd-${ROOTFS_NAME}.tar.zst"
ABS_ROOTFS="$(cd "$ROOTFS" && pwd)"

TAR_ARGS=(-a -cf "$ABS_DEST" -C "$ABS_ROOTFS")

if [ -f "$ABS_ROOTFS/METALOG.world" ] && [ -f "$ABS_ROOTFS/METALOG.kernel" ]; then
    TAR_ARGS+=("@$ABS_ROOTFS/METALOG.world" "@$ABS_ROOTFS/METALOG.kernel")
elif [ -f "$ABS_ROOTFS/METALOG.world" ] || [ -f "$ABS_ROOTFS/METALOG.kernel" ]; then
    printf 'Error: found only one of METALOG.world / METALOG.kernel in %s (both are required).\n' "$ABS_ROOTFS" >&2
    exit 1
elif [ -f "$ABS_ROOTFS/METALOG" ]; then
    TAR_ARGS+=("@$ABS_ROOTFS/METALOG")
else
    printf 'Error: no METALOG.world/METALOG.kernel (or METALOG) found in %s\n' "$ABS_ROOTFS" >&2
    exit 1
fi

EXTRA_MTREE=""
cleanup() {
    if [ -n "$EXTRA_MTREE" ] && [ -f "$EXTRA_MTREE" ]; then
        rm -f "$EXTRA_MTREE"
    fi
}
trap cleanup EXIT INT TERM

mtree_escape() {
    printf '%s' "$1" | awk '{ gsub(/\\/, "\\\\"); gsub(/ /, "\\040"); gsub(/\t/, "\\011"); print }'
}

if [ -n "$EXTRA_DIR" ]; then
    ABS_EXTRA="$(cd "$EXTRA_DIR" && pwd)"
    EXTRA_MTREE=$(mktemp /tmp/extra-files.XXXXXX.mtree)
    printf '#mtree\n' > "$EXTRA_MTREE"

    (
        cd "$ABS_EXTRA"
        find . -mindepth 1 | sort | while IFS= read -r rel; do
            esc_rel=$(mtree_escape "$rel")
            if [ -L "$rel" ]; then
                target=$(readlink "$rel")
                esc_target=$(mtree_escape "$target")
                printf '%s type=link uname=root gname=wheel mode=0755 link=%s\n' \
                    "$esc_rel" "$esc_target" >> "$EXTRA_MTREE"
            elif [ -d "$rel" ]; then
                printf '%s type=dir uname=root gname=wheel mode=0755\n' \
                    "$esc_rel" >> "$EXTRA_MTREE"
            elif [ -f "$rel" ]; then
                if [ -x "$rel" ]; then
                    mode=0755
                else
                    mode=0644
                fi
                esc_src=$(mtree_escape "$ABS_EXTRA/${rel#./}")
                printf '%s type=file uname=root gname=wheel mode=%s contents=%s\n' \
                    "$esc_rel" "$mode" "$esc_src" >> "$EXTRA_MTREE"
            fi
        done
    )

    TAR_ARGS+=("@$EXTRA_MTREE")
    printf 'Archiving %s + %s -> %s ...\n' "$ABS_ROOTFS" "$ABS_EXTRA" "$ABS_DEST"
else
    printf 'Archiving %s -> %s ...\n' "$ABS_ROOTFS" "$ABS_DEST"
fi

bsdtar "${TAR_ARGS[@]}"
printf 'Done.\n'
