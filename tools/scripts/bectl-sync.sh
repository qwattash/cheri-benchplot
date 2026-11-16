#!/usr/bin/env sh

#
# Sync a given rootfs archive as a new boot environment on a Morello host.
#

set -eu

usage() {
    printf 'Usage: %s [-r] <archive> <be-name>\n' "$(basename "$0")" >&2
    printf '  -r    Replace <be-name> if it already exists (disallowed for "default")\n' >&2
    exit 1
}

REPLACE=0

while getopts "rh" opt; do
    case "$opt" in
        r)
            REPLACE=1
            ;;
        h|\?)
            usage
            ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -ne 2 ]; then
    usage
fi

ARCHIVE=$1
BE_NAME=$2

if [ "$(id -u)" -ne 0 ]; then
    printf 'Error: this script must be run as root.\n' >&2
    exit 1
fi

if ! command -v bectl >/dev/null 2>&1; then
    printf 'Error: bectl not found.\n' >&2
    exit 1
fi

if [ ! -f "$ARCHIVE" ]; then
    printf 'Error: archive does not exist: %s\n' "$ARCHIVE" >&2
    exit 1
fi

case "$BE_NAME" in
    ""|*/*|*[[:space:]]*)
        printf 'Error: invalid boot environment name: %s\n' "$BE_NAME" >&2
        exit 1
        ;;
    default)
        printf 'Error: refusing to create or overwrite the "default" boot environment.\n' >&2
        exit 1
        ;;
esac

# Check whether the BE exists and inspect its active flags (N=now, R=reboot, T=temporary)
BE_EXISTS=0
BE_FLAGS=""
if bectl list -H | cut -f1 | grep -Fxq "$BE_NAME"; then
    BE_EXISTS=1
    BE_FLAGS=$(bectl list -H | grep -E "^${BE_NAME}[[:blank:]]" | cut -f2)
fi

if [ "$BE_EXISTS" -eq 1 ]; then
    if [ "$REPLACE" -ne 1 ]; then
        printf 'Error: boot environment "%s" already exists (pass -r to overwrite).\n' "$BE_NAME" >&2
        exit 1
    fi

    case "$BE_FLAGS" in
        *N*|*R*|*T*)
            printf 'Error: refusing to replace "%s" because it is active or scheduled for boot (%s).\n' "$BE_NAME" "$BE_FLAGS" >&2
            exit 1
            ;;
    esac

    printf 'Replacing existing boot environment "%s"...\n' "$BE_NAME"
    bectl umount "$BE_NAME" >/dev/null 2>&1 || true
    bectl destroy "$BE_NAME"
fi

MNT=$(mktemp -d /tmp/be-deploy.XXXXXX)
MOUNTED=0

cleanup() {
    if [ "$MOUNTED" -eq 1 ]; then
        bectl umount "$BE_NAME" >/dev/null 2>&1 || true
    fi
    rmdir "$MNT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

printf 'Creating and mounting boot environment "%s"...\n' "$BE_NAME"
bectl create "$BE_NAME"
bectl mount "$BE_NAME" "$MNT"
MOUNTED=1

printf 'Clearing schg flags and extracting %s...\n' "$ARCHIVE"
chflags -R noschg "$MNT"
tar -xpf "$ARCHIVE" --exclude='./etc*' -C "$MNT"

printf 'Unmounting and setting one-time boot target...\n'
bectl umount "$BE_NAME"
MOUNTED=0

printf 'Done. Activate with `bectl activate -t "%s"` and reboot to test.\n' "$BE_NAME"
