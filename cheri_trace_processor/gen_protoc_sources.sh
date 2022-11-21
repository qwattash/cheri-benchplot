#!/bin/sh

function usage() {
    echo "Usage: gen_protoc_sources.sh <cheri-perfetto-protobuf-dir>"
    echo "Example: gen_protoc_sources.sh $HOME/cheri/cheri-perfetto/protos/perfetto/trace/track_event"
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

protoc --python_out=. -I=$1 "${1}/qemu_log_entry.proto"
