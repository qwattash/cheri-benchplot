#!/bin/sh

OPT_FORCE=

usage() {
    echo "Run benchmark workload variations for target hybrid-kernel"
    echo "Usage: $0 [-f]"
    echo "-f\tRe-run all workloads overwriting existing results."
}

while :; do
    case $1 in
        -h)
            usage
            exit
            ;;
        -f)
            OPT_FORCE="yes"
            ;;
        -*)
            usage
            exit
            ;;
        *)
            break
            ;;
    esac

    shift
done

echo "Run benchmarks for hybrid-kernel"

ROOT_WORKDIR="$(pwd)"
# List of workload variations for this target
TARGET_WORKLOADS="iperf-599954d8-bb8a-4e92-9d7a-e6436dd67cac
iperf-9739dacd-e83c-4a40-82eb-995237504585
iperf-5510e2ca-c10c-445d-a6aa-09070d4a9595
iperf-0f6c6fd7-0d6a-4d74-9cb3-1bcc1c7eb078
iperf-5dfa11a1-ee21-44c3-8aed-d381f1742b8f
iperf-7298eed5-c118-48bc-ad0a-b7e17f7943ee
iperf-952dee70-57c7-47c8-af09-81adb45b0d3f
iperf-3774e0bd-2452-4b33-8d2c-39b6f0ce4620"
# Record failed workloads
FAILED=""

# Iterate over each target workload.
for WORKDIR in ${TARGET_WORKLOADS}; do
    if [ -n "${OPT_FORCE}" ] || [ ! -f "${WORKDIR}/.run_completed" ]; then
        if ! (cd ${WORKDIR} && ./runner-script.sh); then
            FAILED="${FAILED} ${WORKDIR}"
        fi
    else
        echo -n "Skip workload ${WORKDIR}, last run:"
        cat "${WORKDIR}/.run_completed"
    fi
done

if [ -n "${FAILED}" ]; then
    echo "Benchmark workloads failed:"
    for WORKDIR in ${FAILED}; do
        echo "${WORKDIR}"
    done
    exit 1
fi
