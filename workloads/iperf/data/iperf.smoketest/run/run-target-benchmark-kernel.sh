#!/bin/sh

OPT_FORCE=

usage() {
    echo "Run benchmark workload variations for target benchmark-kernel"
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

echo "Run benchmarks for benchmark-kernel"

ROOT_WORKDIR="$(pwd)"
# List of workload variations for this target
TARGET_WORKLOADS="iperf-a8451fe1-a901-4c28-a9f2-1b99186c4275
iperf-c1496bff-e8d4-4571-8809-21796c1e0c34
iperf-50b216ba-c1a7-4a61-90b5-53f3667a20dd
iperf-3e7dcfe7-56bf-42bd-8951-b22f1d49a0f4
iperf-e8537d75-a95b-4377-b43d-46ecd7395259
iperf-9aa8bb20-d95c-470f-aab6-e7da97b1291a
iperf-702c21ee-60ee-4577-a13d-bca89361be78
iperf-8d977d07-259d-4e6d-9b9f-b8fedfc476b0"
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
