#!/bin/sh

${CHERIBUILD} --skip-update netperf-riscv64-purecap --netperf-riscv64-purecap/configure-options='--enable-pmc-profile'
${CHERIBUILD} --skip-update cheribsd-riscv64-purecap
${CHERIBUILD} --skip-update disk-image-mfs-root-riscv64-purecap
${CHERIBUILD} --skip-update --skip-buildworld cheribsd-mfs-root-kernel-riscv64-purecap --cheribsd/build-bench-kernels --cheribsd/build-fpga-kernels --cheribsd/extra-kernel-config CHERI-GFE-BUCKET-ADJUST CHERI-PURECAP-GFE-BUCKET-ADJUST CHERI-PURECAP-NOSUBOBJ-GFE-NODEBUG
