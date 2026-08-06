#include "cheri_micro_benchmark.h"

extern "C" {
void copycommon_loadstore_8B(const void *src, void *dst, size_t len);
void copycommon_pair_64B(const void *src, void *dst, size_t len);
}

BENCHMARK_WITH_COUNTERS(copycommon_loadstore_8B);
BENCHMARK_WITH_COUNTERS(copycommon_pair_64B);
