#include "cheri_micro_benchmark.h"

static void CopyCommon_Simple(benchmark::State& state) {
    const size_t buffer_size = state.range(0);

    void *src = nullptr;
    void *dst = nullptr;
    if (posix_memalign(&src, CACHE_LINE_SIZE, buffer_size) != 0 ||
        posix_memalign(&dst, CACHE_LINE_SIZE, buffer_size) != 0) {
        state.SkipWithError("Aligned allocation failed");
        return;
    }
    std::memset(src, 0xAB, buffer_size);
    std::memset(dst, 0x00, buffer_size);

    uint64_t start_cycles = read_cntvct();

    for (auto _ : state) {
        copycommon_simple(src, dst, buffer_size);
        benchmark::ClobberMemory();
    }

    uint64_t end_cycles = read_cntvct();

    state.SetBytesProcessed(buffer_size);
    double cycles_per_iter = static_cast<double>(end_cycles - start_cycles) /
	    static_cast<double>(state.iterations());
    state.counters["Cycles/Iter"] = benchmark::Counter(cycles_per_iter,
	benchmark::Counter::kDefaults);

    std::free(src);
    std::free(dst);
}
BENCHMARK(CopyCommon_Simple)->Apply(CustomArguments);
