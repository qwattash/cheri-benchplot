#include <benchmark/benchmark.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <unistd.h>
#include <inttypes.h>

// Cache line alignment constraint for CHERI capabilities (128 bits = 16 bytes).
// Aligning memory to 64-byte boundaries prevents split cache-line penalties.
#define CACHE_LINE_SIZE 64

// Function to read the AArch64 virtual cycle counter register (cntvct_el0)
static inline uint64_t read_cntvct() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}

// Function to read the AArch64 counter frequency register (cntfrq_el0)
static inline uint64_t read_cntfrq() {
    uint64_t val;
    asm volatile("mrs %0, cntfrq_el0" : "=r" (val));
    return val;
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
  // b->Arg(64);
  // b->Arg(256);
  b->Arg(512);
  b->Arg(64 * 1024);
  b->Arg(1024 * 1024);
  b->Arg(16 * 1024 * 1024);
}

/**
 * Benchmark loop using single scalar 64-bit LDR and STR instructions.
 * Loop is unrolled 8x manually in assembly (64 bytes read + 64 bytes written per iteration).
 */
static void bench_scalar_ldr_str(const void *src, void *dst, size_t count) {
#ifdef __CHERI__
    register uintptr_t s_ptr asm("c0") = (uintptr_t)src;
    register uintptr_t d_ptr asm("c1") = (uintptr_t)dst;
#else
    register uintptr_t s_ptr asm("x0") = (uintptr_t)src;
    register uintptr_t d_ptr asm("x1") = (uintptr_t)dst;
#endif

    asm volatile (
        "1:\n\t"
        // 8x Unrolled 64-bit scalar loads
        "ldr x3,  [%[src], #0]\n\t"
        "ldr x4,  [%[src], #8]\n\t"
        "ldr x5,  [%[src], #16]\n\t"
        "ldr x6,  [%[src], #24]\n\t"
        "ldr x7,  [%[src], #32]\n\t"
        "ldr x8,  [%[src], #40]\n\t"
        "ldr x9,  [%[src], #48]\n\t"
        "ldr x10, [%[src], #56]\n\t"
        "add %[src], %[src], #64\n\t"

        // 8x Unrolled 64-bit scalar stores
        "str x3,  [%[dst], #0]\n\t"
        "str x4,  [%[dst], #8]\n\t"
        "str x5,  [%[dst], #16]\n\t"
        "str x6,  [%[dst], #24]\n\t"
        "str x7,  [%[dst], #32]\n\t"
        "str x8,  [%[dst], #40]\n\t"
        "str x9,  [%[dst], #48]\n\t"
        "str x10, [%[dst], #56]\n\t"
        "add %[dst], %[dst], #64\n\t"

        "subs %[count], %[count], #64\n\t"
        "b.ne 1b\n\t"
        : [count] "+r" (count)
#ifdef __CHERI__
        : [src] "C" (s_ptr), [dst] "C" (d_ptr)
#else
        : [src] "r" (s_ptr), [dst] "r" (d_ptr)
#endif
        : "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "memory", "cc"
    );
}

/**
 * Benchmark loop using pair scalar 64-bit LDP and STP instructions.
 * Loop is unrolled 4x manually in assembly (64 bytes read + 64 bytes written per iteration).
 */
static void bench_scalar_ldp_stp(const void *src, void *dst, size_t count) {
#ifdef __CHERI__
    register uintptr_t s_ptr asm("c0") = (uintptr_t)src;
    register uintptr_t d_ptr asm("c1") = (uintptr_t)dst;
#else
    register uintptr_t s_ptr asm("x0") = (uintptr_t)src;
    register uintptr_t d_ptr asm("x1") = (uintptr_t)dst;
#endif

    asm volatile (
        "1:\n\t"
        // 4x Unrolled 128-bit scalar loads
        "ldp x3, x4,  [%[src], #0]\n\t"
        "ldp x5, x6,  [%[src], #16]\n\t"
        "ldp x7, x8,  [%[src], #32]\n\t"
        "ldp x9, x10, [%[src], #48]\n\t"
        "add %[src], %[src], #64\n\t"

        // 4x Unrolled 128-bit scalar stores
        "stp x3, x4,  [%[dst], #0]\n\t"
        "stp x5, x6,  [%[dst], #16]\n\t"
        "stp x7, x8,  [%[dst], #32]\n\t"
        "stp x9, x10, [%[dst], #48]\n\t"
        "add %[dst], %[dst], #64\n\t"

        "subs %[count], %[count], #64\n\t"
        "b.ne 1b\n\t"
        : [count] "+r" (count)
#ifdef __CHERI__
        : [src] "C" (s_ptr), [dst] "C" (d_ptr)
#else
        : [src] "r" (s_ptr), [dst] "r" (d_ptr)
#endif
        : "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "memory", "cc"
    );
}

#ifdef __CHERI__
/**
 * Benchmark loop using single scalar 64-bit LDR and STR instructions.
 * Loop is unrolled 8x manually in assembly (64 bytes read + 64 bytes written per iteration).
 */
static void bench_capability_ldr_str(const void *src, void *dst, size_t count) {
    register uintptr_t s_ptr asm("c0") = (uintptr_t)src;
    register uintptr_t d_ptr asm("c1") = (uintptr_t)dst;

    asm volatile (
        "1:\n\t"
        // 8x Unrolled 128-bit scalar loads
        "ldr c3,  [%[src], #0]\n\t"
        "ldr c4,  [%[src], #16]\n\t"
        "ldr c5,  [%[src], #32]\n\t"
        "ldr c6,  [%[src], #48]\n\t"
        "ldr c7,  [%[src], #64]\n\t"
        "ldr c8,  [%[src], #80]\n\t"
        "ldr c9,  [%[src], #96]\n\t"
        "ldr c10, [%[src], #112]\n\t"
        "add %[src], %[src], #128\n\t"

        // 8x Unrolled 128-bit scalar stores
        "str c3,  [%[dst], #0]\n\t"
        "str c4,  [%[dst], #16]\n\t"
        "str c5,  [%[dst], #32]\n\t"
        "str c6,  [%[dst], #48]\n\t"
        "str c7,  [%[dst], #64]\n\t"
        "str c8,  [%[dst], #80]\n\t"
        "str c9,  [%[dst], #96]\n\t"
        "str c10, [%[dst], #112]\n\t"
        "add %[dst], %[dst], #128\n\t"

        "subs %[count], %[count], #128\n\t"
        "b.ne 1b\n\t"
        : [count] "+r" (count)
        : [src] "C" (s_ptr), [dst] "C" (d_ptr)
        : "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "memory", "cc"
    );
}

/**
 * Benchmark loop using pair capability 128-bit LDP and STP instructions.
 * Loop is unrolled 4x manually in assembly (128 bytes read + 128 bytes written per iteration).
 */
static void bench_capability_ldp_stp(const void *src, void *dst, size_t count) {
    register uintptr_t s_ptr asm("c0") = (uintptr_t)src;
    register uintptr_t d_ptr asm("c1") = (uintptr_t)dst;

    asm volatile (
        "1:\n\t"
        // 4x Unrolled 256-bit scalar loads
        "ldp c3, c4,  [%[src], #0]\n\t"
        "ldp c5, c6,  [%[src], #32]\n\t"
        "ldp c7, c8,  [%[src], #64]\n\t"
        "ldp c9, c10, [%[src], #96]\n\t"
        "add %[src], %[src], #128\n\t"

        // 4x Unrolled 256-bit scalar stores
        "stp c3, c4,  [%[dst], #0]\n\t"
        "stp c5, c6,  [%[dst], #32]\n\t"
        "stp c7, c8,  [%[dst], #64]\n\t"
        "stp c9, c10, [%[dst], #96]\n\t"
        "add %[dst], %[dst], #128\n\t"

        "subs %[count], %[count], #128\n\t"
        "b.ne 1b\n\t"
        : [count] "+r" (count)
        : [src] "C" (s_ptr), [dst] "C" (d_ptr)
        : "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "memory", "cc"
    );
}

static void CopyLoop_LdrStr_Cap(benchmark::State& state) {
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
        bench_capability_ldr_str(src, dst, buffer_size);
        benchmark::ClobberMemory();
    }

    uint64_t end_cycles = read_cntvct();
    uint64_t total_unroll_iterations = state.iterations() * (buffer_size / 128);

    state.SetBytesProcessed(int64_t(total_unroll_iterations) * 128); // 128 bytes worth of data moved
    state.SetItemsProcessed(int64_t(total_unroll_iterations) * 16); // 16 memory instructions

    double cycles_per_iter = static_cast<double>(end_cycles - start_cycles) / static_cast<double>(total_unroll_iterations);
    state.counters["Cycles/Iter"] = benchmark::Counter(cycles_per_iter, benchmark::Counter::kDefaults);

    std::free(src);
    std::free(dst);
}
BENCHMARK(CopyLoop_LdrStr_Cap)->Apply(CustomArguments);

static void CopyLoop_LdpStp_Cap(benchmark::State& state) {
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
        bench_capability_ldp_stp(src, dst, buffer_size);
        benchmark::ClobberMemory();
    }

    uint64_t end_cycles = read_cntvct();
    uint64_t total_unroll_iterations = state.iterations() * (buffer_size / 128);

    state.SetBytesProcessed(int64_t(total_unroll_iterations) * 128); // 128 bytes worth of data moved
    state.SetItemsProcessed(int64_t(total_unroll_iterations) * 8); // 8 memory instructions

    double cycles_per_iter = static_cast<double>(end_cycles - start_cycles) / static_cast<double>(total_unroll_iterations);
    state.counters["Cycles/Iter"] = benchmark::Counter(cycles_per_iter, benchmark::Counter::kDefaults);

    std::free(src);
    std::free(dst);
}
BENCHMARK(CopyLoop_LdpStp_Cap)->Apply(CustomArguments);
#endif

static void CopyLoop_LdpStp_64bit(benchmark::State& state) {
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
        bench_scalar_ldp_stp(src, dst, buffer_size);
        benchmark::ClobberMemory();
    }

    uint64_t end_cycles = read_cntvct();
    uint64_t total_unroll_iterations = state.iterations() * (buffer_size / 64);

    state.SetBytesProcessed(int64_t(total_unroll_iterations) * 64); // 64 bytes worth of data moved
    state.SetItemsProcessed(int64_t(total_unroll_iterations) * 8); // 8 memory instructions

    double cycles_per_iter = static_cast<double>(end_cycles - start_cycles) / static_cast<double>(total_unroll_iterations);
    state.counters["Cycles/Iter"] = benchmark::Counter(cycles_per_iter, benchmark::Counter::kDefaults);

    std::free(src);
    std::free(dst);
}
BENCHMARK(CopyLoop_LdpStp_64bit)->Apply(CustomArguments);

static void CopyLoop_LdrStr_64bit(benchmark::State& state) {
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
        bench_scalar_ldr_str(src, dst, buffer_size);
        benchmark::ClobberMemory();
    }

    uint64_t end_cycles = read_cntvct();
    uint64_t total_unroll_iterations = state.iterations() * (buffer_size / 64);

    state.SetBytesProcessed(int64_t(total_unroll_iterations) * 64); // 64 bytes worth of data moved
    state.SetItemsProcessed(int64_t(total_unroll_iterations) * 16); // 16 memory instructions

    double cycles_per_iter = static_cast<double>(end_cycles - start_cycles) / static_cast<double>(total_unroll_iterations);
    state.counters["Cycles/Iter"] = benchmark::Counter(cycles_per_iter, benchmark::Counter::kDefaults);

    std::free(src);
    std::free(dst);
}
BENCHMARK(CopyLoop_LdrStr_64bit)->Apply(CustomArguments);

int main(int argc, char** argv) {
    std::cout << "========================================================================================\n";
    std::cout << "             Google Benchmark: Arm Morello CHERI Load/Store Microbenchmark             \n";
    std::cout << "========================================================================================\n";

#if defined(__CHERI__)
#if defined(__ARM_MORELLO_PURECAP_BENCHMARK_ABI)
    std::cout << "[Target ABI] CHERI Pure-Capability Benchmark (-mabi=purecap-benchmark)\n";
#else
    std::cout << "[Target ABI] CHERI Pure-Capability (-mabi=purecap)\n";
#endif
#elif __has_feature(capabilities)
    std::cout << "[Target ABI] Morello Hybrid\n";
#else
    std::cout << "[Target ABI] Standard AArch64\n";
#endif

    std::cout << "Counter Frequency: " << read_cntfrq() << " Hz\n";
    std::cout << "========================================================================================\n\n";

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
