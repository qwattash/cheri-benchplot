/*
 * Basic copy loop benchmarks.
 * These are intended to validate very simple assumptions about loads and stores.
 */

#include <concepts>
#include <unistd.h>

#include "cheri_micro_benchmark.h"

namespace {
/*
 * Benchmark loop using single scalar 64-bit LDR and STR instructions.
 * Loop is unrolled 8x manually in assembly (64 bytes read + 64 bytes written
 * per iteration).
 */
static void scalar_loadstore_loop_64B(const void *src, void *dst, size_t count) {
#ifdef __CHERI__
  register uintptr_t s_ptr asm("c0") = (uintptr_t)src;
  register uintptr_t d_ptr asm("c1") = (uintptr_t)dst;
#else
  register uintptr_t s_ptr asm("x0") = (uintptr_t)src;
  register uintptr_t d_ptr asm("x1") = (uintptr_t)dst;
#endif

  asm volatile("1:\n\t"
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
               : [count] "+r"(count)
#ifdef __CHERI__
               : [src] "C"(s_ptr), [dst] "C"(d_ptr)
#else
                 : [src] "r"(s_ptr), [dst] "r"(d_ptr)
#endif
               : "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "memory",
                 "cc");
}

/*
 * Benchmark loop using pair scalar 64-bit LDP and STP instructions.
 * Loop is unrolled 4x manually in assembly (64 bytes read + 64 bytes written
 * per iteration).
 */
static void scalar_pair_loop_64B(const void *src, void *dst, size_t count) {
#ifdef __CHERI__
  register uintptr_t s_ptr asm("c0") = (uintptr_t)src;
  register uintptr_t d_ptr asm("c1") = (uintptr_t)dst;
#else
  register uintptr_t s_ptr asm("x0") = (uintptr_t)src;
  register uintptr_t d_ptr asm("x1") = (uintptr_t)dst;
#endif

  asm volatile("1:\n\t"
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
               : [count] "+r"(count)
#ifdef __CHERI__
               : [src] "C"(s_ptr), [dst] "C"(d_ptr)
#else
                 : [src] "r"(s_ptr), [dst] "r"(d_ptr)
#endif
               : "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "memory",
                 "cc");
}

#ifdef __CHERI__
/**
 * Benchmark loop using single scalar 64-bit LDR and STR instructions.
 * Loop is unrolled 8x manually in assembly (128 bytes read + 128 bytes written
 * per iteration).
 */
static void cap_loadstore_loop_128B(const void *src, void *dst, size_t count) {
  register uintptr_t s_ptr asm("c0") = (uintptr_t)src;
  register uintptr_t d_ptr asm("c1") = (uintptr_t)dst;

  asm volatile("1:\n\t"
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
               : [count] "+r"(count)
               : [src] "C"(s_ptr), [dst] "C"(d_ptr)
               : "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "memory",
                 "cc");
}

/**
 * Benchmark loop using single capability  LDR and STR instructions.
 * Loop is unrolled 4x manually in assembly (64 bytes read + 64 bytes written
 * per iteration).
 */
static void cap_loadstore_loop_64B(const void *src, void *dst,
                                       size_t count) {
  register uintptr_t s_ptr asm("c0") = (uintptr_t)src;
  register uintptr_t d_ptr asm("c1") = (uintptr_t)dst;

  asm volatile("1:\n\t"
               // 4x Unrolled 128-bit scalar loads
               "ldr c3,  [%[src], #0]\n\t"
               "ldr c4,  [%[src], #16]\n\t"
               "ldr c5,  [%[src], #32]\n\t"
               "ldr c6,  [%[src], #48]\n\t"
               "add %[src], %[src], #64\n\t"

               // 4x Unrolled 128-bit scalar stores
               "str c3,  [%[dst], #0]\n\t"
               "str c4,  [%[dst], #16]\n\t"
               "str c5,  [%[dst], #32]\n\t"
               "str c6,  [%[dst], #48]\n\t"
               "add %[dst], %[dst], #64\n\t"

               "subs %[count], %[count], #64\n\t"
               "b.ne 1b\n\t"
               : [count] "+r"(count)
               : [src] "C"(s_ptr), [dst] "C"(d_ptr)
               : "c3", "c4", "c5", "c6", "memory", "cc");
}

/**
 * Benchmark loop using pair capability 128-bit LDP and STP instructions.
 * Loop is unrolled 4x manually in assembly (128 bytes read + 128 bytes written
 * per iteration).
 */
static void cap_pair_loop_128B(const void *src, void *dst, size_t count) {
  register uintptr_t s_ptr asm("c0") = (uintptr_t)src;
  register uintptr_t d_ptr asm("c1") = (uintptr_t)dst;

  asm volatile("1:\n\t"
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
               : [count] "+r"(count)
               : [src] "C"(s_ptr), [dst] "C"(d_ptr)
               : "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "memory",
                 "cc");
}
#endif /* __CHERI__ */

} /* namespace anonymous */

/*
 * Test hardware performance counters access.
 */
static void CounterAccess(benchmark::State &state) {
  PerfCounterReader reader(std::to_array<std::string>({"INST_RETIRED"}));
  const size_t buffer_size = 512 * 1024;

  void *src = std::malloc(buffer_size);
  void *dst = std::malloc(buffer_size);
  std::memset(src, 0xAB, buffer_size);
  std::memset(dst, 0x00, buffer_size);

  reader.start();
  for (auto _ : state) {
    scalar_loadstore_loop_64B(src, dst, buffer_size);
    benchmark::ClobberMemory();
  }
  reader.stop();

  state.counters.merge(reader.get_counters());

  std::free(src);
  std::free(dst);
}
BENCHMARK(CounterAccess);

BENCHMARK_WITH_COUNTERS(scalar_loadstore_loop_64B);
BENCHMARK_WITH_COUNTERS(scalar_pair_loop_64B);
#ifdef __CHERI__
BENCHMARK_WITH_COUNTERS(cap_loadstore_loop_128B);
BENCHMARK_WITH_COUNTERS(cap_loadstore_loop_64B);
BENCHMARK_WITH_COUNTERS(cap_pair_loop_128B);
#endif
