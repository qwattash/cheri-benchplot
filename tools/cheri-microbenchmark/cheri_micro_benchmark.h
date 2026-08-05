#pragma once

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <cheriintrin.h>

#ifdef __aarch64__
#include <machine/armreg.h>
#include <sys/sysctl.h>
#endif

void CopyBenchmarkArgs(benchmark::internal::Benchmark *b);

#define ALWAYS_INLINE __attribute__((always_inline))

extern std::unordered_map<std::string, uint64_t> EventRegistry;

#ifdef __aarch64__

/* Cache line alignment for the target architecture */
static constexpr size_t CACHE_LINE_SIZE = 64;

static inline void init_counters() {
  uint64_t pmu_enabled;
  size_t size = sizeof(pmu_enabled);
  if (sysctlbyname("machdep.pmu_user_access", &pmu_enabled, &size, NULL, 0) ==
      -1) {
    std::cerr << "No machdep.pmu_user_access" << std::endl;
    abort();
  }
  if ((pmu_enabled & 0x1) == 0) {
    std::cerr << "PMU user access disabled" << std::endl;
    abort();
  }
}

static inline uint64_t read_cntvct() { return READ_SPECIALREG(cntvct_el0); }

static inline uint64_t read_cntfrq() { return READ_SPECIALREG(cntfrq_el0); }

static inline uint64_t read_cycles() { return READ_SPECIALREG(pmccntr_el0); }

#if 0
#define DEBUG(x) (x)
#define DEBUG_COUNTER(idx, val) do {                                    \
    std::cerr << "DEBUG: counter[" << idx << "] = " << val <<           \
      std::endl;                                                        \
  } while (0)

#define DEBUG_EVENT(idx, event) do {                          \
    std::cerr << "DEBUG: config counter[" << idx << "] = " << \
      std::hex << event << std::endl;                         \
  } while (0)
#else
#define DEBUG(x)
#define DEBUG_COUNTER(idx, val)
#define DEBUG_EVENT(idx, event)
#endif

#define EVAL_PMEVCNTR_CASE(target_idx, current_idx)                            \
  if constexpr (current_idx == target_idx) {                                   \
    uint64_t val;                                                              \
    asm volatile("mrs %0, pmevcntr" #target_idx "_el0" : "=r"(val));           \
    DEBUG_COUNTER(target_idx, val);                                     \
    return val;                                                                \
  }

#define EVAL_PMEVTYPER_CASE(target_idx, current_idx, event)                    \
  if constexpr (current_idx == target_idx) {                                   \
    asm volatile("msr pmevtyper" #target_idx                                   \
                 "_el0, %0" ::"r"(PMEVTYPER_NSH | event));                     \
    DEBUG_EVENT(target_idx, event);                                     \
  }

template <size_t NumCounters> class PerfCounterReader {
private:
  std::array<std::string, NumCounters> counter_names;
  std::array<uint64_t, NumCounters> event_ids;
  std::array<uint64_t, NumCounters + 1> values;
  const uint64_t CNTR_MASK = ((1 << NumCounters) - 1) | (1 << 31);

  // This function compiles down to a single instruction for any valid Index
  template <size_t Index> uint64_t read_counter() const {
    // Expand the macro for all possible ARM64 performance counter registers (0
    // to 30)
    EVAL_PMEVCNTR_CASE(0, Index)
    EVAL_PMEVCNTR_CASE(1, Index)
    EVAL_PMEVCNTR_CASE(2, Index)
    EVAL_PMEVCNTR_CASE(3, Index)
    EVAL_PMEVCNTR_CASE(4, Index)
    EVAL_PMEVCNTR_CASE(5, Index)
    EVAL_PMEVCNTR_CASE(6, Index)
    EVAL_PMEVCNTR_CASE(7, Index)
    EVAL_PMEVCNTR_CASE(8, Index)
    EVAL_PMEVCNTR_CASE(9, Index)
    EVAL_PMEVCNTR_CASE(10, Index)
    EVAL_PMEVCNTR_CASE(11, Index)
    EVAL_PMEVCNTR_CASE(12, Index)
    EVAL_PMEVCNTR_CASE(13, Index)
    EVAL_PMEVCNTR_CASE(14, Index)
    EVAL_PMEVCNTR_CASE(15, Index)
    EVAL_PMEVCNTR_CASE(16, Index)
    EVAL_PMEVCNTR_CASE(17, Index)
    EVAL_PMEVCNTR_CASE(18, Index)
    EVAL_PMEVCNTR_CASE(19, Index)
    EVAL_PMEVCNTR_CASE(20, Index)
    EVAL_PMEVCNTR_CASE(21, Index)
    EVAL_PMEVCNTR_CASE(22, Index)
    EVAL_PMEVCNTR_CASE(23, Index)
    EVAL_PMEVCNTR_CASE(24, Index)
    EVAL_PMEVCNTR_CASE(25, Index)
    EVAL_PMEVCNTR_CASE(26, Index)
    EVAL_PMEVCNTR_CASE(27, Index)
    EVAL_PMEVCNTR_CASE(28, Index)
    EVAL_PMEVCNTR_CASE(29, Index)
    EVAL_PMEVCNTR_CASE(30, Index)

    // Fallback for out-of-bounds indices
    return 0;
  }

  // This function compiles down to a single instruction for any valid Index
  template <size_t Index> void config_counter(uint64_t event) const {
    // Expand the macro for all possible ARM64 performance counter registers (0
    // to 30)
    EVAL_PMEVTYPER_CASE(0, Index, event)
    EVAL_PMEVTYPER_CASE(1, Index, event)
    EVAL_PMEVTYPER_CASE(2, Index, event)
    EVAL_PMEVTYPER_CASE(3, Index, event)
    EVAL_PMEVTYPER_CASE(4, Index, event)
    EVAL_PMEVTYPER_CASE(5, Index, event)
    EVAL_PMEVTYPER_CASE(6, Index, event)
    EVAL_PMEVTYPER_CASE(7, Index, event)
    EVAL_PMEVTYPER_CASE(8, Index, event)
    EVAL_PMEVTYPER_CASE(9, Index, event)
    EVAL_PMEVTYPER_CASE(10, Index, event)
    EVAL_PMEVTYPER_CASE(11, Index, event)
    EVAL_PMEVTYPER_CASE(12, Index, event)
    EVAL_PMEVTYPER_CASE(13, Index, event)
    EVAL_PMEVTYPER_CASE(14, Index, event)
    EVAL_PMEVTYPER_CASE(15, Index, event)
    EVAL_PMEVTYPER_CASE(16, Index, event)
    EVAL_PMEVTYPER_CASE(17, Index, event)
    EVAL_PMEVTYPER_CASE(18, Index, event)
    EVAL_PMEVTYPER_CASE(19, Index, event)
    EVAL_PMEVTYPER_CASE(20, Index, event)
    EVAL_PMEVTYPER_CASE(21, Index, event)
    EVAL_PMEVTYPER_CASE(22, Index, event)
    EVAL_PMEVTYPER_CASE(23, Index, event)
    EVAL_PMEVTYPER_CASE(24, Index, event)
    EVAL_PMEVTYPER_CASE(25, Index, event)
    EVAL_PMEVTYPER_CASE(26, Index, event)
    EVAL_PMEVTYPER_CASE(27, Index, event)
    EVAL_PMEVTYPER_CASE(28, Index, event)
    EVAL_PMEVTYPER_CASE(29, Index, event)
    EVAL_PMEVTYPER_CASE(30, Index, event)
  }

  // Compile-time loop using a fold expression to fill the results array
  // sequentially
  template <size_t... Is>
  void read_all_impl(std::array<uint64_t, NumCounters + 1> &out,
                     std::index_sequence<Is...>) const {
    // Use __builtin_add_overflow?
    ((out[Is] += read_counter<Is>()), ...);
    out[NumCounters] += READ_SPECIALREG(pmccntr_el0);
  }

  // Compile-time loop using a fold expression to initialize all counters
  template <size_t... Is>
  void config_all_impl(const std::array<uint64_t, NumCounters> &events,
                       std::index_sequence<Is...>) const {
    (config_counter<Is>(events[Is]), ...);
  }

  template <size_t... Is>
  static std::array<uint64_t, NumCounters>
  map_names_to_ids(const std::array<std::string, NumCounters> &names,
                   std::index_sequence<Is...>) {
    auto lookup_id = [](std::string name) -> uint64_t {
      auto it = EventRegistry.find(name);
      assert(it != EventRegistry.end() && "Invalid PMC event");
      return it->second;
    };

    return {lookup_id(names[Is])...};
  }

public:
  PerfCounterReader(const std::array<std::string, NumCounters> &names)
      : counter_names(names),
        event_ids(
            map_names_to_ids(names, std::make_index_sequence<NumCounters>{})) {
    values.fill(0);
    size_t max_supported =
        (READ_SPECIALREG(pmcr_el0) & PMCR_N_MASK) >> PMCR_N_SHIFT;
    assert(NumCounters < max_supported && "Too many counters for platform");

    config_all_impl(event_ids, std::make_index_sequence<NumCounters>{});
  }

  PerfCounterReader(const std::array<std::string_view, NumCounters> &names) {
    std::copy(names.begin(), names.end(), counter_names.begin());

    event_ids = map_names_to_ids(counter_names,
                                 std::make_index_sequence<NumCounters>{});
    values.fill(0);
    size_t max_supported =
        (READ_SPECIALREG(pmcr_el0) & PMCR_N_MASK) >> PMCR_N_SHIFT;
    assert(NumCounters < max_supported && "Too many counters for platform");

    config_all_impl(event_ids, std::make_index_sequence<NumCounters>{});
  }

  // Compile-time safe interface, note that this expands to a direct register
  // read
  template <size_t Index> uint64_t read() const {
    static_assert(Index <= NumCounters,
                  "Counter index out of bounds for this reader configuration.");
    if constexpr (Index == NumCounters) {
      return READ_SPECIALREG(pmccntr_el0);
    }
    return read_counter<Index>();
  }

  std::array<uint64_t, NumCounters + 1> read_all() const {
    std::array<uint64_t, NumCounters + 1> results{};
    read_all_impl(results, std::make_index_sequence<NumCounters>{});
    return results;
  }

  void checkpoint() {
    asm volatile("isb" ::: "memory");
    read_all_impl(values, std::make_index_sequence<NumCounters>{});
    // Reset counters to zero
    reset();
    // asm volatile("isb" ::: "memory");
  }

  std::unordered_map<std::string, uint64_t> get() const {
    std::unordered_map<std::string, uint64_t> out;
    auto results = read_all();

    for (size_t i = 0; i < NumCounters; i++) {
      out[counter_names[i]] = results[i];
    }
    out["CPU_CYCLES"] = results[NumCounters];
    return out;
  }

  std::map<std::string, benchmark::Counter> get_counters() const {
    std::map<std::string, benchmark::Counter> out;
    auto results = read_all();

    for (size_t i = 0; i < NumCounters; i++) {
      out[counter_names[i]] = benchmark::Counter(results[i]);
    }
    out["CPU_CYCLES"] = benchmark::Counter(results[NumCounters]);
    return out;
  }

  std::map<std::string, benchmark::Counter> get_avg_counters() const {
    std::map<std::string, benchmark::Counter> out;

    for (size_t i = 0; i < NumCounters; i++) {
      out[counter_names[i]] = benchmark::Counter(static_cast<double>(values[i]), benchmark::Counter::kAvgIterations);
    }
    out["CPU_CYCLES"] = benchmark::Counter(static_cast<double>(values[NumCounters]), benchmark::Counter::kAvgIterations);
    return out;
  }

  ALWAYS_INLINE void reset() const noexcept {
    // pmcr control
    // LC = long cycle counter (overflow on u64 overflow)
    // C = reset pmccntr_el0 to 0
    // P = reset event counters to 0
    // E = enable event counters
    WRITE_SPECIALREG(pmcr_el0, PMCR_LC | PMCR_C | PMCR_P | PMCR_E);
  }

  ALWAYS_INLINE void start() const noexcept {
    DEBUG(std::cerr << "start" << std::endl);
    // Cycle counter filter setup, non-secure hypervisor filtering
    WRITE_SPECIALREG(pmccfiltr_el0, PMEVTYPER_NSH);
    reset();
    // Clear unsigned overflow flag for all counters
    WRITE_SPECIALREG(pmovsclr_el0, ~0ULL);
    // Enable all configured counters alongside pmccntr_el0
    WRITE_SPECIALREG(pmcntenset_el0, CNTR_MASK);
    asm volatile("isb" ::: "memory");
  }

  ALWAYS_INLINE void stop() const noexcept {
    uint64_t val = READ_SPECIALREG(pmcr_el0);
    DEBUG(std::cerr << "stop" << std::endl);
    val &= PMCR_E;
    WRITE_SPECIALREG(pmcr_el0, val);
    asm volatile("isb" ::: "memory");
    /* Validate that we did not overflow the counters */
    uint64_t overflowed = READ_SPECIALREG(pmovsset_el0);
    if ((overflowed & CNTR_MASK) != 0) {
      for (size_t i = 0; i < NumCounters; i++) {
        if (overflowed & (1 << i)) {
          std::cerr << counter_names[i] << " overflow detected" << std::endl;
        }
      }
      if ((overflowed >> 31) & 0x01) {
        std::cerr << "CPU_CYCLES overflow detected" << std::endl;
      }
    }
  }
};

#endif /* __aarch64__ */

/*
 * Common memcpy-like benchmark body.
 * This allocates two cache-aligned buffers of a given size and runs the given copy-like
 * function in the benchmark loop, capturing the performance counters.
 */
template<auto Func, size_t NumCounters>
requires std::regular_invocable<decltype(Func), const void *, void *, size_t>
static void BenchmarkBody(benchmark::State &state, const std::array<std::string_view, NumCounters> &counters) {
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

  PerfCounterReader reader(counters);

  /* Note: should all be inlined from here... */
  reader.start();
  for (auto _ : state) {
    Func(src, dst, buffer_size);
    benchmark::ClobberMemory();
  }
  reader.stop();
  reader.checkpoint();
  /* ...to here, to minimize counters perturbations. */

  state.counters.merge(reader.get_avg_counters());

  state.SetBytesProcessed(state.iterations() * buffer_size);
  // double cycles_per_iter = static_cast<double>(end_cycles - start_cycles) /
  // static_cast<double>(total_unroll_iterations); state.counters["Cycles/Iter"]
  // = benchmark::Counter(cycles_per_iter, benchmark::Counter::kDefaults);

  std::free(src);
  std::free(dst);
}

template<auto Func, typename Counters, size_t... Is>
std::array<benchmark::internal::Benchmark *, sizeof...(Is)>
RegisterBenchmarkWithCounters(std::string name, std::index_sequence<Is...>) {
  /* Fold expression to register a benchmark with each counter set */
  return std::to_array({
      [&](){
        using CounterSet = std::tuple_element_t<Is, Counters>;
        constexpr size_t NumCounters = std::tuple_size<decltype(CounterSet::counters)>{};

        auto b = benchmark::RegisterBenchmark(name + "/" + std::string(CounterSet::name),
                                              [](benchmark::State &state) {
                                                BenchmarkBody<Func, NumCounters>(state, CounterSet::counters);
                                              });
        b->Apply(CopyBenchmarkArgs);
        return b;
      }()...
  });
}

struct InstrCounters {
  static constexpr std::string_view name = "InstrCounters";
  static constexpr auto counters = std::to_array<std::string_view>({
      "INST_RETIRED",
      "BR_MIS_PRED",
      "BR_PRED",
      "LD_SPEC",
      "ST_SPEC",
      "CAP_LD_SPEC",
      "CAP_ST_SPEC",
    });
};
using AllCounters = std::tuple<InstrCounters>;

#define BENCHMARK_WITH_COUNTERS(loop_fn)                                \
  static auto _benchmark_with_counters_ ## loop_fn =                    \
    RegisterBenchmarkWithCounters<loop_fn, AllCounters>(                \
      # loop_fn, \
      std::make_index_sequence<std::tuple_size_v<AllCounters>>{})
