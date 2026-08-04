#pragma once

#include <benchmark/benchmark.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <cheriintrin.h>

#ifdef __aarch64__
#include <machine/armreg.h>
#include <sys/sysctl.h>
#endif

void CustomArguments(benchmark::internal::Benchmark *b);

extern "C" {
void copycommon_simple(void *src, void *dst, size_t len);
void copycommon_scalar_unrolled(void *src, void *dst, size_t len);
}

// Cache line alignment constraint for CHERI capabilities (128 bits = 16 bytes).
// Aligning memory to 64-byte boundaries prevents split cache-line penalties.
#define CACHE_LINE_SIZE 64

#define ALWAYS_INLINE __attribute__((always_inline))

extern std::unordered_map<std::string, uint64_t> EventRegistry;

#ifdef __aarch64__

static inline void init_counters() {
  uint64_t pmu_enabled;
  size_t size = sizeof(pmu_enabled);
  if (sysctlbyname("machdep.pmu_user_access", &pmu_enabled, &size, NULL, 0) ==
      -1) {
    throw std::runtime_error("No machdep.pmu_user_access");
  }
  if ((pmu_enabled & 0x1) == 0) {
    throw std::runtime_error("PMU user access disabled");
  }
}

static inline uint64_t read_cntvct() { return READ_SPECIALREG(cntvct_el0); }

static inline uint64_t read_cntfrq() { return READ_SPECIALREG(cntfrq_el0); }

static inline uint64_t read_cycles() { return READ_SPECIALREG(pmccntr_el0); }

#define EVAL_PMEVCNTR_CASE(target_idx, current_idx)                            \
  if constexpr (current_idx == target_idx) {                                   \
    uint64_t val;                                                              \
    asm volatile("mrs %0, pmevcntr" #target_idx "_el0" : "=r"(val));           \
    return val;                                                                \
  }

#define EVAL_PMEVTYPER_CASE(target_idx, current_idx, event)                    \
  if constexpr (current_idx == target_idx) {                                   \
    asm volatile("msr pmevtyper" #target_idx                                   \
                 "_el0, %0" ::"r"(PMEVTYPER_NSH | event));                     \
  }

template <size_t NumCounters> class PerfCounterReader {
private:
  std::array<std::string, NumCounters> counter_names;
  std::array<uint64_t, NumCounters> event_ids;

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
  void read_all_impl(std::array<uint64_t, NumCounters> &out,
                     std::index_sequence<Is...>) const {
    ((out[Is] = read_counter<Is>()), ...);
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
    size_t max_supported =
        (READ_SPECIALREG(pmcr_el0) & PMCR_N_MASK) >> PMCR_N_SHIFT;
    assert(NumCounters < max_supported && "Too many counters for platform");

    config_all_impl(event_ids, std::make_index_sequence<NumCounters>{});
  }

  PerfCounterReader(std::array<std::string, NumCounters> &&names)
      : counter_names(std::move(names)),
        event_ids(map_names_to_ids(counter_names,
                                   std::make_index_sequence<NumCounters>{})) {
    size_t max_supported =
        (READ_SPECIALREG(pmcr_el0) & PMCR_N_MASK) >> PMCR_N_SHIFT;
    assert(NumCounters < max_supported && "Too many counters for platform");

    config_all_impl(event_ids, std::make_index_sequence<NumCounters>{});
  }

  // Compile-time safe interface, note that this expands to a direct register
  // read
  template <size_t Index> uint64_t read() const {
    static_assert(Index < NumCounters,
                  "Counter index out of bounds for this reader configuration.");
    return read_counter<Index>();
  }

  std::array<uint64_t, NumCounters> read_all() const {
    std::array<uint64_t, NumCounters> results{};
    read_all_impl(results, std::make_index_sequence<NumCounters>{});
    return results;
  }

  std::unordered_map<std::string, uint64_t> get() const {
    std::unordered_map<std::string, uint64_t> out;
    auto results = read_all();

    for (size_t i = 0; i < NumCounters; i++) {
      out[counter_names[i]] = results[i];
    }
    return out;
  }

  std::map<std::string, benchmark::Counter> get_counters() const {
    std::map<std::string, benchmark::Counter> out;
    auto results = read_all();

    for (size_t i = 0; i < NumCounters; i++) {
      out[counter_names[i]] = benchmark::Counter(results[i]);
    }
    return out;
  }

  ALWAYS_INLINE void start() const noexcept {
    WRITE_SPECIALREG(pmccfiltr_el0, PMEVTYPER_NSH);
    WRITE_SPECIALREG(pmcr_el0, PMCR_LC | PMCR_C | PMCR_P | PMCR_E);
    WRITE_SPECIALREG(pmovsclr_el0, ~0ULL);
  }

  ALWAYS_INLINE void stop() const noexcept {
    uint64_t val = READ_SPECIALREG(pmcr_el0);
    val &= PMCR_E;
    WRITE_SPECIALREG(pmcr_el0, val);
    asm volatile("isb" ::: "memory");
  }
};

#endif /* __aarch64__ */
