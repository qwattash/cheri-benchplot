#include <cstdlib>
#include <cstring>
#include <ctime>
#include <inttypes.h>
#include <iostream>

#include "cheri_micro_benchmark.h"

void CopyBenchmarkArgs(benchmark::internal::Benchmark *b) {
  b->ArgsProduct({{512, 1472, 1 << 16, 1 << 20, 1 << 24}, {0, 4, 8}});
  b->ArgNames({"bytes", "offset"});
}

int main(int argc, char **argv) {
  std::cout << "==============================================================="
               "=========================\n";
  std::cout << "                           Arm Morello CHERI Microbenchmark    "
               "                         \n";
  std::cout << "==============================================================="
               "=========================\n";

#ifdef __CHERI__
#ifndef __ARM_MORELLO_PURECAP_BENCHMARK_ABI
  std::cerr << "[ WARNING ] Kernel built for Morello purecap ABI " << std::endl;
  std::cerr << "[ WARNING ] Maybe you meant to use the benchmark ABI "
               "(-mabi=purecap-benchmark)?"
            << std::endl;
#endif
#endif

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  init_counters();

  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
