#include <cstdlib>
#include <cstring>
#include <ctime>
#include <inttypes.h>
#include <iostream>

#include "cheri_micro_benchmark.h"

void CopyBenchmarkArgs(benchmark::internal::Benchmark *b) {
  b->Arg(512);
  b->Arg(1472);
  b->Arg(1 << 16);
  b->Arg(1 << 20);
  b->Arg(1 << 24);
  b->ArgName("bytes");
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
