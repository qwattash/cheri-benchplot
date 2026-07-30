
#include "cheri_micro_benchmark.h"

#ifdef __aarch64__
std::unordered_map<std::string, uint64_t> EventRegistry = {
  { "INST_RETIRED", 0x08 },
  { "BR_MIS_PRED", 0x10 },
  { "BR_PRED", 0x12 },
  { "LD_SPEC", 0x70 },
  { "ST_SPEC", 0x71 },
  { "CAP_LD_SPEC", 0x210 },
  { "CAP_ST_SPEC", 0x211 }
};
#endif
