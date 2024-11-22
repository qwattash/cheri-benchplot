#pragma once

#include <optional>

#include "parser.hh"

namespace cheri {

// Allocate these on the side because atomics can not move.
struct RegionStats {
  unsigned long totalAccessCount() {
    return read_count + write_count + fetch_count;
  }
  unsigned long totalBytesCount() {
    return read_bytes + write_bytes + fetch_bytes;
  }

  std::atomic<unsigned long> read_count;
  std::atomic<unsigned long> write_count;
  std::atomic<unsigned long> fetch_count;
  std::atomic<unsigned long> read_bytes;
  std::atomic<unsigned long> write_bytes;
  std::atomic<unsigned long> fetch_bytes;
};

struct RegionInfo {
  RegionInfo() : start(0), end(0), stats(std::make_unique<RegionStats>()) {}

  // Base information from config file
  std::string name;
  unsigned long start;
  unsigned long end;
  // Statistics
  std::unique_ptr<RegionStats> stats;
};

class MemcountTraceParser : public TraceParser {
public:
  MemcountTraceParser(ParserOptions options, std::filesystem::path mem_regions);

  void onTraceEntry(unsigned long offset, const QEMULogEntry &entry) override;
  void report() override;

private:
  void recordMemAccess(std::optional<const QEMULogEntryMem *> &opt,
                       const RegionInfo &ri);

  std::vector<RegionInfo> regions_;
};

} /* namespace cheri */
