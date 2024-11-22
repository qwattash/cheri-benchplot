
#include <cstdint>
#include <fstream>
#include <string>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/format.hpp>

#include "memcount.hh"

namespace fs = std::filesystem;

namespace {} // namespace

namespace cheri {

MemcountTraceParser::MemcountTraceParser(ParserOptions options,
                                         fs::path mem_regions)
    : TraceParser(options), regions_file_(mem_regions) {
  LOG(info) << "Read memory regions from" << mem_regions;

  std::ifstream region_file(mem_regions);
  std::string line;
  while (std::getline(region_file, line)) {
    std::vector<std::string> parts;
    boost::algorithm::split(parts, line, [](char c) { return c == ','; });
    if (std::size(parts) != 3) {
      LOG(error) << "Invalid memory region line: " << line;
      continue;
    }
    RegionInfo &ri = regions_.emplace_back();
    ri.name = boost::algorithm::trim_copy(parts[0]);
    std::string base = boost::algorithm::trim_copy(parts[1]);
    ri.start = std::stoull(base, nullptr, 0);

    std::string top = boost::algorithm::trim_copy(parts[2]);
    if (top.starts_with('+')) {
      top.erase(0, 1);
      ri.end = ri.start;
    }
    ri.end += std::stoull(top, nullptr, 0);
    LOG(info) << "Add region [" << boost::format("%#x") % ri.start << ", "
              << boost::format("%#x") % ri.end << "] " << ri.name;
  }

  // Sort regions
  std::sort(regions_.begin(), regions_.end(),
            [](RegionInfo &l, RegionInfo &r) { return l.start < r.start; });
}

void MemcountTraceParser::onTraceEntry(unsigned long offset,
                                       const QEMULogEntry &entry) {

  // Skip event-only entries
  if (entry.insn_case() != QEMULogEntry::InsnCase::kDisas) {
    LOG(debug) << "Skip non-instruction event";
    return;
  }

  auto pc = entry.pc();
  if (pc == 0) {
    LOG(warning) << "Entry without PC";
  }
  std::optional<const QEMULogEntryMem *> mem;
  std::optional<const QEMULogEntryMem *> mem2;
  if (auto naccess = entry.mem_size()) {
    mem = &entry.mem(0);
    if (naccess > 1) {
      mem2 = &entry.mem(1);
    }
    if (naccess > 2) {
      LOG(warning) << "Entry with too many memory accesses pc="
                   << boost::format("%#x") % pc;
    }
  }
  for (const auto &ri : regions_) {
    if (ri.start <= pc && pc < ri.end) {
      ri.stats->fetch_count.fetch_add(1, std::memory_order_relaxed);
      ri.stats->fetch_bytes.fetch_add(4, std::memory_order_relaxed);
    }
    recordMemAccess(mem, ri);
    recordMemAccess(mem2, ri);
  }
}

void MemcountTraceParser::recordMemAccess(
    std::optional<const QEMULogEntryMem *> &opt, const RegionInfo &ri) {
  if (!opt)
    return;
  auto mem = *opt;
  if (ri.start <= mem->addr() && mem->addr() < ri.end) {
    if (mem->is_load()) {
      ri.stats->read_count.fetch_add(1, std::memory_order_relaxed);
      ri.stats->read_bytes.fetch_add(mem->size(), std::memory_order_relaxed);
    } else {
      ri.stats->write_count.fetch_add(1, std::memory_order_relaxed);
      ri.stats->write_bytes.fetch_add(mem->size(), std::memory_order_relaxed);
    }
  }
}

void MemcountTraceParser::report() {
  LOG(info) << "Memory access count report";
  LOG(info) << "Decoded " << state_->decoded << " entries";
  std::ofstream csv(
      regions_file_.replace_filename(regions_file_.filename().concat(".csv")));
  csv << "Total,Region,Range,#Read,#Write,#Fetch,Read (KiB),Write (KiB),Fetch "
         "(KiB)\n";
  for (auto &ri : regions_) {
    LOG(info) << "Region " << ri.name << " [" << boost::format("%#x") % ri.start
              << ", " << boost::format("%#x") % ri.end << "]";
    LOG(info) << "Access counts: R=" << ri.stats->read_count
              << " W=" << ri.stats->write_count
              << " X=" << ri.stats->fetch_count;
    LOG(info) << "Bytes touched: R=" << ri.stats->read_bytes / 1024 << " KiB"
              << " W=" << ri.stats->write_bytes / 1024 << " KiB"
              << " X=" << ri.stats->fetch_bytes / 1024 << " KiB";
    csv << state_->decoded << "," << ri.name << ","
        << "[" << boost::format("%#x") % ri.start << ":"
        << boost::format("%#x") % ri.end << "]," << ri.stats->read_count << ","
        << ri.stats->write_count << "," << ri.stats->fetch_count << ","
        << ri.stats->read_bytes << "," << ri.stats->write_bytes << ","
        << ri.stats->fetch_bytes << "\n";
  }
  csv.close();
}

} /* namespace cheri */
