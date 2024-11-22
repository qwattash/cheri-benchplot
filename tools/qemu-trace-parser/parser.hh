#pragma once

#include <atomic>
#include <filesystem>
#include <vector>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/log/trivial.hpp>

#include "pool.hh"
#include "qemu_log_entry.pb.h"

#define LOG(level) BOOST_LOG_TRIVIAL(level)

namespace cheri {

/**
 * Parser configuration
 */
struct ParserOptions {
  std::filesystem::path trace_path;
  std::filesystem::path index_path;
  int workers;
  bool debug_dump;
};

/**
 * Shared state information
 */
struct ParserState {
  ParserState(ParserOptions opts) : options(opts), trace(opts.trace_path) {}

  ParserOptions options;
  std::atomic<long long> decoded;
  std::atomic<long long> decode_fail;
  std::atomic<long long> processed_bytes;
  std::vector<IndexFrame> iframes;
  boost::iostreams::mapped_file_source trace;
};

/**
 * Base class for parsing QEMU traces.
 */
class TraceParser {
public:
  TraceParser(ParserOptions options);
  TraceParser(const TraceParser &) = delete;
  virtual ~TraceParser();

  void createIndex();
  void readIndex(std::filesystem::path path);
  void writeIndex(std::filesystem::path path);
  void run();
  void finalize();
  void cancel();
  virtual void report() {}
  virtual void onTraceEntry(unsigned long offset,
                            const QEMULogEntry &entry) = 0;

protected:
  WorkerPool pool_;
  std::shared_ptr<ParserState> state_;
  std::thread progress_;
};

} /* namespace cheri */
