#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <string>

#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/restrict.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "dump.hh"
#include "memcount.hh"

namespace bl = boost::log;
namespace fs = std::filesystem;
namespace io = boost::iostreams;
namespace po = boost::program_options;

namespace {

std::function<void(int)> shutdown_handler;
void sigint_handler(int signal) {
  if (shutdown_handler)
    shutdown_handler(signal);
}

/**
 * Boost iostreams device source that reads up to N bytes from a
 * parent istream.
 */
class SliceSource : public io::source {
public:
  SliceSource(std::basic_istream<char> *src, std::streamsize max_bytes)
      : io::source(), parent_(src), remaining_(max_bytes) {}

  std::streamsize read(char *s, std::streamsize n) {
    auto nbytes = std::min(n, remaining_);
    if (nbytes == 0)
      return -1; // EOF
    if (parent_->read(s, nbytes)) {
      remaining_ -= nbytes;
      return nbytes;
    } else {
      auto nread = parent_->gcount();
      if (nread == 0)
        return -1; // EOF
      return nread;
    }
  }

private:
  std::basic_istream<char> *parent_;
  std::streamsize remaining_;
};

} // namespace

namespace cheri {

constexpr long FRAME_INTERVAL = 5000000;
constexpr long MAX_IFRAME_SIZE = 1024;

bool traceIndexer(std::shared_ptr<ParserState> state, std::stop_token stop) {
  std::uint32_t preamble;

  LOG(info) << "Indexing trace: frame interval = " << FRAME_INTERVAL;
  io::stream<io::mapped_file_source> trace(state->trace);

  long long entries = 0;
  while (!stop.stop_requested() && trace) {
    // Skip preamble that encodes entry size
    if (entries % FRAME_INTERVAL == 0) {
      LOG(info) << "Index frame @" << entries;
      IndexFrame iframe;
      iframe.set_offset(trace.tellg());
      state->iframes.emplace_back(std::move(iframe));
    }

    if (!trace.read(reinterpret_cast<char *>(&preamble), sizeof(preamble))) {
      LOG(info) << "End of trace";
      break;
    }
    // Skip the contents, we don't care now
    trace.seekg(preamble, std::ios_base::cur);
    state->decoded += 1;
    state->processed_bytes += preamble;
    entries++;
  }
  LOG(info) << "Indexed trace: iframes = " << std::size(state->iframes);
  return true;
}

bool traceDecoder(
    std::shared_ptr<ParserState> state, long offset, long count,
    std::function<void(unsigned long, const QEMULogEntry &)> callback,
    std::stop_token stop) {
  std::uint32_t preamble;

  LOG(info) << "Start decoder: offset = " << offset << " blocksize = " << count;
  io::stream<io::mapped_file_source> trace(state->trace);
  trace.set_auto_close(false);

  // Convert instruction offset to an actual file offset
  long iframe_index = offset / FRAME_INTERVAL;
  long file_offset = state->iframes[iframe_index].offset();
  trace.seekg(file_offset, std::ios_base::beg);

  LOG(debug) << "Decoder loop at file offset " << file_offset;
  long decoded = 0;
  while (!stop.stop_requested() && trace && decoded < count) {
    // Skip preamble that encodes entry size
    if (!trace.read(reinterpret_cast<char *>(&preamble), sizeof(preamble))) {
      LOG(info) << "End of trace";
      break;
    }
    auto saved_offset = trace.tellg();
    io::stream<SliceSource> slice(&trace, preamble);
    // io::stream slice(io::restrict(trace, saved_offset, preamble));
    // slice.set_auto_close(false);

    QEMULogEntry entry;
    if (!entry.ParseFromIstream(&slice)) {
      LOG(error) << "Failed to parse trace entry (+" << decoded << ") @"
                 << offset + decoded;
      if (state->options.debug_dump) {
        fs::path dump_path(state->options.trace_path);
        dump_path.concat("." + std::to_string(offset + decoded));
        LOG(warning) << "Dumping failed trace entry to " << dump_path;
        io::stream<io::file_sink> dump(dump_path);
        trace.seekg(saved_offset);
        std::vector<char> tmp(preamble);
        trace.read(tmp.data(), preamble);
        dump.write(tmp.data(), preamble);
        dump.close();

        LOG(warning) << "Dump done, skipping entry";
      }
      state->decode_fail.fetch_add(1, std::memory_order_relaxed);
    } else {
      callback(offset + decoded, entry);
    }

    // Increment this every N entries to reduce conention
    state->decoded.fetch_add(1, std::memory_order_relaxed);
    state->processed_bytes.fetch_add(preamble, std::memory_order_relaxed);
    decoded++;
  }
  LOG(debug) << "Worker done exit conditions:"
             << " decoded = " << decoded
             << " trace closed = " << !trace.is_open()
             << " trace EOF = " << trace.eof()
             << " decoded >= count = " << (decoded >= count)
             << " stop req = " << stop.stop_requested();
  LOG(info) << "Exit decoder loop";
  return true;
}

void traceProgress(std::stop_token stop, std::shared_ptr<ParserState> state) {
  using namespace std::chrono_literals;

  long long progress = 0;
  std::condition_variable cv;
  std::mutex mutex;
  std::stop_callback stop_wait(stop, [&]() {
    // Prevent race between notification and stop_request check
    // in the loop.
    { std::lock_guard _(mutex); }
    cv.notify_one();
  });

  std::unique_lock<std::mutex> lock(mutex);
  while (!stop.stop_requested()) {
    auto count = state->decoded.load(std::memory_order_relaxed);
    double keps = (count - progress) / 5000;
    LOG(info) << "Processed " << count << " entries (" << keps << " Ke/s | "
              << (state->processed_bytes.load(std::memory_order_relaxed) >> 20)
              << " MiB) " << state->decode_fail << " fails";
    progress = count;
    // poll statistics every 5 seconds
    cv.wait_for(lock, 5000ms, [&stop]() { return stop.stop_requested(); });
  }
}

TraceParser::TraceParser(ParserOptions options) {
  LOG(info) << "Initialize trace parser for" << options.trace_path;

  if (options.workers == 0)
    options.workers = 1;

  state_ = std::make_shared<ParserState>(options);
}

TraceParser::~TraceParser() {}

void TraceParser::run() {
  shutdown_handler = [this](int _signal) { cancel(); };
  std::signal(SIGINT, sigint_handler);

  // Check if we have the trace index
  if (fs::exists(state_->options.index_path)) {
    readIndex(state_->options.index_path);
  } else {
    createIndex();
    writeIndex(state_->options.index_path);
  }

  unsigned long total_entries = std::size(state_->iframes) * FRAME_INTERVAL;
  unsigned long chunk_size = total_entries / state_->options.workers;
  auto callback = std::bind(&TraceParser::onTraceEntry, this,
                            std::placeholders::_1, std::placeholders::_2);
  std::vector<std::future<bool>> results;

  for (int i = 0; i < state_->options.workers + 1; i++) {
    auto offset = i * chunk_size;
    LOG(debug) << "Schedule worker " << i << " [" << offset << ", "
               << offset + chunk_size << "]"
               << " / " << total_entries;
    auto result = pool_.submit([=, s = state_](std::stop_token stop) {
      return traceDecoder(s, offset, chunk_size, callback, stop);
    });
    results.emplace_back(std::move(result));
  }

  pool_.submit(
      [=, s = state_](std::stop_token stop) { return traceProgress(stop, s); });

  // Wait for the results to pop up.
  for (auto &r : results) {
    r.wait();
  }
  // The only remaining task is the progress metering, cancel it
  cancel();
  finalize();
}

void TraceParser::createIndex() {
  auto result =
      pool_.submit(std::bind(traceIndexer, state_, std::placeholders::_1));

  pool_.submit(
      [=, s = state_](std::stop_token stop) { return traceProgress(stop, s); });
  result.wait();
  pool_.cancel();
  pool_.drain();
}

void TraceParser::readIndex(fs::path path) {
  io::stream<io::file_source> iframes(path);
  std::uint32_t iframe_size;
  char buffer[MAX_IFRAME_SIZE];

  LOG(info) << "Read iframes from " << path;
  while (iframes) {
    if (!iframes.read(reinterpret_cast<char *>(&iframe_size),
                      sizeof(iframe_size))) {
      LOG(debug) << "End of iframes";
      break;
    }
    if (iframe_size > MAX_IFRAME_SIZE) {
      LOG(error) << "Invalid iframe size: " << iframe_size << " > "
                 << MAX_IFRAME_SIZE;
      throw std::runtime_error("Invalid iframe record");
    }
    iframes.read(buffer, iframe_size);

    auto &frame = state_->iframes.emplace_back();
    if (!frame.ParseFromArray(buffer, iframe_size)) {
      LOG(error) << "Invalid iframe";
      throw std::runtime_error("Decoding iframe error");
    }
  }
  LOG(info) << "Loaded " << std::size(state_->iframes) << " iframes";
}

void TraceParser::writeIndex(fs::path path) {
  io::stream<io::file_sink> iframes(path);

  LOG(info) << "Write iframes to " << path;
  for (auto &frame : state_->iframes) {
    auto size = frame.ByteSizeLong();
    if (size > std::numeric_limits<std::uint32_t>::max()) {
      LOG(error) << "IFrame is too long";
      throw std::runtime_error("Invalid iframe size");
    }
    std::uint32_t wire_size = size;
    if (!iframes.write(reinterpret_cast<char *>(&wire_size),
                       sizeof(wire_size))) {
      LOG(error) << "Can not serialize iframe wire size";
      throw std::runtime_error("Encoding iframe size error");
    }
    if (!frame.SerializeToOstream(&iframes)) {
      LOG(error) << "IFrame serialization error";
      throw std::runtime_error("Encoding iframe error");
    }
  }
  LOG(info) << "Serialized iframes";
}

void TraceParser::finalize() { pool_.drain(); }

void TraceParser::cancel() { pool_.cancel(); }

} /* namespace cheri */

int main(int argc, char *argv[]) {
  fs::path trace_file;
  fs::path index_file;
  fs::path memory_regions;
  std::string action;
  int workers = std::thread::hardware_concurrency();

  GOOGLE_PROTOBUF_VERIFY_VERSION;

  po::options_description desc("Qemu protobuf trace parser and scraper");
  /* clang-format: off */
  desc.add_options()("help", "produce help message")(
      "trace", po::value<fs::path>(&trace_file)->required(),
      "Path to the trace file")(
      "action", po::value<std::string>(&action)->required(),
      "Action to perform, valid values are {memcount,dump}")(
      "threads", po::value<int>(&workers), "Number of parallel threads to use")(
      "memory-regions", po::value<fs::path>(&memory_regions),
      "Path to a file specifying the memory regions to consider for memcount")(
      "index", po::value<fs::path>(&index_file),
      "Path to the trace index file")(
      "decode-fail-dump",
      "Dump entries that fail to decode for inspection (see protoscope)")(
      "verbose", "verbose logging");
  /* clang-format: on */

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  try {
    po::notify(vm);
  } catch (po::error &ex) {
    LOG(error) << "Invalid command line options";
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("verbose")) {
    bl::core::get()->set_filter(bl::trivial::severity >= bl::trivial::debug);
  } else {
    bl::core::get()->set_filter(bl::trivial::severity >= bl::trivial::info);
  }

  cheri::ParserOptions opts;
  opts.trace_path = trace_file;
  opts.workers = workers;
  if (vm.count("index") == 0) {
    opts.index_path = trace_file;
    opts.index_path.replace_filename(trace_file.filename().concat(".index"));
  } else {
    opts.index_path = index_file;
  }
  opts.debug_dump = (vm.count("decode-fail-dump") == 1);

  if (action == "dump") {
    LOG(info) << "TODO Dump trace";
  } else if (action == "memcount") {
    LOG(info) << "Prepare to count memory access statistics";
    if (vm.count("memory-regions") == 0) {
      LOG(error) << "--memory-regions is required";
      return 1;
    }
    cheri::MemcountTraceParser parser(opts, memory_regions);
    parser.run();
    parser.report();
  } else {
    LOG(error) << "Invalid action " << action << ", valid values are: ";
    LOG(error) << "dump: Dump trace entries to stdout";
    LOG(error) << "memcount: Collect memory access statistics";
  }

  return 0;
}
