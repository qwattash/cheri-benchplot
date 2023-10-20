#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <llvm/Support/CommandLine.h>

#include "log.hh"
#include "pool.hh"
#include "scraper.hh"
#include "storage.hh"
#include "struct_layout_scraper.hh"

namespace fs = std::filesystem;
namespace cl = llvm::cl;

namespace {

/**
 * Maps command line arguments to an internal identifier for
 * a specific scraper.
 */
enum class ScraperID { StructLayout };

/**
 * Helper context for the scraping session
 */
struct ScrapeContext {
  using TaskResult = std::optional<cheri::ScrapeResult>;
  using TaskFuture = std::shared_future<TaskResult>;

  ScrapeContext(unsigned long workers, fs::path db_file)
      : pool(workers), storage(db_file) {}

  /* Thread pool where work is submitted */
  cheri::ThreadPool pool;
  /* Vector of future results */
  std::vector<TaskFuture> future_results;
  /* Storage manager */
  cheri::StorageManager storage;
};

std::unique_ptr<cheri::DwarfScraper>
MakeScraper(ScraperID id, ScrapeContext &ctx,
            std::shared_ptr<cheri::DwarfSource> dwsrc) {
  switch (id) {
  case ScraperID::StructLayout:
    return std::make_unique<cheri::StructLayoutScraper>(ctx.storage, dwsrc);
  default:
    throw std::runtime_error("Unexpected scraper ID");
  }
}

void Scrape(ScrapeContext &ctx, fs::path target,
            std::vector<ScraperID> *scrapers) {
  LOG(cheri::kInfo) << "Create DWARF scraping job for " << target;

  /* DWARF source is shared among all scrapers, which run concurrently */
  auto dwsrc = std::make_shared<cheri::DwarfSource>(target);

  for (auto id : *scrapers) {
    auto future = ctx.pool.Async(
        [&ctx, dwsrc,
         id](std::stop_token stop_tok) -> std::optional<cheri::ScrapeResult> {
          try {
            auto scr = MakeScraper(id, ctx, dwsrc);
            scr->Extract(stop_tok);
            return scr->Result();
          } catch (std::exception &ex) {
            LOG(cheri::kError)
                << "DWARF scraper failed for " << dwsrc->GetPath().string();
            return std::nullopt;
          }
        });
    ctx.future_results.emplace_back(future);
  }
}

void TryScrape(ScrapeContext &ctx, fs::path target,
               std::vector<ScraperID> *scrapers) {
  try {
    Scrape(ctx, target, scrapers);
  } catch (std::exception &ex) {
    LOG(cheri::kError) << "Failed to setup scraping for " << target.string() <<
        " reason: " << ex.what();
  }
}

} // namespace

// clang-format off
static cl::OptionCategory cat_cheri_scraper("DWARF Scraper Options");

static cl::opt<bool> opt_verbose(
    "verbose",
    cl::desc("Enable verbose output"),
    cl::cat(cat_cheri_scraper));
static cl::alias alias_verbose(
    "v",
    cl::desc("Alias for --verbose"),
    cl::aliasopt(opt_verbose),
    cl::cat(cat_cheri_scraper));

static cl::opt<unsigned int> opt_workers(
    "threads",
    cl::desc("Use parallel threads for DWARF traversal, (defaults to #CPU)"),
    cl::init(std::thread::hardware_concurrency()),
    cl::cat(cat_cheri_scraper));
static cl::alias alias_workers(
    "t",
    cl::desc("Alias for --threads"),
    cl::aliasopt(opt_workers),
    cl::cat(cat_cheri_scraper));

static cl::opt<fs::path> opt_database(
    "database",
    cl::desc("Database file to store the information (defaults to cheri-dwarf.sqlite)"),
    cl::init("cheri-dwarf.sqlite"),
    cl::cat(cat_cheri_scraper));

static cl::opt<bool> opt_stdin(
    "stdin",
    cl::desc("Read input files from stdin instead of looking for --input options"),
    cl::cat(cat_cheri_scraper));

static cl::list<fs::path> opt_input(
    "input",
    cl::desc("Specify input file path(s)"),
    cl::cat(cat_cheri_scraper),
    cl::Required);
static cl::alias alias_input(
    "i",
    cl::desc("Alias for --input"),
    cl::aliasopt(opt_input),
    cl::cat(cat_cheri_scraper));

static cl::list<ScraperID> opt_scrapers(
    "scrapers",
    cl::desc("Select the scrapers to run:"),
    cl::values(
        clEnumValN(ScraperID::StructLayout, "struct-layout", "Extract structure layouts")),
    cl::cat(cat_cheri_scraper),
    cl::Required);
static cl::alias alias_scrapers(
    "s",
    cl::desc("Alias for --scrapers"),
    cl::aliasopt(opt_scrapers),
    cl::cat(cat_cheri_scraper));
// clang-format on

int main(int argc, char **argv) {
  cheri::Logger &logger = cheri::Logger::Default();

  cl::HideUnrelatedOptions(cat_cheri_scraper);
  cl::ParseCommandLineOptions(argc, argv, "CHERI debug info scraping tool");

  if (opt_verbose) {
    logger.SetLevel(cheri::kDebug);
  }

  LOG(cheri::kDebug) << "Initialize thread pool with " << opt_workers << " workers";
  ScrapeContext ctx(opt_workers, opt_database);

  if (opt_stdin) {
    LOG(cheri::kDebug) << "Reading target files from STDIN";
    fs::path path;
    while (std::cin >> path) {
      TryScrape(ctx, path, &opt_scrapers);
    }
  } else {
    LOG(cheri::kDebug) << "Reading target files from --input args";
    for (auto path : opt_input) {
      TryScrape(ctx, path, &opt_scrapers);
    }
  }

  ctx.pool.Join();

  return 0;
}
