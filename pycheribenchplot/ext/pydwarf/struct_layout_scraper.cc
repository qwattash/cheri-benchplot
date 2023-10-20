
#include "struct_layout_scraper.hh"

namespace cheri {

bool StructLayoutScraper::visit_structure_type(llvm::DWARFDie &die) {
  return false;
}

bool StructLayoutScraper::visit_class_type(llvm::DWARFDie &die) {
  return false;
}

bool StructLayoutScraper::visit_union_type(llvm::DWARFDie &die) {
  return false;
}

bool StructLayoutScraper::visit_typedef(llvm::DWARFDie &die) {
  return false;
}

void StructLayoutScraper::InitSchema() {}
void StructLayoutScraper::BeginUnit(llvm::DWARFDie &unit_die) {}
void StructLayoutScraper::EndUnit(llvm::DWARFDie &unit_die) {}

} /* namespace cheri */
