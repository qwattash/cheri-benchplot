
#pragma once

#include "scraper.hh"

namespace cheri {

/**
 * Specialized scraper that extract DWARF structure layout information.
 *
 * The structure layouts are nested tree-like objects, where the top-level
 * structure definition is the root of the tree.
 *
 * The scraper initialized the storage with the following schema:
 * - The Types table records all types we have seen, if a Type is aggregate
 * (i.e. struct, union, class), there will be a one-to-many relationship
 * with the Members table.
 * - The Members table records structure members. These are associated to
 * one and only one entry in Types table, which is the containing object.
 */
class StructLayoutScraper : public DwarfScraper {
 public:
  StructLayoutScraper(StorageManager &sm, std::shared_ptr<const DwarfSource> dwsrc)
      : DwarfScraper(sm, dwsrc) {}

  bool visit_structure_type(llvm::DWARFDie &die);
  bool visit_class_type(llvm::DWARFDie &die);
  bool visit_union_type(llvm::DWARFDie &die);
  bool visit_typedef(llvm::DWARFDie &die);

 protected:
  void InitSchema() override;
  void BeginUnit(llvm::DWARFDie &unit_die) override;
  void EndUnit(llvm::DWARFDie &unit_die) override;
};

} /* namespace cheri */
