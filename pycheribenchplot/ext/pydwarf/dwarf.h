/*
 * Common interface to the DWARF information visitor.
 */
#pragma once

#include <concepts>
#include <exception>
#include <string>
#include <type_traits>

#include <llvm/DebugInfo/DWARF/DWARFContext.h>
#include <llvm/Object/Binary.h>
#include <llvm/Support/Error.h>

// #include "dwarf_interface.h"

namespace cheri_benchplot {

int getABIPointerSize(const llvm::DWARFContext &DICtx);
int getABICapabilitySize(const llvm::DWARFContext &DICtx);

/*
 * Base class for inspection of DWARF Die information.
 * This is the core of the inspector that extract different information
 * from the object file.
 */
class DWARFVisitorBase {
public:
  DWARFVisitorBase(llvm::DWARFContext &DICtx);
  virtual ~DWARFVisitorBase() = default;

  /* Start scanning a new DWARF compile unit */
  virtual llvm::Error beginUnit(llvm::DWARFDie &CUDie);

  /* Implement in subclasses any number of visit_<TAG> functions with the
   *  following signature:
   *  llvm::Expected<bool> visit_<TAG>(llvm::DWARFDie &Die);
   *  where the tag is one of the names in llvm/BinaryFormat/Dwarf.def
   *
   * If the visitor function returns true, the visit is stopped.
   */

protected:
  llvm::DWARFContext &DICtx;
  llvm::DWARFUnit *CurrentUnit;
  std::string CurrentUnitName;
};

/*
 * Top-level interface to DWARF information.
 */
class DWARFInspector {
public:
  static DWARFInspector loadDWARF(std::string Path);
  DWARFInspector(DWARFInspector &&DWI)
      : Path(std::move(DWI.Path)), DICtx(std::move(DWI.DICtx)),
        OwnedBinary(std::move(DWI.OwnedBinary)) {}
  ~DWARFInspector() = default;

  int getABIPointerSize() const;
  int getABICapabilitySize() const;

  template <typename V, typename... Args>
  std::unique_ptr<V> createVisitor(Args... args) {
    return std::make_unique<V>(*DICtx, std::forward<Args>(args)...);
  }

  template <typename V>
  void visit(
      V &visitor,
      typename std::enable_if<std::is_base_of<DWARFVisitorBase, V>::value>::type
          * = nullptr) {
    for (auto &Unit : DICtx->info_section_units()) {
      if (!isCompileUnit(Unit)) {
        continue;
      }
      if (Unit->getVersion() < 4) {
        throw std::runtime_error("Unsupported DWARF version");
      }

      llvm::DWARFDie CUDie = Unit->getUnitDIE(false);
      auto Result = visitor.beginUnit(CUDie);
      if (Result) {
        throw std::runtime_error(llvm::toString(std::move(Result)));
      }
      // Iterate over DIEs in the unit
      llvm::DWARFDie Child = CUDie.getFirstChild();
      while (Child) {
        auto StopOrErr = dispatch(visitor, Child);
        if (auto E = StopOrErr.takeError()) {
          throw std::runtime_error(llvm::toString(std::move(E)));
        } else if (*StopOrErr) {
          break;
        }
        Child = Child.getSibling();
      }
    }
  }

protected:
  DWARFInspector(std::unique_ptr<llvm::DWARFContext> &&DICtx,
                 llvm::object::OwningBinary<llvm::object::Binary> &&Binary)
      : Path(Binary.getBinary()->getFileName()), DICtx(std::move(DICtx)),
        OwnedBinary(std::move(Binary)) {}

  template <typename V>
  llvm::Expected<bool> dispatch(V &Visitor, llvm::DWARFDie &Die) {
#define HANDLE_DW_TAG(ID, NAME, VERSION, VENDOR, KIND)                         \
  case ID: {                                                                   \
    constexpr bool HasVisit = requires(V & VRef, llvm::DWARFDie & DieRef) {    \
                                {                                              \
                                  VRef.visit_##NAME(DieRef)                    \
                                  } -> std::same_as<llvm::Expected<bool>>;     \
                              };                                               \
    if constexpr (HasVisit)                                                    \
      return Visitor.visit_##NAME(Die);                                        \
    break;                                                                     \
  }

    switch (Die.getTag()) {
#include <llvm/BinaryFormat/Dwarf.def>
    }
#undef HANDLE_DW_TAG

    return false;
  }

private:
  std::string Path;
  std::unique_ptr<llvm::DWARFContext> DICtx;
  llvm::object::OwningBinary<llvm::object::Binary> OwnedBinary;
};

} // namespace cheri_benchplot
