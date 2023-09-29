#include <exception>
#include <format>
#include <mutex>

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/DebugInfo/DIContext.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/Error.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include "dwarf.h"

namespace object = llvm::object;
namespace dwarf = llvm::dwarf;

namespace {

std::once_flag LLVMInitFlag;

} // namespace

namespace cheri_benchplot {

/*
 * Helpers to determine the pointer/capability size
 */
int getABIPointerSize(const llvm::DWARFContext &DICtx) {
  auto *Obj = DICtx.getDWARFObj().getFile();
  if (Obj == nullptr) {
    return -1;
  }
  auto Triple = Obj->makeTriple();

  if (Triple.getEnvironment() == llvm::Triple::CheriPurecap) {
    return getABICapabilitySize(DICtx);
  } else {
    if (Triple.getArch() == llvm::Triple::aarch64 ||
        Triple.getArch() == llvm::Triple::riscv64) {
      return 8;
    } else if (Triple.getArch() == llvm::Triple::riscv32) {
      return 4;
    }
  }
  return -1;
}

int getABICapabilitySize(const llvm::DWARFContext &DICtx) {
  auto *Obj = DICtx.getDWARFObj().getFile();
  if (Obj == nullptr) {
    return -1;
  }
  auto Triple = Obj->makeTriple();

  if (Triple.getArch() == llvm::Triple::aarch64 ||
      Triple.getArch() == llvm::Triple::riscv64) {
    return 16;
  } else if (Triple.getArch() == llvm::Triple::riscv32) {
    return 8;
  }
  return -1;
}

/*
 * DWARF Inspector implementation
 */
DWARFInspector DWARFInspector::loadDWARF(std::string Path) {
  // static LLVMInitFlag
  std::call_once(LLVMInitFlag, []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
  });

  llvm::Expected<object::OwningBinary<object::Binary>> BinOrErr =
      object::createBinary(Path);
  if (auto E = BinOrErr.takeError()) {
    throw std::runtime_error(llvm::toString(std::move(E)));
  }
  auto &OwnedBinary = *BinOrErr;
  auto *Obj = llvm::dyn_cast<object::ObjectFile>(OwnedBinary.getBinary());
  if (Obj == nullptr) {
    throw std::runtime_error(
        std::format("Invalid binary at %s, not an object", Path));
  }

  auto DICtx = llvm::DWARFContext::create(
      *Obj, llvm::DWARFContext::ProcessDebugRelocations::Process, nullptr, "",
      nullptr);

  // Check DWARF version
  if (DICtx->getMaxVersion() < 4) {
    throw std::runtime_error("Unsupported DWARF version, use 4 or above");
  }

  return DWARFInspector(std::move(DICtx), std::move(OwnedBinary));
}

int DWARFInspector::getABIPointerSize() const {
  return cheri_benchplot::getABIPointerSize(*DICtx);
}

int DWARFInspector::getABICapabilitySize() const {
  return cheri_benchplot::getABICapabilitySize(*DICtx);
}

/*
 * Common DWARF visitor data and methods.
 * The visitors can define functions that are invoked for specific Die tags,
 * the visitor should accumulate or process the Die information as
 * desired during the traversal.
 */

DWARFVisitorBase::DWARFVisitorBase(llvm::DWARFContext &DICtx) : DICtx(DICtx) {}

llvm::Error DWARFVisitorBase::beginUnit(llvm::DWARFDie &CUDie) {
  // Check for validity
  if (!CUDie) {
    return llvm::createStringError(object::object_error::parse_failed,
                                   "Invalid Compile Unit DIE");
  }

  auto AtName = CUDie.find(dwarf::DW_AT_name);
  CurrentUnitName = "<unknown CU>";
  if (AtName) {
    llvm::Expected NameOrErr = AtName->getAsCString();
    if (NameOrErr)
      CurrentUnitName = *NameOrErr;
  }
  return llvm::Error::success();
}

} // namespace cheri_benchplot
