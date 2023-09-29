
#include <gtest/gtest.h>

#include "dwarf.h"
#include "dwarf_type_layout.h"

using namespace cheri_benchplot;

struct MyVisitor : public DWARFVisitorBase {
  MyVisitor(llvm::DWARFContext &DICtx) : DWARFVisitorBase(DICtx) {}

  llvm::Expected<bool> visit_structure_type(llvm::DWARFDie &Die) {
    visitCalled = true;
    return false;
  }

  bool visitCalled = false;
};

TEST(DWARFTest, Simple) {
  auto DWI = DWARFInspector::loadDWARF(
      "../../../tests/assets/riscv_purecap_test_dwarf_simple");

  auto Visitor = DWI.createVisitor<MyVisitor>();
  DWI.visit(*Visitor);

  EXPECT_TRUE(Visitor->visitCalled);
}

TEST(DWARFTest, VisitStructLayout) {
  auto DWI = DWARFInspector::loadDWARF(
      "../../../tests/assets/riscv_purecap_test_dwarf_simple");

  auto Visitor = DWI.createVisitor<StructLayoutVisitor>();
  DWI.visit(*Visitor);

  // Try to iterate through the results
  int NumEntries = 0;
  for (auto &LayoutInfo : *Visitor) {
    EXPECT_GT(LayoutInfo.second->Handle, 0);
    NumEntries++;
  }

  EXPECT_EQ(NumEntries, 3);
}
