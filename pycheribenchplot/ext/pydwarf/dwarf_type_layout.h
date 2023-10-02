/*
 * Visitor and data types for structure layout inspection.
 */
#pragma once

#include <set>
#include <unordered_map>
#include <vector>

#include "dwarf.h"

namespace cheri_benchplot {

enum TypeInfoFlags {
  kNone = 0,
  kIsAnonymous = 1,
  kIsStruct = 1 << 1,
  kIsUnion = 1 << 2,
  kIsPtr = 1 << 3,
  kIsFnPtr = 1 << 4,
  kIsArray = 1 << 5,
  kIsConst = 1 << 6,
  kIsDecl = 1 << 7
};

TypeInfoFlags operator|(TypeInfoFlags &L, const TypeInfoFlags &R);
TypeInfoFlags operator&(TypeInfoFlags &L, const TypeInfoFlags &R);
TypeInfoFlags &operator|=(TypeInfoFlags &L, const TypeInfoFlags &R);

struct TypeInfo;

/*
 * Note that the offset and size is normalized so that
 * the Offset/Size contain the byte-aligned portion of the value
 * and the BitOffset/BitSize holds the remainder.
 */
struct Member {
  /* Name of the member */
  std::string Name;
  /* Full name including all ancestors up to the structure root */
  std::string FullName;
  /* Line where the member is defined */
  unsigned long Line;
  /* Offset into the root structure (Byte part)*/
  unsigned long Offset;
  /* Size of the member (Byte part)*/
  unsigned long Size;
  /* Bit offset of the member relative to the last byte boundary */
  unsigned long BitOffset;
  /* Bit size of the member past the last byte boundary */
  unsigned long BitSize;
  /* Reference to the member type information */
  std::weak_ptr<TypeInfo> Type;
};

struct TypeInfo {
  // Unique identifier for the typeinfo, this is the offset in the debug
  // section.
  unsigned long Handle;
  // File where the type is defined
  std::string File;
  // Line where the type is defined
  unsigned long Line;
  // Reported aggregate size
  unsigned long Size;
  // Flags identifying the type
  TypeInfoFlags Flags;
  // For arrays, this is the number of items. 0 if it is a flexible array.
  unsigned long ArrayItems;
  // Unmodified base type name
  std::string BaseName;
  // Type name modified by CV-qualifiers pointers and references.
  std::string TypeName;
  // For composite types, member descriptions
  std::vector<Member> Layout;
  // Alias names from typedefs
  std::set<std::string> AliasNames;

  TypeInfo() = default;
};

using TypeInfoPtr = std::shared_ptr<TypeInfo>;

class StructLayoutVisitor : public DWARFVisitorBase {
public:
  using Iterator = std::unordered_map<unsigned long, TypeInfoPtr>::iterator;

  StructLayoutVisitor(llvm::DWARFContext &DICtx)
      : DWARFVisitorBase(DICtx), ArchPointerSize(getABIPointerSize(DICtx)) {}

  llvm::Expected<bool> visit_structure_type(llvm::DWARFDie &Die);
  llvm::Expected<bool> visit_class_type(llvm::DWARFDie &Die);
  llvm::Expected<bool> visit_union_type(llvm::DWARFDie &Die);
  llvm::Expected<bool> visit_typedef(llvm::DWARFDie &Die);

  /* Inspect/access the structure layouts */
  TypeInfoPtr findLayout(unsigned long Handle) const;
  Iterator begin();
  Iterator end();

private:
  llvm::Expected<bool> visitCommon(const llvm::DWARFDie &Die);
  llvm::Expected<TypeInfoPtr> getTypeInfo(const llvm::DWARFDie &Die,
                                          bool Nested);
  llvm::Error extractCompositeTypeInfo(const llvm::DWARFDie &Die,
                                       TypeInfoFlags KindFlag, TypeInfoPtr TI);
  llvm::Error extractSubroutineTypeInfo(const llvm::DWARFDie &Die,
                                        TypeInfoPtr TI);
  llvm::Error extractMemberInfo(const llvm::DWARFDie &Die, bool Nested,
                                TypeInfoPtr TI);

  const int ArchPointerSize;
  // Collect type information for composite types
  std::unordered_map<unsigned long, TypeInfoPtr> StructLayoutInfo;
  // Collect all type information by DIE offset to avoid duplication
  std::unordered_map<unsigned long, TypeInfoPtr> AllTypeInfo;
};

/*
 * Helper to visit struct layouts.
 * The callback function is invoked for every discovered layout
 * type information.
 */
void visitStructLayouts(DWARFInspector &DWI,
                        std::function<void(TypeInfoPtr)> Callback);

} /* namespace cheri_benchplot */
