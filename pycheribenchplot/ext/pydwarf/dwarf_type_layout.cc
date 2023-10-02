
#include <format>

#include <llvm/Object/Error.h>

#include "dwarf_type_layout.h"

namespace object = llvm::object;
namespace dwarf = llvm::dwarf;

using FileLineInfoKind = llvm::DILineInfoSpecifier::FileLineInfoKind;

namespace {

/*
 * Helper to extract name attribute from DIE.
 */
llvm::Optional<std::string> extractName(const llvm::DWARFDie &Die) {
  if (auto Name = dwarf::toString(Die.find(dwarf::DW_AT_name))) {
    if (!Name)
      return llvm::None;
    return std::string(*Name);
  }
  return llvm::None;
}

/*
 * Helper to extract name attribute from DIE or build an anonymous identifier
 * from the source file/line.
 */
std::string extractNameOrAnon(const llvm::DWARFDie &Die) {
  auto MaybeName = extractName(Die);
  if (MaybeName)
    return *MaybeName;
  std::string File = Die.getDeclFile(FileLineInfoKind::AbsoluteFilePath);
  unsigned long Line = Die.getDeclLine();
  return std::format("<anon>.{}.{:d}", File, Line);
}

} // namespace

namespace cheri_benchplot {
/*
 * Helper TypeInfoFlags binary operators
 */
TypeInfoFlags operator|(TypeInfoFlags &L, const TypeInfoFlags &R) {
  return static_cast<TypeInfoFlags>(static_cast<int>(L) | static_cast<int>(R));
}

TypeInfoFlags operator&(TypeInfoFlags &L, const TypeInfoFlags &R) {
  return static_cast<TypeInfoFlags>(static_cast<int>(L) & static_cast<int>(R));
}

TypeInfoFlags &operator|=(TypeInfoFlags &L, const TypeInfoFlags &R) {
  L = L | R;
  return L;
}

/*
 * StructLayoutVisitor implementation
 */
llvm::Expected<bool>
StructLayoutVisitor::visit_structure_type(llvm::DWARFDie &Die) {
  return visitCommon(Die);
}

llvm::Expected<bool>
StructLayoutVisitor::visit_union_type(llvm::DWARFDie &Die) {
  return visitCommon(Die);
}

llvm::Expected<bool>
StructLayoutVisitor::visit_class_type(llvm::DWARFDie &Die) {
  return visitCommon(Die);
}

llvm::Expected<bool> StructLayoutVisitor::visit_typedef(llvm::DWARFDie &Die) {
  auto Name = extractName(Die);
  if (!Name)
    return llvm::createStringError(
        object::object_error::parse_failed,
        std::format("typedef without a name at {:#x}", Die.getOffset()));
  auto TypeDie = Die.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
                     .resolveTypeUnitReference();
  if (!TypeDie) {
    /* This is a declaration, skip the typedef */
    return false;
  }
  auto TIOrErr = getTypeInfo(TypeDie, /*Nested=*/false);
  if (auto E = TIOrErr.takeError())
    return E;

  /*
   * If the typedef is aliasing a structure/union name,
   * we create another entry in the StructLayoutInfo map.
   * Otherwise, just ignore the typedef as it is not interesting here.
   */
  std::shared_ptr<TypeInfo> TI = *TIOrErr;
  TI->AliasNames.insert(*Name);

  return false;
}

llvm::Expected<bool>
StructLayoutVisitor::visitCommon(const llvm::DWARFDie &Die) {
  // Skip declarations, we don't care.
  if (Die.find(dwarf::DW_AT_declaration)) {
    return false;
  }

  auto TIOrErr = getTypeInfo(Die, /*Nested=*/false);
  if (auto E = TIOrErr.takeError()) {
    return E;
  }
  std::shared_ptr<TypeInfo> TI = *TIOrErr;

  // Skip duplicate definitions
  auto Item = StructLayoutInfo.find(TI->Handle);
  if (Item == StructLayoutInfo.end()) {
    StructLayoutInfo[TI->Handle] = TI;
  }

  return false;
}

/*
 * This function is the main entry point for parsing a Die subtree.
 * The Nested parameter should be true if this is called for nested structures
 * or data types within another structure or union.
 * Build type info from a Die subtree defining a type as referenced by
 * DW_AT_type
 */
llvm::Expected<TypeInfoPtr>
StructLayoutVisitor::getTypeInfo(const llvm::DWARFDie &Die, bool Nested) {
  // Try to find a cached typeinfo
  auto Pos = AllTypeInfo.find(Die.getOffset());
  if (Pos != AllTypeInfo.end()) {
    return Pos->second;
  }
  // We don't have one cached, so we have to make a new one
  auto TI = std::make_shared<TypeInfo>();
  TI->Handle = Die.getOffset();
  // Prevent re-entering for the same Die
  AllTypeInfo[TI->Handle] = TI;

  std::vector<llvm::DWARFDie> Chain;
  llvm::DWARFDie Next = Die;
  while (Next) {
    Chain.push_back(Next);
    if (Next.getTag() == dwarf::DW_TAG_structure_type ||
        Next.getTag() == dwarf::DW_TAG_class_type ||
        Next.getTag() == dwarf::DW_TAG_union_type ||
        Next.getTag() == dwarf::DW_TAG_subroutine_type ||
        Next.getTag() == dwarf::DW_TAG_enumeration_type ||
        Next.getTag() == dwarf::DW_TAG_base_type) {
      break;
    }
    Next = Next.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
               .resolveTypeUnitReference();
  }
  // Handle void if the chain does not break with one of the tags above
  if (!Next) {
    TI->BaseName = "void";
    TI->TypeName = "void";
  }

  // Evaluate the type definition
  if (Chain.size() == 0)
    return llvm::createStringError(object::object_error::parse_failed,
                                   "Could not resolve type");

  for (auto IterDie = Chain.rbegin(); IterDie != Chain.rend(); ++IterDie) {
    llvm::DWARFDie &D = *IterDie;
    auto TagStr = dwarf::TagString(IterDie->getTag());

    switch (IterDie->getTag()) {
    case dwarf::DW_TAG_base_type: {
      auto Name = extractName(*IterDie);
      if (!Name)
        return llvm::createStringError(object::object_error::parse_failed,
                                       "Base type without a name");
      auto Size = dwarf::toUnsigned(IterDie->find(dwarf::DW_AT_byte_size));
      if (!Size)
        return llvm::createStringError(object::object_error::parse_failed,
                                       "Base type without a size");
      TI->BaseName = *Name;
      TI->TypeName = *Name;
      TI->Size = *Size;
      break;
    }
    case dwarf::DW_TAG_pointer_type: {
      TI->Size = dwarf::toUnsigned(IterDie->find(dwarf::DW_AT_byte_size),
                                   ArchPointerSize);
      TI->TypeName += " *";
      // Note, reset the flags because this is now a pointer
      TI->Flags = kIsPtr;
      break;
    }
    case dwarf::DW_TAG_const_type: {
      TI->TypeName += " const";
      TI->Flags |= kIsConst;
      break;
    }
    case dwarf::DW_TAG_volatile_type: {
      TI->TypeName += " volatile";
      break;
    }
    case dwarf::DW_TAG_array_type: {
      llvm::DWARFDie SubrangeDie;
      for (auto &Child : *IterDie) {
        if (Child.getTag() == dwarf::DW_TAG_subrange_type) {
          SubrangeDie = Child;
          break;
        }
      }

      if (!SubrangeDie)
        return llvm::createStringError(object::object_error::parse_failed,
                                       "Array type without subrange");
      auto Count = SubrangeDie.find(dwarf::DW_AT_count);
      auto UpperBound = SubrangeDie.find(dwarf::DW_AT_upper_bound);
      unsigned long NItems = 0;
      if (Count) {
        auto UnsignedCount = dwarf::toUnsigned(*Count);
        if (!UnsignedCount)
          return llvm::createStringError(object::object_error::parse_failed,
                                         "Unexpected item count type");
        NItems = *UnsignedCount;
      } else if (UpperBound) {
        auto UnsignedCount = dwarf::toUnsigned(*UpperBound);
        if (!UnsignedCount)
          return llvm::createStringError(object::object_error::parse_failed,
                                         "Unexpected array upper bound type");
        NItems = *UnsignedCount + 1;
      }
      TI->TypeName += std::format(" [{:d}]", NItems);
      TI->ArrayItems = NItems;
      TI->Size = TI->Size * NItems;
      TI->Flags |= kIsArray;
      break;
    }
    case dwarf::DW_TAG_subroutine_type: {
      if (auto E = extractSubroutineTypeInfo(*IterDie, TI)) {
        return E;
      }
      break;
    }
    case dwarf::DW_TAG_typedef: {
      auto Name = extractName(*IterDie);
      if (!Name)
        return llvm::createStringError(object::object_error::parse_failed,
                                       "Typedef without a name");
      TI->TypeName = *Name;
      break;
    }
    case dwarf::DW_TAG_structure_type:
    case dwarf::DW_TAG_class_type: {
      if (auto E = extractCompositeTypeInfo(*IterDie, kIsStruct, TI)) {
        return E;
      }
      break;
    }
    case dwarf::DW_TAG_union_type: {
      if (auto E = extractCompositeTypeInfo(*IterDie, kIsUnion, TI)) {
        return E;
      }
      break;
    }
    case dwarf::DW_TAG_enumeration_type: {
      auto Name = extractNameOrAnon(*IterDie);
      TI->TypeName = "enum " + Name;
      TI->BaseName = "enum " + Name;
      if (IterDie->find(dwarf::DW_AT_declaration))
        continue;
      auto Size = dwarf::toUnsigned(IterDie->find(dwarf::DW_AT_byte_size));
      if (!Size)
        return llvm::createStringError(object::object_error::parse_failed,
                                       "Enum type without a size");
      TI->Size = *Size;
      break;
    }
    default:
      auto TagStr = dwarf::TagString(IterDie->getTag());
      return llvm::createStringError(object::object_error::parse_failed,
                                     "Type resolver did not handle %s",
                                     TagStr.str().c_str());
    }
  }

  /*
   * Note, we must not recursively enter structures and unions unless
   * this is not a pointer/reference type.
   * When this is the case, the last Die in the type chain is always the
   * definition.
   */
  if ((TI->Flags & kIsPtr) == 0 && (TI->Flags & (kIsStruct | kIsUnion))) {
    if (auto E = extractMemberInfo(Chain.back(), Nested, TI)) {
      return E;
    }
  }

  return TI;
}

llvm::Error StructLayoutVisitor::extractCompositeTypeInfo(
    const llvm::DWARFDie &Die, TypeInfoFlags KindFlag, TypeInfoPtr TI) {
  std::string Prefix;
  if (KindFlag == kIsStruct)
    Prefix = "struct ";
  else if (KindFlag == kIsUnion)
    Prefix = "union ";
  TI->Flags |= KindFlag;

  // This is a forward declaration, mark it as such.
  // Do not generate anything else because it will not be used. We could
  // resolve forward declarations after all typeinfo are collected if we needed.
  if (Die.find(dwarf::DW_AT_declaration)) {
    TI->BaseName = extractNameOrAnon(Die);
    TI->TypeName = Prefix + TI->BaseName;
    TI->Flags |= kIsDecl;
    return llvm::Error::success();
  }

  auto MaybeSize = dwarf::toUnsigned(Die.find(dwarf::DW_AT_byte_size));
  if (!MaybeSize)
    return llvm::createStringError(object::object_error::parse_failed,
                                   "Missing struct size at %lx",
                                   Die.getOffset());
  TI->Size = *MaybeSize;
  TI->File = Die.getDeclFile(FileLineInfoKind::AbsoluteFilePath);
  TI->Line = Die.getDeclLine();

  TI->BaseName = extractNameOrAnon(Die);
  TI->TypeName = Prefix + TI->BaseName;
  if (!Die.find(dwarf::DW_AT_name).hasValue()) {
    TI->Flags |= kIsAnonymous;
  }

  return llvm::Error::success();
}

llvm::Error StructLayoutVisitor::extractMemberInfo(const llvm::DWARFDie &Die,
                                                   bool Nested,
                                                   TypeInfoPtr TI) {
  // Fail if we find a specification, we need to handle this case with
  // findRecursively()
  if (Die.find(dwarf::DW_AT_specification))
    return llvm::createStringError(object::object_error::parse_failed,
                                   "DW_AT_specification unsupported");

  // Now extract the structure layout
  for (auto &Child : Die.children()) {
    if (Child.getTag() == dwarf::DW_TAG_member) {
      auto MType = Child.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
                       .resolveTypeUnitReference();
      auto MemberInfoOrErr = getTypeInfo(MType, /*Nested=*/true);
      if (auto E = MemberInfoOrErr.takeError()) {
        return E;
      }

      unsigned long ByteSize = dwarf::toUnsigned(
          Child.find(dwarf::DW_AT_byte_size), (*MemberInfoOrErr)->Size);
      unsigned long BitSize =
          dwarf::toUnsigned(Child.find(dwarf::DW_AT_bit_size), 0);
      unsigned long DataOffset =
          dwarf::toUnsigned(Child.find(dwarf::DW_AT_data_member_location), 0);
      unsigned long BitDataOffset =
          dwarf::toUnsigned(Child.find(dwarf::DW_AT_data_bit_offset), 0);
      unsigned long BitOffset = DataOffset * 8 + BitDataOffset;

      auto OldStyleBitOffset = Child.find(dwarf::DW_AT_bit_offset);
      if (OldStyleBitOffset) {
        if (DICtx.isLittleEndian()) {
          BitOffset += ByteSize * 8 -
                       (dwarf::toUnsigned(OldStyleBitOffset, 0) + BitSize);
        } else {
          BitOffset += dwarf::toUnsigned(OldStyleBitOffset, 0);
        }
      }
      unsigned long ByteOffset = BitOffset / 8;
      BitOffset = BitOffset % 8;
      std::string DefaultName = std::format("<anon>.{:d}", ByteOffset);
      if (BitOffset) {
        DefaultName += std::format(":{:d}", BitOffset);
      }

      Member MI{.Name = dwarf::toString(Child.find(dwarf::DW_AT_name),
                                        DefaultName.c_str()),
                .Line = Die.getDeclLine(),
                .Offset = ByteOffset,
                .Size = (BitSize) ? BitSize / 8 : ByteSize,
                .BitOffset = BitOffset,
                .BitSize = BitSize % 8,
                .Type = *MemberInfoOrErr};
      TI->Layout.push_back(MI);
    }
  }

  return llvm::Error::success();
}

llvm::Error
StructLayoutVisitor::extractSubroutineTypeInfo(const llvm::DWARFDie &Die,
                                               TypeInfoPtr TI) {
  std::string ReturnName;
  auto ReturnType = Die.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
                        .resolveTypeUnitReference();
  if (ReturnType) {
    auto ReturnTIOrErr = getTypeInfo(ReturnType, /*Nested=*/false);
    if (auto E = ReturnTIOrErr.takeError()) {
      return E;
    }
    ReturnName = (*ReturnTIOrErr)->TypeName;
  } else {
    ReturnName = "void";
  }

  std::vector<TypeInfoPtr> Params;
  for (auto &Child : Die.children()) {
    if (Child.getTag() != dwarf::DW_TAG_formal_parameter) {
      // XXX variadics are not supported yet, DW_TAG_unspecified_type
      continue;
    }
    auto ParamType = Child.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
                         .resolveTypeUnitReference();
    auto ParamTIOrErr = getTypeInfo(ParamType, /*Nested=*/false);
    if (auto E = ParamTIOrErr.takeError()) {
      return E;
    }
    Params.push_back(*ParamTIOrErr);
  }

  std::string ParamString;
  for (int Idx = 0; Idx < Params.size(); ++Idx) {
    ParamString += Params[Idx]->TypeName;
    if (Idx != Params.size() - 1) {
      ParamString += ", ";
    }
  }

  std::string Name = std::format("{}({})", ReturnName, ParamString);
  TI->TypeName = Name;
  TI->BaseName = Name;
  TI->Flags |= kIsFnPtr;

  return llvm::Error::success();
}

TypeInfoPtr StructLayoutVisitor::findLayout(unsigned long Handle) const {
  auto Item = StructLayoutInfo.find(Handle);
  if (Item != StructLayoutInfo.end()) {
    return Item->second;
  }
  return nullptr;
}

StructLayoutVisitor::Iterator StructLayoutVisitor::begin() {
  return StructLayoutInfo.begin();
}

StructLayoutVisitor::Iterator StructLayoutVisitor::end() {
  return StructLayoutInfo.end();
}

void visitStructLayouts(DWARFInspector &DWI,
                        std::function<void(TypeInfoPtr)> Callback) {
  // Note that this runs without holding the GIL
  auto Visitor = DWI.createVisitor<StructLayoutVisitor>();
  DWI.visit(*Visitor);

  for (auto LayoutInfo : *Visitor) {
    Callback(LayoutInfo.second);
  }
}

} /* namespace cheri_benchplot */
