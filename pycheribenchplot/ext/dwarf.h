

#pragma once

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace cheri_benchplot {

enum TypeInfoFlags {
  kIsAnonymous = 1,
  kIsStruct = 1 << 1,
  kIsUnion = 1 << 2,
  kIsPtr = 1 << 3,
  kIsFnPtr = 1 << 4,
  kIsArray = 1 << 5,
};

TypeInfoFlags operator|(TypeInfoFlags &L, const TypeInfoFlags &R);
TypeInfoFlags operator&(TypeInfoFlags &L, const TypeInfoFlags &R);
TypeInfoFlags& operator|=(TypeInfoFlags &L, const TypeInfoFlags &R);

struct TypeInfo;

struct Member {
  std::string Name;
  unsigned long Offset;
  unsigned long Size;
  unsigned long BitOffset;
  unsigned long BitSize;
  std::weak_ptr<TypeInfo> Type;
};

struct TypeInfo {
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

class OwnedTypeInfo {
 public:
  using TypeInfoByName = std::map<std::string, std::shared_ptr<TypeInfo>>;
  using TypeInfoByOffset = std::map<unsigned long, std::shared_ptr<TypeInfo>>;

  OwnedTypeInfo() = default;
  OwnedTypeInfo(TypeInfoByName &&CompositeTI, TypeInfoByOffset &&AllTI)
      : CompositeTypeInfo(CompositeTI), AllTypeInfo(AllTI) {}

  const TypeInfoByName& getCompositeTypeInfo() const {
    return CompositeTypeInfo;
  }

  const TypeInfoByOffset& getAllTypeInfo() const {
    return AllTypeInfo;
  }

 private:
  TypeInfoByName CompositeTypeInfo;
  TypeInfoByOffset AllTypeInfo;
};

/// Top level DWARF parser
class DWARFHelper {
  class HelperImpl;

public:
  DWARFHelper(std::string Path);
  ~DWARFHelper();

  // int getABIPointerSize() const;
  // int getABICapabilitySize() const;
  OwnedTypeInfo collectTypeInfo();

private:
  std::unique_ptr<HelperImpl> Impl;
};

std::ostream &operator<<(std::ostream &OS, const Member &M);
std::ostream &operator<<(std::ostream &OS, const TypeInfo &TI);

} // namespace cheri_benchplot

extern "C" {};
