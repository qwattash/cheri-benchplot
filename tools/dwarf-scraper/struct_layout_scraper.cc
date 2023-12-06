#include <format>
#include <iomanip>

#include "log.hh"
#include "struct_layout_scraper.hh"

namespace fs = std::filesystem;
namespace dwarf = llvm::dwarf;

/* Shorthand for the kind of file-line spec we want */
using FileLineInfoKind = llvm::DILineInfoSpecifier::FileLineInfoKind;

namespace cheri {

constexpr auto record_type_mask = TypeInfoFlags::kTypeIsStruct |
                                  TypeInfoFlags::kTypeIsUnion |
                                  TypeInfoFlags::kTypeIsClass;

StructTypeRow StructTypeRow::FromSql(SqlRowView view) {
  StructTypeRow row;

  view.Fetch("id", row.id);
  view.Fetch("file", row.file);
  view.Fetch("line", row.line);
  view.Fetch("name", row.name);
  view.Fetch("size", row.size);
  view.Fetch("flags", row.flags);
  return row;
}

StructMemberRow StructMemberRow::FromSql(SqlRowView view) {
  StructMemberRow row;

  view.Fetch("id", row.id);
  view.Fetch("owner", row.owner);
  view.Fetch("nested", row.nested);
  view.Fetch("name", row.name);
  view.Fetch("type_name", row.type_name);
  view.Fetch("line", row.line);
  view.Fetch("size", row.byte_size);
  view.Fetch("bit_size", row.bit_size);
  view.Fetch("offset", row.byte_offset);
  view.Fetch("bit_offset", row.bit_offset);
  view.Fetch("flags", row.flags);
  view.Fetch("array_items", row.array_items);
  return row;
}

std::ostream &operator<<(std::ostream &os, const StructMemberRow &row) {
  os << "StructMemberRow{"
     << "id=" << row.id << ", "
     << "owner=" << row.owner << ", "
     << "nested=" << (row.nested ? std::to_string(*row.nested) : "NULL") << ", "
     << "name=" << std::quoted(row.name) << ", "
     << "tname=" << std::quoted(row.type_name) << ", "
     << "line=" << row.line << ", "
     << "off=" << row.byte_offset << "/" << row.bit_offset.value_or(0) << ", "
     << "size=" << row.byte_size << "/" << row.bit_size.value_or(0) << ", "
     << "flags=0x" << std::hex << row.flags << std::dec << ", "
     << "arrcnt="
     << (row.array_items ? std::to_string(*row.array_items) : "NULL");

  return os;
}

/**
 * Initialize the storage schema.
 */
void StructLayoutScraper::InitSchema() {
  LOG(kDebug) << "Initialize StructLayout scraper database";

  // clang-format off
  /*
   * Structure, unions and classes are collected here.
   * Note that we consider two struct to be the same if:
   * 1. Have the same name
   * 2. Have the same size
   * 3. Are defined in the same file, at the same line.
   */
  sm_.SqlExec("CREATE TABLE IF NOT EXISTS struct_type ("
               "id INTEGER NOT NULL PRIMARY KEY,"
               // File where the struct is defined
               "file TEXT NOT NULL,"
               // Line where the struct is defined
               "line INTEGER NOT NULL,"
               // Name of the type.
               // If this is anonymous, a synthetic name is created.
               "name TEXT,"
               // Size of the strucutre including any padding
               "size INTEGER NOT NULL,"
               // Flags that determine whether this is a struct/union/class
               "flags INTEGER DEFAULT 0 NOT NULL,"
               // Flag that is set if the structure type layout contains
               // at least one field that is not precisely representably by
               // a sub-object capability
               "has_imprecise BOOLEAN DEFAULT 0,"
               "UNIQUE(name, file, line))");

  /*
   * Pre-compiled queries for struct_type.
   */
  insert_struct_query_ = sm_.Sql(
      "INSERT INTO struct_type (id, file, line, name, size, flags) "
      "VALUES(@id, @file, @line, @name, @size, @flags) "
      "ON CONFLICT DO NOTHING RETURNING id");

  select_struct_query_ = sm_.Sql(
      "SELECT * FROM struct_type WHERE file = @file AND line = @line "
      "AND name = @name");

  /*
   * Expresses the composition between struct types and
   * their memebrs.
   * There is a one-to-many relationship between StructTypes
   * and StructMembers.
   * If the member is an aggregate type (e.g. another struct),
   * it is associated to the corresponding sturcture in the StructTypes.
   * This forms another many-to-one relationship between the tables,
   * as for each member there is a single associated structure but a
   * structure may be associated to many members.
   */
  sm_.SqlExec("CREATE TABLE IF NOT EXISTS struct_member ("
              "id INTEGER NOT NULL PRIMARY KEY,"
              // Index of the owning structure
              "owner INTEGER NOT NULL,"
              // Optional index of the nested structure
              "nested int,"
              // Member name, anonymous members have synthetic names
              "name TEXT NOT NULL,"
              // Type name of the member, for nested structures, this is the
              // same as struct_type.name
              "type_name TEXT NOT NULL,"
              // Line in the file where the member is defined
              "line INTEGER NOT NULL,"
              // Size (bytes) of the member, this may or may not include internal
              // padding
              "size INTEGER NOT NULL,"
              // Bit remainder of the size, only valid for bitfields
              "bit_size int,"
              // Offset (bytes) of the member with respect to the owner
              "offset INTEGER NOT NULL,"
              // Bit remainder of the offset, only valid for bitfields
              "bit_offset int,"
              // Type flags
              "flags INTEGER DEFAULT 0 NOT NULL,"
              "array_items int,"
              "FOREIGN KEY (owner) REFERENCES struct_type (id),"
              "FOREIGN KEY (nested) REFERENCES struct_type (id),"
              "UNIQUE(owner, name, offset),"
              "CHECK(owner != nested))");

  /*
   * Pre-compiled queries for struct_member
   */
  insert_member_query_ = sm_.Sql(
      "INSERT INTO struct_member ("
      "  id, owner, nested, name, type_name, line, size, "
      "  bit_size, offset, bit_offset, flags, array_items"
      ") VALUES("
      "  @id, @owner, @nested, @name, @type_name, @line, @size,"
      "  @bit_size, @offset, @bit_offset, @flags, @array_items) "
      "ON CONFLICT DO NOTHING RETURNING id");

  /*
   * Create a table holding the representable bounds for each (nested) member
   * of a structure.
   */
  sm_.SqlExec("CREATE TABLE IF NOT EXISTS member_bounds ("
               // ID of the flattened layout entry
               "id INTEGER NOT NULL PRIMARY KEY,"
               // ID of the struct_type containing this member
               "owner INTEGER NOT NULL,"
               // Flattened name for the layout entry
               "name TEXT NOT NULL,"
               // ID of the corresponding member entry in struct_members
               "member INTEGER NOT NULL,"
               // Cumulative offset of this member from the start of owner
               "offset INTEGER NOT NULL,"
               // Representable sub-object base
               "base INTEGER NOT NULL,"
               // Representable top of the sub-object
               "top INTEGER NOT NULL,"
               // Mark whether the member is not precisely representable
               "is_imprecise BOOL DEFAULT 0,"
               // Require number of precision bits required to exactly represent
               // the capability
               "precision INTEGER,"
               "FOREIGN KEY (owner) REFERENCES struct_type (id),"
               "FOREIGN KEY (member) REFERENCES struct_member (id))");

  /*
   * Pre-compiled queries for member_bounds
   */
  insert_member_bounds_query_ = sm_.Sql(
      "INSERT INTO member_bounds ("
      "  owner, member, offset, name, base, top, is_imprecise, precision) "
      "VALUES(@owner, @member, @offset, @name, @base, @top, @is_imprecise,"
      "  @precision)");

  /*
   * Create table holding imprecise sub-objects for each structure
   */
  sm_.SqlExec("CREATE TABLE IF NOT EXISTS subobject_alias ("
               // Member bounds for which the sub-object capability aliases
               // a set of other members
               "subobj INTEGER NOT NULL,"
               // Member bounds entry that is accessible from the subobj
               // capability
               "alias INTEGER NOT NULL,"
               "PRIMARY KEY (subobj, alias),"
               "FOREIGN KEY (subobj) REFERENCES member_bounds (id),"
               "FOREIGN KEY (alias) REFERENCES member_bounds (id))");

  /*
   * Create view to produce combinations of member_bounds to check for
   * sub-object bounds aliasing.
   */
  sm_.SqlExec("CREATE VIEW IF NOT EXISTS alias_bounds AS "
               "WITH impl ("
               "  owner, id, alias_id, name, alias_name, base, check_base,"
               "  top, check_top) "
               "AS ("
               "SELECT "
               "  mb.owner,"
               "  mb.id,"
               "  alb.id AS alias_id,"
               "  mb.name,"
               "  alb.name AS alias_name,"
               "  mb.base,"
               "  alb.offset AS check_base,"
               "  mb.top,"
               "  (alb.offset + alm.size + IIF(alm.bit_size, 1, 0)) AS check_top "
               "FROM member_bounds alb"
               "  JOIN struct_member alm ON alb.member = alm.id"
               "  JOIN member_bounds mb ON "
               "    mb.owner = alb.owner AND mb.id != alb.id) "
               "SELECT owner, id AS subobj_id, alias_id "
               "FROM impl "
               "WHERE "
               "  MAX(check_base, base) < MIN(check_top, top) AND"
               "  NOT (name LIKE alias_name || '%') AND"
               "  NOT (alias_name LIKE name || '%')");

  /*
   * Pre-compiled queries for subobject alias discovery
   */
  find_imprecise_alias_query_ = sm_.Sql(
      "INSERT INTO subobject_alias (subobj, alias)"
      "  SELECT ab.subobj_id AS subobj, ab.alias_id AS alias"
      "  FROM alias_bounds ab"
      "  WHERE ab.owner = @owner");
  // clang-format on
}

bool StructLayoutScraper::visit_structure_type(llvm::DWARFDie &die) {
  VisitCommon(die, StructTypeFlags::kTypeIsStruct);
  return false;
}

bool StructLayoutScraper::visit_class_type(llvm::DWARFDie &die) {
  VisitCommon(die, StructTypeFlags::kTypeIsClass);
  return false;
}

bool StructLayoutScraper::visit_union_type(llvm::DWARFDie &die) {
  VisitCommon(die, StructTypeFlags::kTypeIsUnion);
  return false;
}

bool StructLayoutScraper::visit_typedef(llvm::DWARFDie &die) { return false; }

void StructLayoutScraper::BeginUnit(llvm::DWARFDie &unit_die) {
  auto at_name = unit_die.find(dwarf::DW_AT_name);
  std::string unit_name;
  if (at_name) {
    llvm::Expected name_or_err = at_name->getAsCString();
    if (name_or_err) {
      unit_name = *name_or_err;
    } else {
      LOG(kError) << "Invalid compilation unit, can't extract AT_name";
      throw std::runtime_error("Invalid compliation unit");
    }
  } else {
    LOG(kError) << "Invalid compliation unit, missing AT_name";
    throw std::runtime_error("Invalid compliation unit");
  }
  LOG(kDebug) << "Enter compilation unit " << unit_name;
}

void StructLayoutScraper::EndUnit(llvm::DWARFDie &unit_die) {
  // Drain the struct_type_map_ and push the data to the database

  // Temporary mapping between struct_type IDs and entries
  std::unordered_map<uint64_t, StructTypeEntry *> entry_by_id;

  sm_.Transaction([this, &entry_by_id](StorageManager *_) {
    entry_by_id.clear();

    // Insert structure layouts first, this allows us to fixup
    // the row ID with the real database ID.
    // The remap_id is used to fixup structure type IDs for duplicate
    // structures that already exist in the database.
    std::unordered_map<uint64_t, uint64_t> remap_id;

    for (auto &[_, entry] : struct_type_map_) {
      LOG(kDebug) << "Try insert struct " << entry.data.name;
      uint64_t local_id = entry.data.id;
      assert(local_id != 0 && "Unassigned local ID");
      bool new_entry = InsertStructLayout(entry.data);
      assert(entry.data.id != 0 && "Unassigned global ID");
      if (!new_entry) {
        // Need to remap this ID
        remap_id.insert({local_id, entry.data.id});
        entry.skip_postprocess = true;
      }
      entry_by_id.insert({entry.data.id, &entry});
    }

    // Now that we have stable struct IDs, deal with the members
    // Note that we have filtered out duplicate structs
    for (auto &[_, entry] : struct_type_map_) {
      // New entry, need to add members as well
      uint64_t owner = entry.data.id;
      assert(owner != 0 && "Unassigned owner global ID");
      for (auto &m : entry.members) {
        LOG(kDebug) << "Try insert member" << m.name;
        assert(m.id != 0 && "Unassigned member local ID");
        m.owner = owner;
        if (m.nested) {
          auto mapped = remap_id.find(m.nested.value());
          if (mapped != remap_id.end()) {
            assert(m.owner != mapped->second && "Recursive member!");
            m.nested = mapped->second;
          }
        }
        // XXX sync
        InsertStructMember(m);
        assert(m.id != 0 && "Unassigned member global ID");
      }
    }
  });

  // Now that we are done and we know exactly which structures we are
  // responsible for, generate the flattened layout
  // Compute the flattened layout data for the structures in this CU.
  for (auto p : struct_type_map_) {
    if (p.second.skip_postprocess)
      continue;
    FindSubobjectCapabilities(entry_by_id, p.second);
  }

  sm_.Transaction([this](StorageManager *_) {
    for (auto &[_, entry] : struct_type_map_) {
      if (entry.skip_postprocess)
        continue;
      for (auto mb_row : entry.flattened_layout) {
        InsertMemberBounds(mb_row);
      }
      // Determine the alias groups for the member capabilities
      auto find_imprecise = find_imprecise_alias_query_->TakeCursor();
      find_imprecise.Bind(entry.data.id);
      find_imprecise.Run();
    }
  });

  struct_type_map_.clear();
}

uint64_t StructLayoutScraper::GetStructTypeId() {
  static std::atomic<uint64_t> struct_id(1);

  return ++struct_id;
}

uint64_t StructLayoutScraper::GetStructMemberId() {
  static std::atomic<uint64_t> struct_id(1);

  return ++struct_id;
}

std::optional<int64_t>
StructLayoutScraper::VisitCommon(const llvm::DWARFDie &die,
                                 StructTypeFlags kind) {
  /* Skip declarations, we don't care. */
  if (die.find(dwarf::DW_AT_declaration)) {
    return std::nullopt;
  }
  // Fail if we find a specification, we need to handle this case with
  // findRecursively()
  if (die.find(dwarf::DW_AT_specification)) {
    LOG(kError) << "DW_AT_specification unsupported";
    throw std::runtime_error("Unsupported");
  }

  StructTypeRow row;

  row.flags |= kind;

  /*
   * Need to extract the following in order to determine whether this is a
   * duplicate: (Name, File, Line, Size)
   */
  auto opt_size = GetULongAttr(die, dwarf::DW_AT_byte_size);
  if (!opt_size) {
    LOG(kWarn) << "Missing struct size for DIE @ 0x" << std::hex
               << die.getOffset();
    return std::nullopt;
  }

  row.size = *opt_size;
  row.file = die.getDeclFile(FileLineInfoKind::AbsoluteFilePath);
  row.line = die.getDeclLine();
  if (strip_prefix_) {
    row.file = fs::relative(row.file, *strip_prefix_);
  }

  auto name = GetStrAttr(die, dwarf::DW_AT_name);
  if (name) {
    row.name = *name;
  } else {
    row.name = AnonymousName(die, strip_prefix_);
    row.flags |= StructTypeFlags::kTypeIsAnonymous;
  }

  auto key = std::make_tuple(row.name, row.file, row.line);
  auto entry = struct_type_map_.find(key);
  if (entry == struct_type_map_.end()) {
    // Assign the global ID to the row, this is needed in VisitMember().
    row.id = GetStructTypeId();
    // Not a duplicate, we must collect the members
    int member_index = 0;
    StructTypeEntry e;
    e.data = row;
    for (auto &child : die.children()) {
      if (child.getTag() == dwarf::DW_TAG_member) {
        e.members.emplace_back(VisitMember(child, row, member_index++));
      }
    }

    struct_type_map_.insert({key, e});
  } else {
    return entry->second.data.id;
  }

  return row.id;
}

StructMemberRow StructLayoutScraper::VisitMember(const llvm::DWARFDie &die,
                                                 const StructTypeRow &row,
                                                 int member_index) {
  StructMemberRow member;
  member.line = die.getDeclLine();
  member.owner = row.id;
  if (member.owner == 0) {
    LOG(kError) << "Can not visit member of " << std::quoted(row.name)
                << " with invalid owner ID";
    throw std::runtime_error("Invalid member owner ID");
  }
  member.id = GetStructMemberId();

  auto member_type_die = die.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
                             .resolveTypeUnitReference();
  /*
   * This is expected to set the following fields:
   * - type_name
   * - array_items
   * - flags
   * - byte_size
   * It will return the ID of the nested structure type, if this is
   * a nested union/struct/class.
   */
  VisitMemberType(member_type_die, member);

  /* Extract offsets, taking into account bitfields */
  member.byte_size =
      dwarf::toUnsigned(die.find(dwarf::DW_AT_byte_size), member.byte_size);
  member.bit_size = GetULongAttr(die, dwarf::DW_AT_bit_size);

  auto data_offset =
      GetULongAttr(die, dwarf::DW_AT_data_member_location).value_or(0);
  auto bit_data_offset = GetULongAttr(die, dwarf::DW_AT_data_bit_offset);
  std::optional<unsigned long> bit_offset =
      (bit_data_offset) ? std::make_optional(data_offset * 8 + *bit_data_offset)
                        : std::nullopt;

  auto old_style_bit_offset = GetULongAttr(die, dwarf::DW_AT_bit_offset);
  if (old_style_bit_offset) {
    if (dwsrc_->GetContext().isLittleEndian()) {
      auto shift = *old_style_bit_offset + member.bit_size.value_or(0);
      bit_offset = bit_offset.value_or(0) + member.byte_size * 8 - shift;
    } else {
      bit_offset = bit_offset.value_or(0) + *old_style_bit_offset;
    }
  }
  member.byte_offset = (bit_offset) ? *bit_offset / 8 : data_offset;
  member.bit_offset =
      (bit_offset) ? std::make_optional(*bit_offset % 8) : std::nullopt;

  std::string name;
  if (!!(row.flags & StructTypeFlags::kTypeIsUnion)) {
    name = std::format("<anon>@{:d}", member_index);
  } else {
    name = std::format("<anon>@{:d}", member.byte_offset);
    if (member.bit_offset) {
      name += std::format(":{:d}", *member.bit_offset);
    }
  }
  member.name = GetStrAttr(die, dwarf::DW_AT_name).value_or(name);

  return member;
}

std::optional<uint64_t>
StructLayoutScraper::VisitMemberType(const llvm::DWARFDie &die,
                                     StructMemberRow &member) {
  /* Returned ID for the nested type, if any */
  std::optional<uint64_t> nested_type_id = std::nullopt;

  TypeInfo member_type = GetTypeInfo(die);

  member.type_name = member_type.type_name;
  member.byte_size = member_type.byte_size;
  member.flags = member_type.flags;
  member.array_items = member_type.array_items;

  /*
   * In this case, we want to reference the nested aggregate type,
   * if this does not exist yet, we must visit it to create an entry
   * in the database.
   */
  if (!!(member.flags & record_type_mask)) {
    StructTypeFlags flags = StructTypeFlags::kTypeNone;
    if (!!(member.flags & TypeInfoFlags::kTypeIsStruct))
      flags |= StructTypeFlags::kTypeIsStruct;
    else if (!!(member.flags & TypeInfoFlags::kTypeIsUnion))
      flags |= StructTypeFlags::kTypeIsUnion;
    else if (!!(member.flags & TypeInfoFlags::kTypeIsClass))
      flags |= StructTypeFlags::kTypeIsClass;

    member.nested = VisitCommon(member_type.type_die, flags);
    assert(member.nested && *member.nested != 0 &&
           "Structure type ID must be set");
    nested_type_id = member.nested;
  }
  return nested_type_id;
}

void StructLayoutScraper::InsertMemberBounds(const MemberBoundsRow &row) {
  auto cursor = insert_member_bounds_query_->TakeCursor();
  cursor.Bind(row.owner, row.member, row.offset, row.name, row.base, row.top,
              row.is_imprecise, row.required_precision);
  cursor.Run();

  LOG(kDebug) << "Record member bounds for " << row.name << std::hex
              << " base=0x" << row.base << " off=0x" << row.offset << " top=0x"
              << row.top << std::dec << " p=" << row.required_precision;
}

bool StructLayoutScraper::InsertStructLayout(StructTypeRow &row) {
  bool new_entry = false;
  auto timing = stats_.Timing("insert_type");
  auto cursor = insert_struct_query_->TakeCursor();
  cursor.Bind(row.id, row.file, row.line, row.name, row.size, row.flags);
  cursor.Run([&new_entry, &row](SqlRowView result) {
    result.Fetch("id", row.id);
    LOG(kDebug) << "Insert record type for " << row.name << " at " << row.file
                << ":" << row.line << " with ID=" << row.id;
    new_entry = true;
    return true;
  });

  if (!new_entry) {
    // Need to extract the ID from the database
    // XXX this may be done lazily maybe? as we do not always use it.
    auto cursor = select_struct_query_->TakeCursor();
    cursor.Bind(row.file, row.line, row.name);
    cursor.Run([&row](SqlRowView result) {
      result.Fetch("id", row.id);
      return true;
    });
    stats_.dup_structs++;
  }
  return new_entry;
}

void StructLayoutScraper::InsertStructMember(StructMemberRow &row) {
  bool new_entry = false;
  auto timing = stats_.Timing("insert_member");
  auto cursor = insert_member_query_->TakeCursor();
  cursor.BindAt("@id", row.id);
  cursor.BindAt("@owner", row.owner);
  cursor.BindAt("@nested", row.nested);
  cursor.BindAt("@name", row.name);
  cursor.BindAt("@type_name", row.type_name);
  cursor.BindAt("@line", row.line);
  cursor.BindAt("@size", row.byte_size);
  cursor.BindAt("@bit_size", row.bit_size);
  cursor.BindAt("@offset", row.byte_offset);
  cursor.BindAt("@bit_offset", row.bit_offset);
  cursor.BindAt("@flags", row.flags);
  cursor.BindAt("@array_items", row.array_items);

  cursor.Run([&new_entry, &row](SqlRowView result) {
    new_entry = true;
    result.Fetch("id", row.id);
    return true;
  });

  if (!new_entry) {
    std::string q =
        std::format("SELECT id FROM struct_member WHERE owner={} AND "
                    "name='{}' AND offset={}",
                    row.owner, row.name, row.byte_offset);
    sm_.SqlExec(q, [&row](SqlRowView result) {
      result.Fetch("id", row.id);
      return true;
    });
  }
}

void StructLayoutScraper::FindSubobjectCapabilities(
    std::unordered_map<uint64_t, StructTypeEntry *> &entry_by_id,
    StructTypeEntry &entry) {
  if (!entry.flattened_layout.empty()) {
    // Already scanned, skip
    return;
  }

  std::function<void(StructTypeEntry &, uint64_t, std::string)> FlattenedLayout;

  FlattenedLayout = [&, this](StructTypeEntry &curr, uint64_t offset,
                              std::string prefix) {
    for (auto m : curr.members) {
      MemberBoundsRow mb_row;
      mb_row.owner = entry.data.id;
      mb_row.member = m.id;
      mb_row.name = prefix + "::" + m.name;
      mb_row.offset = offset + m.byte_offset;
      uint64_t req_length = m.byte_size + (m.bit_size ? 1 : 0);
      auto [base, length] =
          dwsrc_->FindRepresentableRange(mb_row.offset, req_length);
      mb_row.required_precision =
          dwsrc_->FindRequiredPrecision(mb_row.offset, req_length);
      mb_row.base = base;
      mb_row.top = base + length;
      mb_row.is_imprecise = mb_row.offset != base || length != req_length;
      if (mb_row.is_imprecise) {
        entry.data.has_imprecise;
      }
      if (m.nested) {
        assert(*m.nested != 0 && "Missing member nested ID");
        auto it = entry_by_id.find(m.nested.value());
        assert(it != entry_by_id.end() &&
               "Entry is not in the compilation unit?");
        StructTypeEntry &nested = *it->second;
        if (nested.flattened_layout.empty()) {
          FlattenedLayout(nested, 0, nested.data.name);
        }
        // Merge the nested flat layout back here
        for (auto flat_nested : nested.flattened_layout) {
          entry.flattened_layout.push_back(flat_nested);
          auto flat_curr = entry.flattened_layout.back();
          flat_curr.offset += offset;
          flat_curr.name =
              prefix + flat_curr.name.substr(nested.data.name.length());
        }
      }
      entry.flattened_layout.emplace_back(std::move(mb_row));
    }
  };
  FlattenedLayout(entry, 0, entry.data.name);
}

} /* namespace cheri */
