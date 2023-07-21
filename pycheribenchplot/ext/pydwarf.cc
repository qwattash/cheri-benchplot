
#include <boost/python.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include "dwarf.h"

using namespace boost::python;
using namespace cheri_benchplot;

BOOST_PYTHON_MODULE(pydwarf)
{
  // Expose the main interface class
  class_<DWARFHelper, boost::noncopyable>("DWARFHelper", init<std::string>())
      .def("collect_typeinfo", &DWARFHelper::collectTypeInfo);

  // Helper to wrap string => type info map
  class_<OwnedTypeInfo::TypeInfoByName>("PyTypeInfoByName")
      .def(map_indexing_suite<OwnedTypeInfo::TypeInfoByName>());

  // Helper to wrap offset => type info map
  class_<OwnedTypeInfo::TypeInfoByOffset>("PyTypeInfoByOffset")
      .def(map_indexing_suite<OwnedTypeInfo::TypeInfoByOffset>());

  class_<OwnedTypeInfo>("OwnedTypeInfo")
      .def("get_composite_types", &OwnedTypeInfo::getCompositeTypeInfo,
           return_value_policy<copy_const_reference>())
      .def("get_all_types", &OwnedTypeInfo::getAllTypeInfo,
           return_value_policy<copy_const_reference>());

  class_<TypeInfo, std::shared_ptr<TypeInfo>>("TypeInfo")
      .def_readonly("file", &TypeInfo::File)
      .def_readonly("line", &TypeInfo::Line)
      .def_readonly("size", &TypeInfo::Size)
      .def_readonly("array_items", &TypeInfo::ArrayItems)
      .def_readonly("base_name", &TypeInfo::BaseName)
      .def_readonly("type_name", &TypeInfo::TypeName)
      .def_readonly("layout", &TypeInfo::Layout)
      .def_readonly("aliases", &TypeInfo::AliasNames);

  class_<Member>("Member")
      .def_readonly("name", &Member::Name)
      .def_readonly("offset", &Member::Offset)
      .def_readonly("size", &Member::Size)
      .def_readonly("bit_offset", &Member::BitOffset)
      .def_readonly("bit_size", &Member::BitSize)
      .def_readonly("type_info", &Member::Type);
}
