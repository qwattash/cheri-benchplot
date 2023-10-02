
#include <exception>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dwarf.h"
#include "dwarf_type_layout.h"

namespace py = pybind11;
using namespace cheri_benchplot;

PYBIND11_MODULE(_pydwarf, m) {
  // Expose the main interface class
  py::class_<DWARFInspector>(m, "DWARFInspector")
      .def_static("load", &DWARFInspector::loadDWARF)
      .def("visit_struct_layouts", visitStructLayouts,
           py::call_guard<py::gil_scoped_release>());

  py::enum_<TypeInfoFlags>(m, "TypeInfoFlags", py::arithmetic())
      .value("kNone", TypeInfoFlags::kNone)
      .value("kIsAnonymous", TypeInfoFlags::kIsAnonymous)
      .value("kIsStruct", TypeInfoFlags::kIsStruct)
      .value("kIsUnion", TypeInfoFlags::kIsUnion)
      .value("kIsPtr", TypeInfoFlags::kIsPtr)
      .value("kIsFnPtr", TypeInfoFlags::kIsFnPtr)
      .value("kIsArray", TypeInfoFlags::kIsArray)
      .value("kIsConst", TypeInfoFlags::kIsConst)
      .value("kIsDecl", TypeInfoFlags::kIsDecl)
      .export_values();

  py::class_<TypeInfo, std::shared_ptr<TypeInfo>>(m, "TypeInfo")
      .def_readonly("handle", &TypeInfo::Handle)
      .def_readonly("file", &TypeInfo::File)
      .def_readonly("line", &TypeInfo::Line)
      .def_readonly("size", &TypeInfo::Size)
      .def_readonly("flags", &TypeInfo::Flags)
      .def_readonly("array_items", &TypeInfo::ArrayItems)
      .def_readonly("base_name", &TypeInfo::BaseName)
      .def_readonly("type_name", &TypeInfo::TypeName)
      .def_readonly("layout", &TypeInfo::Layout)
      .def_readonly("aliases", &TypeInfo::AliasNames);

  py::class_<Member>(m, "Member")
      .def_readonly("name", &Member::Name)
      .def_readonly("line", &Member::Line)
      .def_readonly("offset", &Member::Offset)
      .def_readonly("size", &Member::Size)
      .def_readonly("bit_offset", &Member::BitOffset)
      .def_readonly("bit_size", &Member::BitSize)
      .def_property_readonly("type_info",
                             [](const Member &M) { return M.Type.lock(); });
}
