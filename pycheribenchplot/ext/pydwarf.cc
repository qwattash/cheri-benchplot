
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dwarf.h"

namespace py = pybind11;
using namespace cheri_benchplot;

PYBIND11_MODULE(pydwarf, m)
{
  // Expose the main interface class
  py::class_<DWARFHelper>(m, "DWARFHelper")
      .def(py::init<std::string>())
      .def("collect_type_info", &DWARFHelper::collectTypeInfo);

  // Container that exposes collected type information for a dwarf object file.
  py::class_<TypeInfoContainer>(m, "TypeInfoContainer")
      .def("find_composite", &TypeInfoContainer::findComposite)
      .def("iter_composite", [](const TypeInfoContainer &Ref) {
        return py::make_iterator(Ref.beginComposite(), Ref.endComposite());
      }, py::keep_alive<0, 1>());

  py::class_<TypeInfo>(m, "TypeInfo")
      .def_readonly("file", &TypeInfo::File)
      .def_readonly("line", &TypeInfo::Line)
      .def_readonly("size", &TypeInfo::Size)
      .def_readonly("array_items", &TypeInfo::ArrayItems)
      .def_readonly("base_name", &TypeInfo::BaseName)
      .def_readonly("type_name", &TypeInfo::TypeName)
      .def_readonly("layout", &TypeInfo::Layout)
      .def_readonly("aliases", &TypeInfo::AliasNames);

  py::class_<Member>(m, "Member")
      .def_readonly("name", &Member::Name)
      .def_readonly("offset", &Member::Offset)
      .def_readonly("size", &Member::Size)
      .def_readonly("bit_offset", &Member::BitOffset)
      .def_readonly("bit_size", &Member::BitSize)
      .def_readonly("type_info", &Member::Type);
}
