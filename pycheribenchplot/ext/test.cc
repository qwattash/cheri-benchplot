
#include <iostream>
#include "dwarf.h"

using namespace cheri_benchplot;

int main(int argc, char **argv) {
  std::string Path(argv[1]);
  DWARFHelper Helper(Path);

  OwnedTypeInfo Info = Helper.collectTypeInfo();
  for (auto &[ID, TI] : Info.getCompositeTypeInfo()) {
    std::cout << *TI << "\n";
  }
}
