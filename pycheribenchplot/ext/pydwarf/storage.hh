#pragma once

#include <filesystem>

namespace cheri {

class StorageManager {
 public:
  StorageManager(std::filesystem::path db_path);
  StorageManager(StorageManager &other) = delete;

 private:
  
};

} /* namespace cheri */
