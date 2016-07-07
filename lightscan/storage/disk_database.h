#pragma once

#include "vale/storage/database_backend.h"

#include <string>

namespace vale {
namespace internal {

//////////////////////////////////////////////////////////////////////
/// DiskDatabaseConfig
struct DiskDatabaseConfig : public DatabaseConfig {
  std::string name() override { return database_name; }

  std::string root_directory;
  std::string database_name;
};

////////////////////////////////////////////////////////////////////////////////
/// DiskDatabase
class DiskDatabase : public DatabaseBackend {
public:
  DiskDatabase(DiskDatabaseConfig config);

  ~DiskDatabase();

  std::string name() override;

  bool has_table(const std::string &table) override;

  TableDescriptor fetch_table_descriptor(const std::string &table) override;

  bool insert_table(const TableDescriptor &descriptor) override;

  bool delete_table(const std::string &name) override;

protected:
  std::string root_table_directory_path();

  std::string table_directory_path(const std::string &table);

  std::string table_file_path(const std::string &table);

  std::string attribute_file_path(const std::string &table);

  const std::string root_directory_;
  const std::string database_name_;
  const std::string database_directory_;

private:
  void setup_database_directory();

};

}
}
