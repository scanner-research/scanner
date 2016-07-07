/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "lightscan/storage/disk_database.h"
#include "lightscan/util/util.h"

#include <string>
#include <cstdio>
#include <sys/stat.h>
#include <unistd.h>

namespace lightscan {
namespace internal {

namespace {

std::string database_directory_path(
  const std::string &root,
  const std::string &database_name)
{
  return root + "/" + database_name;
}

template <typename T>
void write_line(T data, FILE *fp) {
  std::string s = std::to_string(data) + "\n";
  size_t size_written = fwrite(s.c_str(), sizeof(char), s.size(), fp);
  if (size_written != s.size()) {
    log_ls.fatal("FATAL: size written %lu does not equal %lu",
                  size_written, s.size());
    exit(EXIT_FAILURE);
  }
}

template <>
void write_line(std::string data, FILE *fp) {
  std::string s = data + "\n";
  size_t size_written = fwrite(s.c_str(), sizeof(char), s.size(), fp);
  if (size_written != s.size()) {
    log_ls.fatal("FATAL: size written %lu does not equal %lu",
                  size_written, s.size());
    exit(EXIT_FAILURE);
  }
}

std::string read_line(FILE *fp) {
  char *line = NULL;
  size_t n;
  ssize_t size_read = getline(&line, &n, fp);
  if (size_read < 0) {
    log_ls.fatal("FATAL: could not getline");
    exit(EXIT_FAILURE);
  }
  assert(size_read > 0);

  std::string out(line, line + size_read - 1);

  free(line);

  return out;
}

}

////////////////////////////////////////////////////////////////////////////////
/// DiskDatabase
DiskDatabase::DiskDatabase(DiskDatabaseConfig config)
  : root_directory_(config.root_directory),
    database_name_(config.database_name),
    database_directory_(
      database_directory_path(root_directory_, database_name_))
{
  // TODO(abpoms): ensure the directory is an absolute path. Should probably
  //               be done when the config is created
  // Create directory for holding databases if it does not exist
  if (mkdir_p(root_directory_.c_str(), S_IRWXU) != 0) {
    log_ls.fatal("FATAL: could not make root disk database directory %s",
                  root_directory_.c_str());
    exit(EXIT_FAILURE);
  }
  // Create directory for specific database if it does not exist
  if (mkdir(database_directory_.c_str(), S_IRWXU) != 0) {
    if (errno != EEXIST) {
      log_ls.fatal("FATAL: could not make directory %s for disk database %s",
                    database_directory_.c_str(), database_name_.c_str());
      exit(EXIT_FAILURE);
    }
  } else {
    // Setup directory
    setup_database_directory();
  }
  log_ls.debug("DiskDatabase: connected to disk based database %s",
                database_name_.c_str());
}

DiskDatabase::~DiskDatabase() {
}

std::string DiskDatabase::name() {
  return database_name_;
}

bool DiskDatabase::has_table(const std::string &table) {
  struct stat buffer;
  return (stat(table_directory_path(table).c_str(), &buffer) == 0);
}

TableDescriptor DiskDatabase::fetch_table_descriptor(const std::string &table) {
  TableDescriptor descriptor;

  std::string table_dir_path = table_directory_path(table);
  // Write table info to file
  std::string table_path = table_file_path(table);
  {
    FILE *table_fp = fopen(table_path.c_str(), "r");
    if (table_fp == NULL) {
      log_ls.fatal("FATAL: Can not get table descriptor for table %s "
                    "because that table does not exist.",
                    table.c_str());
      exit(EXIT_FAILURE);
    }

    std::string s = read_line(table_fp);
    descriptor.name = s;

    s = read_line(table_fp);
    descriptor.type_name = s;

    s = read_line(table_fp);
    descriptor.num_rows = std::stoi(s);

    s = read_line(table_fp);
    descriptor.total_bytes = std::stoi(s);

    fclose(table_fp);
  }
  // Write the attribute info to file
  std::string attribute_path = attribute_file_path(table);
  {
    FILE *attribute_fp = fopen(attribute_path.c_str(), "r");
    if (attribute_fp == NULL) {
      log_ls.fatal("FATAL: could not open %s for reading",
                    attribute_path.c_str());
      exit(EXIT_FAILURE);
    }

    std::string s = read_line(attribute_fp);
    size_t num_attributes = std::stoul(s);
    descriptor.attributes.reserve(num_attributes);

    for (size_t i = 0; i < num_attributes; ++i) {
      AttributeDescriptor attr;

      s = read_line(attribute_fp);
      attr.name = s;

      s = read_line(attribute_fp);
      attr.type_name = s;

      descriptor.attributes.push_back(attr);
    }

    fclose(attribute_fp);
  }

  return descriptor;
}

bool DiskDatabase::insert_table(const TableDescriptor &descriptor) {
  std::string table_name = descriptor.name;
  // Setup the table directory for storing metadata
  std::string table_dir_path = table_directory_path(table_name);
  if (mkdir(table_dir_path.c_str(), S_IRWXU) != 0) {
    if (errno == EEXIST) {
      return false;
    } else {
      log_ls.fatal("FATAL: could not create directory %s for table %s",
                    table_dir_path.c_str(), table_name.c_str());
      exit(EXIT_FAILURE);
    }
  }
  // Write table info to file
  log_ls.debug("DiskDatabase: writing to table file "
                "(name %s, numRows %d, bytes %d)",
                descriptor.name.c_str(),
                descriptor.num_rows,
                descriptor.total_bytes);
  std::string table_path = table_file_path(table_name);
  {
    FILE *table_fp = fopen(table_path.c_str(), "w");
    if (table_fp == NULL) {
      log_ls.fatal("FATAL: could not open %s for writing",
                    table_path.c_str());
      exit(EXIT_FAILURE);
    }

    write_line(descriptor.name, table_fp);
    write_line(descriptor.type_name, table_fp);
    write_line(descriptor.num_rows, table_fp);
    write_line(descriptor.total_bytes, table_fp);

    fclose(table_fp);
  }
  // Write the attribute info to file
  std::string attribute_path = attribute_file_path(table_name);
  {
    FILE *attribute_fp = fopen(attribute_path.c_str(), "w");
    if (attribute_fp == NULL) {
      log_ls.fatal("FATAL: could not open %s for writing",
                    attribute_path.c_str());
      exit(EXIT_FAILURE);
    }

    write_line(descriptor.attributes.size(), attribute_fp);
    for (const AttributeDescriptor &attr : descriptor.attributes) {
      log_ls.debug("DiskDatabase: writing to attribute file "
                    "(name %s, type %s)",
                    attr.name.c_str(),
                    attr.type_name.c_str());
      write_line(attr.name, attribute_fp);
      write_line(attr.type_name, attribute_fp);
    }

    fclose(attribute_fp);
  }

  return true;
}

bool DiskDatabase::delete_table(const std::string &table) {
  std::string table_path = table_file_path(table);
  if (remove(table_path.c_str()) != 0) {
    return false;
  }
  std::string attribute_path = attribute_file_path(table);
  if (remove(attribute_path.c_str()) != 0) {
    return false;
  }
  std::string table_dir_path = table_directory_path(table);
  if (rmdir(table_dir_path.c_str()) != 0) {
    return false;
  }

  return true;
}

std::string DiskDatabase::root_table_directory_path() {
  return database_directory_ + "/tables";
}

std::string DiskDatabase::table_directory_path(const std::string &table) {
  return root_table_directory_path() + "/" + table;
}

std::string DiskDatabase::table_file_path(const std::string &table) {
  return table_directory_path(table) + "/table.txt";
}

std::string DiskDatabase::attribute_file_path(const std::string &table) {
  return table_directory_path(table) + "/attributes.txt";
}

void DiskDatabase::setup_database_directory() {
  // Create table directory for holding table information
  std::string root_table_dir_path = root_table_directory_path();
  if (mkdir(root_table_dir_path.c_str(), S_IRWXU) != 0) {
    if (errno != EEXIST) {
      log_ls.fatal("FATAL: could not make root table directory %s for disk "
                    "database %s",
                    root_table_dir_path.c_str(), database_name_.c_str());
      exit(EXIT_FAILURE);
    }
  }
}

}
}
