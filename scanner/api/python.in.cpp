#include "scanner/util/common.h"
#include "scanner/api/commands.h"

#include <boost/python.hpp>
#include <dlfcn.h>

namespace scanner {

namespace bp = boost::python;

DatabaseParameters make_database_parameters(
  storehouse::StorageConfig* storage_config,
  std::string memory_config_serialized,
  std::string db_path) {
  MemoryPoolConfig memory_config;
  memory_config.ParseFromArray(
    memory_config_serialized.data(), memory_config_serialized.size());
  DatabaseParameters params = {storage_config, memory_config, db_path};
  return params;
}

void start_master_wrapper(DatabaseParameters& params) {
  start_master(params);
}

void start_worker_wrapper(DatabaseParameters& params,
                          const std::string& master_address) {
  start_worker(params, master_address);
}

void load_evaluator(const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  LOG_IF(FATAL, handle == NULL)
    << "dlopen of " << path << " failed: " << dlerror();
}

const std::string get_include() {
  return "@dirs@";
}

BOOST_PYTHON_MODULE(scanner_bindings) {
  using namespace bp;
  class_<DatabaseParameters>("DatabaseParameters", no_init);
  def("make_database_parameters", make_database_parameters);
  def("start_master", start_master_wrapper);
  def("start_worker", start_worker_wrapper);
  def("load_evaluator", load_evaluator);
  def("get_include", get_include);
}

}
