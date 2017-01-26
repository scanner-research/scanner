#include "scanner/util/common.h"
#include "scanner/api/commands.h"

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <dlfcn.h>

namespace scanner {

namespace py = boost::python;

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

struct PyServerState {
  std::shared_ptr<grpc::Server> server;
  std::shared_ptr<grpc::Service> service;
};

PyServerState unwrap(ServerState state) {
  PyServerState new_state;
  new_state.server = std::move(state.server);
  new_state.service = std::move(state.service);
  return new_state;
}

PyServerState start_master_wrapper(DatabaseParameters& params) {
  return unwrap(start_master(params, false));
}

PyServerState start_worker_wrapper(DatabaseParameters& params,
                          const std::string& master_address) {
  return unwrap(start_worker(params, master_address, false));
}

void load_evaluator(const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  LOG_IF(FATAL, handle == NULL)
    << "dlopen of " << path << " failed: " << dlerror();
}

const std::string get_include() {
  // This variable is filled in at compile time by CMake.
  return "@dirs@";
}

template <typename T>
inline std::vector<T> to_std_vector(const py::object& iterable) {
  return std::vector<T>(py::stl_input_iterator<T>(iterable),
                        py::stl_input_iterator<T>());
}

void ingest_videos_wrapper(
  storehouse::StorageConfig* storage_config,
  const std::string& db_path,
  const py::list table_names,
  const py::list paths) {
  ingest_videos(storage_config, db_path,
                to_std_vector<std::string>(table_names),
                to_std_vector<std::string>(paths));
}

BOOST_PYTHON_MODULE(scanner_bindings) {
  using namespace py;
  class_<DatabaseParameters>("DatabaseParameters", no_init);
  class_<PyServerState>("ServerState", no_init);
  def("make_database_parameters", make_database_parameters);
  def("start_master", start_master_wrapper);
  def("start_worker", start_worker_wrapper);
  def("load_evaluator", load_evaluator);
  def("get_include", get_include);
  def("ingest_videos", ingest_videos_wrapper);
  def("create_database", create_database);
}

}
