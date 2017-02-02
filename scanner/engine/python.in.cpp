#include "scanner/util/common.h"
#include "scanner/api/commands.h"
#include "scanner/engine/evaluator_info.h"
#include "scanner/engine/evaluator_registry.h"

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

PyServerState start_master_wrapper(DatabaseParameters& params, bool block) {
  return unwrap(start_master(params, block));
}

PyServerState start_worker_wrapper(DatabaseParameters& db_params,
                                   const std::string& worker_params_s,
                                   const std::string& master_address,
                                   bool block) {
  proto::WorkerParameters worker_params;
  worker_params.ParseFromArray(worker_params_s.data(), worker_params_s.size());
  return unwrap(start_worker(db_params, worker_params, master_address, block));
}

void load_evaluator(const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  LOG_IF(FATAL, handle == NULL)
    << "dlopen of " << path << " failed: " << dlerror();
}

std::string get_include() {
  // This variable is filled in at compile time by CMake.
  return "@dirs@";
}

std::string other_flags() {
#ifdef HAVE_CUDA
  return "-DHAVE_CUDA";
#else
  return "";
#endif
}

template <typename T>
inline std::vector<T> to_std_vector(const py::object& iterable) {
  return std::vector<T>(py::stl_input_iterator<T>(iterable),
                        py::stl_input_iterator<T>());
}

template <class T>
py::list to_py_list(std::vector<T> vector) {
  typename std::vector<T>::iterator iter;
  py::list list;
  for (iter = vector.begin(); iter != vector.end(); ++iter) {
    list.append(*iter);
  }
  return list;
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

py::list get_output_columns(const std::string& evaluator_name) {
  internal::EvaluatorRegistry* registry = internal::get_evaluator_registry();
  internal::EvaluatorInfo* info = registry->get_evaluator_info(evaluator_name);
  return to_py_list(info->output_columns());
}

bool has_evaluator(const std::string& name) {
  internal::EvaluatorRegistry* registry = internal::get_evaluator_registry();
  return registry->has_evaluator(name);
}

std::string default_worker_params_wrapper() {
  proto::WorkerParameters params = default_worker_params();
  std::string output;
  bool success = params.SerializeToString(&output);
  LOG_IF(FATAL, !success) << "Failed to serialize worker params";
  return output;
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
  def("other_flags", other_flags);
  def("ingest_videos", ingest_videos_wrapper);
  def("create_database", create_database);
  def("get_output_columns", get_output_columns);
  def("has_evaluator", has_evaluator);
  def("default_worker_params", default_worker_params_wrapper);
}

}
