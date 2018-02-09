#include "scanner/api/database.h"
#include "scanner/engine/op_info.h"
#include "scanner/engine/op_registry.h"
#include "scanner/util/common.h"

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/numpy.hpp>
#include <thread>

namespace scanner {
namespace {
class GILRelease {
 public:
  inline GILRelease() {
    PyEval_InitThreads();
    m_thread_state = PyEval_SaveThread();
  }

  inline ~GILRelease() {
    PyEval_RestoreThread(m_thread_state);
    m_thread_state = NULL;
  }

 private:
  PyThreadState* m_thread_state;
};
}

namespace py = boost::python;

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

std::string default_machine_params_wrapper() {
  MachineParameters params = default_machine_params();
  proto::MachineParameters params_proto;
  params_proto.set_num_cpus(params.num_cpus);
  params_proto.set_num_load_workers(params.num_load_workers);
  params_proto.set_num_save_workers(params.num_save_workers);
  for (auto gpu_id : params.gpu_ids) {
    params_proto.add_gpu_ids(gpu_id);
  }

  std::string output;
  bool success = params_proto.SerializeToString(&output);
  LOG_IF(FATAL, !success) << "Failed to serialize machine params";
  return output;
}

proto::Result start_master_wrapper(Database& db, const std::string& port,
                                   bool watchdog, bool prefetch_table_metadata,
                                   i64 no_workers_timeout) {
  GILRelease r;
  return db.start_master(default_machine_params(), port, watchdog,
                         prefetch_table_metadata,
                         no_workers_timeout);
}

proto::Result start_worker_wrapper(Database& db, const std::string& params_s,
                                   const std::string& port, bool watchdog,
                                   bool prefetch_table_metadata, bool stream_mode) {
  GILRelease r;
  proto::MachineParameters params_proto;
  params_proto.ParseFromString(params_s);
  MachineParameters params;
  params.num_cpus = params_proto.num_cpus();
  params.num_load_workers = params_proto.num_load_workers();
  params.num_save_workers = params_proto.num_save_workers();
  for (auto gpu_id : params_proto.gpu_ids()) {
    params.gpu_ids.push_back(gpu_id);
  }

  return db.start_worker(params, port, watchdog, prefetch_table_metadata, stream_mode);
}

py::list ingest_videos_wrapper(Database& db, const py::list table_names,
                               const py::list paths,
                               bool inplace) {
  std::vector<FailedVideo> failed_videos;
  {
    GILRelease r;
    db.ingest_videos(to_std_vector<std::string>(table_names),
                     to_std_vector<std::string>(paths), inplace, failed_videos);
  }
  return to_py_list<FailedVideo>(failed_videos);
}

Result wait_for_server_shutdown_wrapper(Database& db) {
  GILRelease r;
  return db.wait_for_server_shutdown();
}

boost::shared_ptr<Database> initWrapper(storehouse::StorageConfig* sc,
                                        const std::string& db_path,
                                        const std::string& master_addr) {
  GILRelease r;
  return boost::shared_ptr<Database>( new Database(sc, db_path, master_addr) );
}

BOOST_PYTHON_MODULE(libscanner) {
  boost::python::numpy::initialize();
  using namespace py;
  class_<Database, boost::noncopyable>("Database", no_init)
      .def("__init__", make_constructor(&initWrapper))
      .def("ingest_videos", &Database::ingest_videos);
  class_<FailedVideo>("FailedVideo", no_init)
      .def_readonly("path", &FailedVideo::path)
      .def_readonly("message", &FailedVideo::message);
  class_<proto::Result>("Result", no_init)
      .def("success", &proto::Result::success,
           return_value_policy<return_by_value>())
      .def("msg", &proto::Result::msg, return_value_policy<return_by_value>());
  def("start_master", start_master_wrapper);
  def("start_worker", start_worker_wrapper);
  def("ingest_videos", ingest_videos_wrapper);
  def("wait_for_server_shutdown", wait_for_server_shutdown_wrapper);
  def("default_machine_params", default_machine_params_wrapper);
}
}
