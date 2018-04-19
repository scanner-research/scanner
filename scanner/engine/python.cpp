#include "scanner/api/kernel.h"
#include "scanner/api/database.h"
#include "scanner/engine/op_info.h"
#include "scanner/engine/op_registry.h"
#include "scanner/util/common.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace scanner {

  namespace py = pybind11;

  py::bytes default_machine_params_wrapper() {
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
    return py::bytes(output);
  }

  proto::Result start_master_wrapper(Database& db, const std::string& port,
                                     bool watchdog, bool prefetch_table_metadata,
                                     i64 no_workers_timeout) {
    py::gil_scoped_release release;
    return db.start_master(default_machine_params(), port, watchdog,
                           prefetch_table_metadata,
                           no_workers_timeout);
  }

  proto::Result start_worker_wrapper(Database& db, const std::string& params_s,
                                     const std::string& port, bool watchdog,
                                     bool prefetch_table_metadata) {
    py::gil_scoped_release release;

    proto::MachineParameters params_proto;
    params_proto.ParseFromString(params_s);
    MachineParameters params;
    params.num_cpus = params_proto.num_cpus();
    params.num_load_workers = params_proto.num_load_workers();
    params.num_save_workers = params_proto.num_save_workers();
    for (auto gpu_id : params_proto.gpu_ids()) {
      params.gpu_ids.push_back(gpu_id);
    }

    return db.start_worker(params, port, watchdog, prefetch_table_metadata);
  }

  std::vector<FailedVideo> ingest_videos_wrapper(Database& db, std::vector<std::string> table_names,
                                                 std::vector<std::string> paths,
                                                 bool inplace) {
    py::gil_scoped_release release;
    std::vector<FailedVideo> failed_videos;
    db.ingest_videos(table_names, paths, inplace, failed_videos);
    return failed_videos;
  }

  Result wait_for_server_shutdown_wrapper(Database& db) {
    py::gil_scoped_release release;
    return db.wait_for_server_shutdown();
  }

  PYBIND11_MODULE(libscanner, m) {
    m.doc() = "Scanner C library";

    py::class_<Database>(m, "Database")
      .def(py::init<storehouse::StorageConfig*, const std::string&, const std::string&>())
      .def("ingest_videos", &Database::ingest_videos);

    py::class_<FailedVideo>(m, "FailedVideo")
      .def_readonly("path", &FailedVideo::path)
      .def_readonly("message", &FailedVideo::message);

    py::class_<proto::Result>(m, "Result")
      .def("success", &proto::Result::success)
      .def("msg", &proto::Result::msg);

    py::class_<KernelConfig>(m, "KernelConfig")
      .def_readonly("devices", &KernelConfig::devices)
      .def_readonly("input_columns", &KernelConfig::input_columns)
      .def_readonly("input_column_types", &KernelConfig::input_column_types)
      .def_readonly("output_columns", &KernelConfig::output_columns)
      .def_readonly("output_column_types", &KernelConfig::output_column_types)
      .def_readonly("node_id", &KernelConfig::node_id)
      .def("args", [](const KernelConfig& config) {
          std::string s(config.args.begin(), config.args.end());
          return py::bytes(s);
        });

    py::class_<DeviceHandle>(m, "DeviceHandle")
      .def_readonly("id", &DeviceHandle::id)
      .def_readonly("type", &DeviceHandle::type);

    py::enum_<DeviceType>(m, "DeviceType")
      .value("GPU", DeviceType::GPU)
      .value("CPU", DeviceType::CPU);

    py::enum_<proto::ColumnType>(m, "ColumnType")
      .value("Other", ColumnType::Other)
      .value("Video", ColumnType::Video)
      .value("Image", ColumnType::Image);

    m.def("start_master", &start_master_wrapper);
    m.def("start_worker", &start_worker_wrapper);
    m.def("ingest_videos", &ingest_videos_wrapper);
    m.def("wait_for_server_shutdown", &wait_for_server_shutdown_wrapper);
    m.def("default_machine_params", &default_machine_params_wrapper);
  }
}
