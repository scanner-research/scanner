#include "scanner/engine/python_kernel.h"
#include "scanner/util/util.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace scanner {

namespace py = boost::python;
namespace np = boost::python::numpy;

std::string handle_pyerror() {
  using namespace boost::python;
  using namespace boost;

  PyObject *exc, *val, *tb;
  object formatted_list, formatted;
  PyErr_Fetch(&exc, &val, &tb);
  handle<> hexc(exc), hval(allow_null(val)), htb(allow_null(tb));
  object traceback(import("traceback"));
  if (!tb) {
    object format_exception_only(traceback.attr("format_exception_only"));
    formatted_list = format_exception_only(hexc, hval);
  } else {
    object format_exception(traceback.attr("format_exception"));
    formatted_list = format_exception(hexc, hval, htb);
  }
  formatted = str("\n").join(formatted_list);
  return extract<std::string>(formatted);
}

PythonKernel::PythonKernel(const KernelConfig& config,
                           const std::string& kernel_str,
                           const std::string& pickled_config,
                           const int preferred_batch)
  : BatchedKernel(config), config_(config), device_(config.devices[0]) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  can_batch_ = (preferred_batch > 1);
  try {
    py::object main = py::import("__main__");
    main.attr("kernel_str") = py::str(kernel_str);
    main.attr("config_str") = py::str(pickled_config);

    py::list devices;
    py::list device_ids;
    for (auto& handle : config.devices) {
      devices.append(py::object(handle.type == DeviceType::CPU ? 0 : 1));
      device_ids.append(py::object(handle.id));
    }
    py::list input_columns;
    for (auto& inc : config.input_columns) {
      input_columns.append(inc);
    }
    py::list input_column_types;
    for (auto& inc : config.input_column_types) {
      input_column_types.append(py::object(inc == ColumnType::Other ? 0 : 1));
    }
    py::list output_columns;
    for (auto& outc : config.output_columns) {
      output_columns.append(outc);
    }
    py::str args((const char*)config.args.data(), config.args.size());
    py::object node_id(config.node_id);

    main.attr("devices") = devices;
    main.attr("device_ids") = device_ids;
    main.attr("input_columns") = input_columns;
    main.attr("input_column_types") = input_column_types;
    main.attr("output_columns") = output_columns;
    main.attr("args") = args;
    main.attr("node_id") = node_id;

    py::object main_namespace = main.attr("__dict__");
    py::exec(
        "import pickle\n"
        "from scannerpy import Config, DeviceType, DeviceHandle, KernelConfig, "
        "ColumnType\n"
        "from scannerpy.protobuf_generator import ProtobufGenerator\n"
        "config = pickle.loads(config_str)\n"
        "protobufs = ProtobufGenerator(config)\n"
        "handles = [DeviceHandle(DeviceType(d), di)\n"
        "           for d, di in zip(devices, device_ids)]\n"
        "input_types = [ColumnType(c) for c in input_column_types]\n"
        "kernel_config = KernelConfig(handles, input_columns,\n"
        "                             input_column_types, output_columns,\n"
        "                             args, node_id)\n"
        "exec(kernel_str)\n"
        "kernel = KERNEL(kernel_config, protobufs)",
        main_namespace);
  } catch (py::error_already_set& e) {
    LOG(FATAL) << handle_pyerror();
  }
  PyGILState_Release(gstate);
}

PythonKernel::~PythonKernel() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  try {
    py::object main = py::import("__main__");
    py::object kernel = main.attr("kernel");
    kernel.attr("close")();
  } catch (py::error_already_set& e) {
    LOG(FATAL) << handle_pyerror();
  }
  PyGILState_Release(gstate);
}

void PythonKernel::reset() {
  PyGILState_STATE gstate = PyGILState_Ensure();
  try {
    py::object main = py::import("__main__");
    py::object kernel = main.attr("kernel");
    kernel.attr("reset")();
  } catch (py::error_already_set& e) {
    LOG(FATAL) << handle_pyerror();
  }
  PyGILState_Release(gstate);
}

void PythonKernel::batched_python_execute(const BatchedElements& input_columns,
                                          BatchedElements& output_columns) {
  i32 input_count = (i32)num_rows(input_columns[0]);
  PyGILState_STATE gstate = PyGILState_Ensure();

  try {
    py::object main = py::import("__main__");
    py::object kernel = main.attr("kernel");

    py::list batched_cols;
    for (i32 j = 0; j < input_columns.size(); ++j) {
      py::list rows;
      // HACK(wcrichto): should pass column type in config and check here
      if (config_.input_column_types[j] == proto::ColumnType::Video) {
        for (i32 i = 0; i < input_count; ++i) {
          const Frame *frame = input_columns[j][i].as_const_frame();
          np::ndarray frame_np =
              np::from_data(frame->data, np::dtype::get_builtin<uint8_t>(),
                            py::make_tuple(frame->height(), frame->width(),
                                           frame->channels()),
                            py::make_tuple(frame->width() * frame->channels(),
                                           frame->channels(), 1),
                            py::object());
          rows.append(frame_np);
        }
      } else {
        for (i32 i = 0; i < input_count; ++i) {
          rows.append(py::str((char const*)input_columns[j][i].buffer,
                              input_columns[j][i].size));
        }
      }
      batched_cols.append(rows);
    }

    py::list batched_out_cols =
        py::extract<py::list>(kernel.attr("execute")(batched_cols));
    LOG_IF(FATAL, py::len(batched_out_cols) != output_columns.size())
        << "Incorrect number of output columns. Expected "
        << output_columns.size();

    for (i32 j = 0; j < output_columns.size(); ++j) {
      // push all rows to that column
      LOG_IF(FATAL, py::len(batched_out_cols[j]) != input_count)
          << "Incorrect number of output rows. Expected "
          << input_count;
      if (config_.output_columns[j] == "frame") {
        for (i32 i = 0; i < input_count; ++i) {
          np::ndarray frame_np =
              py::extract<np::ndarray>(batched_out_cols[j][i]);
          FrameType frame_type;
          {
            np::dtype dtype = frame_np.get_dtype();
            if (dtype == np::dtype::get_builtin<uint8_t>()) {
              frame_type = FrameType::U8;
            } else if (dtype == np::dtype::get_builtin<f32>()) {
              frame_type = FrameType::F32;
            } else if (dtype == np::dtype::get_builtin<f64>()) {
              frame_type = FrameType::F64;
            } else {
              LOG(FATAL) << "Invalid numpy dtype: "
                         << py::extract<char const*>(py::str(dtype));
            }
          }
          i32 ndim = frame_np.get_nd();
          if (ndim > 3) {
            LOG(FATAL) << "Invalid number of dimensions (must be less than 4): "
                       << ndim;
          }
          std::vector<i32> shapes;
          std::vector<i32> strides;
          for (int n = 0; n < ndim; ++n) {
            shapes.push_back(frame_np.shape(n));
            strides.push_back(frame_np.strides(n));
          }
          FrameInfo frame_info(shapes, frame_type);
          Frame* frame = new_frame(CPU_DEVICE, frame_info);
          const char* frame_data = frame_np.get_data();

          if (ndim == 3) {
            assert(strides[1] % strides[2] == 0);
            for (int i = 0; i < shapes[0]; ++i) {
              u64 offset = strides[0] * i;
              memcpy(frame->data + offset, frame_data + offset,
                     shapes[2] * shapes[1] * strides[2]);
            }
          } else {
            LOG(FATAL) << "Can not support ndim != 3.";
          }
          insert_frame(output_columns[j], frame);
        }
      } else {
        std::vector<std::string> outputs;
        size_t total_size = 0;
        for (i32 i = 0; i < input_count; ++i) {
          std::string field = py::extract<std::string>(batched_out_cols[j][i]);
          outputs.push_back(field);
          total_size += field.size();
        }

        u8* output_block = new_block_buffer(CPU_DEVICE, total_size, input_count);
        for (i32 i = 0; i < input_count; ++i) {
          u8* buf = output_block;
          memcpy_buffer(buf, CPU_DEVICE, (u8*)outputs[i].data(), CPU_DEVICE, outputs[i].size());
          insert_element(output_columns[j], buf, outputs[i].size());
          output_block += outputs[i].size();
        }
      }
    }

  } catch (py::error_already_set& e) {
    LOG(FATAL) << handle_pyerror();
  }

  PyGILState_Release(gstate);
}

void PythonKernel::single_python_execute(const BatchedElements& input_columns,
                                         BatchedElements& output_columns) {
  i32 input_count = (i32)num_rows(input_columns[0]);

  PyGILState_STATE gstate = PyGILState_Ensure();

  try {
    py::object main = py::import("__main__");
    py::object kernel = main.attr("kernel");

    for (i32 i = 0; i < input_count; ++i) {
      py::list cols;
      for (i32 j = 0; j < input_columns.size(); ++j) {
        // HACK(wcrichto): should pass column type in config and check here
        if (config_.input_column_types[j] == proto::ColumnType::Video) {
          const Frame* frame = input_columns[j][i].as_const_frame();
          np::ndarray frame_np =
              np::from_data(frame->data, np::dtype::get_builtin<uint8_t>(),
                            py::make_tuple(frame->height(), frame->width(),
                                           frame->channels()),
                            py::make_tuple(frame->width() * frame->channels(),
                                           frame->channels(), 1),
                            py::object());
          cols.append(frame_np);
        } else {
          cols.append(py::str((char const*)input_columns[j][i].buffer,
                              input_columns[j][i].size));
        }
      }

      py::list out_cols = py::extract<py::list>(kernel.attr("execute")(cols));
      LOG_IF(FATAL, py::len(out_cols) != output_columns.size())
          << "Incorrect number of output columns. Expected "
          << output_columns.size();

      for (i32 j = 0; j < output_columns.size(); ++j) {
        // HACK(wcrichto): should pass column type in config and check here
        if (config_.output_columns[j] == "frame") {
          np::ndarray frame_np = py::extract<np::ndarray>(out_cols[j]);
          FrameType frame_type;
          {
            np::dtype dtype = frame_np.get_dtype();
            if (dtype == np::dtype::get_builtin<uint8_t>()) {
              frame_type = FrameType::U8;
            } else if (dtype == np::dtype::get_builtin<f32>()) {
              frame_type = FrameType::F32;
            } else if (dtype == np::dtype::get_builtin<f64>()) {
              frame_type = FrameType::F64;
            } else {
              LOG(FATAL) << "Invalid numpy dtype: "
                         << py::extract<char const*>(py::str(dtype));
            }
          }
          i32 ndim = frame_np.get_nd();
          if (ndim > 3) {
            LOG(FATAL) << "Invalid number of dimensions (must be less than 4): "
                       << ndim;
          }
          std::vector<i32> shapes;
          std::vector<i32> strides;
          for (int n = 0; n < ndim; ++n) {
            shapes.push_back(frame_np.shape(n));
            strides.push_back(frame_np.strides(n));
          }
          FrameInfo frame_info(shapes, frame_type);
          Frame* frame = new_frame(CPU_DEVICE, frame_info);
          const char* frame_data = frame_np.get_data();

          if (ndim == 3) {
            assert(strides[1] % strides[2] == 0);
            for (int i = 0; i < shapes[0]; ++i) {
              u64 offset = strides[0] * i;
              memcpy(frame->data + offset, frame_data + offset,
                     shapes[2] * shapes[1] * strides[2]);
            }
          } else {
            LOG(FATAL) << "Can not support ndim != 3.";
          }
          insert_frame(output_columns[j], frame);
        } else {
          std::string field = py::extract<std::string>(out_cols[j]);
          size_t size = field.size();
          u8* buf = new_buffer(CPU_DEVICE, size);
          memcpy_buffer(buf, CPU_DEVICE, (u8*)field.data(), CPU_DEVICE, size);
          insert_element(output_columns[j], buf, size);
        }
      }
    }
  } catch (py::error_already_set& e) {
    LOG(FATAL) << handle_pyerror();
  }

  PyGILState_Release(gstate);
}

void PythonKernel::execute(const BatchedElements& input_columns,
                           BatchedElements& output_columns) {
  if (can_batch_) {
    batched_python_execute(input_columns, output_columns);
  } else {
    single_python_execute(input_columns, output_columns);
  }

}

}
