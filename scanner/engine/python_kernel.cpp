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
                           const std::string& pickled_config)
  : BatchedKernel(config), config_(config), device_(config.devices[0]) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  try {
    py::object main = py::import("__main__");
    main.attr("kernel_str") = py::str(kernel_str);
    main.attr("args") =
        py::str((const char*)config.args.data(), config.args.size());
    main.attr("config_str") = py::str(pickled_config);
    py::object main_namespace = main.attr("__dict__");
    // TODO(wcrichto): pass kernel config in as well (e.g. device info)
    py::exec(
        "import pickle\n"
        "from scannerpy import Config\n"
        "from scannerpy.protobuf_generator import ProtobufGenerator\n"
        "config = pickle.loads(config_str)\n"
        "protobufs = ProtobufGenerator(config)\n"
        "exec(kernel_str)\n"
        "kernel = KERNEL(args, protobufs)",
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

void PythonKernel::execute(const BatchedColumns& input_columns,
                           BatchedColumns& output_columns) {
  i32 input_count = (i32)num_rows(input_columns[0]);

  PyGILState_STATE gstate = PyGILState_Ensure();

  try {
    py::object main = py::import("__main__");
    py::object kernel = main.attr("kernel");

    for (i32 i = 0; i < input_count; ++i) {
      py::list cols;
      for (i32 j = 0; j < input_columns.size(); ++j) {
        // HACK(wcrichto): should pass column type in config and check here
        if (config_.input_columns[j] == "frame") {
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
          Frame* frame = new_frame(device_, frame_info);
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
          u8* buf = new_buffer(device_, size);
          memcpy_buffer(buf, device_, (u8*)field.data(), CPU_DEVICE, size);
          insert_element(output_columns[j], buf, size);
        }
      }
    }
  } catch (py::error_already_set& e) {
    LOG(FATAL) << handle_pyerror();
  }

  PyGILState_Release(gstate);
}

}
