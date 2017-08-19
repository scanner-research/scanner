#include "scanner/engine/python_kernel.h"

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
                           const std::string& kernel_str)
  : BatchedKernel(config), config_(config), device_(config.devices[0]) {
  if (!args_.ParseFromArray(config.args.data(), config.args.size())) {
    LOG(FATAL) << "Failed to parse args";
  }

  PyGILState_STATE gstate = PyGILState_Ensure();
  try {
    py::object main = py::import("__main__");
    main.attr("kernel") = py::str(kernel_str);
    main.attr("args") = py::str(args_.py_args());
    py::object main_namespace = main.attr("__dict__");
    // TODO(wcrichto): pass kernel config in as well (e.g. device info)
    py::exec(
        "import pickle\n"
        "exec(kernel)\n"
        "kernel = Kernel(**pickle.loads(args))",
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

      for (i32 j = 0; j < py::len(out_cols); ++j) {
        std::string field = py::extract<std::string>(out_cols[j]);
        size_t size = field.size();
        u8* buf = new_buffer(device_, size);
        memcpy_buffer(buf, device_, (u8*)field.data(), CPU_DEVICE, size);
        insert_element(output_columns[j], buf, size);
      }
    }
  } catch (py::error_already_set& e) {
    LOG(FATAL) << handle_pyerror();
  }

  PyGILState_Release(gstate);
}

}
