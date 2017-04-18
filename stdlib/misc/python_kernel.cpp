#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "stdlib/stdlib.pb.h"

#include <boost/python.hpp>

namespace scanner {

namespace py = boost::python;

std::string handle_pyerror()
{
  using namespace boost::python;
  using namespace boost;

  PyObject *exc,*val,*tb;
  object formatted_list, formatted;
  PyErr_Fetch(&exc,&val,&tb);
  handle<> hexc(exc),hval(allow_null(val)),htb(allow_null(tb));
  object traceback(import("traceback"));
  if (!tb) {
    object format_exception_only(traceback.attr("format_exception_only"));
    formatted_list = format_exception_only(hexc,hval);
  } else {
    object format_exception(traceback.attr("format_exception"));
    formatted_list = format_exception(hexc,hval,htb);
  }
  formatted = str("\n").join(formatted_list);
  return extract<std::string>(formatted);
}

class PythonKernel : public Kernel {
 public:
  PythonKernel(const Kernel::Config& config)
      : Kernel(config),
        device_(config.devices[0]) {
    if (!args_.ParseFromArray(config.args.data(), config.args.size())) {
      LOG(FATAL) << "Failed to parse args";
    }
  }

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override {
    i32 input_count = (i32)NUM_ROWS(input_columns[0]);

    PyGILState_STATE gstate = PyGILState_Ensure();

    try {
      py::object main = py::import("__main__");
      main.attr("code") = py::str(args_.kernel());
      py::object main_namespace = main.attr("__dict__");
      py::exec(
        "import types, marshal\n"
        "f = types.FunctionType(marshal.loads(code), globals(), \"ignore\")\n",
        main_namespace);
      for (i32 i = 0; i < input_count; ++i) {
        py::list cols;
        for (i32 j = 0; j < input_columns.size(); ++j) {
          cols.append(
            py::str(
              (char const*) input_columns[j][i].buffer,
              input_columns[j][i].size));
        }
        main.attr("columns") = cols;

        py::list out_cols = py::extract<py::list>(
          py::eval(
            "f(columns)",
            main_namespace));
        LOG_IF(FATAL, py::len(out_cols) != output_columns.size())
          << "Incorrect number of output columns. Expected "
          << output_columns.size();

        for (i32 j = 0; j < py::len(out_cols); ++j) {
          std::string field = py::extract<std::string>(out_cols[j]);
          size_t size = field.size();
          u8* buf = new_buffer(device_, size);
          memcpy_buffer(buf, device_,
                        (u8*) field.data(), CPU_DEVICE,
                        size);
          INSERT_ELEMENT(output_columns[j], buf, size);
        }
      }
    } catch (py::error_already_set& e) {
      LOG(FATAL) << handle_pyerror();
    }

    PyGILState_Release(gstate);
  }

 private:
  DeviceHandle device_;
  proto::PythonArgs args_;
};

REGISTER_OP(Python)
  .frame_input("frame")
  .output("dummy");

REGISTER_KERNEL(Python, PythonKernel).device(DeviceType::CPU).num_devices(1);

}
