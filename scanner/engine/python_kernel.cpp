#include "scanner/engine/python_kernel.h"
#include "scanner/util/tinyformat.h"
#include "scanner/util/util.h"

#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace scanner {

namespace py = pybind11;

PythonKernel::PythonKernel(const KernelConfig &config,
                           const std::string &op_name,
                           const std::string &kernel_str,
                           const std::string &pickled_config,
                           const int preferred_batch)
    : BatchedKernel(config), config_(config), device_(config.devices[0]),
      op_name_(op_name) {
  py::gil_scoped_acquire acquire;
  can_batch_ = (preferred_batch > 1);
  kernel_name_ = tfm::format("%s_kernel", op_name_);

  try {
    py::module main = py::module::import("__main__");
    py::object scope = main.attr("__dict__");
    main.attr("kernel_str") = kernel_str;
    main.attr("user_config_str") = py::bytes(pickled_config);
    main.attr("config") = config;

    std::string pycode = tfm::format(R"(
import pickle
import traceback
from scannerpy import Config, DeviceType, DeviceHandle, KernelConfig, ColumnType
from scannerpy.protobuf_generator import ProtobufGenerator

user_config = pickle.loads(user_config_str)
protobufs = ProtobufGenerator(user_config)
kernel_config = KernelConfig(config)
exec(kernel_str)
%s = KERNEL(kernel_config, protobufs))",
                                     kernel_name_);
    py::exec(pycode, scope);
  } catch (py::error_already_set &e) {
    LOG(FATAL) << e.what();
  }
}

PythonKernel::~PythonKernel() {
  py::gil_scoped_acquire acquire;
  try {
    py::module::import("__main__").attr(kernel_name_.c_str()).attr("close")();
  } catch (py::error_already_set &e) {
    LOG(FATAL) << e.what();
  }
}

void PythonKernel::reset() {
  py::gil_scoped_acquire acquire;
  try {
    py::module::import("__main__").attr(kernel_name_.c_str()).attr("reset")();
  } catch (py::error_already_set &e) {
    LOG(FATAL) << e.what();
  }
}

void PythonKernel::new_stream(const std::vector<u8> &args) {
  py::gil_scoped_acquire acquire;

  try {
    py::module main = py::module::import("__main__");
    py::object kernel = main.attr(kernel_name_.c_str());
    main.attr("args_str") =
        py::bytes(reinterpret_cast<const char *>(args.data()), args.size());
    std::string pycode = tfm::format(R"(
import pickle
if len(args_str) == 0:
  args = None
else:
  args = pickle.loads(args_str)
%s.new_stream(args))",
                                     kernel_name_);
    py::exec(pycode.c_str(), main.attr("__dict__"));
  } catch (py::error_already_set &e) {
    LOG(FATAL) << e.what();
  }
}

void PythonKernel::execute(const BatchedElements &input_columns,
                           BatchedElements &output_columns) {
  i32 input_count = (i32)num_rows(input_columns[0]);
  py::gil_scoped_acquire acquire;

  try {
    py::object kernel =
        py::module::import("__main__").attr(kernel_name_.c_str());

    std::vector<std::vector<py::object>> batched_cols;
    for (i32 j = 0; j < input_columns.size(); ++j) {
      batched_cols.emplace_back();
    }

    for (i32 j = 0; j < input_columns.size(); ++j) {
      std::vector<py::object> &col = batched_cols[j];
      if (config_.input_column_types[j] == proto::ColumnType::Video) {
        for (i32 i = 0; i < input_count; ++i) {
          const Frame *frame = input_columns[j][i].as_const_frame();
          std::string dtype;
          u32 dtype_size;
          if (frame->type == FrameType::U8) {
            dtype = py::format_descriptor<u8>::format();
            dtype_size = 1;
          } else if (frame->type == FrameType::F32) {
            dtype = py::format_descriptor<f32>::format();
            dtype_size = 4;
          } else if (frame->type == FrameType::F64) {
            dtype = py::format_descriptor<f64>::format();
            dtype_size = 8;
          }

          py::buffer_info buffer(
              frame->data, (size_t)dtype_size, dtype, 3,
              {(long int)frame->height(), (long int)frame->width(),
               (long int)frame->channels()},
              {(long int)frame->width() * frame->channels() * dtype_size,
               (long int)frame->channels() * dtype_size, (long int)dtype_size});

          if (frame->type == FrameType::U8) {
            col.push_back(py::object(py::array_t<u8>(buffer)));
          } else if (frame->type == FrameType::F32) {
            col.push_back(py::object(py::array_t<f32>(buffer)));
          } else if (frame->type == FrameType::F64) {
            col.push_back(py::object(py::array_t<f64>(buffer)));
          }
        }
      } else {
        for (i32 i = 0; i < input_count; ++i) {
          std::string s((char const *)input_columns[j][i].buffer,
                        input_columns[j][i].size);
          col.push_back(py::object(py::bytes(s)));
        }
      }
    }

    std::vector<std::vector<py::object>> batched_out_cols;

    if (can_batch_) {
      batched_out_cols = kernel.attr("execute")(batched_cols)
                             .cast<std::vector<std::vector<py::object>>>();

      LOG_IF(FATAL, batched_out_cols.size() != output_columns.size())
          << "Incorrect number of output columns. Expected "
          << output_columns.size() << ", got " << batched_out_cols.size();
    } else {
      for (i32 j = 0; j < output_columns.size(); ++j) {
        batched_out_cols.emplace_back();
      }

      for (i32 i = 0; i < input_count; ++i) {
        std::vector<py::object> in_row;
        for (i32 j = 0; j < batched_cols.size(); ++j) {
          in_row.push_back(batched_cols[j][i]);
        }
        std::vector<py::object> out_row =
            kernel.attr("execute")(in_row).cast<std::vector<py::object>>();
        for (i32 j = 0; j < out_row.size(); ++j) {
          batched_out_cols[j].push_back(out_row[j]);
        }
      }
    }

    for (i32 j = 0; j < output_columns.size(); ++j) {
      // push all rows to that column
      LOG_IF(FATAL, batched_out_cols[j].size() != input_count)
          << "Incorrect number of output rows at column " << j << ". Expected "
          << input_count << ", got " << batched_out_cols[j].size();
      if (config_.output_column_types[j] == proto::ColumnType::Video) {
        for (i32 i = 0; i < input_count; ++i) {
          py::array frame_np = batched_out_cols[j][i].cast<py::array>();
          FrameType frame_type;
          if (frame_np.dtype().is(py::dtype("uint8"))) {
            frame_type = FrameType::U8;
          } else if (frame_np.dtype().is(py::dtype("float"))) {
            frame_type = FrameType::F32;
          } else if (frame_np.dtype().is(py::dtype("double"))) {
            frame_type = FrameType::F64;
          } else {
            LOG(FATAL) << "Invalid numpy dtype";
          }

          i32 ndim = frame_np.ndim();
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
          Frame *frame = new_frame(CPU_DEVICE, frame_info);
          const char *frame_data = (const char *)frame_np.data();

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
          std::string field = batched_out_cols[j][i].cast<std::string>();
          outputs.push_back(field);
          total_size += field.size();
        }

        u8 *output_block =
            new_block_buffer(CPU_DEVICE, total_size, input_count);
        for (i32 i = 0; i < input_count; ++i) {
          u8 *buf = output_block;
          memcpy_buffer(buf, CPU_DEVICE, (u8 *)outputs[i].data(), CPU_DEVICE,
                        outputs[i].size());
          insert_element(output_columns[j], buf, outputs[i].size());
          output_block += outputs[i].size();
        }
      }
    }

  } catch (py::error_already_set &e) {
    LOG(FATAL) << e.what();
  }
}
}
