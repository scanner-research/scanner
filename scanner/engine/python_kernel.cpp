#include "scanner/engine/python_kernel.h"
#include "scanner/util/tinyformat.h"
#include "scanner/util/util.h"

#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace scanner {

namespace py = pybind11;
using namespace pybind11::literals;

void gen_random(char* s, const int len) {
  static const char alphanum[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";

  for (int i = 0; i < len; ++i) {
    s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
  }

  s[len] = 0;
}

PythonKernel::PythonKernel(const KernelConfig &config,
                           const std::string &op_name,
                           const std::string &kernel_code,
                           const std::string &pickled_config,
                           const bool can_batch,
                           const bool can_stencil)
    : StenciledBatchedKernel(config), config_(config),
      device_(config.devices[0]), op_name_(op_name) {
  py::gil_scoped_acquire acquire;
  can_batch_ = can_batch;
  can_stencil_ = can_stencil;
  char rand[256];
  gen_random(rand, 256);
  kernel_name_ = tfm::format("%s_kernel_%s", op_name_, rand);
  std::string kernel_ns_name_ = tfm::format("ns_%s", op_name_);

  try {
    py::module main = py::module::import("__main__");
    py::object scope = main.attr("__dict__");
    main.attr(kernel_ns_name_.c_str()) = py::dict(
        "kernel_code"_a=py::bytes(kernel_code),
        "user_config_str"_a=py::bytes(pickled_config),
        "config"_a=config);

    std::string pycode = tfm::format(R"(
def %s_fn():
  import pickle
  import cloudpickle
  import traceback
  from scannerpy import Config, DeviceType, DeviceHandle, KernelConfig
  from scannerpy.protobuf_generator import ProtobufGenerator

  n = %s
  user_config = pickle.loads(n['user_config_str'])
  protobufs = ProtobufGenerator(user_config)
  kernel_config = KernelConfig(n['config'])
  kernel_config.protobufs = protobufs
  return cloudpickle.loads(n['kernel_code'])(kernel_config)
%s = %s_fn()
)", kernel_name_, kernel_ns_name_, kernel_name_, kernel_name_);
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

void PythonKernel::execute(const StenciledBatchedElements &input_columns,
                           BatchedElements &output_columns) {
  i32 input_count = (i32)input_columns[0].size();
  py::gil_scoped_acquire acquire;

  try {
    py::object kernel =
        py::module::import("__main__").attr(kernel_name_.c_str());

    std::vector<std::vector<std::vector<py::object>>> batched_cols;
    for (i32 j = 0; j < input_columns.size(); ++j) {
      batched_cols.emplace_back();
    }

    for (i32 j = 0; j < input_columns.size(); ++j) {
      std::vector<std::vector<py::object>> &batched_col = batched_cols[j];
      if (config_.input_column_types[j] == proto::ColumnType::Video) {
        for (i32 i = 0; i < input_count; ++i) {
          batched_col.emplace_back();
          std::vector<py::object> &col = batched_col.back();
          for (i32 k = 0; k < input_columns[j][i].size(); ++k) {
            const Frame *frame = input_columns[j][i][k].as_const_frame();
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
                 (long int)frame->channels() * dtype_size,
                 (long int)dtype_size});

            if (frame->type == FrameType::U8) {
              col.push_back(py::object(py::array_t<u8>(buffer)));
            } else if (frame->type == FrameType::F32) {
              col.push_back(py::object(py::array_t<f32>(buffer)));
            } else if (frame->type == FrameType::F64) {
              col.push_back(py::object(py::array_t<f64>(buffer)));
            }
          }
        }
      } else {
        for (i32 i = 0; i < input_count; ++i) {
          batched_col.emplace_back();
          std::vector<py::object> &col = batched_col.back();
          for (i32 k = 0; k < input_columns[j][i].size(); ++k) {
            std::string s((char const *)input_columns[j][i][k].buffer,
                          input_columns[j][i][k].size);
            col.push_back(py::object(py::bytes(s)));
          }
        }
      }
    }

    std::vector<std::vector<py::object>> batched_out_cols;

    if (can_batch_ && can_stencil_) {
      py::tuple out_cols = kernel.attr("execute")(batched_cols)
          .cast<py::tuple>();
      for (i32 j = 0; j < out_cols.size(); ++j) {
        batched_out_cols.push_back(out_cols[j].cast<std::vector<py::object>>());
      }
    } else if (can_stencil_) {
      // Remove the batch dimension since we are only stenciling
      std::vector<std::vector<py::object>> in_row(batched_cols.size());
      for (size_t i = 0; i < batched_cols.size(); ++i) {
        in_row[i] = batched_cols[i][0];
      }
      py::tuple out_row = kernel.attr("execute")(in_row)
          .cast<py::tuple>();
      for (i32 j = 0; j < output_columns.size(); ++j) {
        batched_out_cols.emplace_back();
      }
      for (i32 j = 0; j < out_row.size(); ++j) {
        batched_out_cols[j].push_back(out_row[j]);
      }

    } else if (can_batch_) {
      // Remove the stencil dimension since we are only batching
      std::vector<std::vector<py::object>> in_row(batched_cols.size());
      for (size_t i = 0; i < batched_cols.size(); ++i) {
        for (size_t j = 0; j < batched_cols[i].size(); ++j) {
          in_row[i].push_back(batched_cols[i][j][0]);
        }
      }
      py::tuple out_cols = kernel.attr("execute")(in_row)
          .cast<py::tuple>();
      for (i32 j = 0; j < out_cols.size(); ++j) {
        batched_out_cols.push_back(out_cols[j].cast<std::vector<py::object>>());
      }
    } else {
      // Remove both the stencil and batch dimension
      std::vector<py::object> in_row;
      for (i32 j = 0; j < batched_cols.size(); ++j) {
        in_row.push_back(batched_cols[j][0][0]);
      }
      py::tuple out_row =
          kernel.attr("execute")(in_row).cast<py::tuple>();

      for (i32 j = 0; j < output_columns.size(); ++j) {
        batched_out_cols.emplace_back();
      }
      for (i32 j = 0; j < out_row.size(); ++j) {
        batched_out_cols[j].push_back(out_row[j]);
      }
    }
    LOG_IF(FATAL, batched_out_cols.size() != output_columns.size())
        << "Incorrect number of output columns. Expected "
        << output_columns.size() << ", got " << batched_out_cols.size();

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
          } else if (frame_np.dtype().is(py::dtype("float32"))) {
            frame_type = FrameType::F32;
          } else if (frame_np.dtype().is(py::dtype("float64"))) {
            frame_type = FrameType::F64;
          } else {
            LOG(FATAL) << "Invalid numpy dtype: " << frame_np.dtype();
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
            if (frame_np.strides(n) < 0) {
              LOG(FATAL) << "We do not yet support negative strides in PythonKernel!";
            }
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
