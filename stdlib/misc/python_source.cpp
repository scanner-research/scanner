/* Copyright 2018 Carnegie Mellon University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scanner/api/source.h"
#include "scanner/api/enumerator.h"
#include "scanner/util/storehouse.h"
#include "stdlib/stdlib.pb.h"

#include <glog/logging.h>
#include <vector>

#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace scanner {

namespace py = pybind11;
using namespace pybind11::literals;

class PythonEnumerator : public Enumerator {
 public:
  PythonEnumerator(const EnumeratorConfig& config)
      : Enumerator(config) {
    scanner::proto::PythonEnumeratorArgs args;
    bool parsed = args.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse PythonEnumeratorArgs");
      return;
    }

    py::gil_scoped_acquire acquire;
    // Unpickle arguments and repickle per array element
    try {
      py::module main = py::module::import("__main__");
      py::object scope = main.attr("__dict__");

      py::module pickle = py::module::import("pickle");

      // Pass pickled data into var
      py::bytes pickled_data(reinterpret_cast<const char*>(args.data().data()),
                             args.data().size());

      std::vector<py::object> unpickled_data =
          pickle.attr("loads")(pickled_data).cast<std::vector<py::object>>();

      for (const auto& obj : unpickled_data) {
        // If this is a python 'bytes' object already, we don't need to pickle it
        if (py::isinstance<py::bytes>(obj)) {
          pickled_data_.push_back(obj.cast<py::bytes>());
        } else {
          pickled_data_.push_back(pickle.attr("dumps")(obj).cast<std::string>());
        }
      }
    } catch (py::error_already_set& e) {
      LOG(FATAL) << e.what();
    } catch (py::cast_error& e) {
      LOG(FATAL) << e.what();
    }
  }

  i64 total_elements() override {
    return pickled_data_.size();
  }

  ElementArgs element_args_at(i64 element_idx) override {
    proto::PythonElementArgs args;
    args.set_data(pickled_data_.at(element_idx));
    size_t size = args.ByteSizeLong();

    ElementArgs element_args;
    element_args.args.resize(size);
    args.SerializeToArray(element_args.args.data(), size);
    element_args.row_id = element_idx;

    return element_args;
  }

 private:
  Result valid_;
  std::vector<std::string> pickled_data_;
};

class PythonSource : public Source {
 public:
  PythonSource(const SourceConfig& config) :
      Source(config) {
    scanner::proto::PythonSourceArgs args;
  }

  void read(const std::vector<ElementArgs>& element_args,
            std::vector<Elements>& output_columns) override {
    // Deserialize all ElementArgs
    std::vector<std::string> elements;
    size_t total_size = 0;
    std::vector<size_t> sizes;
    for (size_t i = 0; i < element_args.size(); ++i) {
      proto::PythonElementArgs a;
      bool parsed = a.ParseFromArray(element_args[i].args.data(),
                                     element_args[i].args.size());
      assert(parsed);
      LOG_IF(FATAL, !parsed) << "Could not parse element args in FilesSource";

      elements.push_back(a.data());

      total_size += a.data().size();
      sizes.push_back(a.data().size());
    }

    // Allocate a buffer for all the data
    u8* block_buffer = new_block_buffer(CPU_DEVICE, total_size, elements.size());

    u64 offset = 0;
    for (size_t i = 0; i < element_args.size(); ++i) {
      u8* dest_buffer = block_buffer + offset;

      std::memcpy(dest_buffer, elements[i].data(), sizes[i]);

      insert_element(output_columns[0], dest_buffer, sizes[i]);

      offset += sizes[i];
    }
  }

 private:
  Result valid_;
};

REGISTER_ENUMERATOR(Python, PythonEnumerator)
    .protobuf_name("PythonEnumeratorArgs");

REGISTER_SOURCE(Python, PythonSource)
    .output("output")
    .protobuf_name("PythonSourceArgs");

}  // namespace scanner
