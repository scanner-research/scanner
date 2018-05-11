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

#include "scanner/api/sink.h"
#include "scanner/util/json.hpp"
#include "scanner/util/tinyformat.h"
#include "stdlib/stdlib.pb.h"
#include "stdlib/misc/sql.h"

#include <glog/logging.h>
#include <pqxx/pqxx>
#include <vector>

using nlohmann::json;

namespace scanner {

class SQLSink : public Sink {
 public:
  SQLSink(const SinkConfig& config) : Sink(config) {
    bool parsed = args_.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse SQLSinkArgs");
      return;
    }
  }

  void new_stream(const std::vector<u8>& args) {
    scanner::proto::SQLSinkStreamArgs sargs;
    if (args.size() != 0) {
      bool parsed = sargs.ParseFromArray(args.data(), args.size());
      if (!parsed) {
        RESULT_ERROR(&valid_, "Could not parse SQLSinkStreamArgs");
        return;
      }
      job_name_ = sargs.job_name();
    }
  }

  void finished() override {
    std::string job_table = args_.query().job_table();
    if (job_table != "") {x
      std::unique_ptr<pqxx::connection> conn = sql_connect(args_.config());
      pqxx::work txn{*conn};
      txn.exec(tfm::format("INSERT INTO %s (name) VALUES ('%s')", job_table, job_name_));
      txn.commit();
    }
  }

  void write(const BatchedElements& input_columns) override {
    std::unique_ptr<pqxx::connection> conn = sql_connect(args_.config());
    pqxx::work txn{*conn};

    for (size_t i = 0; i < input_columns[0].size(); ++i) {
      auto& element = input_columns[0][i];
      json jrows = json::parse(std::string((char*)element.buffer, element.size));

      for (json& jrow : jrows) {
        std::vector<std::string> updates;
        i64 id = -1;
        for (json::iterator it = jrow.begin(); it != jrow.end(); ++it) {
          try {
            if (it.key() == "id") {
              id = it.value();
            } else {
              updates.push_back(tfm::format("%s = %s", it.key(), it.value()));
            }
          } catch (nlohmann::detail::invalid_iterator) {
            LOG(FATAL) << "JSON had invalid structure. Each element should be a list of dictionaries.";
          }
        }

        auto query = args_.query();
        std::string query_str;
        if (id == -1) {
          LOG(FATAL) << "TODO(wcrichto): insert case";
        } else {
          std::ostringstream stream;
          std::copy(updates.begin(), updates.end(), std::ostream_iterator<std::string>(stream, ","));
          std::string update_str = stream.str();
          update_str.erase(update_str.length()-1);
          query_str = tfm::format("UPDATE %s SET %s WHERE id = %d",
                                  query.table(), update_str, id);
        }

        txn.exec(query_str);
      }
    }

    txn.commit();
  }

 private:
  scanner::proto::SQLSinkArgs args_;
  std::string job_name_;
  Result valid_;
};

REGISTER_SINK(SQL, SQLSink)
    .input("input")
    .per_element_output()
    .protobuf_name("SQLSinkArgs")
    .stream_protobuf_name("SQLSinkStreamArgs");
}  // namespace scanner
