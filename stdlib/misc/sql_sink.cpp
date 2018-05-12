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

std::string join(const std::vector<std::string>& v, const std::string& c) {
  std::stringstream ss;
  for(size_t i = 0; i < v.size(); ++i) {
    ss << v[i];
    if(i != v.size() - 1) { ss << c; }
  }
  return ss.str();
}

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
    bool parsed = sargs_.ParseFromArray(args.data(), args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse SQLSinkStreamArgs");
      return;
    }
  }

  void finished() override {
    std::string job_table = args_.query().job_table();
    if (job_table != "") {
      std::unique_ptr<pqxx::connection> conn = sql_connect(args_.config());
      pqxx::work txn{*conn};
      txn.exec(tfm::format("INSERT INTO %s (name) VALUES ('%s')", job_table, sargs_.job_name()));
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
        std::map<std::string, std::string> updates;
        i64 id = -1;
        for (json::iterator it = jrow.begin(); it != jrow.end(); ++it) {
          auto k = it.key();
          auto v = it.value();
          try {
            if (k == "id") {
              id = v;
            } else {
              // Have to special case strings since SQL queries expect single quotes, while
              // json to string formatter uses double quotes
              if (v.is_string()) {
                updates[k] = tfm::format("'%s'", v.get<std::string>());
              } else {
                updates[k] = tfm::format("%s", v);
              }
            }
          } catch (nlohmann::detail::invalid_iterator) {
            LOG(FATAL) << "JSON had invalid structure. Each element should be a list of dictionaries.";
          }
        }

        auto query = args_.query();
        std::string query_str;
        if (sargs_.insert()) {
          std::vector<std::string> column_list;
          std::vector<std::string> value_list;
          for (auto it = updates.begin(); it != updates.end(); it++) {
            column_list.push_back(it->first);
            value_list.push_back(it->second);
          }

          query_str =
              tfm::format("INSERT INTO %s (%s) VALUES (%s)", sargs_.table(),
                          join(column_list, ", "), join(value_list, ", "));
        } else {
          std::vector<std::string> update_list;
          for (auto it = updates.begin(); it != updates.end(); it++) {
            update_list.push_back(tfm::format("%s = %s", it->first, it->second));
          }

          LOG_IF(FATAL, id == -1) << "SQLSink updates must have an `id` field set to know which row to update.";
          query_str = tfm::format("UPDATE %s SET %s WHERE id = %d", query.table(), join(update_list, ", "), id);
        }

        txn.exec(query_str);
      }
    }

    txn.commit();
  }

 private:
  scanner::proto::SQLSinkArgs args_;
  scanner::proto::SQLSinkStreamArgs sargs_;
  Result valid_;
};

REGISTER_SINK(SQL, SQLSink)
    .input("input")
    .per_element_output()
    .protobuf_name("SQLSinkArgs")
    .stream_protobuf_name("SQLSinkStreamArgs");
}  // namespace scanner
