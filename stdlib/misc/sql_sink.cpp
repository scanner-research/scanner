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
#include "stdlib/stdlib.pb.h"
#include "scanner/util/tinyformat.h"

#include <glog/logging.h>
#include <vector>
#include <pqxx/pqxx>

namespace scanner {

class SQLSink : public Sink {
 public:
  SQLSink(const SinkConfig& config) :
      Sink(config) {
    bool parsed = args_.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse SQLSinkArgs");
      return;
    }

    // Setup connection to database
    auto sql_config = args_.config();
    LOG_IF(FATAL, sql_config.adapter() != "postgres")
        << "Requested adapter " << sql_config.adapter()
        << " does not exist. Only \"postgres\" is supported right now.";
    conn_.reset(new pqxx::connection{tfm::format(
        "hostaddr=%s port=%d dbname=%s user=%s password=%s",
        sql_config.hostaddr(), sql_config.port(), sql_config.dbname(),
        sql_config.user(), sql_config.password())});
  }

  void new_stream(const std::vector<u8>& args) override {
    scanner::proto::SQLSinkStreamArgs sargs;
    if (args.size() != 0) {
      bool parsed = sargs.ParseFromArray(args.data(), args.size());
      if (!parsed) {
        RESULT_ERROR(&valid_, "Could not parse SQLSinkStreamArgs");
        return;
      }
    }
  }

  void write(const BatchedElements& input_columns) override {
    pqxx::work txn{*conn_};

    for (size_t i = 0; i < input_columns[0].size(); ++i) {
      auto& element = input_columns[0][i];
      txn.exec("UPDATE table SET a = b WHERE c = d");
    }

    txn.commit();
  }

 private:
  scanner::proto::SQLSinkArgs args_;
  Result valid_;
  std::unique_ptr<pqxx::connection> conn_;
};

REGISTER_SINK(SQL, SQLSink)
    .input("input")
    .per_element_output()
    .protobuf_name("SQLSinkArgs")
    .stream_protobuf_name("SQLSinkStreamArgs");
}  // namespace scanner
