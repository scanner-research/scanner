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
#include "stdlib/stdlib.pb.h"

#include "scanner/util/tinyformat.h"
#include "scanner/util/serialize.h"
#include "scanner/util/json.hpp"

#include <glog/logging.h>
#include <vector>
#include <pqxx/pqxx>

using nlohmann::json;

namespace scanner {

// NOTE: SQLSource/Enumerator currently only supports Postgres.

class SQLEnumerator : public Enumerator {
 public:
  SQLEnumerator(const EnumeratorConfig& config) : Enumerator(config), conn_(nullptr) {
    bool parsed = args_.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse SQLEnumeratorArgs");
      return;
    }

    // Setup connection to database
    auto sql_config = args_.config();
    LOG_IF(FATAL, sql_config.adapter() != "postgres")
        << "Requested adapter " << sql_config.adapter()
        << " does not exist. Only \"postgres\" is supported right now.";

    try {
      conn_.reset(new pqxx::connection{tfm::format(
          "hostaddr=%s port=%d dbname=%s user=%s password=%s",
          sql_config.hostaddr(), sql_config.port(), sql_config.dbname(),
          sql_config.user(), sql_config.password())});
    } catch (pqxx::pqxx_exception& e) {
      LOG(FATAL) << e.base().what();
    }
  }

  i64 total_elements() override {
    try {
    pqxx::work txn{*conn_};
    // Count the number the number of groups
    auto query = args_.query();
    pqxx::row r = txn.exec1(tfm::format(
        "SELECT COUNT(DISTINCT(%s)) FROM %s %s WHERE %s",
        query.group(), query.table(), query.joins(), args_.filter()));
    return r[0].as<i32>();
    } catch (pqxx::pqxx_exception& e) {
      LOG(FATAL) << e.base().what();
    }
  }

  ElementArgs element_args_at(i64 element_idx) override {
    proto::SQLElementArgs args;
    args.set_filter(args_.filter());
    size_t size = args.ByteSizeLong();

    ElementArgs element_args;
    element_args.args.resize(size);
    args.SerializeToArray(element_args.args.data(), size);
    element_args.row_id = element_idx;

    return element_args;
  }

 private:
  Result valid_;
  scanner::proto::SQLEnumeratorArgs args_;
  std::unique_ptr<pqxx::connection> conn_;
};

class SQLSource : public Source {
 public:
  SQLSource(const SourceConfig& config) :
      Source(config), conn_(nullptr) {
    bool parsed = args_.ParseFromArray(config.args.data(), config.args.size());
    if (!parsed) {
      RESULT_ERROR(&valid_, "Could not parse SQLSourceArgs");
      return;
    }

    // Setup connection to database
    auto sql_config = args_.config();
    conn_.reset(new pqxx::connection{
      tfm::format("hostaddr=%s port=%d dbname=%s user=%s password=%s",
                  sql_config.hostaddr(), sql_config.port(), sql_config.dbname(), sql_config.user(), sql_config.password())
    });

    pqxx::work txn{*conn_};
    pqxx::result types = txn.exec("SELECT oid, typname FROM pg_type");
    for (auto row : types) {
      pq_types_[row["oid"].as<pqxx::oid>()] = row["typname"].as<std::string>();
    }
  }

  json row_to_json(pqxx::row row) {
    json jrow;

    for (pqxx::field field : row) {
      std::string name(field.name());
      if (name == "_scanner_id" || name == "_scanner_number") {
        continue;
      }

      pqxx::oid type = field.type();
      std::string& typname = pq_types_[type];
      bool assigned = false;

      #define ASSIGN_IF(NAME, TYPE) \
        if (typname == NAME) { jrow[name] = field.as<TYPE>(); assigned = true; }

      ASSIGN_IF("int4", i32);
      ASSIGN_IF("float8", f32);

      LOG_IF(FATAL, !assigned)
          << "Requested row has field " << name
          << " with a type we can't serialize yet: " << typname;
    }

    return std::move(jrow);
  }

  void read(const std::vector<ElementArgs>& element_args,
            std::vector<Elements>& output_columns) override {
    LOG_IF(FATAL, element_args.size() == 0) << "Asked to read zero elements";

    // Deserialize all ElementArgs
    std::string filter;
    std::vector<i64> row_ids;
    for (size_t i = 0; i < element_args.size(); ++i) {
      proto::SQLElementArgs a;
      bool parsed = a.ParseFromArray(element_args[i].args.data(),
                                     element_args[i].args.size());
      LOG_IF(FATAL, !parsed) << "Could not parse element args in SQL";

      row_ids.push_back(element_args[i].row_id);
      filter = a.filter();
    }

    auto query = args_.query();

    // If we received elements from a new job, then flush our cached results and run a new query
    if (last_filter_ != filter) {
      last_filter_ = filter;

      // Execute SELECT to fetch all the rows
      pqxx::work txn{*conn_};
      std::string query_str = tfm::format(
          "SELECT %s, %s AS _scanner_id, %s AS _scanner_number FROM %s %s WHERE %s ORDER BY _scanner_number, _scanner_id",
          query.fields(), query.id(), query.group(), query.table(), query.joins(), filter);

      pqxx::result result = txn.exec(query_str);
      LOG_IF(FATAL, result.size() == 0) << "Query returned zero rows. Executed query was:\n" << query_str;

      // Group the rows based on the provided key
      cached_rows_.clear();
      std::vector<pqxx::row> cur_group;
      i32 last_group = result[0]["_scanner_number"].as<i32>();
      for (auto row : result) {
        i32 num = row["_scanner_number"].as<i32>();
        if (num != last_group) {
          last_group = num;
          cached_rows_.push_back(cur_group);
          cur_group = std::vector<pqxx::row>();
        }
        cur_group.push_back(row);
      }

      cached_rows_.push_back(cur_group);
    }

    size_t total_size = 0;
    std::vector<size_t> sizes;

    std::vector<std::string> rows_serialized;
    for (auto row_id : row_ids) {
      json jrows = json::array();
      auto& row_group = cached_rows_[row_id];
      for (auto row : row_group) {
        jrows.push_back(row_to_json(row));
      }
      std::string serialized = jrows.dump();
      rows_serialized.push_back(serialized);
      sizes.push_back(serialized.size());
      total_size += serialized.size();
    }

    // Pack serialized results into a single block buffer;
    u8* block_buffer = new_block_buffer(CPU_DEVICE, total_size, sizes.size());
    u8* cursor = block_buffer;
    for (i32 i = 0; i < sizes.size(); ++i) {
      memcpy_buffer(cursor, CPU_DEVICE, (u8*) rows_serialized[i].c_str(), CPU_DEVICE, sizes[i]);
      insert_element(output_columns[0], cursor, sizes[i]);
      cursor += sizes[i];
    }
  }

 private:
  Result valid_;
  std::unique_ptr<pqxx::connection> conn_;
  scanner::proto::SQLSourceArgs args_;
  std::string last_filter_;
  std::vector<std::vector<pqxx::row>> cached_rows_;
  std::map<pqxx::oid, std::string> pq_types_;
};

REGISTER_ENUMERATOR(SQL, SQLEnumerator)
    .protobuf_name("SQLEnumeratorArgs");

REGISTER_SOURCE(SQL, SQLSource)
    .output("output")
    .protobuf_name("SQLSourceArgs");

}  // namespace scanner
