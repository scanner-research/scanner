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

#include "scanner/engine/column_enumerator.h"
#include "scanner/source_args.pb.h"

#include <glog/logging.h>
#include <vector>

namespace scanner {
namespace internal {

ColumnEnumerator::ColumnEnumerator(const EnumeratorConfig& config) :
    Enumerator(config) {
  // Deserialize ColumnSourceConfig
  scanner::proto::ColumnEnumeratorArgs args;
  bool parsed = args.ParseFromArray(config.args.data(), config.args.size());
  if (!parsed || config.args.size() == 0) {
    RESULT_ERROR(&valid_, "Could not parse ColumnEnumeratorArgs");
    return;
  }
  table_name_ = args.table_name();
  column_name_ = args.column_name();
}

// void validate(proto::Result* result) override {
// }

i64 ColumnEnumerator::total_elements() {
  assert(table_metadata_ != nullptr);
  return total_rows_;
}

ElementArgs ColumnEnumerator::element_args_at(i64 element_idx) {
  assert(table_metadata_ != nullptr);
  proto::ColumnElementArgs args;
  args.set_table_id(table_id_);
  args.set_column_id(column_id_);
  args.set_row_id(element_idx);
  size_t size = args.ByteSizeLong();

  ElementArgs element_args;
  element_args.args.resize(size);
  args.SerializeToArray(element_args.args.data(), size);
  element_args.row_id = element_idx;

  return element_args;
}

void ColumnEnumerator::set_table_meta(TableMetaCache* cache) {
  table_metadata_ = cache;
  // Verify column input table exists
  if (!table_metadata_->exists(table_name_)) {
    RESULT_ERROR(&valid_,
                 "Tried to sample from non-existent table "
                 "%s.",
                 table_name_.c_str());
    return;
  }
  const TableMetadata& table_meta = table_metadata_->at(table_name_);
  // Verify column input column exists in the requested table
  if (!table_meta.has_column(column_name_)) {
    RESULT_ERROR(&valid_,
                 "Tried to sample column %s from table %s, "
                 "but that column is not in that table.",
                 column_name_.c_str(),
                 table_name_.c_str());
    return;
  }

  total_rows_ = table_meta.num_rows();
  table_id_ = table_meta.id();
  column_id_ = table_meta.column_id(column_name_);

}

REGISTER_ENUMERATOR(Column, ColumnEnumerator);

REGISTER_ENUMERATOR(FrameColumn, ColumnEnumerator);

}
}  // namespace scanner
