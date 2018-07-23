#pragma once

#include "scanner/engine/metadata.h"
#include "scanner/engine/table_meta_cache.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/dag_analysis.h"

#include <deque>


namespace scanner {
namespace internal {

class Scheduler {
public:
  // This function should do dependency analysis and fill up global_data_map,
  // global_dependency_map and global_status_map.
  void fill_global_maps(DatabaseMetadata& meta, TableMetaCache& table_metas,
                           const std::vector<proto::Job>& jobs,
                           const std::vector<proto::Op>& ops,
                           DAGAnalysisInfo& info);

  // If the finished op is sink, write to storage (and update global_data_map);
  // if the finished op is not sink, update global_status_map so its successors
  // can be assigned.
  Result receive_finished_work();

  // Give work to an actor based on the global maps we have.
  Result give_actor_work();

private:
  // A unit is a frame on a sequence of a stage.
  enum unit_status {
    ASSIGNABLE, // This unit can be assigned to any actor right now.
    FINISHED,   // An actor just finished processing this unit but it's not retired yet.
    RETIRED,    // This unit is retired, i.e. it's no longer needed by any other unit.
    WAITING     // This unit is needed by another unit but we can't process it yet because
                // some of its dependencies are not finished yet.
  };

  // (job_id, stage_id, seq_id) -> address of data (using pointer for now,
  // but will possibly change it to S3 storage address or incorporate storehouse)
  std::map<std::tuple<i32, i32, i32>, Element*> global_data_map;

  // For each unit of (job_id, stage_id, seq_id), this variable stores
  // all other units (job_id, stage_id, seq_id) it depends on.
  std::map<std::tuple<i32, i32, i32>, std::vector<std::tuple<i32, i32, i32>>> global_dependency_map;

  std::map<std::tuple<i32, i32, i32>, enum unit_status> global_status_map;

};
}
}