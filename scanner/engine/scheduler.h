#pragma once

#include "scanner/engine/metadata.h"
#include "scanner/engine/table_meta_cache.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/dag_analysis.h"
#include "scanner/engine/sampler.h"

#include <deque>


namespace scanner {
namespace internal {

enum class TaskStatus {
  READY,      // This element can be assigned to any actor right now.
  ASSIGNED,   // This element is assigned to a worker and currently being processed.
  DONE,       // This element is done. We keep it here for future dependents' use.
  WAITING     // This element is needed by another element but we can't process it yet
              // because some of its dependencies are not finished yet.
};

struct TaskData {
  Element* data;

  TaskStatus status = TaskStatus::WAITING;

  i64 slice_group;

  TaskStream taskStream;

  // For each element, this variable stores all other elements (job_id, op_id, seq_id[])
  // it depends on.
//  std::vector<std::tuple<i64, i64, std::vector<i64>>> dependency_map;

  // For each element, all the other elements that depend on it
//  std::vector<std::tuple<i64, i64, std::vector<i64>>> inverse_dependency_map;

  // reference_count is the number of dependencies which are not finished.
  // It will be initialized to be the number of dependencies the element will be needed;
  // if the reference_count turns 0, this element is no longer needed.
  i32 reference_count = -1;
};

class Scheduler {
public:

  Scheduler() {
    // fill up parameters such as number of cores and number of workers
  }

  // This infinite loop shoule be invoked in a new thread; it calls fill_global_maps()
  // to fill the map for new output_rows whenever necessary.
  void start_scheduler_loop();

  // This function should do dependency analysis and fill up global_data_map,
  // global_dependency_map and global_status_map.
  // This should perform functionality described in both perform_liveness_analysis()
  // and derive_stencil_requirements().
  // This should be done incrementally. e.g. do the first 100 output rows first.
  // We may use output_rows to specify which elements to fill each time
  Result fill_global_maps(DatabaseMetadata &meta,
                          TableMetaCache &table_meta,
                          const proto::Job &job,
                          const std::vector<proto::Op> &ops,
                          DAGAnalysisInfo &analysis_results,
                          proto::BulkJobParameters::BoundaryCondition boundary_condition,
                          i64 table_id, i64 job_idx, i64 task_idx,
                          const std::vector<i64> &output_rows,
                          LoadWorkEntry &output_entry,
                          std::deque<TaskStream> &task_streams,
                          std::map<i64, std::vector<std::unique_ptr<DomainSampler>>>& domain_samplers,
                          std::vector<std::unique_ptr<Enumerator>>& enumerators);

  // If the finished op is sink, write to storage (and update global_data_map);
  // if the finished op is not sink, update global_status_map so its successors
  // can be assigned.
  void report_done(const std::tuple<i64, i64, std::vector<i64>>& output_id);

  // Give work to an actor based on the global maps we have.
  // The op_id is simply the second value in the tuple.
  const std::tuple<i64, i64, std::vector<i64>>& give_actor_work();

private:

  i32 num_of_cores{};

  i32 num_of_workers{};

  // Job -> Op -> Output Rows
  std::map<std::tuple<i64, i64, std::vector<i64>>, TaskData> global_task_data;

};
}
}