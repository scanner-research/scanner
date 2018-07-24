#pragma once

#include "scanner/engine/metadata.h"
#include "scanner/engine/table_meta_cache.h"
#include "scanner/engine/runtime.h"
#include "scanner/engine/dag_analysis.h"

#include <deque>


namespace scanner {
namespace internal {

enum class ElementStatus {
  UNTOUCHED,  // This element is not touched at all yet.
  ASSIGNABLE, // This element can be assigned to any actor right now.
  FINISHED,   // An actor just finished processing this element but it's not retired yet.
  RETIRED,    // This element is retired, i.e. it's no longer needed by any other element.
  WAITING     // This element is needed by another element but we can't process it yet
              // because some of its dependencies are not finished yet.
};

struct ElementData {
  Element* data;

  ElementStatus status = ElementStatus::UNTOUCHED;

  // For each element, this variable stores all other elements (job_id, stage_id, seq_id)
  // it depends on.
  std::vector<std::tuple<i64, i64, i64>> dependency_map;

  // For each element, all the other elements that depend on it
  std::vector<std::tuple<i64, i64, i64>> inverse_dependency_map;

  // For each element, the number of dependencies which are not finished
  i32 num_unfinished_dependencies;

  // reference_count is initialized to be the number of times the element will be needed;
  // if the reference_count turns 0, this element is no longer needed.
  i32 reference_count;
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
                          const std::vector<proto::Job> &jobs,
                          const std::vector<proto::Op> &ops,
                          DAGAnalysisInfo &analysis_results,
                          proto::BulkJobParameters::BoundaryCondition boundary_condition,
                          i64 table_id, i64 job_idx, i64 task_idx,
                          const std::vector<i64> &output_rows,
                          LoadWorkEntry &output_entry,
                          std::deque<TaskStream> &task_streams);

  // If the finished op is sink, write to storage (and update global_data_map);
  // if the finished op is not sink, update global_status_map so its successors
  // can be assigned.
  Result receive_finished_work();

  // Give work to an actor based on the global maps we have.
  Result give_actor_work();

private:

  i32 num_of_cores{};

  i32 num_of_workers{};

  std::map<std::tuple<i64, i64, i64>, ElementData> global_element_data;

};
}
}