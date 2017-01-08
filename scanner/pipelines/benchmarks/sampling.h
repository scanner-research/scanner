#pragma once

#include "scanner/eval/pipeline_description.h"

namespace scanner {

inline void benchmark_sampling(const DatasetInformation &info,
                               PipelineDescription &desc, bool is_flow) {
  const char *JOB_NAME = std::getenv("SC_JOB_NAME");
  std::string job_name{JOB_NAME};

  const JobInformation &job = info.job(job_name);
  for (const std::string &table_name : job.table_names()) {
    Task task;
    task.table_name = table_name;
    TableSample sample;
    sample.job_name = job_name;
    sample.table_name = table_name;
    sample.columns = job.column_names();
    const TableInformation &table = job.table(table_name);
    for (i64 r = 0; r < table.num_rows() / (is_flow ? 20 : 1); r++) {
      sample.rows.push_back(r);
    }
    task.samples.push_back(sample);
    desc.tasks.push_back(task);
  }
}
}
