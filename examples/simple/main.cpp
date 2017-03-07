#include "scanner/api/op.h"
#include "scanner/api/database.h"
#include "stdlib/stdlib.pb.h"

#include <grpc/grpc_posix.h>

int main(int argc, char** argv) {
  grpc_use_signal(-1);

  std::string worker_port(argv[1]);

  std::string db_path = "/tmp/scanner_db";
  std::unique_ptr<storehouse::StorageConfig> sc(
      storehouse::StorageConfig::make_posix_config());
  std::string master_address = "localhost";
  const std::string master_port = "5001";

  scanner::Database db(sc.get(), db_path, master_address,
                       master_port,
                       worker_port);

  // Ingest video
  scanner::Result result;
  std::vector<scanner::FailedVideo> failed_videos;
  result = db.ingest_videos(
      {"mean"}, {"/n/scanner/wcrichto.new/videos/meanGirls_medium.mp4"},
      failed_videos);
  assert(failed_videos.empty());

  // Initialize master and one worker
  scanner::MachineParameters machine_params = scanner::default_machine_params();
  db.start_master(machine_params);
  db.start_worker(machine_params, worker_port);

  // Construct job parameters
  scanner::JobParameters params;
  params.job_name = "test_job";
  params.memory_pool_config.mutable_cpu()->set_use_pool(false);
  params.memory_pool_config.mutable_gpu()->set_use_pool(false);
  params.pipeline_instances_per_node = 1;
  params.work_item_size = 512;

  // Specify job tasks
  scanner::Task task;
  task.output_table_name = "blurred_mean";
  scanner::TableSample sample;
  sample.table_name = "mean";
  sample.column_names = {"frame", "frame_info"};

  sample.sampling_function = "Gather";
  scanner::proto::GatherSamplerArgs args;
  auto& gather_sample = *args.add_samples();
  for (int i = 0; i < 100; i += 1) {
    gather_sample.add_rows(i);
  }
  std::vector<scanner::u8> args_data(args.ByteSize());
  args.SerializeToArray(args_data.data(), args_data.size());
  sample.sampling_args = args_data;

  task.samples.push_back(sample);
  params.task_set.tasks.push_back(task);

  scanner::proto::BlurArgs blur_args;
  blur_args.set_sigma(0.5);
  blur_args.set_kernel_size(3);

  size_t blur_args_size = blur_args.ByteSize();
  char* blur_args_buff = new char[blur_args_size];
  blur_args.SerializeToArray(blur_args_buff, blur_args_size);

  scanner::Op *input =
      scanner::make_input_op({"frame", "frame_info"});

  scanner::Op *blur = new scanner::Op(
      "Blur", {scanner::OpInput(input, {"frame", "frame_info"})},
      scanner::DeviceType::CPU, blur_args_buff, blur_args_size);

  scanner::Op *output = scanner::make_output_op(
      {scanner::OpInput(blur, {"frame", "frame_info"})});

  // Launch job
  params.task_set.output_op = output;
  result = db.new_job(params);
  assert(result.success());
}
