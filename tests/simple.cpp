#include "scanner/api/op.h"
#include "scanner/kernels/args.pb.h"
#include "scanner/kernels/caffe_kernel.h"
#include "scanner/api/commands.h"
#include "scanner/util/fs.h"

#include <grpc/grpc_posix.h>

namespace {
std::string download_video(const std::string& video_url) {
  std::string local_video_path;
  scanner::temp_file(local_video_path);
  scanner::download(video_url, local_video_path);
  return local_video_path;
}
}

int main(int argc, char** argv) {
  grpc_use_signal(-1);

  std::unique_ptr<storehouse::StorageConfig> sc(
      storehouse::StorageConfig::make_posix_config());

  std::string db_path;
  scanner::temp_dir(db_path);

  std::string video_path = download_video(
      "https://storage.googleapis.com/scanner-data/test/short_video.mp4");

  // Create database
  scanner::create_database(sc.get(), db_path);
  // Ingest video
  scanner::ingest_videos(sc.get(), db_path, {"mean"}, {video_path});
  // Initialize master and one worker
  scanner::DatabaseParameters db_params;
  db_params.storage_config = sc.get();
  db_params.memory_pool_config.mutable_cpu()->set_use_pool(false);
  db_params.db_path = db_path;
  scanner::ServerState master_state = scanner::start_master(db_params, false);
  scanner::proto::WorkerParameters worker_params =
      scanner::default_worker_params();
  scanner::ServerState worker_state =
      scanner::start_worker(db_params, worker_params, "localhost:5001", false);

  // Construct job parameters
  scanner::JobParameters params;
  params.master_address = "localhost:5001";
  params.job_name = "test_job";
  params.kernel_instances_per_node = 1;
  params.io_item_size = 100;
  params.work_item_size = 25;

  // Specify job tasks
  scanner::Task task;
  task.output_table_name = "blurred_mean";
  scanner::TableSample sample;
  sample.table_name = "mean";
  sample.column_names = {"frame", "frame_info"};
  for (int i = 0; i < 100; i += 1) {
    sample.rows.push_back(i);
  }
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
  scanner::new_job(params);
}
