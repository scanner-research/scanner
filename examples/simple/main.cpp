#include "scanner/api/op.h"
#include "scanner/kernels/args.pb.h"
#include "scanner/kernels/caffe_kernel.h"
#include "scanner/api/commands.h"

#include <grpc/grpc_posix.h>

int main(int argc, char** argv) {
  grpc_use_signal(-1);

  std::string db_path = "/tmp/test_db";
  std::unique_ptr<storehouse::StorageConfig> sc(
      storehouse::StorageConfig::make_posix_config());

  // Create database
  scanner::create_database(sc.get(), db_path);
  // Ingest video
  scanner::ingest_videos(sc.get(), db_path, {"mean"},
                         {"/n/scanner/wcrichto.new/videos/meanGirls_short.mp4"});
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

  printf("after start workers\n");
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
  for (int i = 0; i < 720; i += 1) {
    sample.rows.push_back(i);
  }
  task.samples.push_back(sample);
  params.task_set.tasks.push_back(task);

  scanner::proto::NetDescriptor descriptor =
      scanner::descriptor_from_net_file("features/googlenet.toml");
  scanner::proto::CaffeInputArgs caffe_input_args;
  scanner::proto::CaffeArgs caffe_args;
  caffe_input_args.mutable_net_descriptor()->CopyFrom(descriptor);
  caffe_input_args.set_batch_size(96);
  caffe_args.mutable_net_descriptor()->CopyFrom(descriptor);
  caffe_args.set_batch_size(96);

  size_t caffe_input_args_size = caffe_input_args.ByteSize();
  char* caffe_input_args_buff = new char[caffe_input_args_size];
  caffe_input_args.SerializeToArray(caffe_input_args_buff,
                                    caffe_input_args_size);

  size_t caffe_args_size = caffe_args.ByteSize();
  char* caffe_args_buff = new char[caffe_args_size];
  caffe_args.SerializeToArray(caffe_args_buff, caffe_args_size);

  scanner::Op *input =
      scanner::make_input_op({"frame", "frame_info"});

  scanner::Op *caffe_input = new scanner::Op(
      "CaffeInput", {scanner::OpInput(input, {"frame", "frame_info"})},
      scanner::DeviceType::GPU,
      caffe_input_args_buff, caffe_input_args_size);

  scanner::Op *caffe = new scanner::Op(
      "Caffe", {scanner::OpInput(caffe_input, {"caffe_frame"}),
                scanner::OpInput(input, {"frame_info"})},
      scanner::DeviceType::GPU, caffe_args_buff, caffe_args_size);

  scanner::Op *output = scanner::make_output_op(
      {scanner::OpInput(caffe, {"caffe_output"})});

  // Launch job
  params.task_set.output_op = output;
  scanner::new_job(params);
}
