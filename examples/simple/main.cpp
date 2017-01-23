#include "scanner/api/evaluator.h"
#include "scanner/kernels/args.pb.h"
#include "scanner/api/commands.h"

int main(int argc, char** argv) {
  std::string db_path = "/tmp/new_scanner_db";
  std::unique_ptr<storehouse::StorageConfig> sc(
      storehouse::StorageConfig::make_posix_config());

  // Create database
  scanner::create_database(sc.get(), db_path);
  // Ingest video
  scanner::ingest_videos(sc.get(), db_path, {"mean"},
                         {"/n/scanner/apoms/videos/meanGirls_50k.mp4"});
  // Initialize master and one worker
  scanner::DatabaseParameters db_params;
  db_params.storage_config = sc.get();
  db_params.memory_pool_config.set_use_pool(false);
  db_params.db_path = db_path;
  scanner::ServerState master_state = scanner::start_master(db_params, false);
  scanner::ServerState worker_state =
      scanner::start_worker(db_params, "localhost:5001", false);

  // Construct job parameters
  scanner::Evaluator *input =
      scanner::make_input_evaluator({"frame", "frame_info"});

  scanner::proto::BlurArgs args;
  args.set_kernel_size(3);
  args.set_sigma(0.5);
  size_t arg_size = args.ByteSize();
  char* arg = new char[arg_size];
  args.SerializeToArray(arg, arg_size);
  scanner::Evaluator *blur = new scanner::Evaluator(
      "Blur", {scanner::EvalInput(input, {"frame", "frame_info"})}, arg,
      arg_size);

  // Launch job
  scanner::JobParameters params;
  params.master_address = "localhost:5001";
  scanner::new_job(params);
}
