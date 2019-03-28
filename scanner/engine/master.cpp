/* Copyright 2016 Carnegie Mellon University
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

#include "scanner/engine/master.h"
#include "scanner/engine/ingest.h"
#include "scanner/engine/sampler.h"
#include "scanner/engine/dag_analysis.h"
#include "scanner/util/cuda.h"
#include "scanner/util/util.h"
#include "scanner/util/glog.h"
#include "scanner/util/grpc.h"
#include "scanner/engine/python_kernel.h"
#include "scanner/engine/source_registry.h"
#include "scanner/engine/sink_registry.h"
#include "scanner/engine/enumerator_registry.h"
#include "scanner/util/thread_pool.h"
#include "scanner/util/storehouse.h"

#include <grpc/support/log.h>
#include <set>
#include <mutex>
#include <pybind11/embed.h>

using storehouse::StoreResult;
using storehouse::WriteFile;
namespace py = pybind11;

namespace scanner {
namespace internal {

static const i32 GRPC_THREADS = 64;

// Timeouts for GRPC requests
static const u32 LOAD_OP_WORKER_TIMEOUT = 15;
static const u32 REGISTER_OP_WORKER_TIMEOUT = 15;
static const u32 REGISTER_PYTHON_KERNEL_WORKER_TIMEOUT = 15;
static const u32 POKE_WATCHDOG_WORKER_TIMEOUT = 5;
static const u32 PING_WORKER_TIMEOUT = 5;
static const u32 NEW_JOB_WORKER_TIMEOUT = 30;

MasterServerImpl::MasterServerImpl(DatabaseParameters& params, const std::string& port)
    : db_params_(params), port_(port), profiler_(now()) {
  VLOG(1) << "Creating master...";

  {
    // HACK(apoms): to fix this issue: https://github.com/pybind/pybind11/issues/1364
    pybind11::get_shared_data("");
  }

  pool_.reset(new ThreadPool(GRPC_THREADS));

  init_glog("scanner_master");
  storage_ =
      storehouse::StorageBackend::make_from_config(db_params_.storage_config);
  set_database_path(params.db_path);

  // Perform database consistency checks on startup
  recover_and_init_database();

  // Ping workers every 10 seconds to make sure they are alive
  start_worker_pinger();

  start_job_processor();

  VLOG(1) << "Master created.";
}

MasterServerImpl::~MasterServerImpl() {
  trigger_shutdown_.set();
  {
    std::unique_lock<std::mutex> lock(finished_mutex_);
    finished_ = true;
  }
  finished_cv_.notify_all();

  {
    std::unique_lock<std::mutex> lk(work_mutex_);
  }

  stop_job_processor();

  stop_worker_pinger();
  delete storage_;
  if (shutdown_alarm_) {
    delete shutdown_alarm_;
  }
}

// Expects context->peer() to return a string in the format
// ipv4:<peer_address>:<random_port>
// Returns the <peer_address> from the above format.
std::string MasterServerImpl::get_worker_address_from_grpc_context(
    grpc::ServerContext* context) {
  std::string worker_address = context->peer();
  std::size_t portSep = worker_address.find_last_of(':');
  if (portSep == std::string::npos) {
  }
  std::string worker_address_base = worker_address.substr(0, portSep);

  portSep = worker_address_base.find_first_of(':');
  if (portSep == std::string::npos) {
  }

  std::string worker_address_actual = worker_address_base.substr(portSep + 1);

  return worker_address_actual;
}

void MasterServerImpl::run() {
  std::string server_address("0.0.0.0:" + port_);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service_" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *asynchronous* service.
  builder.RegisterService(&service_);
  builder.SetMaxSendMessageSize(1024*1024*1024);
  builder.SetMaxReceiveMessageSize(1024*1024*1024);
  // Get hold of the completion queue used for the asynchronous communication
  // with the gRPC runtime.
  cq_ = builder.AddCompletionQueue();
  server_ = builder.BuildAndStart();
  LOG_IF(FATAL, server_.get() == nullptr) << "Failed to start server";
}

#define REQUEST_RPC(name__, request__, reply__)                              \
  do {                                                                       \
    if (!trigger_shutdown_.raised()) {                                       \
      auto call = new Call<MasterServerImpl, request__, reply__>(            \
          #name__, &MasterServerImpl::name__##Handler);                      \
      service_.Request##name__(&call->ctx, &call->request, &call->responder, \
                               cq_.get(), cq_.get(),                         \
                               (void*)&call->received_tag);                  \
    }                                                                        \
  } while (0);

void MasterServerImpl::handle_rpcs(i32 watchdog_timeout_ms) {
  // Spawn instances for receiving and returning requests
  REQUEST_RPC(Shutdown, proto::Empty, Result);
  REQUEST_RPC(ListTables, proto::Empty, proto::ListTablesResult);
  REQUEST_RPC(GetTables, proto::GetTablesParams, proto::GetTablesResult);
  REQUEST_RPC(DeleteTables, proto::DeleteTablesParams, proto::Empty);
  REQUEST_RPC(NewTable, proto::NewTableParams, proto::Empty);
  REQUEST_RPC(GetVideoMetadata, proto::GetVideoMetadataParams, proto::GetVideoMetadataResult);
  REQUEST_RPC(IngestVideos, proto::IngestParameters, proto::IngestResult);
  // handle more worker registrations efficiently
  for (int i = 0; i < 10; ++i) {
    REQUEST_RPC(RegisterWorker, proto::WorkerParams, proto::Registration);
  }
  REQUEST_RPC(UnregisterWorker, proto::UnregisterWorkerRequest, proto::Empty);
  REQUEST_RPC(ActiveWorkers, proto::Empty, proto::RegisteredWorkers);
  REQUEST_RPC(GetOpInfo, proto::OpInfoArgs, proto::OpInfo);
  REQUEST_RPC(GetSourceInfo, proto::SourceInfoArgs, proto::SourceInfo);
  REQUEST_RPC(GetEnumeratorInfo, proto::EnumeratorInfoArgs, proto::EnumeratorInfo);
  REQUEST_RPC(GetSinkInfo, proto::SinkInfoArgs, proto::SinkInfo);
  REQUEST_RPC(LoadOp, proto::OpPath, proto::Result);
  REQUEST_RPC(RegisterOp, proto::OpRegistration, proto::Result);
  REQUEST_RPC(RegisterPythonKernel, proto::PythonKernelRegistration, proto::Result);
  REQUEST_RPC(ListLoadedOps, proto::Empty, proto::ListLoadedOpsReply);
  REQUEST_RPC(ListRegisteredOps, proto::Empty, proto::ListRegisteredOpsReply);
  REQUEST_RPC(ListRegisteredPythonKernels, proto::Empty, proto::ListRegisteredPythonKernelsReply);
  REQUEST_RPC(NextWork, proto::NextWorkRequest, proto::NextWorkReply);
  REQUEST_RPC(FinishedWork, proto::FinishedWorkRequest, proto::Empty);
  REQUEST_RPC(FinishedJob, proto::FinishedJobRequest, proto::Empty);
  REQUEST_RPC(NewJob, proto::BulkJobParameters, proto::NewJobReply);
  REQUEST_RPC(GetJobs, proto::GetJobsRequest, proto::GetJobsReply);
  REQUEST_RPC(GetJobStatus, proto::GetJobStatusRequest, proto::GetJobStatusReply);
  REQUEST_RPC(Ping, proto::Empty, proto::Empty);
  REQUEST_RPC(PokeWatchdog, proto::Empty, proto::Empty);

  // Initialize the watchdog here to avoid instantly shutting down if
  // watchdog poking hasn't started yet.
  last_watchdog_poke_ = now().time_since_epoch();

  // The main event loop for the master. This loop serves two primary functions:
  //
  //  1. Handle GRPC events. The loop first checks if any events have occured
  //     by polling the global completion queue. If so, it calls out to one
  //     of the event handlers to respond to the request (potentially pushing
  //     the request onto the masters thread pool to be handled asynchronously).
  //
  //  2. Monitor watchdog timeout. After checking for an event, the loop then
  //     checks to see if the watchdog timeout has expired. If so, it starts the
  //     shutdown process for the master.
  bool has_called_start_shutdown = false;
  while (true) {
    // Block waiting to read the next event from the completion queue. The
    // event is uniquely identified by its tag, which in this case is the
    // memory address of a CallData instance.
    // The return value of Next should always be checked. This return value
    // tells us whether there is any kind of event or cq_ is shutting down.
    std::chrono::system_clock::time_point cq_deadline =
        std::chrono::system_clock::now() + std::chrono::milliseconds(50);
    void* tag;  // uniquely identifies a request.
    bool ok;
    grpc::CompletionQueue::NextStatus status =
        cq_->AsyncNext(&tag, &ok, cq_deadline);

    // Receive an event (or a shutdown notification)
    if (status == grpc::CompletionQueue::NextStatus::GOT_EVENT) {
      if (auto call_tag = static_cast<BaseCall<MasterServerImpl>::Tag*>(tag)) {
        std::string type;
        switch (call_tag->get_state()) {
          case BaseCall<MasterServerImpl>::Tag::State::Received: {
            type = "Received";
            tag_start_times_[call_tag->get_call()] = now();
            break;
          }
          case BaseCall<MasterServerImpl>::Tag::State::Sent: {
            type = "Sent";
            profiler_.add_interval(call_tag->get_call()->name, tag_start_times_.at(call_tag->get_call()), now());
            tag_start_times_.erase(call_tag->get_call());
            break;
          }
          case BaseCall<MasterServerImpl>::Tag::State::Cancelled: {
            type = "Cancelled";
            break;
          }
        }

        if (call_tag->get_call()->name != "GetJobStatus") {
          VLOG(2) << "Master cq got " << call_tag->get_call()->name << ":" << type
                  << ":" << ok;
        }

        if (ok) {
          call_tag->Advance(this);
        } else {
          delete call_tag->get_call();
        }
      } else {
        // Shutting down
        VLOG(1) << "Master cq got null, shutting down...";
        server_->Shutdown();
        cq_->Shutdown();
      }
    }
    // The cq has shutdown, so exit the loop and prepare to teardown the master
    else if (status == grpc::CompletionQueue::NextStatus::SHUTDOWN) {
      break;
    }

    // If watchdog timeout is enabled (greater than 0), then check if the
    // last watchdog poke occured within the timeout window.
    if (!trigger_shutdown_.raised() && watchdog_timeout_ms > 0) {
      double ms_since =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              now().time_since_epoch() - last_watchdog_poke_.load())
              .count();
      if (ms_since > watchdog_timeout_ms) {
        // Watchdog not woken, time to bail out
        LOG(ERROR) << "Master did not receive heartbeat in "
                   << watchdog_timeout_ms << "ms. Shutting down.";
        trigger_shutdown_.set();
      }
    }

    // Handle shutdown if triggered
    if (trigger_shutdown_.raised() && !has_called_start_shutdown) {
      has_called_start_shutdown = true;
      start_shutdown();
    }
  }
}

void MasterServerImpl::ShutdownHandler(
    MCall<proto::Empty, proto::Result>* call) {
  VLOG(1) << "Master received shutdown!";
  call->reply.set_success(true);
  trigger_shutdown_.set();
  active_cv_.notify_all();

  // FIXME(apoms): Should we even create a new request like this since Shutdown
  // should only be called once?
  REQUEST_RPC(Shutdown, proto::Empty, Result);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::ListTablesHandler(
    MCall<proto::Empty, proto::ListTablesResult>* call) {
  std::unique_lock<std::mutex> lk(work_mutex_);

  for (const auto& table_name : meta_.table_names()) {
    call->reply.add_tables(table_name);
  }

  REQUEST_RPC(ListTables, proto::Empty, proto::ListTablesResult);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::GetTablesHandler(
    MCall<proto::GetTablesParams, proto::GetTablesResult>* call) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  auto params = &call->request;
  auto result = &call->reply;

  result->mutable_result()->set_success(true);

  std::vector<std::string> table_names;
  for (const auto& table_name : params->tables()) {
    table_names.push_back(table_name);
  }
  // table_metas_->prefetch(table_names);

  VLOG(1) << "Creating output";
  for (const auto& table_name : params->tables()) {
    // Check if has table
    if (!meta_.has_table(table_name)) {
      RESULT_ERROR(result->mutable_result(),
                   "Requested table %s is not in database.",
                   table_name.c_str());
      result->clear_tables();
      break;
    } else {
      // Add table descriptor to result
      const TableMetadata& table_meta = table_metas_->at(table_name);
      proto::TableDescriptor& descriptor = table_meta.get_descriptor();
      proto::TableDescriptor* desc = result->add_tables();
      desc->CopyFrom(descriptor);
    }
  }

  REQUEST_RPC(GetTables, proto::GetTablesParams, proto::GetTablesResult);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::DeleteTablesHandler(
    MCall<proto::DeleteTablesParams, proto::Empty>* call) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  auto params = &call->request;

  // For each table, remove the entry from the database
  std::vector<i32> table_ids;
  for (const auto& table_name : params->tables()) {
    if (meta_.has_table(table_name)) {
      i32 table_id = meta_.get_table_id(table_name);
      meta_.remove_table(table_id);
      table_ids.push_back(table_id);
    }
  }

  write_database_metadata(storage_, meta_);

  // Delete the table data
  for (i32 table_id : table_ids) {
    pool_->enqueue_front([this, table_id]() {
      storage_->delete_dir(table_directory(table_id), true);
    });
  }

  REQUEST_RPC(DeleteTables, proto::DeleteTablesParams, proto::Empty);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::NewTableHandler(
    MCall<proto::NewTableParams, proto::Empty>* call) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  auto params = &call->request;

  const std::string& table_name = params->table_name();
  const auto& columns = params->columns();
  const auto& rows = params->rows();

  i32 table_id = meta_.add_table(table_name);
  LOG_IF(FATAL, table_id == -1) << "failed to add table";
  proto::TableDescriptor table_desc;
  table_desc.set_id(table_id);
  table_desc.set_name(table_name);
  table_desc.set_timestamp(
      std::chrono::duration_cast<std::chrono::seconds>(now().time_since_epoch())
          .count());
  for (size_t i = 0; i < columns.size(); ++i) {
    proto::Column* col = table_desc.add_columns();
    col->set_id(i);
    col->set_name(columns[i]);
    col->set_type(proto::ColumnType::Bytes);
  }

  table_desc.add_end_rows(rows.size());
  table_desc.set_job_id(-1);
  meta_.commit_table(table_id);

  write_table_metadata(storage_, TableMetadata(table_desc));
  write_database_metadata(storage_, meta_);

  LOG_IF(FATAL, rows[0].columns().size() != columns.size())
      << "Row 0 doesn't have # entries == # columns";
  for (size_t j = 0; j < columns.size(); ++j) {
    const std::string output_path = table_item_output_path(table_id, j, 0);

    const std::string output_metadata_path =
        table_item_metadata_path(table_id, j, 0);

    std::unique_ptr<storehouse::WriteFile> output_file;
    storehouse::make_unique_write_file(storage_, output_path,
                                       output_file);

    std::unique_ptr<storehouse::WriteFile> output_metadata_file;
    storehouse::make_unique_write_file(storage_, output_metadata_path,
                                       output_metadata_file);

    u64 num_rows = rows.size();
    s_write(output_metadata_file.get(), num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
      u64 buffer_size = rows[i].columns()[j].size();
      s_write(output_metadata_file.get(), buffer_size);
    }

    for (size_t i = 0; i < num_rows; ++i) {
      i64 buffer_size = rows[i].columns()[j].size();
      u8* buffer = (u8*)rows[i].columns()[j].data();
      s_write(output_file.get(), buffer, buffer_size);
    }

    BACKOFF_FAIL(output_file->save(),
                 "while trying to save " + output_file->path());
    BACKOFF_FAIL(output_metadata_file->save(),
                 "while trying to save " + output_metadata_file->path());
  }

  REQUEST_RPC(NewTable, proto::NewTableParams, proto::Empty);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::GetVideoMetadataHandler(
    MCall<proto::GetVideoMetadataParams, proto::GetVideoMetadataResult>* call) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  auto params = &call->request;
  auto result = &call->reply;

  std::vector<std::string> table_names;
  for (const auto& table_name : params->tables()) {
    table_names.push_back(table_name);
  }

  std::vector<proto::VideoDescriptor*> video_descriptors;
  for (const auto& table_name : params->tables()) {
    video_descriptors.push_back(result->add_videos());
  }

  VLOG(1) << "Prefetching video metadata";
  auto load_video_meta = [&](i32 i) {
    const std::string& table_name = params->tables(i);
    const TableMetadata& table_meta = table_metas_->at(table_name);
    proto::VideoDescriptor* desc_dst = video_descriptors[i];
    if (table_meta.columns().size() == 2 &&
        table_meta.column_type(1) == ColumnType::Video) {
      VideoMetadata video_meta = read_video_metadata(
          storage_, VideoMetadata::descriptor_path(table_meta.id(), 1, 0));
      proto::VideoDescriptor& desc = video_meta.get_descriptor();
      desc.clear_sample_offsets();
      desc.clear_sample_sizes();
      desc.clear_keyframe_indices();
      desc.clear_frames_per_video();
      desc.clear_keyframes_per_video();
      desc.clear_size_per_video();
      desc_dst->CopyFrom(desc);
    } else {
      desc_dst->set_table_id(-1);
    }
  };

  ThreadPool prefetch_pool(16);
  std::vector<std::future<void>> futures;
  for (i32 i = 0; i < params->tables().size(); ++i) {
    futures.emplace_back(prefetch_pool.enqueue(load_video_meta, i));
  }

  for (auto& future : futures) {
    future.wait();
  }

  REQUEST_RPC(GetVideoMetadata, proto::GetVideoMetadataParams,
              proto::GetVideoMetadataResult);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::IngestVideosHandler(
    MCall<proto::IngestParameters, proto::IngestResult>* call) {
  auto params = &call->request;
  auto result = &call->reply;
  std::vector<FailedVideo> failed_videos;
  result->mutable_result()->CopyFrom(
      ingest_videos(db_params_.storage_config, db_params_.db_path,
                    std::vector<std::string>(params->table_names().begin(),
                                             params->table_names().end()),
                    std::vector<std::string>(params->video_paths().begin(),
                                             params->video_paths().end()),
                    params->inplace(), failed_videos));
  for (auto& failed : failed_videos) {
    result->add_failed_paths(failed.path);
    result->add_failed_messages(failed.message);
  }

  // HACK(apoms): instead of doing this, we should just add tables to db and
  //              table cache.
  recover_and_init_database();

  REQUEST_RPC(IngestVideos, proto::IngestParameters, proto::IngestResult);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::RegisterWorkerHandler(
    MCall<proto::WorkerParams, proto::Registration>* call) {
  pool_->enqueue_front([this, call]() {
    std::unique_lock<std::mutex> lk(work_mutex_);
    auto worker_info = &call->request;
    auto registration = &call->reply;

    set_database_path(db_params_.db_path);

    std::string worker_address =
        get_worker_address_from_grpc_context(&call->ctx);
    worker_address += ":" + worker_info->port();

    i32 node_id = next_worker_id_++;
    VLOG(1) << "Adding worker: " << node_id << ", " << worker_address;

    // wcrichto 10-17-18: providing the node_id in the ChannelArguments prevents a bug where
    // when a worker dies/restarts on the same node/port, the connection would be reused and
    // all messages would bounce with GOAWAY.
    // See: https://github.com/grpc/grpc/issues/14260#issuecomment-362298290
    auto chan_args = grpc::ChannelArguments();
    chan_args.SetInt("node_id", node_id);
    auto stub = proto::Worker::NewStub(grpc::CreateCustomChannel(
        worker_address, grpc::InsecureChannelCredentials(), chan_args));

    std::shared_ptr<WorkerState> worker_state(
        new WorkerState(node_id, std::move(stub), worker_address));
    worker_state->state = WorkerState::IDLE;

    workers_[node_id] = worker_state;

    std::unique_lock<std::mutex> lock(active_mutex_);
    if (active_bulk_job_) {
      auto& state = bulk_jobs_state_.at(active_bulk_job_id_);
      state->worker_histories[node_id].start_time = now();
      state->worker_histories[node_id].tasks_assigned = 0;
      state->worker_histories[node_id].tasks_retired = 0;
      state->unstarted_workers.push_back(node_id);
    }

    registration->set_node_id(node_id);
    REQUEST_RPC(RegisterWorker, proto::WorkerParams, proto::Registration);
    call->Respond(grpc::Status::OK);
  });
}

void MasterServerImpl::UnregisterWorkerHandler(
    MCall<proto::UnregisterWorkerRequest, proto::Empty>* call) {
  pool_->enqueue_front([this, call]() {
    std::unique_lock<std::mutex> lk(work_mutex_);
    auto request = &call->request;

    set_database_path(db_params_.db_path);

    i32 node_id = request->node_id();
    remove_worker(node_id);

    REQUEST_RPC(UnregisterWorker, proto::UnregisterWorkerRequest, proto::Empty);
    call->Respond(grpc::Status::OK);
  });
}

void MasterServerImpl::ActiveWorkersHandler(
    MCall<proto::Empty, proto::RegisteredWorkers>* call) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  auto registered_workers = &call->reply;

  set_database_path(db_params_.db_path);

  for (auto& kv : workers_) {
    // Check if worker is not inactive
    if (kv.second->state.load() != WorkerState::UNREGISTERED) {
      i32 worker_id = kv.first;
      proto::WorkerInfo* info = registered_workers->add_workers();
      info->set_id(worker_id);
      info->set_address(workers_.at(worker_id)->address);
    }
  }

  REQUEST_RPC(ActiveWorkers, proto::Empty, proto::RegisteredWorkers);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::GetOpInfoHandler(
    MCall<proto::OpInfoArgs, proto::OpInfo>* call) {
  auto op_info_args = &call->request;
  auto op_info = &call->reply;
  OpRegistry* registry = get_op_registry();
  std::string op_name = op_info_args->op_name();
  if (!registry->has_op(op_name)) {
    op_info->mutable_result()->set_success(false);
    op_info->mutable_result()->set_msg("Op " + op_name + " does not exist");
    REQUEST_RPC(GetOpInfo, proto::OpInfoArgs, proto::OpInfo);
    call->Respond(grpc::Status::OK);
    return;
  }

  OpInfo* info = registry->get_op_info(op_name);

  op_info->set_variadic_inputs(info->variadic_inputs());
  for (auto& input_column : info->input_columns()) {
    Column* info = op_info->add_input_columns();
    info->CopyFrom(input_column);
  }
  for (auto& output_column : info->output_columns()) {
    Column* info = op_info->add_output_columns();
    info->CopyFrom(output_column);
  }
  op_info->set_protobuf_name(info->protobuf_name());
  op_info->set_stream_protobuf_name(info->stream_protobuf_name());

  op_info->mutable_result()->set_success(true);

  REQUEST_RPC(GetOpInfo, proto::OpInfoArgs, proto::OpInfo);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::GetSourceInfoHandler(
    MCall<proto::SourceInfoArgs, proto::SourceInfo>* call) {
  auto source_info_args = &call->request;
  auto source_info = &call->reply;

  SourceRegistry* registry = get_source_registry();
  std::string source_name = source_info_args->source_name();
  if (!registry->has_source(source_name)) {
    source_info->mutable_result()->set_success(false);
    source_info->mutable_result()->set_msg("Source " + source_name +
                                           " does not exist");
    REQUEST_RPC(GetSourceInfo, proto::SourceInfoArgs, proto::SourceInfo);
    call->Respond(grpc::Status::OK);
    return;
  }

  SourceFactory* fact = registry->get_source(source_name);
  for (auto& output_column : fact->output_columns()) {
    Column* info = source_info->add_output_columns();
    info->CopyFrom(output_column);
  }
  source_info->set_protobuf_name(fact->protobuf_name());
  source_info->mutable_result()->set_success(true);

  REQUEST_RPC(GetSourceInfo, proto::SourceInfoArgs, proto::SourceInfo);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::GetEnumeratorInfoHandler(
    MCall<proto::EnumeratorInfoArgs, proto::EnumeratorInfo>* call) {
  auto info_args = &call->request;
  auto info = &call->reply;

  EnumeratorRegistry* registry = get_enumerator_registry();
  std::string enumerator_name = info_args->enumerator_name();
  if (!registry->has_enumerator(enumerator_name)) {
    info->mutable_result()->set_success(false);
    info->mutable_result()->set_msg("Enumerator " + enumerator_name +
                                    " does not exist");
    REQUEST_RPC(GetEnumeratorInfo, proto::EnumeratorInfoArgs, proto::EnumeratorInfo);
    call->Respond(grpc::Status::OK);
    return;
  }
  EnumeratorFactory* fact = registry->get_enumerator(enumerator_name);
  info->set_protobuf_name(fact->protobuf_name());
  info->mutable_result()->set_success(true);

  REQUEST_RPC(GetEnumeratorInfo, proto::EnumeratorInfoArgs, proto::EnumeratorInfo);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::GetSinkInfoHandler(
    MCall<proto::SinkInfoArgs, proto::SinkInfo>* call) {
  auto sink_info_args = &call->request;
  auto sink_info = &call->reply;

  SinkRegistry* registry = get_sink_registry();
  std::string sink_name = sink_info_args->sink_name();
  if (!registry->has_sink(sink_name)) {
    sink_info->mutable_result()->set_success(false);
    sink_info->mutable_result()->set_msg("Sink " + sink_name +
                                           " does not exist");
    REQUEST_RPC(GetSinkInfo, proto::SinkInfoArgs, proto::SinkInfo);
    call->Respond(grpc::Status::OK);
    return;
  }

  SinkFactory* fact = registry->get_sink(sink_name);
  for (auto& output_column : fact->input_columns()) {
    Column* info = sink_info->add_input_columns();
    info->CopyFrom(output_column);
  }
  sink_info->set_variadic_inputs(fact->variadic_inputs());
  sink_info->set_protobuf_name(fact->protobuf_name());
  sink_info->set_stream_protobuf_name(fact->stream_protobuf_name());
  sink_info->mutable_result()->set_success(true);

  REQUEST_RPC(GetSinkInfo, proto::SinkInfoArgs, proto::SinkInfo);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::LoadOpHandler(MCall<proto::OpPath, Result>* call) {
  auto op_path = &call->request;
  auto result = &call->reply;
  std::string so_path = op_path->path();
  VLOG(1) << "Master loading Op: " << so_path;

  std::unique_lock<std::mutex> lk(work_mutex_);

  auto l = std::string("__stdlib").size();
  if (so_path.substr(0, l) == "__stdlib") {
    so_path = db_params_.python_dir + "/lib/libscanner_stdlib" + so_path.substr(l);
  }

  for (auto& loaded_path : so_paths_) {
    if (loaded_path == so_path) {
      LOG(WARNING) << "Master received redundant request to load op " << so_path;
      result->set_success(true);
      REQUEST_RPC(LoadOp, proto::OpPath, proto::Result);
      call->Respond(grpc::Status::OK);
      return;
    }
  }

  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    RESULT_ERROR(result, "Failed to load op library: %s", dlerror());
    REQUEST_RPC(LoadOp, proto::OpPath, proto::Result);
    call->Respond(grpc::Status::OK);
    return;
  }
  so_paths_.push_back(so_path);

  result->set_success(true);
  VLOG(1) << "Master successfully loaded Op: " << op_path->path();

  REQUEST_RPC(LoadOp, proto::OpPath, proto::Result);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::RegisterOpHandler(
    MCall<proto::OpRegistration, proto::Result>* call) {
  std::unique_lock<std::mutex> lk(work_mutex_);

  auto op_registration = &call->request;
  auto result = &call->reply;

  VLOG(1) << "Master registering Op: " << op_registration->name();

  result->set_success(true);
  const std::string& name = op_registration->name();
  {
    const bool variadic_inputs = op_registration->variadic_inputs();
    std::vector<Column> input_columns;
    size_t i = 0;
    for (auto& c : op_registration->input_columns()) {
      Column col;
      col.set_id(i++);
      col.set_name(c.name());
      col.set_type(c.type());
      col.set_type_name(c.type_name());
      input_columns.push_back(col);
    }
    std::vector<Column> output_columns;
    i = 0;
    for (auto& c : op_registration->output_columns()) {
      Column col;
      col.set_id(i++);
      col.set_name(c.name());
      col.set_type(c.type());
      col.set_type_name(c.type_name());
      output_columns.push_back(col);
    }
    bool can_stencil = op_registration->can_stencil();
    std::vector<i32> stencil(op_registration->preferred_stencil().begin(),
                             op_registration->preferred_stencil().end());
    if (stencil.empty()) {
      stencil = {0};
    }
    bool has_bounded_state = op_registration->has_bounded_state();
    i32 warmup = op_registration->warmup();
    bool has_unbounded_state = op_registration->has_unbounded_state();
    OpInfo* info = new OpInfo(name, variadic_inputs, input_columns,
                              output_columns, can_stencil, stencil,
                              has_bounded_state, warmup, has_unbounded_state,
                              "", "");
    OpRegistry* registry = get_op_registry();
    *result = registry->add_op(name, info);
  }
  if (!result->success()) {
    LOG(WARNING) << "Master failed to register op " << name;
    REQUEST_RPC(RegisterOp, proto::OpRegistration, proto::Result);
    call->Respond(grpc::Status::OK);
    return;
  }

  op_registrations_.push_back(*op_registration);
  VLOG(1) << "Master successfully registered Op: " << op_registration->name();

  REQUEST_RPC(RegisterOp, proto::OpRegistration, proto::Result);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::RegisterPythonKernelHandler(
    MCall<proto::PythonKernelRegistration, proto::Result>* call) {
  std::unique_lock<std::mutex> lk(work_mutex_);
  auto python_kernel = &call->request;
  auto result = &call->reply;

  VLOG(1) << "Master registering Python Kernel: " << python_kernel->op_name();

  {
    const std::string& op_name = python_kernel->op_name();
    DeviceType device_type = python_kernel->device_type();
    const std::string& kernel_code = python_kernel->kernel_code();
    const int batch_size = python_kernel->batch_size();

    // Set all input and output columns to be CPU
    std::map<std::string, DeviceType> input_devices;
    std::map<std::string, DeviceType> output_devices;
    bool can_batch = batch_size > 1;
    bool can_stencil;
    {
      OpRegistry* registry = get_op_registry();
      OpInfo* info = registry->get_op_info(op_name);
      if (info->variadic_inputs()) {
        LOG_IF(FATAL, device_type == DeviceType::GPU) << "Cannot have variadic inputs on the GPU";
      } else {
        for (const auto& in_col : info->input_columns()) {
          input_devices[in_col.name()] = DeviceType::CPU;
        }
      }
      for (const auto& out_col : info->output_columns()) {
        output_devices[out_col.name()] = DeviceType::CPU;
      }
      can_stencil = info->can_stencil();
    }

    // Create a kernel builder function
    auto constructor = [op_name, kernel_code, can_batch,
                        can_stencil](const KernelConfig& config) {
      return new PythonKernel(config, op_name, kernel_code, can_batch,
                              can_stencil);
    };

    // Create a new kernel factory
    KernelFactory* factory =
        new KernelFactory(op_name, device_type, 1, input_devices,
                          output_devices, can_batch, batch_size, constructor);

    // Register the kernel
    KernelRegistry* registry = get_kernel_registry();
    registry->add_kernel(op_name, factory);
  }

  result->set_success(true);
  py_kernel_registrations_.push_back(*python_kernel);
  VLOG(1) << "Master successfully registered Python Kernel: " << python_kernel->op_name();

  REQUEST_RPC(RegisterPythonKernel, proto::PythonKernelRegistration, proto::Result);
  call->Respond(grpc::Status::OK);
}


void MasterServerImpl::ListLoadedOpsHandler(MCall<proto::Empty, proto::ListLoadedOpsReply>* call) {
  auto& reply = call->reply;
  for (const std::string& so_path : so_paths_) {
    proto::OpPath* op_path = reply.add_registrations();
    op_path->set_path(so_path);
  }

  REQUEST_RPC(ListLoadedOps, proto::Empty, proto::ListLoadedOpsReply);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::ListRegisteredOpsHandler(MCall<proto::Empty, proto::ListRegisteredOpsReply>* call) {
  auto& reply = call->reply;
  for (auto& reg : op_registrations_) {
    auto r = reply.add_registrations();
    r->CopyFrom(reg);
  }

  REQUEST_RPC(ListRegisteredOps, proto::Empty, proto::ListRegisteredOpsReply);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::ListRegisteredPythonKernelsHandler(
    MCall<proto::Empty, proto::ListRegisteredPythonKernelsReply>* call) {
  auto& reply = call->reply;
  for (auto& reg : py_kernel_registrations_) {
    auto r = reply.add_registrations();
    r->CopyFrom(reg);
  }

  REQUEST_RPC(ListRegisteredPythonKernels, proto::Empty, proto::ListRegisteredPythonKernelsReply);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::NextWorkHandler(
    MCall<proto::NextWorkRequest, proto::NextWorkReply>* call) {
  pool_->enqueue([this, call]() {
    std::unique_lock<std::mutex> lk(work_mutex_);
    auto node_info = &call->request;
    auto new_work = &call->reply;

    VLOG(2) << "Master received NextWork command";
    i32 worker_id = node_info->node_id();
    if (workers_.at(worker_id)->state != WorkerState::RUNNING_JOB) {
      // Worker is not running a job (must be an old request) so ignore
      new_work->set_no_more_work(true);
      REQUEST_RPC(NextWork, proto::NextWorkRequest, proto::NextWorkReply);
      call->Respond(grpc::Status::OK);
      return;
    }

    i32 bulk_job_id = node_info->bulk_job_id();
    {
      std::unique_lock<std::mutex> lock(active_mutex_);
      if (bulk_job_id != active_bulk_job_id_) {
        LOG(WARNING) << "Worker " << worker_id << " ("
                     << workers_.at(worker_id)->address
                     << ") requested NextWork for bulk job " << bulk_job_id
                     << " but active job is " << active_bulk_job_id_;
        new_work->set_no_more_work(true);
        REQUEST_RPC(NextWork, proto::NextWorkRequest, proto::NextWorkReply);
        call->Respond(grpc::Status::OK);
        return;
      }
    }

    auto& state = bulk_jobs_state_.at(bulk_job_id);

    // If we do not have any outstanding work, try and create more
    i32 tasks_to_assign = node_info->num_work();
    bool no_more_work = false;
    bool wait_for_work = false;
    while (tasks_to_assign > 0) {
      if (state->to_assign_job_tasks.empty()) {
        // If we have no more samples for this task, try and get another task
        if (state->next_task == state->num_tasks) {
          // Check if there are any tasks left
          if (state->next_job < state->num_jobs &&
              state->task_result.success()) {
            state->next_task = 0;
            state->num_tasks = state->job_tasks.at(state->next_job).size();
            state->next_job++;
            VLOG(1) << "Tasks left: "
                    << state->total_tasks - state->total_tasks_used;
          }
        }

        // Create more work if possible
        if (state->next_task < state->num_tasks) {
          i64 current_job = state->next_job - 1;
          i64 current_task = state->next_task;

          auto jt = std::make_tuple(current_job, current_task);
          state->active_job_tasks.insert(jt);
          state->to_assign_job_tasks.push_front(jt);
          state->next_task++;
        }
      }

      if (state->to_assign_job_tasks.empty()) {
        if (finished_) {
          // No more work
          no_more_work = true;
        } else {
          // Still have tasks that might be reassigned
          wait_for_work = true;
        }
        break;
      }

      // Grab the next task sample
      std::tuple<i64, i64> job_task_id = state->to_assign_job_tasks.back();
      state->to_assign_job_tasks.pop_back();

      assert(state->next_task <= state->num_tasks);

      i64 job_idx;
      i64 task_idx;
      std::tie(job_idx, task_idx) = job_task_id;

      // If the job was blacklisted, then we throw it away
      if (state->blacklisted_jobs.count(job_idx) > 0) {
        continue;
      }

      proto::NextWorkReply::WorkPacket* work_packet =
          new_work->add_work_packets();
      work_packet->set_job_index(job_idx);
      work_packet->set_task_index(task_idx);
      const auto& task_rows = state->job_tasks.at(job_idx).at(task_idx);
      for (i64 r : task_rows) {
        work_packet->add_output_rows(r);
      }

      auto task_start = std::chrono::duration_cast<std::chrono::seconds>(
                            now().time_since_epoch())
                            .count();
      // Track sample assigned to worker
      state->worker_job_tasks[node_info->node_id()].insert(job_task_id);
      state->worker_job_tasks_starts[std::make_tuple(
          (i64)node_info->node_id(), job_idx, task_idx)] = task_start;
      state->worker_histories[node_info->node_id()].tasks_assigned += 1;

      tasks_to_assign -= 1;
    }

    if (new_work->work_packets_size() == 0) {
      new_work->set_no_more_work(no_more_work);
      new_work->set_wait_for_work(wait_for_work);
    }

    REQUEST_RPC(NextWork, proto::NextWorkRequest, proto::NextWorkReply);
    call->Respond(grpc::Status::OK);
  });
}

void MasterServerImpl::FinishedWorkHandler(
    MCall<proto::FinishedWorkRequest, proto::Empty>* call) {
  pool_->enqueue([this, call]() {
    std::unique_lock<std::mutex> lk(work_mutex_);
    std::unique_lock<std::mutex> lock(active_mutex_);
    VLOG(2) << "Master received FinishedWork command";

    auto params = &call->request;

    i32 worker_id = params->node_id();
    i32 bulk_job_id = params->bulk_job_id();

    for (auto& work_id : params->work_ids()) {
      i64 job_id = work_id.job_id();
      i64 task_id = work_id.task_id();
      i64 num_rows = work_id.num_rows();

      if (workers_.count(worker_id) == 0) {
        LOG(WARNING) << "Master got FinishedWork from non-existent worker id "
                     << worker_id << ". Ignoring.";
        REQUEST_RPC(FinishedWork, proto::FinishedWorkRequest, proto::Empty);
        call->Respond(grpc::Status::OK);
        return;
      }

      {
        if (bulk_job_id != active_bulk_job_id_) {
          LOG(WARNING) << "Worker " << worker_id << " ("
                       << workers_.at(worker_id)->address
                       << ") requested FinishedWork for bulk job "
                       << bulk_job_id << " but active job is "
                       << active_bulk_job_id_;
          REQUEST_RPC(FinishedWork, proto::FinishedWorkRequest, proto::Empty);
          call->Respond(grpc::Status::OK);
          return;
        }
      }

      auto& state = bulk_jobs_state_.at(bulk_job_id);

      auto& worker_tasks = state->worker_job_tasks.at(worker_id);

      std::tuple<i64, i64> job_task = std::make_tuple(job_id, task_id);

      // Remove the task from the list of assigned tasks to the worker
      assert(worker_tasks.count(job_task) > 0);
      worker_tasks.erase(job_task);
      state->worker_job_tasks_starts.erase(
          std::make_tuple((i64)worker_id, job_id, task_id));

      // Increment the number of tasks finished by this worker
      state->worker_histories[worker_id].tasks_retired += 1;

      i64 active_job = state->next_job - 1;

      // If job was blacklisted, then we have already updated total tasks
      // used to reflect that and we should ignore it
      bool not_blacklisted = state->blacklisted_jobs.count(job_id) == 0;
      // It's possible for a task to be assigned to multiple workers, so
      // this condition makes sure that we only mark the task as finished
      // the first time it is processed.
      bool still_active = state->active_job_tasks.count(job_task) > 0;
      if (not_blacklisted && still_active) {
        // Remove from active job task list since it has been finished
        state->active_job_tasks.erase(job_task);

        state->total_tasks_used++;
        state->tasks_used_per_job[job_id]++;

        if (state->tasks_used_per_job[job_id] ==
            state->job_tasks[job_id].size()) {
          if (state->dag_info.has_table_output) {
            for (i32 tid : state->job_uncommitted_tables[job_id]) {
              meta_.commit_table(tid);
            }
          }

          // Commit database metadata every so often
          if (job_id % state->job_params.checkpoint_frequency() == 0) {
            VLOG(1) << "Saving database metadata checkpoint";
            write_database_metadata(storage_, meta_);
          }
        }
      }

      if (state->total_tasks_used == state->total_tasks) {
        VLOG(1) << "Master FinishedWork triggered finished!";
        assert(state->next_job == state->num_jobs);
        {
          std::unique_lock<std::mutex> lock(finished_mutex_);
          finished_ = true;
        }
        finished_cv_.notify_all();
      }
    }

    REQUEST_RPC(FinishedWork, proto::FinishedWorkRequest, proto::Empty);
    call->Respond(grpc::Status::OK);
  });
}

void MasterServerImpl::FinishedJobHandler(
    MCall<proto::FinishedJobRequest, proto::Empty>* call) {
  pool_->enqueue([this, call]() {
    std::unique_lock<std::mutex> lk(work_mutex_);
    VLOG(1) << "Master received FinishedJob command";

    auto params = &call->request;

    i32 worker_id = params->node_id();
    i32 bulk_job_id = params->bulk_job_id();

    {
      std::unique_lock<std::mutex> lock(active_mutex_);
      if (bulk_job_id != active_bulk_job_id_) {
        LOG(WARNING) << "Worker " << worker_id << " ("
                     << workers_.at(worker_id)->address
                     << ") requested FinishedJob for bulk job " << bulk_job_id
                     << " but active job is " << active_bulk_job_id_;
        REQUEST_RPC(FinishedJob, proto::FinishedJobRequest, proto::Empty);
        call->Respond(grpc::Status::OK);
        return;
      }
    }

    auto& state = bulk_jobs_state_.at(bulk_job_id);
    state->unfinished_workers.at(worker_id) = false;

    if (workers_.at(worker_id)->state == WorkerState::UNREGISTERED) {
      REQUEST_RPC(FinishedJob, proto::FinishedJobRequest, proto::Empty);
      call->Respond(grpc::Status::OK);
      return;
    }

    if (!params->result().success()) {
      LOG(WARNING) << "Worker " << worker_id << " sent FinishedJob with error: "
                   << params->result().msg();
    }

    std::unique_lock<std::mutex> lk2(active_mutex_);
    if (active_bulk_job_) {
      stop_job_on_worker(worker_id);
    }
    workers_.at(worker_id)->state = WorkerState::IDLE;

    REQUEST_RPC(FinishedJob, proto::FinishedJobRequest, proto::Empty);
    call->Respond(grpc::Status::OK);
  });
}

void MasterServerImpl::NewJobHandler(
    MCall<proto::BulkJobParameters, proto::NewJobReply>* call) {
  pool_->enqueue([this, call]() {
    VLOG(1) << "Master received NewJob command";
    set_database_path(db_params_.db_path);

    auto job_params = &call->request;
    auto reply = &call->reply;

    {
      std::unique_lock<std::mutex> lock(finished_mutex_);
      finished_ = false;
    }
    finished_cv_.notify_all();

    // Add job name into database metadata so we can look up what jobs have
    // been run
    i32 bulk_job_id = meta_.add_bulk_job(job_params->job_name());
    reply->set_bulk_job_id(bulk_job_id);

    job_params_.Clear();
    job_params_.MergeFrom(*job_params);

    {
      std::unique_lock<std::mutex> lock(active_mutex_);
      active_bulk_job_ = true;
      active_bulk_job_id_ = bulk_job_id;
      new_job_call_ = call;
    }
    active_cv_.notify_all();
  });
}

void MasterServerImpl::GetJobStatusHandler(
    MCall<proto::GetJobStatusRequest, proto::GetJobStatusReply>* call) {
  VLOG(3) << "Master received GetJobStatus command";

  pool_->enqueue_front([this, call]() {
    std::unique_lock<std::mutex> l(work_mutex_);
    std::unique_lock<std::mutex> lock(active_mutex_);
    auto request = &call->request;
    auto reply = &call->reply;

    if (bulk_jobs_state_.count(request->bulk_job_id()) == 0) {
      LOG(WARNING)
          << "GetJobStatus received request for non-existent bulk job id: "
          << request->bulk_job_id();

      REQUEST_RPC(GetJobStatus, proto::GetJobStatusRequest,
                  proto::GetJobStatusReply);
      call->Respond(grpc::Status::OK);
      return;
    }
    std::shared_ptr<BulkJob> state =
        bulk_jobs_state_.at(request->bulk_job_id());

    if (!active_bulk_job_) {
      reply->set_finished(true);
      reply->mutable_result()->CopyFrom(state->job_result);

      reply->set_tasks_done(0);
      reply->set_total_tasks(0);

      reply->set_jobs_done(0);
      reply->set_jobs_failed(0);
      reply->set_total_jobs(0);
    } else {
      reply->set_finished(false);

      reply->set_tasks_done(state->total_tasks_used);
      reply->set_total_tasks(state->total_tasks);

      reply->set_jobs_done(state->next_job - 1);
      reply->set_jobs_failed(0);
      reply->set_total_jobs(state->num_jobs);
    }
    // Num workers
    i32 num_workers = 0;
    for (auto& kv : workers_) {
      if (kv.second->state == WorkerState::RUNNING_JOB) {
        num_workers++;
      }
    }
    reply->set_num_workers(num_workers);
    reply->set_failed_workers(state->num_failed_workers);

    REQUEST_RPC(GetJobStatus, proto::GetJobStatusRequest,
                proto::GetJobStatusReply);
    call->Respond(grpc::Status::OK);
  });
}

void MasterServerImpl::GetJobsHandler(
    MCall<proto::GetJobsRequest, proto::GetJobsReply>* call) {
  VLOG(3) << "Master received GetJobs command";

  pool_->enqueue_front([this, call]() {
    std::unique_lock<std::mutex> l(work_mutex_);
    std::unique_lock<std::mutex> lock(active_mutex_);

    auto request = &call->request;
    auto reply = &call->reply;

    if (active_bulk_job_) {
      reply->add_active_bulk_jobs(active_bulk_job_id_);
    }

    REQUEST_RPC(GetJobs, proto::GetJobsRequest, proto::GetJobsReply);
    call->Respond(grpc::Status::OK);
  });
}



void MasterServerImpl::PingHandler(
    MCall<proto::Empty, proto::Empty>* call) {
  REQUEST_RPC(Ping, proto::Empty, proto::Empty);
  call->Respond(grpc::Status::OK);
}

void MasterServerImpl::PokeWatchdogHandler(
    MCall<proto::Empty, proto::Empty>* call) {
  pool_->enqueue_front([this, call]() {
    VLOG(2) << "Master received PokeWatchdog.";
    last_watchdog_poke_ = now().time_since_epoch();
    REQUEST_RPC(PokeWatchdog, proto::Empty, proto::Empty);
    call->Respond(grpc::Status::OK);
  });
}

void MasterServerImpl::recover_and_init_database() {
  VLOG(1) << "Initializing database...";

  VLOG(1) << "Reading database metadata";
  // TODO(apoms): handle uncommitted database tables
  meta_ = read_database_metadata(storage_, DatabaseMetadata::descriptor_path());

  VLOG(1) << "Setting up table metadata cache";
  // Setup table metadata cache
  table_metas_.reset(new TableMetaCache(storage_, meta_));

  // VLOG(1) << "Writing database metadata";
  // write_database_metadata(storage_, meta_);

  VLOG(1) << "Database initialized.";
}

void MasterServerImpl::start_job_processor() {
  VLOG(1) << "Starting job processor";
  job_processor_thread_ = std::thread([this]() {
    {
      // HACK(apoms): to fix this issue: https://github.com/pybind/pybind11/issues/1364
      py::gil_scoped_acquire acquire;
      pybind11::get_shared_data("");
    }

    while (!trigger_shutdown_.raised()) {
      // Wait on not finished
      {
        std::unique_lock<std::mutex> lock(active_mutex_);
        active_cv_.wait(lock, [this] {
          return active_bulk_job_ || trigger_shutdown_.raised();
        });
      }
      if (trigger_shutdown_.raised()) break;
      VLOG(1) << "Job processor signaled, starting process";
      // Create new bulk job
      // Start processing job
      Result result;
      bool b = process_job(&job_params_, &result);
    }
  });
}

void MasterServerImpl::stop_job_processor() {
  // Wake up job processor
  {
    std::unique_lock<std::mutex> lock(active_mutex_);
    active_bulk_job_ = true;
  }
  active_cv_.notify_all();
  if (job_processor_thread_.joinable()) {
    job_processor_thread_.join();
  }
}

bool MasterServerImpl::process_job(const proto::BulkJobParameters* job_params,
                                   proto::Result* job_result) {
  // Remove old profiling information
  auto job_start = now();
  profiler_.reset(job_start);
  i64 job_start_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(job_start)
    .time_since_epoch()
    .count();
  job_params_.set_base_time(job_start_ns);

  // Set profiling level
  PROFILER_LEVEL = static_cast<ProfilerLevel>(job_params->profiler_level());

  i32 bulk_job_id = active_bulk_job_id_;
  std::shared_ptr<BulkJob> state(new BulkJob);
  {
    std::unique_lock<std::mutex> l(work_mutex_);
    bulk_jobs_state_[bulk_job_id] = state;
  }

  state->job_params.CopyFrom(*job_params);

  // Setup job state
  state->task_result.set_success(true);
  state->job_result.set_success(true);

  // Respond to NewJob only after the state has been created
  REQUEST_RPC(NewJob, proto::BulkJobParameters, proto::NewJobReply);
  new_job_call_->Respond(grpc::Status::OK);

  auto finished_fn = [this, state, job_result]() {
    state->total_tasks_used = 0;
    state->job_result.CopyFrom(*job_result);
    {
      std::unique_lock<std::mutex> lock(finished_mutex_);
      finished_ = true;
    }
    finished_cv_.notify_all();
    {
      std::unique_lock<std::mutex> lock(active_mutex_);
      active_bulk_job_ = false;
    }
    active_cv_.notify_all();
  };

  std::vector<proto::Job> jobs(job_params->jobs().begin(),
                               job_params->jobs().end());
  std::vector<proto::Op> ops(job_params->ops().begin(),
                             job_params->ops().end());

  const i32 work_packet_size = job_params->work_packet_size();
  const i32 io_packet_size = job_params->io_packet_size() != -1
                                 ? job_params->io_packet_size()
                                 : work_packet_size;
  if (io_packet_size > 0 && io_packet_size % work_packet_size != 0) {
    RESULT_ERROR(
        job_result,
        "IO packet size (%d) must be a multiple of work packet size (%d).",
        io_packet_size, work_packet_size);
    finished_fn();
    return false;
  }

  i32 total_rows = 0;

  VLOG(1) << "Validating jobs";
  DAGAnalysisInfo& dag_info = state->dag_info;
  *job_result =
      validate_jobs_and_ops(meta_, *table_metas_.get(), jobs, ops, dag_info);
  if (!job_result->success()) {
    // No database changes made at this point, so just return
    finished_fn();
    return false;
  }

  // Map all source Ops into a single input collection
  const std::map<i64, i64>& input_op_idx_to_column_idx = dag_info.source_ops;

  VLOG(1) << "Finding output columns";
  // Get output columns from sink ops to set as output table columns
  OpRegistry* op_registry = get_op_registry();
  std::vector<std::vector<Column>> job_output_columns;
  for (const auto& job : jobs) {
    // Get input columns from column inputs specified for each job
    std::map<i64, Column> input_op_idx_to_column;
    {
      SourceRegistry* registry = get_source_registry();
      for (auto& ci : job.inputs()) {
        std::string op_name = ops.at(ci.op_index()).name();
        std::vector<Column> source_columns =
            registry->get_source(op_name)->output_columns();
        input_op_idx_to_column[ci.op_index()] = source_columns[0];
      }
    }

    job_output_columns.emplace_back();
    std::vector<Column>& output_columns = job_output_columns.back();
    // For an op column, find the Column info
    std::function<Column(const proto::OpInput&)> determine_column_info =
        [&determine_column_info, &ops, &input_op_idx_to_column,
         op_registry](const proto::OpInput& op_input) -> Column {
      i64 op_idx = op_input.op_index();
      const std::string& col = op_input.column();
      auto& input_op = ops.at(op_idx);
      // For builtin ops, find non bulit-in parent column
      if (!input_op.is_source() && is_builtin_op(input_op.name())) {
        // Find the column
        for (auto& in : input_op.inputs()) {
          if (in.column() == col) {
            return determine_column_info(in);
          }
        }
        assert(false);
      }

      std::vector<Column> input_columns;
      std::vector<Column> actual_columns;
      if (input_op.is_source()) {
        Column col = input_op_idx_to_column.at(op_idx);
        actual_columns = {col};
        col.set_name(input_op.inputs(0).column());
        input_columns = {col};
      } else {
        OpInfo* input_op_info = op_registry->get_op_info(input_op.name());
        input_columns = input_op_info->output_columns();
        actual_columns = input_columns;
      }
      const std::string& name = col;
      bool found = false;
      for (size_t i = 0; i < input_columns.size(); ++i) {
        auto& in_col = input_columns[i];
        if (in_col.name() == name) {
          Column c = actual_columns[i];
          return c;
        }
      }
      assert(false);
    };
    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
      if (dag_info.sink_ops.count(op_idx) > 0) {
        for (const auto& input : ops.at(op_idx).inputs()) {
          Column c = determine_column_info(input);
          c.set_id(output_columns.size());
          output_columns.push_back(c);
        }
      }
    }
    for (size_t i = 0; i < job_params->output_column_names_size(); ++i) {
      output_columns[i].set_name(job_params->output_column_names(i));
    }
  }

  // Tell workers about the output columns
  {
    std::vector<Column>& output_columns = job_output_columns.back();
    for (auto& c : output_columns) {
      Column* col = job_params_.add_output_columns();
      col->CopyFrom(c);
    }
  }

  proto::BulkJobDescriptor job_descriptor;
  job_descriptor.set_io_packet_size(io_packet_size);
  job_descriptor.set_work_packet_size(work_packet_size);
  job_descriptor.set_num_nodes(workers_.size());

  {
    auto& jobs = job_params->jobs();
    job_descriptor.mutable_jobs()->CopyFrom(jobs);
  }

  VLOG(1) << "Determining input rows to slices";
  job_descriptor.set_id(bulk_job_id);
  job_descriptor.set_name(job_params->job_name());
  // Determine total output rows and slice input rows for using to
  // split stream
  *job_result = determine_input_rows_to_slices(meta_, *table_metas_.get(), jobs,
                                               ops, dag_info, db_params_.storage_config);
  state->slice_input_rows_per_job = dag_info.slice_input_rows;
  state->total_output_rows_per_job = dag_info.total_output_rows;

  if (!job_result->success()) {
    // No database changes made at this point, so just return
    finished_fn();
    return false;
  }

  // HACK(apoms): we currently split work into tasks in two ways:
  //  a) align with the natural boundaries defined by the slice partitioner
  //  b) use a user-specified size to chunk up the output sequence

  VLOG(1) << "Building tasks";
  // Job -> task -> rows
  i32 total_tasks_temp = 0;
  for (size_t i = 0; i < jobs.size(); ++i) {
    state->tasks_used_per_job.push_back(0);

    auto& slice_input_rows = state->slice_input_rows_per_job[i];
    i64 total_output_rows = state->total_output_rows_per_job[i];

    std::vector<i64> partition_boundaries;
    if (slice_input_rows.size() == 0) {
      // No slices, so we can split as desired. Currently use IO packet size
      // since it is the smallest granularity we can specify
      for (i64 r = 0; r < total_output_rows; r += io_packet_size) {
        partition_boundaries.push_back(r);
      }
      partition_boundaries.push_back(total_output_rows);
    } else {
      // Split stream into partitions, respecting slice boundaries
      // We assume there is only one slice for now since
      // they all must have the same number of groups

      // Derive the output rows produced by each slice group
      i64 slice_op_idx = slice_input_rows.begin()->first;
      i64 slice_in_rows = slice_input_rows.begin()->second;
      *job_result = derive_slice_final_output_rows(
          jobs.at(i), ops, slice_op_idx, slice_in_rows, dag_info,
          partition_boundaries);
      if (!job_result->success()) {
        // No database changes made at this point, so just return
        finished_fn();
        return false;
      }
    }
    assert(partition_boundaries.back() == total_output_rows);
    state->job_tasks.emplace_back();
    auto& tasks = state->job_tasks.back();
    for (i64 pi = 0; pi < partition_boundaries.size() - 1; ++pi) {
      tasks.emplace_back();
      auto& task_rows = tasks.back();

      i64 s = partition_boundaries[pi];
      i64 e = partition_boundaries[pi + 1];
      for (i64 r = s; r < e; ++r) {
        task_rows.push_back(r);
      }
      total_tasks_temp++;
    }
  }
  state->total_tasks = total_tasks_temp;

  if (!job_result->success()) {
    // No database changes made at this point, so just return
    finished_fn();
    return false;
  }

  // Write out database metadata so that workers can read it
  write_bulk_job_metadata(storage_, BulkJobMetadata(job_descriptor));

  VLOG(1) << "Updating db metadata";
  state->job_uncommitted_tables.clear();
  if (dag_info.has_table_output) {
    for (i64 job_idx = 0; job_idx < job_params->jobs_size(); ++job_idx) {
      auto& job = job_params->jobs(job_idx);
      state->job_uncommitted_tables.emplace_back();
      auto& uncommitted_tables = state->job_uncommitted_tables.back();
      for (i64 sink_op_idx : dag_info.column_sink_ops) {
        std::string& table_name =
            dag_info.column_sink_table_names.at(job_idx).at(
                sink_op_idx);
        i32 table_id = meta_.add_table(table_name);
        state->job_to_table_ids[job_idx].push_back(table_id);
        proto::TableDescriptor table_desc;
        table_desc.set_id(table_id);
        table_desc.set_name(table_name);
        table_desc.set_timestamp(
            std::chrono::duration_cast<std::chrono::seconds>(
                now().time_since_epoch())
                .count());
        // Set columns equal to the last op's output columns
        for (size_t i = 0; i < job_output_columns[job_idx].size(); ++i) {
          Column* col = table_desc.add_columns();
          col->CopyFrom(job_output_columns[job_idx][i]);
        }
        table_metas_->update(TableMetadata(table_desc));

        i64 total_rows = 0;
        std::vector<i64> end_rows;
        auto& tasks = state->job_tasks.at(job_idx);
        for (i64 task_id = 0; task_id < tasks.size(); ++task_id) {
          i64 task_rows = tasks.at(task_id).size();
          total_rows += task_rows;
          end_rows.push_back(total_rows);
        }
        for (i64 r : end_rows) {
          table_desc.add_end_rows(r);
        }
        table_desc.set_job_id(bulk_job_id);
        table_metas_->update(TableMetadata(table_desc));
        uncommitted_tables.push_back(table_id);
      }
    }
    // Write table metadata
    table_metas_->write_megafile();
  }

  // Setup initial task sampler
  state->task_result.set_success(true);
  state->next_task = 0;
  state->num_tasks = 0;
  state->next_job = 0;
  state->num_jobs = jobs.size();

  write_database_metadata(storage_, meta_);
  job_params_.mutable_db_meta()->CopyFrom(meta_.get_descriptor());

  VLOG(1) << "Total jobs: " << state->num_jobs;

  // Send new job command to workers
  VLOG(1) << "Sending new job command to workers";
  {
    std::vector<i32> worker_ids;
    {
      std::unique_lock<std::mutex> lk(work_mutex_);
      for (auto& kv : workers_) {
        if (kv.second->state.load() == WorkerState::IDLE) {
          worker_ids.push_back(kv.first);
        }
      }
      state->unstarted_workers.clear();
    }
    start_job_on_workers(worker_ids);
  }

  // Wait for all workers to finish
  VLOG(1) << "Waiting for workers to finish";

  // Wait until all workers are done and work has been completed
  auto all_workers_finished_start = now();
  auto finished_start = now();
  while (true) {
    if (!finished_) {
      finished_start = now();
    }
    if (finished_) {
      // If we have finished but workers have not responded after 60 seconds,
      // end the job
      double seconds_since = std::chrono::duration_cast<std::chrono::seconds>(
                                 now() - finished_start)
                                 .count();
      if (seconds_since > 60) {
        LOG(WARNING) << "Job has been finished for 60 seconds but not all "
                        "workers confirmed a finished job. This does not "
                        "affect the job results but some profiling information "
                        "may not have been written. Finishing job now.";
        break;
      }
    }
    // Check if we have unfinished workers
    bool all_workers_finished = true;
    {
      std::unique_lock<std::mutex> lk(work_mutex_);
      for (auto& kv : state->unfinished_workers) {
        // If the worker is active and it is not finished, then
        // we need to keep working
        if (workers_.at(kv.first)->state.load() == WorkerState::RUNNING_JOB && kv.second) {
          all_workers_finished = false;
          break;
        }
      }
    }
    if (all_workers_finished && !finished_) {
      // If we have unfinished work but no workers for some period of time,
      // then fail
      double seconds_since = std::chrono::duration_cast<std::chrono::seconds>(
                                 now() - all_workers_finished_start)
                                 .count();
      if (seconds_since > db_params_.no_workers_timeout) {
        RESULT_ERROR(job_result,
                     "No workers but have unfinished work after %ld seconds",
                     db_params_.no_workers_timeout);
        finished_fn();
        return false;
      }
    } else {
      // Reset timer
      all_workers_finished_start = now();
    }
    if (all_workers_finished && finished_) {
      break;
    }
    // Check if any tasks have gone on longer than timeout
    if (job_params_.task_timeout() > 0.0001) {
      std::unique_lock<std::mutex> lk(work_mutex_);
      auto current_time = std::chrono::duration_cast<std::chrono::seconds>(
                              now().time_since_epoch())
                              .count();
      auto& jts = state->worker_job_tasks_starts;
      for (const auto& kv : jts) {
        if (current_time - kv.second > state->job_params.task_timeout()) {
          i64 worker_id;
          i64 job_id;
          i64 task_id;
          std::tie(worker_id, job_id, task_id) = kv.first;
          // Task has timed out, stop the worker
          LOG(WARNING) << "Node " << worker_id << " ("
                       << workers_.at(worker_id)->address << ") "
                       << "failed to finish task (" << job_id << ", " << task_id
                       << ") after " << state->job_params.task_timeout()
                       << " seconds. Removing that worker as an active worker.";
          remove_worker(worker_id);
          state->num_failed_workers++;
          // NOTE(apoms): We must break here because remove_worker modifies
          // state->active_job_tasks_starts, thus invalidating our pointer
          break;
        }
      }
    }
    // Check if we have unstarted workers and start them if so
    {
      std::vector<i32> worker_ids;
      {
        std::unique_lock<std::mutex> lk(work_mutex_);
        for (i32 wid : state->unstarted_workers) {
          worker_ids.push_back(wid);
        }
        state->unstarted_workers.clear();
      }

      if (!worker_ids.empty()) {
        start_job_on_workers(worker_ids);
      }
    }
    std::this_thread::yield();
  }

  {
    std::unique_lock<std::mutex> lock(finished_mutex_);
    finished_cv_.wait(lock, [this] { return finished_.load(); });
  }

  // If we are shutting down, then the job did not finish and we should fail
  if (trigger_shutdown_.raised()) {
    job_result->set_success(false);
  }

  if (job_result->success()) {
    // Commit job since it was successful
    meta_.commit_bulk_job(bulk_job_id);
  }
  write_database_metadata(storage_, meta_);

  if (!state->task_result.success()) {
    job_result->CopyFrom(state->task_result);
  } else {
    assert(state->next_job == state->num_jobs);
  }

  std::fflush(NULL);
  sync();

  // Update job metadata with new # of nodes
  {
    std::unique_lock<std::mutex> lk(work_mutex_);
    job_descriptor.set_num_nodes(workers_.size());
  }
  write_bulk_job_metadata(storage_, BulkJobMetadata(job_descriptor));

  // Write profiler info
  write_profiler(bulk_job_id, job_start, now());

  finished_fn();

  VLOG(1) << "Master finished job";

  return true;
}

void MasterServerImpl::start_worker_pinger() {
  VLOG(1) << "Starting worker pinger";
  pinger_active_ = true;
  pinger_thread_ = std::thread([this]() {
    while (pinger_active_) {
      VLOG(3) << "Start of pinger loop";

      std::shared_ptr<BulkJob> state;
      {
        std::unique_lock<std::mutex> l(work_mutex_);
        if (active_bulk_job_) {
          state = bulk_jobs_state_.at(active_bulk_job_id_);
        }
      }

      std::map<WorkerID, std::shared_ptr<WorkerState>> ws;
      {
        std::unique_lock<std::mutex> lk(work_mutex_);
        for (auto& kv : workers_) {
          i32 worker_id = kv.first;
          auto& worker = kv.second;
          if (worker->state == WorkerState::UNREGISTERED) continue;

          ws[worker_id] = worker;
        }
      }

      struct RequestData {
        grpc::ClientContext context;
        grpc::Status status;
        proto::PingReply reply;
        std::unique_ptr<grpc::ClientAsyncResponseReader<proto::PingReply>> rpc;
      };
      std::map<i32, RequestData> requests;

      grpc::CompletionQueue cq;
      int i = 0;

      VLOG(3) << "Queueing pings to workers";
      for (auto& kv : ws) {
        i64 id = kv.first;
        auto& worker = kv.second;
        proto::Empty em;

        RequestData& request_data = requests[id];

        // Set timeout for Ping call
        u32 timeout = PING_WORKER_TIMEOUT;
        std::chrono::system_clock::time_point deadline =
            std::chrono::system_clock::now() + std::chrono::seconds(timeout);
        request_data.context.set_deadline(deadline);

        request_data.rpc =
            worker->stub->AsyncPing(&request_data.context, em, &cq);
        request_data.rpc->Finish(&request_data.reply, &request_data.status,
                                 (void*)id);
        i++;
        VLOG(3) << "Master sending Ping to worker " << id;
      }

      VLOG(3) << "Waiting on pings from workers";
      for (int i = 0; i < ws.size(); ++i) {
        void* got_tag;
        bool ok = false;
        GPR_ASSERT(cq.Next(&got_tag, &ok));
        i64 worker_id = (i64)got_tag;

        if (requests.count(worker_id) > 0) {
          RequestData& request_data = requests.at(worker_id);
          if (!request_data.status.ok() ||
              request_data.reply.node_id() != worker_id) {
            VLOG(3) << "Master failed to Ping worker " << worker_id;

            // Worker not responding, increment ping count
            i64 num_failed_pings = ++ws[worker_id]->failed_pings;
            const i64 FAILED_PINGS_BEFORE_REMOVAL = 3;
            if (num_failed_pings >= FAILED_PINGS_BEFORE_REMOVAL ||
                (request_data.status.ok() && request_data.reply.node_id() != worker_id)) {

              // remove it from active workers
              if (num_failed_pings >= FAILED_PINGS_BEFORE_REMOVAL) {
                LOG(WARNING)
                  << "Worker " << worker_id << " did not respond to Ping. "
                  << "Removing worker from active list.";
              } else {
                LOG(WARNING)
                    << "Worker " << worker_id << " responded to Ping with "
                    << "worker id " << request_data.reply.node_id()
                    << ". Removing from active "
                    << "list.";
              }

              std::unique_lock<std::mutex> lk(work_mutex_);
              remove_worker(worker_id);
              if (state.get()) {
                state->num_failed_workers++;
              }
            }
          } else {
            VLOG(3) << "Master successfully Pinged worker " << worker_id;
            ws[worker_id]->failed_pings = 0;
          }
        } else {
          LOG(WARNING) << "Got invalid Ping response with tag for non-existent "
                          "worker id "
                       << worker_id << "!";
        }
      }
      cq.Shutdown();

      VLOG(3) << "All pings sent/received";
      // Sleep for 5 seconds or wake up if the job has finished before then
      std::unique_lock<std::mutex> lk(pinger_wake_mutex_);
      pinger_wake_cv_.wait_for(lk, std::chrono::seconds(5),
                               [&] { return pinger_active_ == false; });
    }
  });
}

void MasterServerImpl::stop_worker_pinger() {
  if (pinger_thread_.joinable()) {
    {
      std::unique_lock<std::mutex> lk(pinger_wake_mutex_);
      pinger_active_ = false;
    }
    pinger_wake_cv_.notify_all();
    pinger_thread_.join();
  }
}

void MasterServerImpl::start_job_on_workers(const std::vector<i32>& worker_ids) {
  std::vector<i32> filtered_worker_ids;
  proto::BulkJobParameters w_job_params;
  std::map<WorkerID, std::shared_ptr<WorkerState>> workers_copy;
  {
    std::unique_lock<std::mutex> lk(work_mutex_);
    w_job_params.MergeFrom(job_params_);
    w_job_params.set_bulk_job_id(active_bulk_job_id_);
    for (i32 worker_id : worker_ids) {
      if (workers_.count(worker_id) > 0 &&
          workers_.at(worker_id)->state == WorkerState::IDLE) {
        workers_.at(worker_id)->state = WorkerState::RUNNING_JOB;
        workers_copy[worker_id] = workers_.at(worker_id);
        filtered_worker_ids.push_back(worker_id);
      }
    }
  }

  struct Req {
    i32 worker_id;
    // If not request, then it is an alarm
    bool is_request;
  };

  grpc::CompletionQueue cq;
  std::map<i32, std::unique_ptr<grpc::ClientContext>> client_contexts;
  std::map<i32, std::unique_ptr<grpc::Status>> statuses;
  std::map<i32, std::unique_ptr<proto::Result>> replies;
  std::map<i32, std::unique_ptr<grpc::ClientAsyncResponseReader<proto::Result>>>
      rpcs;
  std::map<i32, std::unique_ptr<grpc::Alarm>> alarms;
  std::map<i32, i32> retry_attempts;

  auto send_new_job = [&](i32 worker_id) {
    auto& worker = workers_copy.at(worker_id)->stub;

    client_contexts[worker_id] =
        std::unique_ptr<grpc::ClientContext>(new grpc::ClientContext);
    u32 timeout = NEW_JOB_WORKER_TIMEOUT;
    std::chrono::system_clock::time_point deadline =
        std::chrono::system_clock::now() + std::chrono::seconds(timeout);
    client_contexts[worker_id]->set_deadline(deadline);

    statuses[worker_id] = std::unique_ptr<grpc::Status>(new grpc::Status);
    replies[worker_id] = std::unique_ptr<proto::Result>(new proto::Result);
    rpcs[worker_id] = worker->AsyncNewJob(client_contexts[worker_id].get(),
                                          w_job_params, &cq);
    rpcs[worker_id]->Finish(replies[worker_id].get(), statuses[worker_id].get(),
                            (void*)new Req{worker_id, true});
    VLOG(2) << "Sent NewJob command to worker " << worker_id;
  };

  for (i32 worker_id : filtered_worker_ids) {
    send_new_job(worker_id);
  }

  i64 workers_processed = 0;
  std::vector<i32> valid_workers;
  std::vector<i32> invalid_workers;
  while (workers_processed < filtered_worker_ids.size()) {
    void* got_tag;
    bool ok = false;
    auto stat = cq.Next(&got_tag, &ok);
    assert(stat != false);
    assert(ok);

    Req* req = (Req*)got_tag;
    i32 worker_id = req->worker_id;
    bool is_request = req->is_request;
    delete req;

    if (is_request) {
      auto status = *statuses[worker_id].get();
      if (status.ok()) {
        VLOG(2) << "Worker " << worker_id << " NewJob returned.";

        if (!replies[worker_id]->success()) {
          LOG(WARNING) << "Worker " << worker_id << " ("
                       << workers_copy.at(worker_id)->address << ") "
                       << " NewJob returned error: " << replies[worker_id]->msg();
          invalid_workers.push_back(worker_id);
        } else {
          valid_workers.push_back(worker_id);
        }
        workers_processed++;
      } else if (status.error_code() == grpc::StatusCode::UNAVAILABLE) {
        // We should retry this request
        i32 retries = retry_attempts[worker_id]++;
        if (retries > db_params_.new_job_retries_limit) {
          // Already retried too many times
          LOG(WARNING) << "Worker " << worker_id << " timed out for NewJob: ("
                       << status.error_code() << "): " << status.error_message();
          invalid_workers.push_back(worker_id);
          workers_processed++;
        } else {
          // Issue a grpc Alarm to alert when we should reissue this NewJob request
          auto alarm = new grpc::Alarm();
          alarms[worker_id].reset(alarm);
          double sleep_time = std::pow(2, retries) +
                              (static_cast<double>(rand()) / RAND_MAX);
          auto deadline =
              std::chrono::system_clock::now();
          deadline += std::chrono::milliseconds((i64)(sleep_time * 1000));
          alarm->Set(&cq, deadline, new Req{worker_id, false});
          LOG(WARNING) << "Worker " << worker_id << " unavailable for NewJob: ("
                       << status.error_code() << "): " << status.error_message()
                       << ". Retrying after " << sleep_time << " seconds.";
        }
      } else {
        // Request failed, so we should ignore this worker
        LOG(WARNING) << "Worker " << worker_id << " did not return NewJob: ("
                     << status.error_code() << "): " << status.error_message();
        invalid_workers.push_back(worker_id);
        workers_processed++;
      }
    } else {
      // This was an alarm, so reissue this request
      send_new_job(worker_id);
    }
  }
  cq.Shutdown();

  void* got_tag;
  bool ok = false;
  while (cq.Next(&got_tag, &ok)) {
    Req* req = (Req*)got_tag;
    delete req;
  }

  {
    std::unique_lock<std::mutex> lk(work_mutex_);
    for (i32 worker_id : invalid_workers) {
      remove_worker(worker_id);
    }
    std::unique_lock<std::mutex> lock(active_mutex_);
    if (active_bulk_job_) {
      auto& state = bulk_jobs_state_.at(active_bulk_job_id_);
      for (i32 worker_id : valid_workers) {
        state->unfinished_workers[worker_id] = true;
      }
    }
  }
}

void MasterServerImpl::stop_job_on_worker(i32 worker_id) {
  // Place workers active tasks back into the unallocated task samples
  auto& state = bulk_jobs_state_.at(active_bulk_job_id_);
  if (state->worker_job_tasks.count(worker_id) > 0) {
    // Place workers active tasks back into the unallocated task samples
    VLOG(1) << "Reassigning worker " << worker_id << "'s "
            << state->worker_job_tasks.at(worker_id).size() << " task samples.";
    for (const std::tuple<i64, i64>& worker_job_task :
         state->worker_job_tasks.at(worker_id)) {
      state->to_assign_job_tasks.push_back(worker_job_task);
      state->worker_job_tasks_starts.erase(
          std::make_tuple((i64)worker_id, std::get<0>(worker_job_task),
                          std::get<1>(worker_job_task)));

      // The worker failure may be due to a bad task. We track number of times
      // a task has failed to detect a bad task and remove it from this bulk
      // job if it exceeds some threshold.
      i64 job_id = std::get<0>(worker_job_task);
      i64 task_id = std::get<1>(worker_job_task);

      i64 num_failures = ++state->job_tasks_num_failures[job_id][task_id];
      const i64 TOTAL_FAILURES_BEFORE_REMOVAL = 3;
      if (num_failures >= TOTAL_FAILURES_BEFORE_REMOVAL) {
        blacklist_job(job_id);
      }
    }
    state->worker_job_tasks.erase(worker_id);
  }

  workers_.at(worker_id)->state = WorkerState::IDLE;
  state->worker_histories[worker_id].end_time = now();
  state->unfinished_workers[worker_id] = false;
}

void MasterServerImpl::remove_worker(i32 node_id) {
  assert(workers_.count(node_id) > 0);

  std::string worker_address = workers_.at(node_id)->address;
  // Remove worker from list
  {
    std::unique_lock<std::mutex> lock(active_mutex_);
    if (active_bulk_job_) {
      stop_job_on_worker(node_id);
    }
  }
  workers_.at(node_id)->state = WorkerState::UNREGISTERED;

  VLOG(1) << "Removing worker " << node_id << " (" << worker_address << ").";
}

void MasterServerImpl::blacklist_job(i64 job_id) {
  auto& state = bulk_jobs_state_.at(active_bulk_job_id_);

  // Check that the job has not been blacklisted yet
  if (state->blacklisted_jobs.count(job_id) > 0) {
    return;
  }
  // All tasks in unallocated_job_tasks_ with this job id will be thrown away
  state->blacklisted_jobs.insert(job_id);
  // Remove all of the job's tasks from active_job_tasks
  for (i64 i = 0; i < state->job_tasks.at(job_id).size(); ++i) {
    state->active_job_tasks.erase(std::make_tuple(job_id, i));
  }
  // Add number of remaining tasks to tasks used
  i64 num_tasks_left_in_job =
      state->job_tasks[job_id].size() - state->tasks_used_per_job[job_id];
  state->total_tasks_used += num_tasks_left_in_job;

  VLOG(1) << "Blacklisted job " << job_id;

  // Check if blacklisting job finished the bulk job
  if (state->total_tasks_used == state->total_tasks) {
    VLOG(1) << "Master blacklisting job triggered finished!";
    assert(state->next_job == state->num_jobs);
    {
      std::unique_lock<std::mutex> lock(finished_mutex_);
      finished_ = true;
    }
    finished_cv_.notify_all();
  }
}  // namespace internal

void MasterServerImpl::start_shutdown() {
  // Shutdown workers
  std::vector<i32> worker_ids;
  std::map<i32, proto::Worker::Stub*> workers_copy;
  {
    std::unique_lock<std::mutex> lk(work_mutex_);
    for (auto& kv : workers_) {
      if (kv.second->state != WorkerState::UNREGISTERED) {
        worker_ids.push_back(kv.first);
        workers_copy[kv.first] = kv.second->stub.get();
      }
    }
  }
  for (i32 i : worker_ids) {
    proto::Empty empty;
    proto::Result wresult;
    grpc::Status status;
    GRPC_BACKOFF_D(workers_copy.at(i)->Shutdown(&ctx, empty, &wresult), status,
                   15);
    const std::string& worker_address = workers_.at(i)->address;
    LOG_IF(WARNING, !status.ok())
        << "Master could not send shutdown message to worker at "
        << worker_address << " (" << status.error_code()
        << "): " << status.error_message();
  }
  // Shutdown self
  {
    std::unique_lock<std::mutex> lock(finished_mutex_);
    finished_ = true;
    shutdown_alarm_ =
        new grpc::Alarm(cq_.get(), gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
  }
}

void MasterServerImpl::write_profiler(int bulk_job_id, timepoint_t job_start, timepoint_t job_end) {
  // Create output file
  std::string profiler_file_name = bulk_job_master_profiler_path(bulk_job_id);
  std::unique_ptr<WriteFile> profiler_output;
  BACKOFF_FAIL(make_unique_write_file(storage_, profiler_file_name, profiler_output),
      "while trying to make write file for " + profiler_file_name);

  // Write profiler data
  i64 start_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(job_start)
          .time_since_epoch()
          .count();
  i64 end_time_ns =
      std::chrono::time_point_cast<std::chrono::nanoseconds>(job_end)
          .time_since_epoch()
          .count();
  s_write(profiler_output.get(), start_time_ns);
  s_write(profiler_output.get(), end_time_ns);

  write_profiler_to_file(profiler_output.get(), 0, "master", "master", 0, profiler_);

  // Save profiler
  BACKOFF_FAIL(profiler_output->save(), "while trying to save " + profiler_output->path());
  std::fflush(NULL);
  sync();
}

}  // namespace internal
}  // namespace scanner
