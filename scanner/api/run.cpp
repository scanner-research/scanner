/* Copyright 2016 Carnegie Mellon University, Stanford University
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

#include "scanner/api/run.h"
#include "scanner/engine/runtime.h"
#include "scanner/metadata.pb.h"
#include "scanner/engine/rpc.grpc.pb.h"
#include "scanner/engine/rpc.pb.h"

#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc++/security/server_credentials.h>
#include <grpc++/security/credentials.h>
#include <grpc++/create_channel.h>

namespace scanner {

namespace {
template <typename T>
std::unique_ptr<grpc::Server> start(T& service, const std::string& port,
                                    bool block) {
  std::string server_address("0.0.0.0:" + port);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  if (block) {
    server->Wait();
  }
  return std::move(server);
}
}

ServerState start_master(DatabaseParameters& params, bool block) {
  ServerState state;
  state.service.reset(scanner::internal::get_master_service(params));
  state.server = start(state.service, "5001", block);
  return state;
}

ServerState start_worker(DatabaseParameters &params,
                  const std::string &master_address, bool block) {
  ServerState state;
  state.service.reset(
      scanner::internal::get_worker_service(params, master_address));
  state.server = start(state.service, "5002", block);
  return state;
}

void new_job(JobParameters& params) {
  auto channel = grpc::CreateChannel(params.master_address,
                                     grpc::InsecureChannelCredentials());
  channel->WaitForConnected(
      gpr_time_add(gpr_now(GPR_CLOCK_REALTIME),
                   gpr_time_from_seconds(30, GPR_TIMESPAN)));
  assert(channel->GetState() != GRPC_CHANNEL_SHUTDOWN);
  std::unique_ptr<proto::Master::Stub> master_ =
      proto::Master::NewStub(channel);

  grpc::ClientContext context;
  proto::JobParameters job_params;
  job_params.set_job_name(params.task_set.job_name);
  proto::TaskSet set = consume_task_set(params.task_set);
  job_params.mutable_task_set()->Swap(&set);
  proto::Empty empty;
  printf("before new job\n");
  grpc::Status status = master_->NewJob(&context, job_params, &empty);
  printf("after new job, %d, %s\n", status.error_code(),
         status.error_message().c_str());
}

}
