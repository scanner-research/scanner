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
#include "scanner/engine/rpc.grpc.pb.h"

#include <grpc++/server.h>
#include <grpc++/server_builder.h>
#include <grpc++/security/server_credentials.h>
#include <grpc++/security/credentials.h>
#include <grpc++/create_channel.h>

namespace scanner {

namespace {
template <typename T>
std::unique_ptr<grpc::Server> start(T& service, bool block) {
  std::string port = "5001";
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
  state.server = start(state.service, block);
  return state;
}

ServerState start_worker(DatabaseParameters &params,
                  const std::string &master_address, bool block) {
  ServerState state;
  state.service.reset(
      scanner::internal::get_worker_service(params, master_address));
  state.server = start(state.service, block);
  return state;
}

void new_job(JobParameters& params) {
}

}
