/* Licensed under the Apache License, Version 2.0 (the "License");
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

#pragma once

#include <cstdlib>
#include <unistd.h>

namespace scanner {

#define GRPC_BACKOFF(expression__, status__)        \
  GRPC_BACKOFF_TIMEOUT(expression__, status__, 64, 0)

#define GRPC_BACKOFF_D(expression__, status__, deadline__)       \
  GRPC_BACKOFF_TIMEOUT(expression__, status__, 64, deadline__)

#define GRPC_BACKOFF_TIMEOUT(expression__, status__, timeout__, deadline__) \
  do {                                                                      \
    int sleep_debt__ = 1;                                                   \
    while (true) {                                                          \
      grpc::ClientContext ctx;                                              \
      if (deadline__ > 0) {                                                 \
        std::chrono::system_clock::time_point deadline =                    \
            std::chrono::system_clock::now() +                              \
            std::chrono::seconds(deadline__);                               \
        ctx.set_deadline(deadline);                                         \
      }                                                                     \
      const grpc::Status result__ = (expression__);                         \
      if (result__.error_code() == grpc::StatusCode::UNAVAILABLE) {         \
        double sleep_time__ =                                               \
            (sleep_debt__ + (static_cast<double>(rand()) / RAND_MAX));      \
        if (sleep_debt__ < (timeout__)) {                                   \
          sleep_debt__ *= 2;                                                \
        } else {                                                            \
          LOG(WARNING) << "GRPC_BACKOFF: reached max backoff.";             \
          status__ = result__;                                              \
          break;                                                            \
        }                                                                   \
        LOG(WARNING) << "GRPC_BACKOFF: transient failure, sleeping for "    \
                     << sleep_time__ << " seconds.";                        \
        usleep(sleep_time__ * 1000000);                                     \
        continue;                                                           \
      }                                                                     \
      status__ = result__;                                                  \
      break;                                                                \
    }                                                                       \
  } while (0);

template <class ServiceImpl>
struct BaseCall {
  virtual ~BaseCall() {}

  virtual void Handle(ServiceImpl* service) = 0;

  class Tag {
   public:
    enum class State { Received, Sent, Cancelled };

    Tag(BaseCall* call, State state) : call_(call), state_(state) {}

    BaseCall* get_call() {
      return call_;
    }

    const State& get_state() {
      return state_;
    }

    void Advance(ServiceImpl* service) {
      switch (state_) {
        case State::Received: {
          call_->Handle(service);
          break;
        }
        case State::Sent: {
          delete call_;
          break;
        }
        case State::Cancelled: {
          delete call_;
          break;
        }
      }
    }

   private:
    BaseCall* call_;
    State state_;
  };

  std::string name;
};

template <class ServiceImpl, class Request, class Reply>
struct Call : BaseCall<ServiceImpl> {
  using HandleFunction =
      void (ServiceImpl::*)(Call<ServiceImpl, Request, Reply>*);

  Call(const std::string& _name, HandleFunction _handler)
    : handler(_handler), responder(&ctx) {
    this->name = _name;
  }

  void Handle(ServiceImpl* service) override {
    (service->*handler)(this);
  }

  void Respond(grpc::Status status) {
    responder.Finish(reply, status, &sent_tag);
  }

  HandleFunction handler;
  grpc::ServerContext ctx;
  Request request;
  Reply reply;
  grpc::ServerAsyncResponseWriter<Reply> responder;

  // Tags
  using Tag = typename BaseCall<ServiceImpl>::Tag;
  Tag received_tag{this, Tag::State::Received};
  Tag sent_tag{this, Tag::State::Sent};
  Tag cancelled_tag{this, Tag::State::Cancelled};
};
}
