#pragma once

#include "stdlib/stdlib.pb.h"
#include <pqxx/pqxx>

namespace scanner {
  std::unique_ptr<pqxx::connection> sql_connect(proto::SQLConfig sql_config);
}
