#include "scanner/util/common.h"
#include "scanner/util/tinyformat.h"
#include "stdlib/misc/sql.h"

namespace scanner {
  std::unique_ptr<pqxx::connection> sql_connect(proto::SQLConfig sql_config) {
    LOG_IF(FATAL, sql_config.adapter() != "postgres")
        << "Requested adapter " << sql_config.adapter()
        << " does not exist. Only \"postgres\" is supported right now.";
    try {
      return std::make_unique<pqxx::connection>(tfm::format(
          "hostaddr=%s port=%d dbname=%s user=%s password=%s",
          sql_config.hostaddr(), sql_config.port(), sql_config.dbname(),
          sql_config.user(), sql_config.password()));
    } catch (pqxx::pqxx_exception& e) {
      LOG(FATAL) << e.base().what();
    }
  }

}
