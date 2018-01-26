#pragma once

#include "scanner/util/common.h"

namespace scanner {

using FnPtr = void (*)();

//! Convenience for dynamic registration of C++ functions.
class UserFunctionRegistry {
 public:
  void add_user_function(const std::string& name, const FnPtr fn);

  template <typename T>
  T get_user_function(const std::string& name);

  bool has_user_function(const std::string& name);

 private:
  std::map<std::string, const FnPtr> fns_;
};

UserFunctionRegistry* get_user_function_registry();

template <typename T>
T UserFunctionRegistry::get_user_function(const std::string& name) {
  return reinterpret_cast<T>(fns_.at(name));
}

namespace internal {

class UserFunctionRegistration {
 public:
  UserFunctionRegistration(const std::string& name, const FnPtr fn);
};
}

#define REGISTER_USER_FUNCTION(name__, function__) \
  REGISTER_USER_FUNCTION_HELPER(__COUNTER__, name__, function__)

#define REGISTER_USER_FUNCTION_HELPER(uid__, name__, function__) \
  REGISTER_USER_FUNCTION_UID(uid__, name__, function__)

#define REGISTER_USER_FUNCTION_UID(uid__, name__, function__) \
  static ::scanner::internal::UserFunctionRegistration         \
      user_function_registration_##uid__ =                    \
          ::scanner::internal::UserFunctionRegistration(      \
              #name__, static_cast<const void*>(function__));
}
