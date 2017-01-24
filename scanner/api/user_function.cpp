#include "scanner/api/user_function.h"

namespace scanner {

void UserFunctionRegistry::add_user_function(
  const std::string &name,
  const FnPtr fn) {
  fns_.insert({name, fn});
}

bool UserFunctionRegistry::has_user_function(const std::string& name) {
  return fns_.count(name) > 0;
}

UserFunctionRegistry* get_user_function_registry() {
  static UserFunctionRegistry* registry = new UserFunctionRegistry;
  return registry;
}

namespace internal {

UserFunctionRegistration::UserFunctionRegistration(const std::string& name,
                                                   const FnPtr fn) {
  UserFunctionRegistry *registry =
    get_user_function_registry();
  registry->add_user_function(name, fn);
}

}
}
