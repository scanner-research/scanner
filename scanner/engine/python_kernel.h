#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/metadata.pb.h"

namespace scanner {

class PythonKernel : public BatchedKernel {
 public:
  PythonKernel(const KernelConfig& config,
               const std::string& op_name,
               const std::string& kernel_str,
               const std::string& pickled_config,
               const int preferred_batch = 1);

  ~PythonKernel();

  void new_stream(const std::vector<u8>& args) override;

  void execute(const BatchedElements& input_columns,
               BatchedElements& output_columns) override;

  void reset() override;

 private:
  KernelConfig config_;
  DeviceHandle device_;
  bool can_batch_;
  std::string op_name_;
  std::string kernel_name_;
};

}
