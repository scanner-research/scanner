#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "scanner/metadata.pb.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace scanner {

class PythonKernel : public BatchedKernel {
 public:
  PythonKernel(const KernelConfig& config, const std::string& kernel_str,
               const std::string& pickled_config);

  ~PythonKernel();

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override;

 private:
  KernelConfig config_;
  DeviceHandle device_;
};

}
