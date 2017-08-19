#include "scanner/api/kernel.h"
#include "scanner/api/op.h"
#include "scanner/util/memory.h"
#include "stdlib/stdlib.pb.h"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace scanner {

class PythonKernel : public BatchedKernel {
 public:
  PythonKernel(const KernelConfig& config, const std::string& kernel_str);

  ~PythonKernel();

  void execute(const BatchedColumns& input_columns,
               BatchedColumns& output_columns) override;

 private:
  KernelConfig config_;
  DeviceHandle device_;
  proto::PythonArgs args_;
};

}
