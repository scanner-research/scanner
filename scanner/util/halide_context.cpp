#include "scanner/util/cuda.h"

#ifdef HAVE_CUDA
namespace Halide {
namespace Runtime {
namespace Internal {
namespace Cuda {
CUcontext context = 0;
}
}
}
}
#endif
