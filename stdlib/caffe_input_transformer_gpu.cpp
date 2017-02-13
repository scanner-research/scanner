#define HALIDE_USE_GPU
#include "caffe_input_transformer_base.h"
Halide::RegisterGenerator<CaffeInputTransformer> register_me{
    "caffe_input_transformer_gpu"};
