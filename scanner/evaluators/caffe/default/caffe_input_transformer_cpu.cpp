#define HALIDE_USE_CPU
#include "caffe_input_transformer_base.h"
Halide::RegisterGenerator<CaffeInputTransformer> register_me{"caffe_input_transformer_cpu"};
