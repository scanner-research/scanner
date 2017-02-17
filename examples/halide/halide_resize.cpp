#include "Halide.h"

using namespace Halide;

// Resize code taken from
// https://github.com/halide/Halide/blob/master/apps/resize/resize.cpp

Expr kernel_box(Expr x) {
  Expr xx = abs(x);
  return select(xx <= 0.5f, 1.0f, 0.0f);
}

class HalideResizeKernel : public Halide::Generator<HalideResizeKernel> {
public:
  ImageParam input{UInt(8), 3, "input"};
  Param<int> input_width{"input_width"}, input_height{"input_height"};
  Param<int> target_width{"target_width"}, target_height{"target_height"};

  Func build() {
    Var x("x"), y("y"), c("c"), k("k");

    Func clamped = BoundaryConditions::repeat_edge(input);

    Func scaled("scaled");
    scaled(x, y, c) = cast<float>(clamped(x, y, c));

    Expr scaleX = target_width / cast<float>(input_width);
    Expr scaleY = target_height / cast<float>(input_height);

    Expr kernelSizeX = 0.5f / scaleX;
    Expr kernelSizeY = 0.5f / scaleY;

    Expr sourcex = (x + 0.5f) / scaleX;
    Expr sourcey = (y + 0.5f) / scaleY;

    Func kernelx("kernelx"), kernely("kernely");
    Expr beginx = cast<int>(sourcex - kernelSizeX + 0.5f);
    Expr beginy = cast<int>(sourcey - kernelSizeY + 0.5f);
    RDom domx(0, 2.0f * kernelSizeX + 1, "domx");
    RDom domy(0, 2.0f * kernelSizeY + 1, "domy");
    {
      Func kx, ky;
      kx(x, k) = kernel_box((k + beginx - sourcex) * scaleX);
      ky(y, k) = kernel_box((k + beginy - sourcey) * scaleY);
      kernelx(x, k) = kx(x, k) / sum(kx(x, domx));
      kernely(y, k) = ky(y, k) / sum(ky(y, domy));
    }

    Func resized_x("resized_x");
    Func resized_y("resized_y");
    resized_x(x, y, c) = sum(kernelx(x, domx) * scaled(domx + beginx, y, c));
    resized_y(x, y, c) = sum(kernely(y, domy) * resized_x(x, domy + beginy, c));

    Func resized_final("resized_final");
    resized_final(x, y, c) = clamp(resized_y(x, y, c), 0.0f, 255.0f);
    resized_final.bound(c, 0, 3);

    input
      .dim(0).set_stride(3)
      .dim(2).set_stride(1);

    Target target = Halide::get_target_from_environment();
    target.set_feature(Target::CUDA);
    target.set_feature(Target::CUDACapability50);
    resized_x.compute_root().reorder(c, x, y).unroll(c).gpu_tile(x, y, 8, 8);
    resized_final.reorder(c, x, y).unroll(c).gpu_tile(x, y, 8, 8);

    return resized_final;
  }
};

Halide::RegisterGenerator<HalideResizeKernel> register_me{
    "halide_resize"};
