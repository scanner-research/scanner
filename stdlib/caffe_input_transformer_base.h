#include "Halide.h"

#define HAS_AUTOSCHEDULER

using namespace Halide;

// Resize code taken from
// https://github.com/halide/Halide/blob/master/apps/resize/resize.cpp
Expr kernel_box(Expr x) {
  Expr xx = abs(x);
  return select(xx <= 0.5f, 1.0f, 0.0f);
}

Expr sinc(Expr x) {
  return sin(float(M_PI) * x) / x;
}

Expr kernel_lanczos(Expr x) {
  Expr value = sinc(x) * sinc(x/3);
  value = select(x == 0.0f, 1.0f, value); // Take care of singularity at zero
  value = select(x > 3 || x < -3, 0.0f, value); // Clamp to zero out of bounds
  return value;
}

struct KernelInfo {
  const char *name;
  float size;
  Expr (*kernel)(Expr);
};

static KernelInfo kernelInfo[] = {
  { "box", 0.5f, kernel_box },
  // { "linear", 1.0f, kernel_linear },
  // { "cubic", 2.0f, kernel_cubic },
  { "lanczos", 3.0f, kernel_lanczos }
};

class CaffeInputTransformer : public Halide::Generator<CaffeInputTransformer> {
public:
  ImageParam input{UInt(8), 3, "input"};
  Param<int> input_width{"input_width"}, input_height{"input_height"};
  Param<int> target_width{"target_width"}, target_height{"target_height"};
  Param<bool> normalize{"normalize"};
  Param<float> mean_r{"mean_r"}, mean_g{"mean_g"}, mean_b{"mean_b"};

  Func build() {
    Var x("x"), y("y"), c("c"), k("k");

    Func clamped = BoundaryConditions::repeat_edge(input);

    Func scaled("scaled");
    scaled(x, y, c) = cast<float>(clamped(x, y, c));

    Expr scaleX = target_width / cast<float>(input_width);
    Expr scaleY = target_height / cast<float>(input_height);

    const KernelInfo &info = kernelInfo[0];
    Expr kernelSizeX = info.size / scaleX;
    Expr kernelSizeY = info.size / scaleY;

    Expr sourcex = (x + 0.5f) / scaleX;
    Expr sourcey = (y + 0.5f) / scaleY;

    Func kernelx("kernelx"), kernely("kernely");
    Expr beginx = cast<int>(sourcex - kernelSizeX + 0.5f);
    Expr beginy = cast<int>(sourcey - kernelSizeY + 0.5f);
    RDom domx(0, 2.0f * kernelSizeX + 1, "domx");
    RDom domy(0, 2.0f * kernelSizeY + 1, "domy");
    {
      Func kx, ky;
      kx(x, k) = info.kernel((k + beginx - sourcex) * scaleX);
      ky(y, k) = info.kernel((k + beginy - sourcey) * scaleY);
      kernelx(x, k) = kx(x, k) / sum(kx(x, domx));
      kernely(y, k) = ky(y, k) / sum(ky(y, domy));
    }

    Func resized_x("resized_x");
    Func resized_y("resized_y");
    resized_x(x, y, c) = sum(kernelx(x, domx) * scaled(domx + beginx, y, c));
    resized_y(x, y, c) = sum(kernely(y, domy) * resized_x(x, domy + beginy, c));

    Func resized_final("resized_final");
    resized_final(x, y, c) = clamp(resized_y(x, y, c), 0.0f, 255.0f);

    Func mean_subtract("mean_subtract");
    mean_subtract(x, y, c) = resized_final(x, y, c) -
      select(c==0, mean_r,
             select(c==1, mean_g, mean_b));

    Func rescaled("rescaled");
    rescaled(x, y, c) = mean_subtract(x, y, 2-c) / select(normalize, 255.0f, 1.0f);
    rescaled.bound(c, 0, 3);

    input
      .dim(0).set_stride(3)
      .dim(2).set_stride(1);

    Target target = Halide::get_target_from_environment();
#ifdef HALIDE_USE_GPU
    target.set_feature(Target::CUDA);
    target.set_feature(Target::CUDACapability50);
    resized_x.compute_root().reorder(c, x, y).unroll(c).gpu_tile(x, y, 8, 8);
    rescaled.reorder(c, x, y).unroll(c).gpu_tile(x, y, 8, 8);
#else
    rescaled.reorder(x, c, y).parallel(y).vectorize(x, 8);
    resized_x.compute_at(rescaled, y).vectorize(x, 8);
    resized_y.compute_at(rescaled, y).vectorize(x, 8);
#endif

    return rescaled;
  }
};
