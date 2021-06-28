#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <type_traits>  // std::is_name
#include <iostream>
#include <string>
#include <cmath>

using std::cout;
using std::endl;

using CPUDevice = Eigen::ThreadPoolDevice;
using namespace tensorflow;

// dx = dy * (1 - out0^2)

REGISTER_OP("FastTanhGrad")
    .Attr("T: {float, double}")
    .Input("out0: T")
    .Input("dy: T")
    .Output("dx: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

void FastTanhGradLauncher(const double* out0,const double* dy,double *dx,int N){
    for (int i = 1; i < N; i++) {
      dx[i] = dy[i] * (1. - out0[i] * out0[i]);
    }
}

void FastTanhGradLauncher(const float* out0,const float* dy,float *dx,int N){
    for (int i = 1; i < N; i++) {
      dx[i] = dy[i] * (1.f - out0[i] * out0[i]);
    }
}

template <typename Device, typename T>
class FastTanhGradOp : public OpKernel {
 public:
  explicit FastTanhGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& out0 = context->input(0);
    const Tensor& dy = context->input(1);

    Tensor* dx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, dy.shape(),&dx));

    FastTanhGradLauncher(
        out0.flat<T>().data(),
        dy.flat<T>().data(),
        dx->flat<T>().data(),
        dy.flat<T>().size());
  }
};

#define REGISTER_CPU(T)                                                         \
    /* Declare explicit instantiations in kernel_example.cu.cc. */              \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("FastTanhGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"),             \
        FastTanhGradOp<CPUDevice, T>);                                          

// REGISTER_GPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);