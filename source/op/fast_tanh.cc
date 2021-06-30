#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <type_traits>  // std::is_name
#include <iostream>
#include <string>
#include <cmath>
// #include "fast_tanh_taylor_2.h"
#include "fast_tanh_poly_2.h"

using std::cout;
using std::endl;

using CPUDevice = Eigen::ThreadPoolDevice;
using namespace tensorflow;

REGISTER_OP("FastTanh")
    .Attr("T: {float, double}")
    .Input("x: T")
    .Output("y: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

void FastTanhLauncher(const double* input,double *output,int N){
    // for (int i = 1; i < N; i++) {
    //   output[i] = tanh(input[i]);
    // }
    // taylor_2_tanh_double(input,output,N);
    poly_2_tanh_double_vector(input,output,N);
// #pragma omp parallel for 
//     for(int i = 0;i<N ;i++){
//         poly_2_tanh_float(input[i],output[i]);
//     }
}

void FastTanhLauncher(const float* input,float *output,int N){
    // for (int i = 1; i < N; i++) {
    //   output[i] = tanhf(input[i]);
    // }
    // taylor_2_tanh_float(input,output,N);
    poly_2_tanh_float_vector(input,output,N);
// #pragma omp parallel for 
//     for(int i = 0;i<N ;i++){
//         poly_2_tanh_float(input[i],output[i]);
//     }
}

template <typename Device, typename T>
class FastTanhOp : public OpKernel {
 public:
  explicit FastTanhOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    FastTanhLauncher(
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data(),
        input_tensor.flat<T>().size());
  }
};

#define REGISTER_CPU(T)                                                         \
    /* Declare explicit instantiations in kernel_example.cu.cc. */              \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("FastTanh").Device(DEVICE_CPU).TypeConstraint<T>("T"),             \
        FastTanhOp<CPUDevice, T>);                                          

// REGISTER_GPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);