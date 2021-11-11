#include "custom_op.h"
#include "fast_tanh.h"


REGISTER_OP("FastTanh")
    .Attr("T: {float, double}")
    .Input("x: T")
    .Output("y: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

template <typename Device, typename T>
class FastTanhOp : public OpKernel {
 public:
  explicit FastTanhOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    DeviceFunctor() (device,context->eigen_device<Device>());
    const Tensor& input_tensor = context->input(0);
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    
    if(device == "CPU"){
        deepmd::fast_tanh_cpu(
            input_tensor.flat<T>().data(),
            output_tensor->flat<T>().data(),
            input_tensor.flat<T>().size());
    }else if(device == "GPU"){
#if GOOGLE_CUDA
        deepmd::fast_tanh_cuda(
            input_tensor.flat<T>().data(),
            output_tensor->flat<T>().data(),
            input_tensor.flat<T>().size());
#endif
    }

  }
  private:
    std::string device;
  
};

#define REGISTER_CPU(T)                                                         \
    /* Declare explicit instantiations in kernel_example.cu.cc. */              \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("FastTanh").Device(DEVICE_CPU).TypeConstraint<T>("T"),             \
        FastTanhOp<CPUDevice, T>);                                          

// REGISTER_GPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);