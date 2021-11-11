#include "custom_op.h"
#include "gemm.h"


REGISTER_OP("GemmLayer")
    .Attr("T: {float, double}")
    .Input("x: T")
    .Input("w: T")
    .Input("b: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle a;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
      ShapeHandle b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));
      DimensionHandle output_rows = c->Dim(a, 0);
      DimensionHandle output_cols = c->Dim(b, 1);
      DimensionHandle inner_a = c->Dim(a, 1);
      DimensionHandle inner_b = c->Dim(b, 0);
      DimensionHandle merged;
      TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged));
      c->set_output(0, c->Matrix(output_rows, output_cols));
      return Status::OK();
    });


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class GemmLayerOp : public OpKernel {
  public :
    explicit GemmLayerOp(OpKernelConstruction* context) : OpKernel(context) {
    }

    void Compute(OpKernelContext* context) override {
      DeviceFunctor() (device,context->eigen_device<Device>());
      // Grab the input tensor
      const Tensor& x = context->input(0);
      const Tensor& w = context->input(1);
      const Tensor& b = context->input(2);
      OP_REQUIRES (context, (x.shape().dims() == 2),	errors::InvalidArgument ("Dim of x should be 2"));
      OP_REQUIRES (context, (w.shape().dims() == 2),	errors::InvalidArgument ("Dim of w should be 2"));
      OP_REQUIRES (context, ((b.shape().dims() == 2 && b.shape().dim_size(0) == 1 && b.shape().dim_size(1) == w.shape().dim_size(1)) 
        || (b.shape().dims() == 1 && b.shape().dim_size(0) == w.shape().dim_size(1))),	errors::InvalidArgument ("Dim of b should be [1,n] or [n]"));
      OP_REQUIRES (context, (x.shape().dim_size(1) == w.shape().dim_size(0)),	errors::InvalidArgument ("dimensions of x and w do not match!"));
      int m = x.shape().dim_size(0);
      int k = x.shape().dim_size(1);
      int n = w.shape().dim_size(1);

      // Create an output tensor 
      // No more output tensors, output tensor is C
      Tensor * output = NULL;
      TensorShape output_shape;
      output_shape.AddDim(m);
      output_shape.AddDim(n);
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,&output));
      
      if(device == "CPU"){
        deepmd::gemm(
                m, n, k,
                x.flat<T>().data(),
                w.flat<T>().data(),
                b.flat<T>().data(),
                output->flat<T>().data());
      }else if(device == "GPU"){
#if GOOGLE_CUDA
        deepmd::gemm_cuda(
                m, n, k,
                x.flat<T>().data(),
                w.flat<T>().data(),
                b.flat<T>().data(),
                output->flat<T>().data());
#endif
      }
    }
    ~GemmLayerOp () {
    }
  private :
    std::string device;

};

#define REGISTER_CPU(T)                                                         \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("GemmLayer").Device(DEVICE_CPU).TypeConstraint<T>("T"),            \
        GemmLayerOp<CPUDevice, T>);                                          

// REGISTER_GPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);

#if  GOOGLE_CUDA
#define REGISTER_GPU(T)                                                         \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("GemmLayer").Device(DEVICE_GPU).TypeConstraint<T>("T"),            \
        GemmLayerOp<GPUDevice, T>);    
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif