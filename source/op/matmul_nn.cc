#include "custom_op.h"
#include "matmul.h"


// C=AB
REGISTER_OP("MatmulNn")
    .Attr("T: {float, double}")
    .Input("a: T")
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


template <typename Device, typename T>
class MatmulNnOp : public OpKernel {
  public :
    explicit MatmulNnOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      DeviceFunctor() (device,context->eigen_device<Device>());
      // Grab the input tensor
      const Tensor& a = context->input(0);
      const Tensor& b = context->input(1);
      OP_REQUIRES (context, (a.shape().dims() == 2),	errors::InvalidArgument ("Dim of x should be 2"));
      OP_REQUIRES (context, (b.shape().dims() == 2),	errors::InvalidArgument ("Dim of w should be 2"));
      OP_REQUIRES (context, (a.shape().dim_size(1) == b.shape().dim_size(0)),	errors::InvalidArgument ("dimensions of x and w do not match!"));
      int m = a.shape().dim_size(0);
      int k = a.shape().dim_size(1);
      int n = b.shape().dim_size(1);
      Tensor * output = NULL;
      TensorShape output_shape;
      output_shape.AddDim(m);
      output_shape.AddDim(n);
      OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,&output));

      if(device == "CPU"){
        deepmd::matmul_nn_row_launcer(
                m, n, k,
                a.flat<T>().data(),
                b.flat<T>().data(),
                output->flat<T>().data());
      }else if(device == "GPU"){
#if GOOGLE_CUDA
        deepmd::matmul_nn_row_launcer_cuda(
                m, n, k,
                a.flat<T>().data(),
                b.flat<T>().data(),
                output->flat<T>().data());
#endif
      }
    
    }
    ~MatmulNnOp () {}
  private :
  
  std::string device;
  
};

#define REGISTER_CPU(T)                                                         \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("MatmulNn").Device(DEVICE_CPU).TypeConstraint<T>("T"),             \
        MatmulNnOp<CPUDevice, T>);                                          
// REGISTER_GPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);

#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                         \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("MatmulNn").Device(DEVICE_GPU).TypeConstraint<T>("T"),             \
        MatmulNnOp<GPUDevice, T>);   
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif 