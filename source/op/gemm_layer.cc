#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <type_traits>  // std::is_name
#include <iostream>
#include <string>
#include <mkl.h>

// REGISTER_OP("GemmLayer")
//     .Attr("T: {half, float, double}")
//     .Input("xyz_scatter: T")
//     .Input("w: T")
//     .Input("b: T")
//     .Output("output: T");

using std::cout;
using std::endl;

REGISTER_OP("GemmLayer")
    .Attr("T: {float, double}")
    .Input("xyz_scatter: T")
    .Input("w: T")
    .Input("b: T")
    .Output("output: T");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

// double
void GemmLauncher(const int m, const int n, const int k,
                const double alpha, const double beta,
                const double * A, const double * B, double * C)
{   
  
  cblas_dgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,C,n);
}

// float 
void GemmLauncher(const int m, const int n, const int k,
                const float alpha, const float beta,
                const float * A, const float * B, float * C)
{   
  cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,C,n);
}

// half
// void GemmLauncher(const int m, consst int n, const int k,
//     const float alpha, const float beta,
//     const Eigen::half * A, const Eigen::half * B, 
//     const Eigen::half * C, Eigen::half * D) 
// {
// }

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class GemmLayerOp : public OpKernel {
  public :
    explicit GemmLayerOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // cout <<  "gemm layer start\n";

        // Grab the input tensor
        const Tensor& xyz_scatter = context->input(0);
        const Tensor& w = context->input(1);
        const Tensor& b = context->input(2);
        
        OP_REQUIRES (context, (xyz_scatter.shape().dims() == 2),	errors::InvalidArgument ("Dim of xyz_scatter should be 2"));
        OP_REQUIRES (context, (w.shape().dims() == 2),	errors::InvalidArgument ("Dim of w should be 2"));
        OP_REQUIRES (context, (b.shape().dims() == 2),	errors::InvalidArgument ("Dim of b should be 2"));
        OP_REQUIRES (context, (xyz_scatter.shape().dim_size(1) == w.shape().dim_size(0)),	errors::InvalidArgument ("dimensions of xyz_scatter and w do not match!"));

        int m = xyz_scatter.shape().dim_size(0);
        int k = xyz_scatter.shape().dim_size(1);
        int n = w.shape().dim_size(1);

        // cout << "(m,k,n):" << "(" << m << "," << k << "," << n << ")" << endl;
        // cout << "b.shape:" << "(" << b.shape().dim_size(0) << "," << b.shape().dim_size(1) << ")" << endl;

        // Create an output tensor 
        // No more output tensors, output tensor is C
        Tensor * output = NULL;
        TensorShape output_shape;
        output_shape.AddDim(m);
        output_shape.AddDim(n);

        int context_output_index = 0;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
					    output_shape,
					    &output));


        auto output_matrix = output->matrix<T> ();
        auto b_matrix = b.matrix<T> ();
        for(int i = 0;i<m;i++){
          for(int j = 0;j<n;j++){
            output_matrix(i,j) = b_matrix(0,j);
          }
        }

        GemmLauncher(
                m, n, k,
                1., 1.,
                xyz_scatter.flat<T>().data(),
                w.flat<T>().data(),
                output->flat<T>().data());
        
        // cout <<  "gemm layer end\n";
    }
    ~GemmLayerOp () {}
  private :
};


#define REGISTER_CPU(T)                                                         \
    /* Declare explicit instantiations in kernel_example.cu.cc. */              \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("GemmLayer").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
        GemmLayerOp<CPUDevice, T>);                                          

// REGISTER_GPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);
