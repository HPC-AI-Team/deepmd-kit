#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <type_traits>  // std::is_name
#include <iostream>
#include <string>
#include <cblas.h>
#include <omp.h>

using std::cout;
using std::endl;
using namespace tensorflow;
using namespace tensorflow::shape_inference;
using CPUDevice = Eigen::ThreadPoolDevice;


// C=(A^T)B

REGISTER_OP("MatmulTn")
    .Attr("T: {float, double}")
    .Input("a: T")
    .Input("b: T")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ShapeHandle a;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
      ShapeHandle b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));
      DimensionHandle output_rows = c->Dim(a, 1);
      DimensionHandle output_cols = c->Dim(b, 1);
      DimensionHandle inner_a = c->Dim(a, 0);
      DimensionHandle inner_b = c->Dim(b, 0);
      DimensionHandle merged;
      TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged));
      c->set_output(0, c->Matrix(output_rows, output_cols));
      return Status::OK();
    });

// double
static void GemmLauncher(const int m, const int n, const int k,
                  const double * A, const double * B, double * C)
{   
    double alpha = 1.;
    double beta = 0.;
    int lda = m;
    int ldb = n;
    int ldc = n;
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
        m,n,k,
        alpha,A,lda,
        B,ldb,
        beta,C,ldc);
}


// float 
static void GemmLauncher(const int m, const int n, const int k,
                  const float * A, const float * B, float * C)
{   
    float alpha = 1.;
    float beta = 0.;
    int lda = m;
    int ldb = n;
    int ldc = n;
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
        m,n,k,
        alpha,A,lda,
        B,ldb,
        beta,C,ldc);
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
class MatmulTnOp : public OpKernel {
  public :
    explicit MatmulTnOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& a = context->input(0);
        const Tensor& b = context->input(1);
        OP_REQUIRES (context, (a.shape().dims() == 2),	errors::InvalidArgument ("Dim of x should be 2"));
        OP_REQUIRES (context, (b.shape().dims() == 2),	errors::InvalidArgument ("Dim of w should be 2"));
        OP_REQUIRES (context, (a.shape().dim_size(0) == b.shape().dim_size(0)),	errors::InvalidArgument ("dimensions of x and w do not match!"));
        int m = a.shape().dim_size(1);
        int k = a.shape().dim_size(0);
        int n = b.shape().dim_size(1);
        Tensor * output = NULL;
        TensorShape output_shape;
        output_shape.AddDim(m);
        output_shape.AddDim(n);
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,&output));
        GemmLauncher(
                m, n, k,
                a.flat<T>().data(),
                b.flat<T>().data(),
                output->flat<T>().data());
    }
    ~MatmulTnOp () {}
  private :
};

#define REGISTER_CPU(T)                                                         \
    /* Declare explicit instantiations in kernel_example.cu.cc. */              \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("MatmulTn").Device(DEVICE_CPU).TypeConstraint<T>("T"),            \
        MatmulTnOp<CPUDevice, T>);                                          

// REGISTER_GPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);