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

// double
static void GemmLauncher(const int m, const int n, const int k,
                  const double alpha, const double beta,
                  const double * A, const double * B, double * C)
{   
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,C,n);
}
// float 
static void GemmLauncher(const int m, const int n, const int k,
                  const float alpha, const float beta,
                  const float * A, const float * B, float * C)
{   
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
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
        double start = omp_get_wtime();
        // cout <<  "gemm layer start\n";

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

        // cout << "(m,k,n):" << "(" << m << "," << k << "," << n << ")" << endl;

        // Create an output tensor 
        // No more output tensors, output tensor is C
        Tensor * output = NULL;
        TensorShape output_shape;
        output_shape.AddDim(m);
        output_shape.AddDim(n);
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,&output));
        auto output_matrix = output->matrix<T> ();
        auto b_vector = b.flat<T>();
        for(int i = 0;i<m;i++){
          for(int j = 0;j<n;j++){
            output_matrix(i,j) = b_vector(j);
          }
        }
        GemmLauncher(
                m, n, k,
                1., 1.,
                x.flat<T>().data(),
                w.flat<T>().data(),
                output->flat<T>().data());

        // cout <<  "gemm layer end\n";
        double end = omp_get_wtime();
        double time = end - start;
        // cout << "(m,k,n):" << "(" << m << "," << k << "," << n << ")" << " time : " << time << " gflops : " << 2.*m*n*k/time/1e9 << endl;
    }
    ~GemmLayerOp () {}
  private :
};

#define REGISTER_CPU(T)                                                         \
    /* Declare explicit instantiations in kernel_example.cu.cc. */              \
    REGISTER_KERNEL_BUILDER(                                                    \
        Name("GemmLayer").Device(DEVICE_CPU).TypeConstraint<T>("T"),            \
        GemmLayerOp<CPUDevice, T>);                                          

// REGISTER_GPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);