#include "gpu_cuda.h"
#include "matmul.h"
#include <iostream>
#include "cutlass/gemm/device/gemm.h"
#include "cublas_v2.h"

// ----------------------------------------------------------------------------------------------
void deepmd::matmul_nn_col_launcer_cuda( 
    const int m, const int n, const int k, const double * A, const double * B, double * C){
    double alpha = 1.;
    double beta = 0.;
    int lda = m;
    int ldb = k;
    int ldc = m;
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<double,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  double,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  double,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix
    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({m,n,k},     // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {NULL, 0},      // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue
    cutlass::Status status = gemm_operator(args);
}

void deepmd::matmul_nn_col_launcer_cuda( 
    const int m, const int n, const int k, const float * A, const float * B, float * C){
    float alpha = 1.;
    float beta = 0.;
    int lda = m;
    int ldb = k;
    int ldc = m;
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                    ColumnMajor,  // Layout of A matrix
                                                    float,        // Data-type of B matrix
                                                    ColumnMajor,  // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix
    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({m,n,k},     // Gemm Problem dimensions
                                {A, lda},    // Tensor-ref for source matrix A
                                {B, ldb},    // Tensor-ref for source matrix B
                                {NULL, 0},      // Tensor-ref for source matrix C
                                {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue
    cutlass::Status status = gemm_operator(args);
}

void deepmd::matmul_nn_row_launcer_cuda(
    const int m, const int n, const int k,
    const double * A, const double * B, double * C)
{
    matmul_nn_col_launcer_cuda(n,m,k,B,A,C);
}

void deepmd::matmul_nn_row_launcer_cuda(
    const int m, const int n, const int k,
    const float * A, const float * B, float * C)
{
    matmul_nn_col_launcer_cuda(n,m,k,B,A,C);
}

// ----------------------------------------------------------------------------------------------

void deepmd::matmul_nt_col_launcer_cuda(
    const int m, const int n, const int k,
    const double * A, const double * B, double * C)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasCreate error!!!" << std::endl;
        exit(-1);
    }
    double alpha = 1.;
    double beta = 0.;
    int lda = m;
    int ldb = n;
    int ldc = m;
    stat = cublasDgemm( handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,   
                        m, n, k,
                        &alpha,
                        A, lda,
                        B, ldb,
                        &beta,
                        C, ldc);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasDgemm error!!!" << std::endl;
        exit(-1);
    }
    cublasDestroy(handle);
}

void deepmd::matmul_nt_col_launcer_cuda(
    const int m, const int n, const int k,
    const float * A, const float * B, float * C)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasCreate error!!!" << std::endl;
        exit(-1);
    }
    float alpha = 1.;
    float beta = 0.;
    int lda = m;
    int ldb = n;
    int ldc = m;
    stat = cublasSgemm( handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,   
                        m, n, k,
                        &alpha,
                        A, lda,
                        B, ldb,
                        &beta,
                        C, ldc);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasDgemm error!!!" << std::endl;
        exit(-1);
    }
    cublasDestroy(handle);
}

void deepmd::matmul_nt_row_launcer_cuda(
    const int m, const int n, const int k, 
    const double * A, const double * B, double * C)
{
    matmul_tn_col_launcer_cuda(n,m,k,B,A,C);
}

void deepmd::matmul_nt_row_launcer_cuda(
    const int m, const int n, const int k, 
    const float * A, const float * B, float * C)
{
    matmul_tn_col_launcer_cuda(n,m,k,B,A,C);
}

// ----------------------------------------------------------------------------------------------

void deepmd::matmul_tn_col_launcer_cuda(
    const int m, const int n, const int k,
    const double * A, const double * B, double * C)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasCreate error!!!" << std::endl;
        exit(-1);
    }
    double alpha = 1.;
    double beta = 0.;
    int lda = k;
    int ldb = k;
    int ldc = m;
    stat = cublasDgemm( handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,   
                        m, n, k,
                        &alpha,
                        A, lda,
                        B, ldb,
                        &beta,
                        C, ldc);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasDgemm error!!!" << std::endl;
        exit(-1);
    }
    cublasDestroy(handle);
}

void deepmd::matmul_tn_col_launcer_cuda(
    const int m, const int n, const int k,
    const float * A, const float * B, float * C)
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasCreate error!!!" << std::endl;
        exit(-1);
    }
    float alpha = 1.;
    float beta = 0.;
    int lda = k;
    int ldb = k;
    int ldc = m;
    stat = cublasSgemm( handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,   
                        m, n, k,
                        &alpha,
                        A, lda,
                        B, ldb,
                        &beta,
                        C, ldc);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasDgemm error!!!" << std::endl;
        exit(-1);
    }
    cublasDestroy(handle);
}

void deepmd::matmul_tn_row_launcer_cuda(
    const int m, const int n, const int k, 
    const double * A, const double * B, double * C)
{
    matmul_nt_col_launcer_cuda(n,m,k,B,A,C);
}

void deepmd::matmul_tn_row_launcer_cuda(
    const int m, const int n, const int k, 
    const float * A, const float * B, float * C)
{
    matmul_nt_col_launcer_cuda(n,m,k,B,A,C);
}