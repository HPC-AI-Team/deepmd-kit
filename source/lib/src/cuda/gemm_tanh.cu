#include "gpu_cuda.h"
#include "gemm_tanh.h"
#include <iostream>
#include "cutlass/gemm/device/gemm.h"

void deepmd::gemm_tanh_cuda(
    const int m, const int n, const int k,
    const double * A, const double * B,const double * C, double * D)
{
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<
        double,        // Data-type of A matrix
        ColumnMajor,  // Layout of A matrix
        double,        // Data-type of B matrix
        ColumnMajor,  // Layout of B matrix
        double,        // Data-type of C matrix
        ColumnMajor,
        double,
        cutlass::arch::OpClassSimtTanh>; // Layout of C matrix
    CutlassGemm gemm_operator;
    double alpha = 1.;
    double beta = 1.;
    CutlassGemm::Arguments args({n,m,k},     // Gemm Problem dimensions
                                {B, n},    // Tensor-ref for source matrix A
                                {A, k},    // Tensor-ref for source matrix B
                                {C, 0},      // Tensor-ref for source matrix C
                                {D, n},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue
    cutlass::Status status = gemm_operator(args);
}

void deepmd::gemm_tanh_cuda(
    const int m, const int n, const int k,
    const float * A, const float * B,const float * C, float * D)
{   
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<
        float,        // Data-type of A matrix
        ColumnMajor,  // Layout of A matrix
        float,        // Data-type of B matrix
        ColumnMajor,  // Layout of B matrix
        float,        // Data-type of C matrix
        ColumnMajor,
        float,
        cutlass::arch::OpClassSimtTanh>; // Layout of C matrix
    CutlassGemm gemm_operator;
    float alpha = 1.;
    float beta = 1.;
    CutlassGemm::Arguments args({n,m,k},     // Gemm Problem dimensions
                                {B, n},    // Tensor-ref for source matrix A
                                {A, k},    // Tensor-ref for source matrix B
                                {C, 0},      // Tensor-ref for source matrix C
                                {D, n},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue
    cutlass::Status status = gemm_operator(args);
}