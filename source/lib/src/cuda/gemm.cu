#include "gpu_cuda.h"
#include "cublas_v2.h"
#include "gemm.h"
#include <iostream>

void deepmd::gemm_launcer_cuda(
    const int m, const int n, const int k,
    const double * A, const double * B,const double * C, double * D)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasCreate error!!!" << std::endl;
        exit(-1);
    }
    double alpha = 1.;
    double beta = 1.;
    for(int i = 0; i < m; i++){
        cudaStat = cudaMemcpy(D + i * n, C, n * sizeof(double), cudaMemcpyDeviceToDevice);
        if(cudaStat != cudaSuccess){
            std::cerr << "cudaMemcpy error!!!" << std::endl;
            exit(-1);
        }
    }
    stat = cublasDgemm( handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,   
                        n, m, k,
                        &alpha,
                        B, n,
                        A, k,
                        &beta,
                        D, n);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasDgemm error!!!" << std::endl;
        exit(-1);
    }
    cublasDestroy(handle);
}

void deepmd::gemm_launcer_cuda(
    const int m, const int n, const int k,
    const float * A, const float * B,const float * C, float * D)
{   
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasCreate error!!!" << std::endl;
        exit(-1);
    }
    float alpha = 1.f;
    float beta = 1.f;
    for(int i = 0; i < m; i++){
        cudaStat = cudaMemcpy(D + i * n, C, n * sizeof(float), cudaMemcpyDeviceToDevice);
        if(cudaStat != cudaSuccess){
            std::cerr << "cudaMemcpy error!!!" << std::endl;
            exit(-1);
        }
    }
    stat = cublasSgemm( handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, m, k,
                        &alpha,
                        B, n,
                        A, k,
                        &beta,
                        D, n);
    if(stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasDgemm error!!!" << std::endl;
        exit(-1);
    }
    cublasDestroy(handle);
}