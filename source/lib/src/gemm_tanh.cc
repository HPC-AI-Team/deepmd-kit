#ifdef __cplusplus
extern "C"
{
#endif
   #include <cblas.h>
#ifdef __cplusplus
}
#endif
#include <cstring>
#include "gemm_tanh.h"

void deepmd::gemm_tanh(
  const int m, const int n, const int k,
  const double * A, const double * B,const double * C, double * D)
{  
  double alpha = 1.;
  double beta = 0.;
  // for(int i = 0; i < m; i++){
  //   std::memcpy(D + i * n, C, n * sizeof(double));
  // }
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,D,n);
  for(int i = 0; i < m; ++i){
    double* DD = D + i * n; 
    for(int j = 0; j < n; ++j){
      DD[j] += C[j];
      poly_2_tanh_float(DD[j],DD[j]);
    }
  }
  
}

void deepmd::gemm_tanh(
  const int m, const int n, const int k,
  const float * A, const float * B,const float * C , float * D)
{   
  float alpha = 1.f;
  float beta = 0.f;
  // for(int i = 0; i < m; i++){
  //   std::memcpy(D + i * n, C, n * sizeof(float));
  // }
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,D,n);
  for(int i = 0; i < m; ++i){
    float* DD = D + i * n; 
    for(int j = 0; j < n; ++j){
      DD[j] += C[j];
      poly_2_tanh_float(DD[j],DD[j]);
    }
  }
}
