#include "gemm.h"
#ifdef __cplusplus
extern "C"
{
#endif
#include <cblas.h>
#ifdef __cplusplus
}
#endif
#include <cstring>

// D = AB+C
// A (m, k)
// B (k, n)
// C (   n)
// D (m, n)
// double
void deepmd::gemm(const int m, const int n, const int k,
                          const double * A, const double * B,const double * C, double * D)
{  
  double alpha = 1.;
  double beta = 1.;
  for(int i = 0; i < m; i++){
    std::memcpy(D + i * n, C, n * sizeof(double));
  }
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,D,n);
}
// float 
void deepmd::gemm(const int m, const int n, const int k,
                          const float * A, const float * B,const float * C , float * D)
{   
  float alpha = 1.f;
  float beta = 1.f;
  for(int i = 0; i < m; i++){
    std::memcpy(D + i * n, C, n * sizeof(float));
  }
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,k,
    B,n,
    beta,D,n);
}
