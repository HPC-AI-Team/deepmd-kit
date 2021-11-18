#include "matmul.h"
#ifdef __cplusplus
extern "C"
{
#endif
#include <cblas.h>
#ifdef __cplusplus
}
#endif
#include <cstring>

// ----------------------------------------------------------------------------------------------
// D = AB
// A (m, k)
// B (k, n)
// C (m, n)
// double

void deepmd::matmul_nn_row_launcer(
  const int m, const int n, const int k,
  const double * A, const double * B, double * C)
{  
  double alpha = 1.;
  double beta = 0.;
  int lda=k;
  int ldb=n;
  int ldc=n;
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,lda,
    B,ldb,
    beta,C,ldc);
}
// float 
void deepmd::matmul_nn_row_launcer(
  const int m, const int n, const int k,
  const float * A, const float * B, float * C)
{   
  float alpha = 1.f;
  float beta = 0.f;
  int lda=k;
  int ldb=n;
  int ldc=n;
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
    m,n,k,
    alpha,A,lda,
    B,ldb,
    beta,C,ldc);
}

// ----------------------------------------------------------------------------------------------

void deepmd::matmul_nt_row_launcer(
  const int m, const int n, const int k,
  const double * A, const double * B, double * C)
{  
    double alpha = 1.;
    double beta = 0.;
    int lda=k;
    int ldb=k;
    int ldc=n;
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
        m,n,k,
        alpha,A,lda,
        B,ldb,
        beta,C,ldc);
}
// float 
void deepmd::matmul_nt_row_launcer(
  const int m, const int n, const int k,
  const float * A, const float * B, float * C)
{   
    float alpha = 1.;
    float beta = 0.;
    int lda=k;
    int ldb=k;
    int ldc=n;
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
        m,n,k,
        alpha,A,lda,
        B,ldb,
        beta,C,ldc);
}

// ----------------------------------------------------------------------------------------------

void deepmd::matmul_tn_row_launcer(
  const int m, const int n, const int k,
  const double * A, const double * B, double * C)
{  
    double alpha = 1.;
    double beta = 0.;
    int lda=m;
    int ldb=n;
    int ldc=n;
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
        m,n,k,
        alpha,A,lda,
        B,ldb,
        beta,C,ldc);
}
// float 
void deepmd::matmul_tn_row_launcer(
  const int m, const int n, const int k,
  const float * A, const float * B, float * C)
{   
    float alpha = 1.;
    float beta = 0.;
    int lda=m;
    int ldb=n;
    int ldc=n;
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,
        m,n,k,
        alpha,A,lda,
        B,ldb,
        beta,C,ldc);
}

// ----------------------------------------------------------------------------------------------

