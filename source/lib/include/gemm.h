#pragma once


namespace deepmd{

    void gemm_launcer(const int m, const int n, const int k, const float * A, const float * B,const float * C , float * D);
    void gemm_launcer(const int m, const int n, const int k, const double * A, const double * B,const double * C, double * D);
#if GOOGLE_CUDA
    void gemm_launcer_cuda( const int m, const int n, const int k, const float * A, const float * B,const float * C , float * D);
    void gemm_launcer_cuda( const int m, const int n, const int k, const double * A, const double * B,const double * C, double * D);
#endif 

}

