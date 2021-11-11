#pragma once

namespace deepmd{

    void matmul_nn_row_launcer(const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_nn_row_launcer(const int m, const int n, const int k, const double * A, const double * B, double * C);
    void matmul_nn_col_launcer(const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_nn_col_launcer(const int m, const int n, const int k, const double * A, const double * B, double * C);


    void matmul_nt_row_launcer(const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_nt_row_launcer(const int m, const int n, const int k, const double * A, const double * B, double * C);
    void matmul_nt_col_launcer(const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_nt_col_launcer(const int m, const int n, const int k, const double * A, const double * B, double * C);
       
    void matmul_tn_row_launcer(const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_tn_row_launcer(const int m, const int n, const int k, const double * A, const double * B, double * C);
    void matmul_tn_col_launcer(const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_tn_col_launcer(const int m, const int n, const int k, const double * A, const double * B, double * C);


#if GOOGLE_CUDA

    void matmul_nn_row_launcer_cuda( const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_nn_row_launcer_cuda( const int m, const int n, const int k, const double * A, const double * B, double * C);
    void matmul_nn_col_launcer_cuda( const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_nn_col_launcer_cuda( const int m, const int n, const int k, const double * A, const double * B, double * C);

    void matmul_nt_row_launcer_cuda( const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_nt_row_launcer_cuda( const int m, const int n, const int k, const double * A, const double * B, double * C);
    void matmul_nt_col_launcer_cuda( const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_nt_col_launcer_cuda( const int m, const int n, const int k, const double * A, const double * B, double * C);

    void matmul_tn_row_launcer_cuda( const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_tn_row_launcer_cuda( const int m, const int n, const int k, const double * A, const double * B, double * C);
    void matmul_tn_col_launcer_cuda( const int m, const int n, const int k, const float * A, const float * B, float * C);
    void matmul_tn_col_launcer_cuda( const int m, const int n, const int k, const double * A, const double * B, double * C);

#endif 

}

