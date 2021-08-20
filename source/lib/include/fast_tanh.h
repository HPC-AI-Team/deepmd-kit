#pragma once
#include <cstdint>
#include <cmath>

#include "poly_table_f.h"
#include "poly_table_d.h"

#define __min__(x,y) (x < y ? x : y)

#define poly_2_tanh_double(input,output)                             \
        double _x = (input);                                         \
        double _x_abs = fabs(_x);                                    \
        int _index = __min__(_x_abs * TABLE_LEN,TABLE_LEN * RANGE - 1);  \
        double _c0 = fast_tanh_poly_table_double[_index][0];         \
        double _c1 = fast_tanh_poly_table_double[_index][1];         \
        double _c2 = fast_tanh_poly_table_double[_index][2];         \
        double _y = ((_c0 * _x_abs) +_c1) * _x_abs + _c2;            \
        _y = _x < 0. ? -_y : _y;                                     \
        (output) = _y;  

#define poly_2_tanh_float(input,output){                            \
        float _x = (input);                                         \
        float _x_abs = fabsf(_x);                                   \
        int _index = __min__(_x_abs * TABLE_LEN,TABLE_LEN * RANGE - 1); \
        float _c0 = fast_tanh_poly_table_float[_index][0];          \
        float _c1 = fast_tanh_poly_table_float[_index][1];          \
        float _c2 = fast_tanh_poly_table_float[_index][2];          \
        float _y = ((_c0 * _x_abs) +_c1) * _x_abs + _c2;            \
        _y = _x < 0.f ? -_y : _y;                                   \
        (output) = _y;                                              \
}

namespace deepmd{

    void fast_tanh_cpu(const double* inputs,double* outputs,size_t N);
    void fast_tanh_cpu(const float* inputs,float* outputs,size_t N);
 
#if GOOGLE_CUDA
    template<typename FPTYPE>
    void fast_tanh_cuda(const FPTYPE* inputs,FPTYPE* outputs,size_t N);
#endif 

}



