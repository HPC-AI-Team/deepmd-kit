#pragma once

#include <cmath>
#include "poly_table_f.h"

#define min(x,y) (x < y ? x : y)

// #define poly_2_tanh_double(input,output)                             \
//         double _x = (input);                                         \
//         double _x_abs = fabs(_x);                                    \
//         int _index = min(_x_abs * TABLE_LEN,TABLE_LEN * RANGE - 1);  \
//         double _c0 = fast_tanh_poly_table_double[_index][0];         \
//         double _c1 = fast_tanh_poly_table_double[_index][1];         \
//         double _c2 = fast_tanh_poly_table_double[_index][2];         \
//         double _y = ((_c0 * _x_abs) +_c1) * _x_abs + _c2;            \
//         _y = _x < 0. ? -_y : _y;                                     \
//         (output) = _y;                                               \


#define poly_2_tanh_float(input,output){                            \
        float _x = (input);                                         \
        float _x_abs = fabsf(_x);                                   \
        int _index = min(_x_abs * TABLE_LEN,TABLE_LEN * RANGE - 1); \
        float _c0 = fast_tanh_poly_table_float[_index][0];          \
        float _c1 = fast_tanh_poly_table_float[_index][1];          \
        float _c2 = fast_tanh_poly_table_float[_index][2];          \
        float _y = ((_c0 * _x_abs) +_c1) * _x_abs + _c2;            \
        _y = _x < 0.f ? -_y : _y;                                   \
        (output) = _y;                                              \
}
    
static void poly_2_tanh_double_vector(const double* inputs,double* outputs,uint64_t N){
// #pragma clang loop vectorize(disable) 
    for(int i = 0;i<N ;i++){
        poly_2_tanh_float(inputs[i],outputs[i]);
    }
}

static void poly_2_tanh_float_vector(const float* inputs,float* outputs,uint64_t N){
// #pragma clang loop vectorize(disable) 
    for(int i = 0;i<N ;i++){
        poly_2_tanh_float(inputs[i],outputs[i]);
    }
}