#include <cmath>
#include "fast_tanh.h"

#define min(x,y) (x < y ? x : y)


void deepmd::fast_tanh_cpu(const double* inputs,double* outputs,size_t N){
// #pragma clang loop vectorize(disable) 
    for(int i = 0;i<N ;i++){
        poly_2_tanh_float(inputs[i],outputs[i]);
    }
}

void deepmd::fast_tanh_cpu(const float* inputs,float* outputs,size_t N){
// #pragma clang loop vectorize(disable) 
    for(int i = 0;i<N ;i++){
        poly_2_tanh_float(inputs[i],outputs[i]);
    }
}