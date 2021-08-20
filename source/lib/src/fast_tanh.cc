#include "fast_tanh.h"
#include <cmath>

void deepmd::fast_tanh_cpu(const double* inputs,double* outputs,size_t N){
// #pragma clang loop vectorize(disable) 
    for(int i = 0;i<N ;i++){
        // poly_2_tanh_float(inputs[i],outputs[i]);
        outputs[i] = 1. - 2. / (std::exp(2. * inputs[i]) + 1.);
    }
}

void deepmd::fast_tanh_cpu(const float* inputs,float* outputs,size_t N){
// #pragma clang loop vectorize(disable) 
    for(int i = 0;i<N ;i++){
        // poly_2_tanh_float(inputs[i],outputs[i]);
        outputs[i] = 1.f - 2.f / (expf(2.f * inputs[i]) + 1.f);
    }
}