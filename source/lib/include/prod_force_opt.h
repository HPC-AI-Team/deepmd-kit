#pragma once

namespace deepmd{

#ifdef __ARM_FEATURE_SVE

void 
prod_force_a_cpu_sve(
    double * force, 
    const double * net_deriv, 
    const double * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

void 
prod_force_a_cpu_sve(
    float * force, 
    const float * net_deriv, 
    const float * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);
    
#endif

}
