#pragma once

namespace deepmd{

#ifdef __ARM_FEATURE_SVE

void prod_virial_a_cpu_sve(
    double * virial, 
    double * atom_virial, 
    const double * net_deriv, 
    const double * env_deriv, 
    const double * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

void prod_virial_a_cpu_sve(
    float * virial, 
    float * atom_virial, 
    const float * net_deriv, 
    const float * env_deriv, 
    const float * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

#endif

} //namespace deepmd
