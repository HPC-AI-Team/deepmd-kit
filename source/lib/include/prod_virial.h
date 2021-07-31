#pragma once

namespace deepmd{

template<typename FPTYPE>
void prod_virial_a_cpu(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * env_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

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

template<typename FPTYPE>
void prod_virial_r_cpu(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * env_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

#if GOOGLE_CUDA
template<typename FPTYPE>
void prod_virial_a_gpu_cuda(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * env_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

template<typename FPTYPE>
void prod_virial_r_gpu_cuda(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * env_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);
#endif // GOOGLE_CUDA


#if TENSORFLOW_USE_ROCM
template<typename FPTYPE>
void prod_virial_a_gpu_rocm(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * env_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

template<typename FPTYPE>
void prod_virial_r_gpu_rocm(
    FPTYPE * virial, 
    FPTYPE * atom_virial, 
    const FPTYPE * net_deriv, 
    const FPTYPE * env_deriv, 
    const FPTYPE * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);
#endif // TENSORFLOW_USE_ROCM

} //namespace deepmd
