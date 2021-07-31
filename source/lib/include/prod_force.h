#pragma once

namespace deepmd{

template<typename FPTYPE>
void prod_force_a_cpu(
    FPTYPE * force, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

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
template<typename FPTYPE>
void prod_force_r_cpu(
    FPTYPE * force, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

#if GOOGLE_CUDA
template<typename FPTYPE> 
void prod_force_a_gpu_cuda(
    FPTYPE * force, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

template<typename FPTYPE> 
void prod_force_r_gpu_cuda(
    FPTYPE * force, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);
#endif // GOOGLE_CUDA

#if TENSORFLOW_USE_ROCM
template<typename FPTYPE> 
void prod_force_a_gpu_rocm(
    FPTYPE * force, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);

template<typename FPTYPE> 
void prod_force_r_gpu_rocm(
    FPTYPE * force, 
    const FPTYPE * net_deriv, 
    const FPTYPE * in_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei);
#endif // TENSORFLOW_USE_ROCM

}
