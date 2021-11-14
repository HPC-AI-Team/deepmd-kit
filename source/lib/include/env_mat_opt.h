#pragma once

#include <vector>

namespace deepmd{

template<typename FPTYPE> 
void env_mat_a_cpu_normalize (
    FPTYPE*	        descrpt_a,
    FPTYPE*	        descrpt_a_deriv,
    FPTYPE*	        rij_a,
    const FPTYPE*      	posi,
    const int*		    type,
    const int &				        i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		sec, 
    const float &			        rmin,
    const float &			        rmax,
    const FPTYPE * avg, 
    const FPTYPE * std
    ) ;

template<typename FPTYPE> 
void env_mat_a_cpu_normalize_preprocessed (
    FPTYPE*	        descrpt_a,
    FPTYPE*	        descrpt_a_deriv,
    FPTYPE*	        rij_a,
    const FPTYPE*      	posi,
    const int*		    type,
    const int &				        i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		sec, 
    const float &			        rmin,
    const float &			        rmax,
    const FPTYPE * avg, 
    const FPTYPE * std
    ) ;

#ifdef __ARM_FEATURE_SVE 

void env_mat_a_cpu_normalize_sve (
    double*	            descrpt_a,
    double*		        descrpt_a_deriv,
    double*		        rij_a,
    const double*      	posi,
    const int*		    type,
    const int &				        i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			        rmin,
    const float &			        rmax,
    const double * avg, 
    const double * std) ;

void env_mat_a_cpu_normalize_sve (
    float*	        descrpt_a,
    float*	        descrpt_a_deriv,
    float*	        rij_a,
    const float*      	posi,
    const int*		    type,
    const int &				        i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			        rmin,
    const float &			        rmax,
    const float * avg, 
    const float * std) ;

void env_mat_a_cpu_normalize_preprocessed_sve (
    double*	            descrpt_a,
    double*		        descrpt_a_deriv,
    double*		        rij_a,
    const double*      	posi,
    const int*		    type,
    const int &				        i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			        rmin,
    const float &			        rmax,
    const double * avg, 
    const double * std) ;

void env_mat_a_cpu_normalize_preprocessed_sve (
    float*	        descrpt_a,
    float*	        descrpt_a_deriv,
    float*	        rij_a,
    const float*      	posi,
    const int*		    type,
    const int &				        i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			        rmin,
    const float &			        rmax,
    const float * avg, 
    const float * std) ;

#endif


}
