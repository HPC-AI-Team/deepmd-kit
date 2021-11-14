#include <cassert>
#include <iostream>
#include <string.h>
#include "prod_env_mat_opt.h"
#include "fmt_nlist.h"
#include "fmt_nlist_opt.h"
#include "env_mat_opt.h"
#include "env_mat.h"
#include "lib_tools.h"

using namespace deepmd;

template<typename FPTYPE>
void
deepmd::
prod_env_mat_a_cpu_opt(
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const InputNlist & inlist,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec) 
{

  // bool have_preprocessed = get_env_preprocessed();
  const int nnei = sec.back();
  const int nem = nnei * 4;

  // build nlist
   std::vector<std::vector<int > > d_nlist_a(nloc);

   assert(nloc == inlist.inum);
   for (unsigned ii = 0; ii < nloc; ++ii) {
     d_nlist_a[ii].reserve(max_nbor_size);
   }
   for (unsigned ii = 0; ii < nloc; ++ii) {
     int i_idx = inlist.ilist[ii];
     for(unsigned jj = 0; jj < inlist.numneigh[ii]; ++jj){
       int j_idx = inlist.firstneigh[ii][jj];
       d_nlist_a[i_idx].push_back (j_idx);
     }
   }

  for (int ii = 0; ii < nloc; ++ii) {
    int*  fmt_nlist_a = &nlist[ii * nnei];
    int ret = format_nlist_i_cpu_opt(fmt_nlist_a, coord, type, ii, d_nlist_a[ii], rcut, sec);
    FPTYPE* d_em_a = &em[ii * nem];
    FPTYPE* d_em_a_deriv = &em_deriv[ii * nem * 3];
    FPTYPE* d_rij_a = &rij[ii * nnei * 3];
//     if(have_preprocessed){
// #ifdef __ARM_FEATURE_SVE
//         env_mat_a_cpu_normalize_preprocessed_sve (d_em_a, d_em_a_deriv, d_rij_a, coord, type, ii, fmt_nlist_a, sec, rcut_smth, rcut, avg, std);
// #else
//         env_mat_a_cpu_normalize_preprocessed (d_em_a, d_em_a_deriv, d_rij_a, coord, type, ii, fmt_nlist_a, sec, rcut_smth, rcut, avg, std);
// #endif
//     }else{
#ifdef __ARM_FEATURE_SVE
        env_mat_a_cpu_normalize_sve (d_em_a, d_em_a_deriv, d_rij_a, coord, type, ii, fmt_nlist_a, sec, rcut_smth, rcut, avg, std);
#else
        env_mat_a_cpu_normalize (d_em_a, d_em_a_deriv, d_rij_a, coord, type, ii, fmt_nlist_a, sec, rcut_smth, rcut, avg, std);
#endif
    // }
  }
}


template
void 
deepmd::
prod_env_mat_a_cpu_opt<double>(
    double * em, 
    double * em_deriv, 
    double * rij, 
    int * nlist, 
    const double * coord, 
    const int * type, 
    const InputNlist & inlist,
    const int max_nbor_size,
    const double * avg, 
    const double * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);

template
void
deepmd::
prod_env_mat_a_cpu_opt<float>(
    float * em, 
    float * em_deriv, 
    float * rij, 
    int * nlist, 
    const float * coord, 
    const int * type, 
    const InputNlist & inlist,
    const int max_nbor_size,
    const float * avg, 
    const float * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec);