#pragma once
#include <vector>
#include "device.h"
#include "neighbor_list.h"

namespace deepmd{

template<typename FPTYPE>
void prod_env_mat_a_cpu_opt(
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
    const std::vector<int> sec);

}

