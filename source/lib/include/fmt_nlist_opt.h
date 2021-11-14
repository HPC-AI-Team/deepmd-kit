
#pragma once

#include <vector>
#include "device.h"
#include "neighbor_list.h"

template<typename FPTYPE> 
 int format_nlist_i_cpu_opt (
     int*                      	  fmt_nei_idx_a,
     const FPTYPE*  posi,
     const int*     type,
     const int &			              i_idx,
     const std::vector<int > &     nei_idx_a, 
     const float &		              rcut,
     const std::vector<int > &     sec_a);

