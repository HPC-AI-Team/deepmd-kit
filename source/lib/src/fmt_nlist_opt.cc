#include <vector>
#include <cassert>
#include <algorithm>
#include "fmt_nlist_opt.h"
#include "SimulationRegion.h"
#include <iostream>
#include <omp.h>

using namespace deepmd;

struct NeighborInfo 
{
  int type;
  double dist;
  int index;
  NeighborInfo () 
      : type (0), dist(0), index(0) 
      {
      }
  NeighborInfo (int tt, double dd, int ii) 
      : type (tt), dist(dd), index(ii) 
      {
      }
  bool operator < (const NeighborInfo & b) const 
      {
	return (type < b.type || 
		(type == b.type && 
		 (dist < b.dist || 
		  (dist == b.dist && index < b.index) ) ) );
      }
};

template<typename FPTYPE> 
 int format_nlist_i_cpu_opt (
     int*                      	  fmt_nei_idx_a,
     const FPTYPE*  posi,
     const int*     type,
     const int &			              i_idx,
     const std::vector<int > &     nei_idx_a, 
     const float &		              rcut,
     const std::vector<int > &     sec_a)
 {
     for(int i = 0;i< sec_a.back();i++){
       fmt_nei_idx_a[i] = -1;
     }

     std::vector<int > nei_idx (nei_idx_a);
     std::vector<NeighborInfo > sel_nei;
     sel_nei.reserve (nei_idx_a.size());

     FPTYPE ix = posi[i_idx * 3 + 0];
     FPTYPE iy = posi[i_idx * 3 + 1];
     FPTYPE iz = posi[i_idx * 3 + 2];
     for (unsigned kk = 0; kk < nei_idx_a.size(); ++kk) {
         FPTYPE diff[3];
         const int & j_idx = nei_idx_a[kk];
         diff[0] = posi[j_idx * 3 + 0] - ix;
         diff[1] = posi[j_idx * 3 + 1] - iy;
         diff[2] = posi[j_idx * 3 + 2] - iz;
         FPTYPE rr = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];    
         if (rr <= rcut * rcut) {
             sel_nei.push_back(NeighborInfo(type[j_idx], rr, j_idx));
         }
     }

     sort(sel_nei.begin(), sel_nei.end());  

     std::vector<int > nei_iter = sec_a;
     int overflowed = -1;
     for (unsigned kk = 0; kk < sel_nei.size(); ++kk) {
        const int & nei_type = sel_nei[kk].type;
        int index = sel_nei[kk].index;
        if (nei_iter[nei_type] < sec_a[nei_type+1]) {
            fmt_nei_idx_a[nei_iter[nei_type] ++] = index;
        }
        else{
          overflowed = nei_type;
 	    }
     }
     return overflowed;
 }

// template<typename FPTYPE> 
//  int format_nlist_i_cpu_opt (
//      int*                      	  fmt_nei_idx_a,
//      const FPTYPE*  posi,
//      const int*     type,
//      const int &			              i_idx,
//      const std::vector<int > &     nei_idx_a, 
//      const float &		              rcut,
//      const std::vector<int > &     sec_a)
//  {
//      for(int i = 0;i< sec_a.back();i++){
//        fmt_nei_idx_a[i] = -1;
//      }

//     //  std::vector<int > nei_idx (nei_idx_a);
//     //  std::vector<NeighborInfo > sel_nei;
//      std::vector<uint64_t > sel_nei;
//      sel_nei.reserve (nei_idx_a.size());

//      FPTYPE ix = posi[i_idx * 3 + 0];
//      FPTYPE iy = posi[i_idx * 3 + 1];
//      FPTYPE iz = posi[i_idx * 3 + 2];
//      for (unsigned kk = 0; kk < nei_idx_a.size(); ++kk) {
//          FPTYPE diff[3];
//          const int & j_idx = nei_idx_a[kk];
//          diff[0] = posi[j_idx * 3 + 0] - ix;
//          diff[1] = posi[j_idx * 3 + 1] - iy;
//          diff[2] = posi[j_idx * 3 + 2] - iz;
//         //  FPTYPE rr = sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);    
//         //  if (rr <= rcut) {
//         //      sel_nei.push_back(NeighborInfo(type[j_idx], rr, j_idx));
//         //  }
//         FPTYPE rr = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);    
//         if (rr <= rcut * rcut) {
//           uint64_t key = (((uint64_t)type[j_idx]    << 61)  & 0xe000000000000000) | 
//                          (((uint64_t)(rr * 1.0E11)  << 20)  & 0x1ffffffffff00000) |
//                          (j_idx                             & 0x00000000000fffff);
//           sel_nei.push_back(key);
//           // int type_ = (key & 0xe000000000000000) >> 61;
//           // uint64_t dist_ = (key & 0x1ffffffffff00000) >> 20;
//           // int index_ = key & 0x00000000000fffff;
//           // assert(type_ == type[j_idx]);
//           // assert(dist_ == (uint64_t)(rr * 1.0E10));
//           // assert(index_ == j_idx);
//         }
//      }

//      sort(sel_nei.begin(), sel_nei.end());  

//      std::vector<int > nei_iter = sec_a;
//      int overflowed = -1;
//      for (unsigned kk = 0; kk < sel_nei.size(); ++kk) {
//         uint64_t compressed_info = sel_nei[kk];
//         int nei_type = (compressed_info & 0xe000000000000000) >> 61;
//         int index = compressed_info & 0x00000000000fffff;
//         //  const int & nei_type = sel_nei[kk].type;
//          if (nei_iter[nei_type] < sec_a[nei_type+1]) {
//              fmt_nei_idx_a[nei_iter[nei_type] ++] = index;
//          }
//          else{
//            overflowed = nei_type;
//  	      }
//      }
//      return overflowed;
//  }

 template
 int format_nlist_i_cpu_opt<double> (
     int*		fmt_nei_idx_a,
     const double*  posi,
     const int*     type,
     const int &			i_idx,
     const std::vector<int > &   nei_idx_a, 
     const float &		rcut,
     const std::vector<int > &   sec_a);


 template
 int format_nlist_i_cpu_opt<float> (
     int*		fmt_nei_idx_a,
     const float*  posi,
     const int*     type,
     const int &			i_idx,
     const std::vector<int > &   nei_idx_a, 
     const float &		rcut,
     const std::vector<int > &   sec_a);