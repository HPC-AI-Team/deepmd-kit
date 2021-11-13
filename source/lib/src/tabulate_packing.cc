#include <vector>
#include <cassert>
#include <iostream>
#include <string.h>
#include "tabulate_packing.h"

/*
    This inline function was designed to get the table info and bias value for current input xx!
    lower:      indicate the lower boundary of the first table;
    upper:      indicate the upper boundary of the first table as well as the lower boundary of the second table;
    max:        indicate the upper boundary of the second table;
    stride0:    indicate the stride of the first table;
    stride1:    indicate the stride of the second table;
    xx:         indicate the inputs value;
    table_idx:  indicate the location of table info of input value xx;
*/
template <typename FPTYPE>
static inline void locate_xx(
    const FPTYPE& lower, 
    const FPTYPE& upper,
    const FPTYPE& max, 
    const FPTYPE& stride0, 
    const FPTYPE& stride1, 
    FPTYPE& xx, 
    int& table_idx) 
{
  if (xx < lower) {
    table_idx = 0;
    xx = 0;
  }
  else if (xx < upper) {
    table_idx = (int)((xx - lower) / stride0);
    xx -= (table_idx * stride0 + lower);
  }
  else if (xx < max) {
    int first_stride = int((upper - lower) / stride0);
    table_idx = first_stride + (int)((xx - upper) / stride1);
    xx -= ((table_idx - first_stride) * stride1 + upper);
  }
  else {
    table_idx = int((upper - lower) / stride0) + (int)((max - upper) / stride1) - 1;
    xx = 0;
  }
}

template <typename FPTYPE>
static inline FPTYPE dot(
    FPTYPE a[4], 
    FPTYPE b[4]) 
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]; 
}


void deepmd::tabulate_fusion_cpu_packing(
    double * out,
    const double * table, 
    const double * table_info, 
    const double * em_x, 
    const double * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size)
{
  
  memset(out, 0.0, sizeof(double) * nloc * 4 * last_layer_size);
  const double lower   = table_info[0];
  const double upper   = table_info[1];
  const double _max    = table_info[2];
  const double stride0 = table_info[3];
  const double stride1 = table_info[4];

  // std::cout << "(nloc,nnei,last_layer_size)" << " : " << "(" << nloc << "," << nnei << "," << last_layer_size << ")" << std::endl;

  // for every atom, execute a small manual gemm ~
  // double * res = new double[4 * last_layer_size];
  // #pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    double ll[4] = {0};
    double ago = em_x[ii * nnei + nnei - 1];
    bool unloop = false; 

    double * out0 = &out[ii * last_layer_size * 4 + 0 * last_layer_size];
    double * out1 = &out[ii * last_layer_size * 4 + 1 * last_layer_size];
    double * out2 = &out[ii * last_layer_size * 4 + 2 * last_layer_size];
    double * out3 = &out[ii * last_layer_size * 4 + 3 * last_layer_size];

    for (int jj = 0; jj < nnei; jj++) { 
      ll[0] = em[ii * nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * nnei * 4 + jj * 4 + 3];
      double xx = em_x[ii * nnei + jj]; 
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      for (int kbs = 0; kbs < last_layer_size; kbs+=16){
        int kbe = kbs + 16;
        const double* table0 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 0];
        const double* table1 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 1];
        const double* table2 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 2];
        const double* table3 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 3];
        const double* table4 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 4];
        const double* table5 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 5];
        for (int kk = kbs; kk < kbe; kk++) {
          double a0  = table0[kk-kbs]; 
          double a1  = table1[kk-kbs]; 
          double a2  = table2[kk-kbs]; 
          double a3  = table3[kk-kbs];
          double a4  = table4[kk-kbs];
          double a5  = table5[kk-kbs];
          double var = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
          if (unloop) {
            out0[kk] += (nnei - jj) * var * ll[0];
            out1[kk] += (nnei - jj) * var * ll[1];
            out2[kk] += (nnei - jj) * var * ll[2];
            out3[kk] += (nnei - jj) * var * ll[3];
          }
          else {
            out0[kk] += var * ll[0];
            out1[kk] += var * ll[1];
            out2[kk] += var * ll[2];
            out3[kk] += var * ll[3];
          }
        }
      }

      if (unloop) break;
    }
  }
}

void deepmd::tabulate_fusion_cpu_packing(
    float * out,
    const float * table, 
    const float * table_info, 
    const float * em_x, 
    const float * em, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size)
{
  memset(out, 0.0, sizeof(float) * nloc * 4 * last_layer_size);
  const float lower   = table_info[0];
  const float upper   = table_info[1];
  const float _max    = table_info[2];
  const float stride0 = table_info[3];
  const float stride1 = table_info[4];

  // std::cout << "(nloc,nnei,last_layer_size)" << " : " << "(" << nloc << "," << nnei << "," << last_layer_size << ")" << std::endl;

  // for every atom, execute a small manual gemm ~
  // float * res = new float[4 * last_layer_size];
  // #pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    float ll[4] = {0};
    float ago = em_x[ii * nnei + nnei - 1];
    bool unloop = false; 

    float * out0 = &out[ii * last_layer_size * 4 + 0 * last_layer_size];
    float * out1 = &out[ii * last_layer_size * 4 + 1 * last_layer_size];
    float * out2 = &out[ii * last_layer_size * 4 + 2 * last_layer_size];
    float * out3 = &out[ii * last_layer_size * 4 + 3 * last_layer_size];

    for (int jj = 0; jj < nnei; jj++) { 
      ll[0] = em[ii * nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * nnei * 4 + jj * 4 + 3];
      float xx = em_x[ii * nnei + jj]; 
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      for (int kbs = 0; kbs < last_layer_size; kbs+=32){
        int kbe = kbs + 32;
        const float* table0 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 0];
        const float* table1 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 1];
        const float* table2 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 2];
        const float* table3 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 3];
        const float* table4 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 4];
        const float* table5 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 5];
        for (int kk = kbs; kk < kbe; kk++) {
          float a0  = table0[kk-kbs]; 
          float a1  = table1[kk-kbs]; 
          float a2  = table2[kk-kbs]; 
          float a3  = table3[kk-kbs];
          float a4  = table4[kk-kbs];
          float a5  = table5[kk-kbs];
          float var = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;
          if (unloop) {
            out0[kk] += (nnei - jj) * var * ll[0];
            out1[kk] += (nnei - jj) * var * ll[1];
            out2[kk] += (nnei - jj) * var * ll[2];
            out3[kk] += (nnei - jj) * var * ll[3];
          }
          else {
            out0[kk] += var * ll[0];
            out1[kk] += var * ll[1];
            out2[kk] += var * ll[2];
            out3[kk] += var * ll[3];
          }
        }
      }

      if (unloop) break;
    }
  }
}

void deepmd::tabulate_fusion_grad_cpu_packing(
    double * dy_dem_x, 
    double * dy_dem,
    const double * table, 
    const double * table_info, 
    const double * em_x, 
    const double * em, 
    const double * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size) 
{
  memset(dy_dem_x, 0.0, sizeof(double) * nloc * nnei);
  memset(dy_dem, 0.0, sizeof(double) * nloc * nnei * 4);
  double const lower   = table_info[0];
  double const upper   = table_info[1];
  double const _max    = table_info[2];
  double const stride0 = table_info[3];
  double const stride1 = table_info[4];
  // for every atom, execute a small gemm~
  // double * res = new double[4 * last_layer_size];
  // #pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    double ll[4];
    double rr[4];
    double ago = em_x[ii * nnei + nnei - 1];
    const double* dy0 = &dy[ii * last_layer_size * 4 + 0 * last_layer_size];
    const double* dy1 = &dy[ii * last_layer_size * 4 + 1 * last_layer_size];
    const double* dy2 = &dy[ii * last_layer_size * 4 + 2 * last_layer_size];
    const double* dy3 = &dy[ii * last_layer_size * 4 + 3 * last_layer_size];
    bool unloop = false;
    for (int jj = 0; jj < nnei; jj++) {
      // construct the dy/dx
      ll[0] = em[ii * nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * nnei * 4 + jj * 4 + 3];
      double xx = em_x[ii * nnei + jj]; 
      if (ago == xx) {
      unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);

      double* dy_dem_tmp = &dy_dem[ii * nnei * 4 + jj * 4];

      double grad = 0.0;
      double dy_dem_0 = 0.0;
      double dy_dem_1 = 0.0;
      double dy_dem_2 = 0.0;
      double dy_dem_3 = 0.0;

      for (int kbs = 0; kbs < last_layer_size; kbs += 16){
        int kbe = kbs + 16;
        const double* table0 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 0];
        const double* table1 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 1];
        const double* table2 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 2];
        const double* table3 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 3];
        const double* table4 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 4];
        const double* table5 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 16 * 5];
        for (int kk = kbs; kk < kbe; kk++) {
          rr[0] = dy0[kk];
          rr[1] = dy1[kk];
          rr[2] = dy2[kk];
          rr[3] = dy3[kk];
          double a0  = table0[kk-kbs]; 
          double a1  = table1[kk-kbs]; 
          double a2  = table2[kk-kbs]; 
          double a3  = table3[kk-kbs];
          double a4  = table4[kk-kbs];
          double a5  = table5[kk-kbs];
          double res = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;

          if (unloop) {
            grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr) * (nnei - jj);
            dy_dem_0 += res * rr[0] * (nnei - jj);
            dy_dem_1 += res * rr[1] * (nnei - jj);
            dy_dem_2 += res * rr[2] * (nnei - jj);
            dy_dem_3 += res * rr[3] * (nnei - jj);
          }
          else {
            grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr);
            dy_dem_0 += res * rr[0];
            dy_dem_1 += res * rr[1];
            dy_dem_2 += res * rr[2];
            dy_dem_3 += res * rr[3];
          }
        }
      }

      dy_dem_x[ii * nnei + jj] = grad;
      dy_dem_tmp[0] = dy_dem_0;
      dy_dem_tmp[1] = dy_dem_1;
      dy_dem_tmp[2] = dy_dem_2;
      dy_dem_tmp[3] = dy_dem_3;

      if (unloop) break;
    }
  }
}



void deepmd::tabulate_fusion_grad_cpu_packing(
    float * dy_dem_x, 
    float * dy_dem,
    const float * table, 
    const float * table_info, 
    const float * em_x, 
    const float * em, 
    const float * dy, 
    const int nloc, 
    const int nnei, 
    const int last_layer_size) 
{
  memset(dy_dem_x, 0.0, sizeof(float) * nloc * nnei);
  memset(dy_dem, 0.0, sizeof(float) * nloc * nnei * 4);
  float const lower   = table_info[0];
  float const upper   = table_info[1];
  float const _max    = table_info[2];
  float const stride0 = table_info[3];
  float const stride1 = table_info[4];
  // for every atom, execute a small gemm~
  // float * res = new float[4 * last_layer_size];
  // #pragma omp parallel for
  for (int ii = 0; ii < nloc; ii++) {
    float ll[4];
    float rr[4];
    float ago = em_x[ii * nnei + nnei - 1];
    const float* dy0 = &dy[ii * last_layer_size * 4 + 0 * last_layer_size];
    const float* dy1 = &dy[ii * last_layer_size * 4 + 1 * last_layer_size];
    const float* dy2 = &dy[ii * last_layer_size * 4 + 2 * last_layer_size];
    const float* dy3 = &dy[ii * last_layer_size * 4 + 3 * last_layer_size];
    bool unloop = false;
    for (int jj = 0; jj < nnei; jj++) {
      // construct the dy/dx
      ll[0] = em[ii * nnei * 4 + jj * 4 + 0];
      ll[1] = em[ii * nnei * 4 + jj * 4 + 1];
      ll[2] = em[ii * nnei * 4 + jj * 4 + 2];
      ll[3] = em[ii * nnei * 4 + jj * 4 + 3];
      float xx = em_x[ii * nnei + jj]; 
      if (ago == xx) {
        unloop = true;
      }
      int table_idx = 0;
      locate_xx(lower, upper, _max, stride0, stride1, xx, table_idx);
      
      float* dy_dem_tmp = &dy_dem[ii * nnei * 4 + jj * 4];

      float grad = 0.0;
      float dy_dem_0 = 0.0;
      float dy_dem_1 = 0.0;
      float dy_dem_2 = 0.0;
      float dy_dem_3 = 0.0;

      for (int kbs = 0; kbs < last_layer_size; kbs += 32){
        int kbe = kbs + 32;
        const float* table0 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 0];
        const float* table1 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 1];
        const float* table2 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 2];
        const float* table3 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 3];
        const float* table4 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 4];
        const float* table5 = &table[table_idx * last_layer_size * 6 + kbs * 6 + 32 * 5];
        for (int kk = kbs; kk < kbe; kk++) {
          rr[0] = dy0[kk];
          rr[1] = dy1[kk];
          rr[2] = dy2[kk];
          rr[3] = dy3[kk];
          float a0  = table0[kk-kbs]; 
          float a1  = table1[kk-kbs]; 
          float a2  = table2[kk-kbs]; 
          float a3  = table3[kk-kbs];
          float a4  = table4[kk-kbs];
          float a5  = table5[kk-kbs];
          float res = a0 + (a1 + (a2 + (a3 + (a4 + a5 * xx) * xx) * xx) * xx) * xx;

          if (unloop) {
            grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr) * (nnei - jj);
            dy_dem_0 += res * rr[0] * (nnei - jj);
            dy_dem_1 += res * rr[1] * (nnei - jj);
            dy_dem_2 += res * rr[2] * (nnei - jj);
            dy_dem_3 += res * rr[3] * (nnei - jj);
          }
          else {
            grad += (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * xx) * xx) * xx) * xx) * dot(ll, rr);
            dy_dem_0 += res * rr[0];
            dy_dem_1 += res * rr[1];
            dy_dem_2 += res * rr[2];
            dy_dem_3 += res * rr[3];
          }
        }
      }

      dy_dem_x[ii * nnei + jj] = grad;
      dy_dem_tmp[0] = dy_dem_0;
      dy_dem_tmp[1] = dy_dem_1;
      dy_dem_tmp[2] = dy_dem_2;
      dy_dem_tmp[3] = dy_dem_3;

      if (unloop) break;
    }
  }
}
