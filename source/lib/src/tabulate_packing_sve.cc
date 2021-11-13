#include <vector>
#include <cassert>
#include <iostream>
#include <string.h>
#include "tabulate_packing.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 

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

void deepmd::tabulate_fusion_cpu_packing_sve(
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

      assert(last_layer_size % svcntd() == 0);

      svbool_t ptrue = svptrue_b64();
      svfloat64_t vnei_sub_jj = svdup_f64((double(nnei - jj)));
      svfloat64_t vxx = svdup_f64(xx);
      svfloat64_t vxx2 = svmul_z(ptrue, vxx, vxx);
      svfloat64_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      svfloat64_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      svfloat64_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      svfloat64_t vll0 = svdup_f64(ll[0]);
      svfloat64_t vll1 = svdup_f64(ll[1]);
      svfloat64_t vll2 = svdup_f64(ll[2]);
      svfloat64_t vll3 = svdup_f64(ll[3]);
      svfloat64_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat64_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat64_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat64_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);

      for(int kk = 0; kk < last_layer_size; kk += svcntd() * 2){
        const double* TABLE = &table[table_idx * last_layer_size * 6 + kk * 6];
        svfloat64_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat64_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat64_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat64_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        svfloat64_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        svfloat64_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        svfloat64_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        svfloat64_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        svfloat64_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        svfloat64_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        svfloat64_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        svfloat64_t va5_1 = svld1_vnum(ptrue, TABLE, 11);

        svfloat64_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat64_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);
        svfloat64_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        svfloat64_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        svfloat64_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        svfloat64_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        svfloat64_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        svfloat64_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        svfloat64_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        svfloat64_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);
        svfloat64_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        svfloat64_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        svfloat64_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        svfloat64_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        svfloat64_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        svfloat64_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);
        svfloat64_t vvar_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        svfloat64_t vvar_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        svfloat64_t vout0_0 = svld1(ptrue, out0 + kk);
        svfloat64_t vout0_1 = svld1(ptrue, out0 + kk + svcntd());
        svfloat64_t vout1_0 = svld1(ptrue, out1 + kk);
        svfloat64_t vout1_1 = svld1(ptrue, out1 + kk + svcntd());
        svfloat64_t vout2_0 = svld1(ptrue, out2 + kk);
        svfloat64_t vout2_1 = svld1(ptrue, out2 + kk + svcntd());
        svfloat64_t vout3_0 = svld1(ptrue, out3 + kk);
        svfloat64_t vout3_1 = svld1(ptrue, out3 + kk + svcntd());

        if(unloop){
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0_);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0_);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1_);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1_);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2_);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2_);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3_);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3_);
        }else{
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3);
        }
        svst1(ptrue, out0 + kk, vout0_0);
        svst1(ptrue, out0 + kk + svcntd(), vout0_1);
        svst1(ptrue, out1 + kk, vout1_0);
        svst1(ptrue, out1 + kk + svcntd(), vout1_1);
        svst1(ptrue, out2 + kk, vout2_0);
        svst1(ptrue, out2 + kk + svcntd(), vout2_1);
        svst1(ptrue, out3 + kk, vout3_0);
        svst1(ptrue, out3 + kk + svcntd(), vout3_1);
      }

      if (unloop) break;
    }
  }
}

void deepmd::tabulate_fusion_cpu_packing_sve(
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

      assert(last_layer_size % svcntw() == 0);

      svbool_t ptrue = svptrue_b32();
      svfloat32_t vnei_sub_jj = svdup_f32((double(nnei - jj)));
      svfloat32_t vxx = svdup_f32(xx);
      svfloat32_t vxx2 = svmul_z(ptrue, vxx, vxx);
      svfloat32_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      svfloat32_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      svfloat32_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      svfloat32_t vll0 = svdup_f32(ll[0]);
      svfloat32_t vll1 = svdup_f32(ll[1]);
      svfloat32_t vll2 = svdup_f32(ll[2]);
      svfloat32_t vll3 = svdup_f32(ll[3]);
      svfloat32_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat32_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat32_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat32_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);

      for(int kk = 0; kk < last_layer_size; kk += svcntw() * 2){
        const float* TABLE = &table[table_idx * last_layer_size * 6 + kk * 6];
        svfloat32_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat32_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat32_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat32_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        svfloat32_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        svfloat32_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        svfloat32_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        svfloat32_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        svfloat32_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        svfloat32_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        svfloat32_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        svfloat32_t va5_1 = svld1_vnum(ptrue, TABLE, 11);

        svfloat32_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat32_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);
        svfloat32_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        svfloat32_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        svfloat32_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        svfloat32_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        svfloat32_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        svfloat32_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        svfloat32_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        svfloat32_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);
        svfloat32_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        svfloat32_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        svfloat32_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        svfloat32_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        svfloat32_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        svfloat32_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);
        svfloat32_t vvar_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        svfloat32_t vvar_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        svfloat32_t vout0_0 = svld1(ptrue, out0 + kk);
        svfloat32_t vout0_1 = svld1(ptrue, out0 + kk + svcntw());
        svfloat32_t vout1_0 = svld1(ptrue, out1 + kk);
        svfloat32_t vout1_1 = svld1(ptrue, out1 + kk + svcntw());
        svfloat32_t vout2_0 = svld1(ptrue, out2 + kk);
        svfloat32_t vout2_1 = svld1(ptrue, out2 + kk + svcntw());
        svfloat32_t vout3_0 = svld1(ptrue, out3 + kk);
        svfloat32_t vout3_1 = svld1(ptrue, out3 + kk + svcntw());

        if(unloop){
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0_);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0_);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1_);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1_);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2_);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2_);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3_);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3_);
        }else{
          vout0_0 = svmla_z(ptrue, vout0_0, vvar_0, vll0);
          vout0_1 = svmla_z(ptrue, vout0_1, vvar_1, vll0);
          vout1_0 = svmla_z(ptrue, vout1_0, vvar_0, vll1);
          vout1_1 = svmla_z(ptrue, vout1_1, vvar_1, vll1);
          vout2_0 = svmla_z(ptrue, vout2_0, vvar_0, vll2);
          vout2_1 = svmla_z(ptrue, vout2_1, vvar_1, vll2);
          vout3_0 = svmla_z(ptrue, vout3_0, vvar_0, vll3);
          vout3_1 = svmla_z(ptrue, vout3_1, vvar_1, vll3);
        }
        svst1(ptrue, out0 + kk, vout0_0);
        svst1(ptrue, out0 + kk + svcntw(), vout0_1);
        svst1(ptrue, out1 + kk, vout1_0);
        svst1(ptrue, out1 + kk + svcntw(), vout1_1);
        svst1(ptrue, out2 + kk, vout2_0);
        svst1(ptrue, out2 + kk + svcntw(), vout2_1);
        svst1(ptrue, out3 + kk, vout3_0);
        svst1(ptrue, out3 + kk + svcntw(), vout3_1);
      }
      if (unloop) break;
    }
  }
}



void deepmd::tabulate_fusion_grad_cpu_packing_sve(
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

      svfloat64_t vgard = svdup_f64(0.);
      svfloat64_t vdy_dem_0 = svdup_f64(0.);
      svfloat64_t vdy_dem_1 = svdup_f64(0.);
      svfloat64_t vdy_dem_2 = svdup_f64(0.);
      svfloat64_t vdy_dem_3 = svdup_f64(0.);

      assert(last_layer_size % svcntd() == 0);
      svfloat64_t vtwo = svdup_f64(2.);
      svfloat64_t vthree = svdup_f64(3.);
      svfloat64_t vfour = svdup_f64(4.);
      svfloat64_t vfive = svdup_f64(5.);

      svbool_t ptrue = svptrue_b64();
      svfloat64_t vnei_sub_jj = svdup_f64((double(nnei - jj)));
      svfloat64_t vxx = svdup_f64(xx);

      svfloat64_t vxx2 = svmul_z(ptrue, vxx, vxx);
      svfloat64_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      svfloat64_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      svfloat64_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      svfloat64_t v2xx1 = svmul_z(ptrue, vtwo, vxx);
      svfloat64_t v3xx2 = svmul_z(ptrue, vthree, vxx2);
      svfloat64_t v4xx3 = svmul_z(ptrue, vfour, vxx3);
      svfloat64_t v5xx4 = svmul_z(ptrue, vfive, vxx4);
      svfloat64_t vll0 = svdup_f64(ll[0]);
      svfloat64_t vll1 = svdup_f64(ll[1]);
      svfloat64_t vll2 = svdup_f64(ll[2]);
      svfloat64_t vll3 = svdup_f64(ll[3]);
      svfloat64_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat64_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat64_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat64_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);
      for(int kk = 0; kk < last_layer_size; kk += svcntd() * 2){
        svfloat64_t vrr0_0 = svld1(ptrue, dy0 + kk);
        svfloat64_t vrr0_1 = svld1(ptrue, dy0 + kk + svcntd());
        svfloat64_t vrr1_0 = svld1(ptrue, dy1 + kk);
        svfloat64_t vrr1_1 = svld1(ptrue, dy1 + kk + svcntd());
        svfloat64_t vrr2_0 = svld1(ptrue, dy2 + kk);
        svfloat64_t vrr2_1 = svld1(ptrue, dy2 + kk + svcntd());
        svfloat64_t vrr3_0 = svld1(ptrue, dy3 + kk);
        svfloat64_t vrr3_1 = svld1(ptrue, dy3 + kk + svcntd());

        const double* TABLE = &table[table_idx * last_layer_size * 6 + kk * 6];
        svfloat64_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat64_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat64_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat64_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        svfloat64_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        svfloat64_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        svfloat64_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        svfloat64_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        svfloat64_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        svfloat64_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        svfloat64_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        svfloat64_t va5_1 = svld1_vnum(ptrue, TABLE, 11);

        // double res = a0 + a1 * xx + a2 * xx2 + a3 * xx3 + a4 * xx4 + a5 * xx5;
        svfloat64_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat64_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);
        svfloat64_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        svfloat64_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        svfloat64_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        svfloat64_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        svfloat64_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        svfloat64_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        svfloat64_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        svfloat64_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);
        svfloat64_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        svfloat64_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        svfloat64_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        svfloat64_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        svfloat64_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        svfloat64_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);
        svfloat64_t vres_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        svfloat64_t vres_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        // a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4
        svfloat64_t tmp9_0 = svmla_z(ptrue, va1_0, va2_0, v2xx1);
        svfloat64_t tmp9_1 = svmla_z(ptrue, va1_1, va2_1, v2xx1);
        svfloat64_t tmp10_0 = svmul_z(ptrue, va3_0, v3xx2);
        svfloat64_t tmp10_1 = svmul_z(ptrue, va3_1, v3xx2);
        svfloat64_t tmp11_0 = svmul_z(ptrue, va4_0, v4xx3);
        svfloat64_t tmp11_1 = svmul_z(ptrue, va4_1, v4xx3);
        svfloat64_t tmp12_0 = svmul_z(ptrue, va5_0, v5xx4);
        svfloat64_t tmp12_1 = svmul_z(ptrue, va5_1, v5xx4);
        svfloat64_t tmp13_0 = svadd_z(ptrue, tmp9_0, tmp10_0);
        svfloat64_t tmp13_1 = svadd_z(ptrue, tmp9_1, tmp10_1);
        svfloat64_t tmp14_0 = svadd_z(ptrue, tmp11_0, tmp12_0);
        svfloat64_t tmp14_1 = svadd_z(ptrue, tmp11_1, tmp12_1);
        svfloat64_t tmp15_0 = svadd_z(ptrue, tmp13_0, tmp14_0); 
        svfloat64_t tmp15_1 = svadd_z(ptrue, tmp13_1, tmp14_1); 

        // dot(ll, rr);
        svfloat64_t tmp16_0 = svmul_z(ptrue, vll0, vrr0_0);
        svfloat64_t tmp16_1 = svmul_z(ptrue, vll0, vrr0_1);
        svfloat64_t tmp17_0 = svmul_z(ptrue, vll1, vrr1_0);
        svfloat64_t tmp17_1 = svmul_z(ptrue, vll1, vrr1_1);
        svfloat64_t tmp18_0 = svmul_z(ptrue, vll2, vrr2_0);
        svfloat64_t tmp18_1 = svmul_z(ptrue, vll2, vrr2_1);
        svfloat64_t tmp19_0 = svmul_z(ptrue, vll3, vrr3_0);
        svfloat64_t tmp19_1 = svmul_z(ptrue, vll3, vrr3_1);
        svfloat64_t tmp20_0 = svadd_z(ptrue, tmp16_0, tmp17_0);
        svfloat64_t tmp20_1 = svadd_z(ptrue, tmp16_1, tmp17_1);
        svfloat64_t tmp21_0 = svadd_z(ptrue, tmp18_0, tmp19_0);
        svfloat64_t tmp21_1 = svadd_z(ptrue, tmp18_1, tmp19_1);
        svfloat64_t tmp22_0 = svadd_z(ptrue, tmp20_0, tmp21_0);
        svfloat64_t tmp22_1 = svadd_z(ptrue, tmp20_1, tmp21_1);

        // grad = (a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4 ) * dot(ll, rr);
        svfloat64_t vgard_0 = svmul_z(ptrue, tmp15_0, tmp22_0);
        svfloat64_t vgard_1 = svmul_z(ptrue, tmp15_1, tmp22_1);

        svfloat64_t vres0_0 = svmul_z(ptrue, vres_0, vrr0_0);
        svfloat64_t vres0_1 = svmul_z(ptrue, vres_1, vrr0_1);
        svfloat64_t vres1_0 = svmul_z(ptrue, vres_0, vrr1_0);
        svfloat64_t vres1_1 = svmul_z(ptrue, vres_1, vrr1_1);
        svfloat64_t vres2_0 = svmul_z(ptrue, vres_0, vrr2_0);
        svfloat64_t vres2_1 = svmul_z(ptrue, vres_1, vrr2_1);
        svfloat64_t vres3_0 = svmul_z(ptrue, vres_0, vrr3_0);
        svfloat64_t vres3_1 = svmul_z(ptrue, vres_1, vrr3_1);
        if(unloop){
          vgard_0 = svmul_z(ptrue, vgard_0, vnei_sub_jj);
          vgard_1 = svmul_z(ptrue, vgard_1, vnei_sub_jj);
          vres0_0 = svmul_z(ptrue, vres0_0, vnei_sub_jj);
          vres0_1 = svmul_z(ptrue, vres0_1, vnei_sub_jj);
          vres1_0 = svmul_z(ptrue, vres1_0, vnei_sub_jj);
          vres1_1 = svmul_z(ptrue, vres1_1, vnei_sub_jj);
          vres2_0 = svmul_z(ptrue, vres2_0, vnei_sub_jj);
          vres2_1 = svmul_z(ptrue, vres2_1, vnei_sub_jj);
          vres3_0 = svmul_z(ptrue, vres3_0, vnei_sub_jj);
          vres3_1 = svmul_z(ptrue, vres3_1, vnei_sub_jj);
        }
        vgard = svadd_z(ptrue, vgard, vgard_0);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_0);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_0);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_0);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_0);
        vgard = svadd_z(ptrue, vgard, vgard_1);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_1);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_1);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_1);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_1);
      }

      dy_dem_x[ii * nnei + jj] = svaddv(ptrue, vgard);
      dy_dem_tmp[0] = svaddv(ptrue, vdy_dem_0);
      dy_dem_tmp[1] = svaddv(ptrue, vdy_dem_1);
      dy_dem_tmp[2] = svaddv(ptrue, vdy_dem_2);
      dy_dem_tmp[3] = svaddv(ptrue, vdy_dem_3);  

      if (unloop) break;
    }
  }
}



void deepmd::tabulate_fusion_grad_cpu_packing_sve(
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

      svfloat32_t vgard = svdup_f32(0.f);
      svfloat32_t vdy_dem_0 = svdup_f32(0.f);
      svfloat32_t vdy_dem_1 = svdup_f32(0.f);
      svfloat32_t vdy_dem_2 = svdup_f32(0.f);
      svfloat32_t vdy_dem_3 = svdup_f32(0.f);

      assert(last_layer_size % svcntw() == 0);

      svfloat32_t vtwo = svdup_f32(2.f);
      svfloat32_t vthree = svdup_f32(3.f);
      svfloat32_t vfour = svdup_f32(4.f);
      svfloat32_t vfive = svdup_f32(5.f);

      svbool_t ptrue = svptrue_b32();
      svfloat32_t vnei_sub_jj = svdup_f32((double(nnei - jj)));
      svfloat32_t vxx = svdup_f32(xx);

      svfloat32_t vxx2 = svmul_z(ptrue, vxx, vxx);
      svfloat32_t vxx3 = svmul_z(ptrue, vxx2, vxx);
      svfloat32_t vxx4 = svmul_z(ptrue, vxx2, vxx2);
      svfloat32_t vxx5 = svmul_z(ptrue, vxx3, vxx2);
      svfloat32_t v2xx1 = svmul_z(ptrue, vtwo, vxx);
      svfloat32_t v3xx2 = svmul_z(ptrue, vthree, vxx2);
      svfloat32_t v4xx3 = svmul_z(ptrue, vfour, vxx3);
      svfloat32_t v5xx4 = svmul_z(ptrue, vfive, vxx4);
      svfloat32_t vll0 = svdup_f32(ll[0]);
      svfloat32_t vll1 = svdup_f32(ll[1]);
      svfloat32_t vll2 = svdup_f32(ll[2]);
      svfloat32_t vll3 = svdup_f32(ll[3]);
      svfloat32_t vll0_ = svmul_z(ptrue, vll0, vnei_sub_jj);
      svfloat32_t vll1_ = svmul_z(ptrue, vll1, vnei_sub_jj);
      svfloat32_t vll2_ = svmul_z(ptrue, vll2, vnei_sub_jj);
      svfloat32_t vll3_ = svmul_z(ptrue, vll3, vnei_sub_jj);
      for(int kk = 0; kk < last_layer_size; kk += svcntw() * 2){
        svfloat32_t vrr0_0 = svld1(ptrue, dy0 + kk);
        svfloat32_t vrr0_1 = svld1(ptrue, dy0 + kk + svcntw());
        svfloat32_t vrr1_0 = svld1(ptrue, dy1 + kk);
        svfloat32_t vrr1_1 = svld1(ptrue, dy1 + kk + svcntw());
        svfloat32_t vrr2_0 = svld1(ptrue, dy2 + kk);
        svfloat32_t vrr2_1 = svld1(ptrue, dy2 + kk + svcntw());
        svfloat32_t vrr3_0 = svld1(ptrue, dy3 + kk);
        svfloat32_t vrr3_1 = svld1(ptrue, dy3 + kk + svcntw());

        const float* TABLE = &table[table_idx * last_layer_size * 6 + kk * 6];
        svfloat32_t va0_0 = svld1_vnum(ptrue, TABLE, 0);
        svfloat32_t va0_1 = svld1_vnum(ptrue, TABLE, 1);
        svfloat32_t va1_0 = svld1_vnum(ptrue, TABLE, 2);
        svfloat32_t va1_1 = svld1_vnum(ptrue, TABLE, 3);
        svfloat32_t va2_0 = svld1_vnum(ptrue, TABLE, 4);
        svfloat32_t va2_1 = svld1_vnum(ptrue, TABLE, 5);
        svfloat32_t va3_0 = svld1_vnum(ptrue, TABLE, 6);
        svfloat32_t va3_1 = svld1_vnum(ptrue, TABLE, 7);
        svfloat32_t va4_0 = svld1_vnum(ptrue, TABLE, 8);
        svfloat32_t va4_1 = svld1_vnum(ptrue, TABLE, 9);
        svfloat32_t va5_0 = svld1_vnum(ptrue, TABLE, 10);
        svfloat32_t va5_1 = svld1_vnum(ptrue, TABLE, 11);

        // double res = a0 + a1 * xx + a2 * xx2 + a3 * xx3 + a4 * xx4 + a5 * xx5;
        svfloat32_t tmp1_0 = svmla_z(ptrue, va0_0, va1_0, vxx);
        svfloat32_t tmp1_1 = svmla_z(ptrue, va0_1, va1_1, vxx);
        svfloat32_t tmp2_0 = svmul_z(ptrue, va2_0, vxx2);
        svfloat32_t tmp2_1 = svmul_z(ptrue, va2_1, vxx2);
        svfloat32_t tmp3_0 = svmul_z(ptrue, va3_0, vxx3);
        svfloat32_t tmp3_1 = svmul_z(ptrue, va3_1, vxx3);
        svfloat32_t tmp4_0 = svmul_z(ptrue, va4_0, vxx4);
        svfloat32_t tmp4_1 = svmul_z(ptrue, va4_1, vxx4);
        svfloat32_t tmp5_0 = svmul_z(ptrue, va5_0, vxx5);
        svfloat32_t tmp5_1 = svmul_z(ptrue, va5_1, vxx5);
        svfloat32_t tmp6_0 = svadd_z(ptrue, tmp1_0, tmp2_0);
        svfloat32_t tmp6_1 = svadd_z(ptrue, tmp1_1, tmp2_1);
        svfloat32_t tmp7_0 = svadd_z(ptrue, tmp3_0, tmp4_0);
        svfloat32_t tmp7_1 = svadd_z(ptrue, tmp3_1, tmp4_1);
        svfloat32_t tmp8_0 = svadd_z(ptrue, tmp6_0, tmp5_0);
        svfloat32_t tmp8_1 = svadd_z(ptrue, tmp6_1, tmp5_1);
        svfloat32_t vres_0 = svadd_z(ptrue, tmp7_0, tmp8_0);
        svfloat32_t vres_1 = svadd_z(ptrue, tmp7_1, tmp8_1);

        // a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4
        svfloat32_t tmp9_0 = svmla_z(ptrue, va1_0, va2_0, v2xx1);
        svfloat32_t tmp9_1 = svmla_z(ptrue, va1_1, va2_1, v2xx1);
        svfloat32_t tmp10_0 = svmul_z(ptrue, va3_0, v3xx2);
        svfloat32_t tmp10_1 = svmul_z(ptrue, va3_1, v3xx2);
        svfloat32_t tmp11_0 = svmul_z(ptrue, va4_0, v4xx3);
        svfloat32_t tmp11_1 = svmul_z(ptrue, va4_1, v4xx3);
        svfloat32_t tmp12_0 = svmul_z(ptrue, va5_0, v5xx4);
        svfloat32_t tmp12_1 = svmul_z(ptrue, va5_1, v5xx4);
        svfloat32_t tmp13_0 = svadd_z(ptrue, tmp9_0, tmp10_0);
        svfloat32_t tmp13_1 = svadd_z(ptrue, tmp9_1, tmp10_1);
        svfloat32_t tmp14_0 = svadd_z(ptrue, tmp11_0, tmp12_0);
        svfloat32_t tmp14_1 = svadd_z(ptrue, tmp11_1, tmp12_1);
        svfloat32_t tmp15_0 = svadd_z(ptrue, tmp13_0, tmp14_0); 
        svfloat32_t tmp15_1 = svadd_z(ptrue, tmp13_1, tmp14_1); 

        // dot(ll, rr);
        svfloat32_t tmp16_0 = svmul_z(ptrue, vll0, vrr0_0);
        svfloat32_t tmp16_1 = svmul_z(ptrue, vll0, vrr0_1);
        svfloat32_t tmp17_0 = svmul_z(ptrue, vll1, vrr1_0);
        svfloat32_t tmp17_1 = svmul_z(ptrue, vll1, vrr1_1);
        svfloat32_t tmp18_0 = svmul_z(ptrue, vll2, vrr2_0);
        svfloat32_t tmp18_1 = svmul_z(ptrue, vll2, vrr2_1);
        svfloat32_t tmp19_0 = svmul_z(ptrue, vll3, vrr3_0);
        svfloat32_t tmp19_1 = svmul_z(ptrue, vll3, vrr3_1);
        svfloat32_t tmp20_0 = svadd_z(ptrue, tmp16_0, tmp17_0);
        svfloat32_t tmp20_1 = svadd_z(ptrue, tmp16_1, tmp17_1);
        svfloat32_t tmp21_0 = svadd_z(ptrue, tmp18_0, tmp19_0);
        svfloat32_t tmp21_1 = svadd_z(ptrue, tmp18_1, tmp19_1);
        svfloat32_t tmp22_0 = svadd_z(ptrue, tmp20_0, tmp21_0);
        svfloat32_t tmp22_1 = svadd_z(ptrue, tmp20_1, tmp21_1);

        // grad = (a1 + 2 * a2 * xx + 3 * a3 * xx2 + 4 * a4 * xx3 + 5 * a5 *xx4 ) * dot(ll, rr);
        svfloat32_t vgard_0 = svmul_z(ptrue, tmp15_0, tmp22_0);
        svfloat32_t vgard_1 = svmul_z(ptrue, tmp15_1, tmp22_1);

        svfloat32_t vres0_0 = svmul_z(ptrue, vres_0, vrr0_0);
        svfloat32_t vres0_1 = svmul_z(ptrue, vres_1, vrr0_1);
        svfloat32_t vres1_0 = svmul_z(ptrue, vres_0, vrr1_0);
        svfloat32_t vres1_1 = svmul_z(ptrue, vres_1, vrr1_1);
        svfloat32_t vres2_0 = svmul_z(ptrue, vres_0, vrr2_0);
        svfloat32_t vres2_1 = svmul_z(ptrue, vres_1, vrr2_1);
        svfloat32_t vres3_0 = svmul_z(ptrue, vres_0, vrr3_0);
        svfloat32_t vres3_1 = svmul_z(ptrue, vres_1, vrr3_1);
        if(unloop){
          vgard_0 = svmul_z(ptrue, vgard_0, vnei_sub_jj);
          vgard_1 = svmul_z(ptrue, vgard_1, vnei_sub_jj);
          vres0_0 = svmul_z(ptrue, vres0_0, vnei_sub_jj);
          vres0_1 = svmul_z(ptrue, vres0_1, vnei_sub_jj);
          vres1_0 = svmul_z(ptrue, vres1_0, vnei_sub_jj);
          vres1_1 = svmul_z(ptrue, vres1_1, vnei_sub_jj);
          vres2_0 = svmul_z(ptrue, vres2_0, vnei_sub_jj);
          vres2_1 = svmul_z(ptrue, vres2_1, vnei_sub_jj);
          vres3_0 = svmul_z(ptrue, vres3_0, vnei_sub_jj);
          vres3_1 = svmul_z(ptrue, vres3_1, vnei_sub_jj);
        }
        vgard = svadd_z(ptrue, vgard, vgard_0);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_0);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_0);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_0);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_0);
        vgard = svadd_z(ptrue, vgard, vgard_1);
        vdy_dem_0 = svadd_z(ptrue, vdy_dem_0, vres0_1);
        vdy_dem_1 = svadd_z(ptrue, vdy_dem_1, vres1_1);
        vdy_dem_2 = svadd_z(ptrue, vdy_dem_2, vres2_1);
        vdy_dem_3 = svadd_z(ptrue, vdy_dem_3, vres3_1);
      }

      dy_dem_x[ii * nnei + jj] = svaddv(ptrue, vgard);
      dy_dem_tmp[0] = svaddv(ptrue, vdy_dem_0);
      dy_dem_tmp[1] = svaddv(ptrue, vdy_dem_1);
      dy_dem_tmp[2] = svaddv(ptrue, vdy_dem_2);
      dy_dem_tmp[3] = svaddv(ptrue, vdy_dem_3);  

      if (unloop) break;
    }
  }
}


#endif /* __ARM_FEATURE_SVE__ */
