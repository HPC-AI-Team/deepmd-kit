#include <iostream>
#include <stdexcept>
#include <cstring>
#include "prod_virial_opt.h"
#include "lib_tools.h"


#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 

void 
deepmd::prod_virial_a_cpu_sve(
    double * virial, 
    double * atom_virial, 
    const double * net_deriv, 
    const double * env_deriv, 
    const double * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
{
  const int ndescrpt = 4 * nnei;

  svfloat64_t tmp_0_0 = svdup_f64(0.);
  svfloat64_t tmp_0_1 = svdup_f64(0.);
  svfloat64_t tmp_0_2 = svdup_f64(0.);
  svfloat64_t tmp_0_3 = svdup_f64(0.);
  svfloat64_t tmp_0_4 = svdup_f64(0.);
  svfloat64_t tmp_0_5 = svdup_f64(0.);
  svfloat64_t tmp_0_6 = svdup_f64(0.);
  svfloat64_t tmp_0_7 = svdup_f64(0.);
  svfloat64_t tmp_0_8 = svdup_f64(0.);

  svfloat64_t tmp_1_0 = svdup_f64(0.);
  svfloat64_t tmp_1_1 = svdup_f64(0.);
  svfloat64_t tmp_1_2 = svdup_f64(0.);
  svfloat64_t tmp_1_3 = svdup_f64(0.);
  svfloat64_t tmp_1_4 = svdup_f64(0.);
  svfloat64_t tmp_1_5 = svdup_f64(0.);
  svfloat64_t tmp_1_6 = svdup_f64(0.);
  svfloat64_t tmp_1_7 = svdup_f64(0.);
  svfloat64_t tmp_1_8 = svdup_f64(0.);

  svfloat64_t tmp_2_0 = svdup_f64(0.);
  svfloat64_t tmp_2_1 = svdup_f64(0.);
  svfloat64_t tmp_2_2 = svdup_f64(0.);
  svfloat64_t tmp_2_3 = svdup_f64(0.);
  svfloat64_t tmp_2_4 = svdup_f64(0.);
  svfloat64_t tmp_2_5 = svdup_f64(0.);
  svfloat64_t tmp_2_6 = svdup_f64(0.);
  svfloat64_t tmp_2_7 = svdup_f64(0.);
  svfloat64_t tmp_2_8 = svdup_f64(0.);

  svfloat64_t tmp_3_0 = svdup_f64(0.);
  svfloat64_t tmp_3_1 = svdup_f64(0.);
  svfloat64_t tmp_3_2 = svdup_f64(0.);
  svfloat64_t tmp_3_3 = svdup_f64(0.);
  svfloat64_t tmp_3_4 = svdup_f64(0.);
  svfloat64_t tmp_3_5 = svdup_f64(0.);
  svfloat64_t tmp_3_6 = svdup_f64(0.);
  svfloat64_t tmp_3_7 = svdup_f64(0.);
  svfloat64_t tmp_3_8 = svdup_f64(0.);

  svuint64_t index0 = svindex_u64(0, 12);
  svuint64_t index1 = svindex_u64(1, 12);
  svuint64_t index2 = svindex_u64(2, 12);
  svuint64_t index3 = svindex_u64(3, 12);
  svuint64_t index4 = svindex_u64(4, 12);
  svuint64_t index5 = svindex_u64(5, 12);
  svuint64_t index6 = svindex_u64(6, 12);
  svuint64_t index7 = svindex_u64(7, 12);
  svuint64_t index8 = svindex_u64(8, 12);
  svuint64_t index9 = svindex_u64(9, 12);
  svuint64_t index10 = svindex_u64(10, 12);
  svuint64_t index11 = svindex_u64(11, 12);

  svfloat64_t
    env_0_0, env_0_1, env_0_2,
    env_1_0, env_1_1, env_1_2,
    env_2_0, env_2_1, env_2_2,
    env_3_0, env_3_1, env_3_2;

  svbool_t pt = svptrue_b64();

  // compute virial of a frame
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;
    // deriv wrt neighbors
    const int* nlist_i = &nlist[i_idx * nnei];
    for (int jj = 0; jj < nnei; jj += svcntd()){

      svbool_t pg0 = svwhilelt_b64(jj, nnei);
      svint64_t vj_idx = svld1sw_s64(pg0, nlist_i + jj);
      svbool_t ge_zero = svcmpge(pg0,vj_idx,0);
      svbool_t pg1 = svand_z(pt,pg0,ge_zero);
      
      // pg1 all false
      if(!svptest_any(pt, pg1)){
        continue;
      }

      // 3*8
      svfloat64x3_t vrij = svld3(pg1, &rij[i_idx * nnei * 3 + jj * 3]);
      svfloat64_t rij_0 = svget3(vrij,0);
      svfloat64_t rij_1 = svget3(vrij,1);
      svfloat64_t rij_2 = svget3(vrij,2);

      // 4*8
      svfloat64x4_t vpref = svld4(pg1, &net_deriv[i_idx * ndescrpt + jj * 4]);
      svfloat64_t vpref_0 = svget4(vpref,0);
      svfloat64_t vpref_1 = svget4(vpref,1);
      svfloat64_t vpref_2 = svget4(vpref,2);
      svfloat64_t vpref_3 = svget4(vpref,3);

      // 12*8
      const double* base = &env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3];

      if(svptest_last(pt, pg1)){
        env_0_0 = svld1_vnum(pt, base, 0);
        env_0_1 = svld1_vnum(pt, base, 1);
        env_0_2 = svld1_vnum(pt, base, 2);
        env_1_0 = svld1_vnum(pt, base, 3);
        env_1_1 = svld1_vnum(pt, base, 4);
        env_1_2 = svld1_vnum(pt, base, 5);
        env_2_0 = svld1_vnum(pt, base, 6);
        env_2_1 = svld1_vnum(pt, base, 7);
        env_2_2 = svld1_vnum(pt, base, 8);
        env_3_0 = svld1_vnum(pt, base, 9);
        env_3_1 = svld1_vnum(pt, base, 10);
        env_3_2 = svld1_vnum(pt, base, 11);
        sve_aos2soa_12x8_inplace(
          env_0_0, env_0_1, env_0_2,
          env_1_0, env_1_1, env_1_2,
          env_2_0, env_2_1, env_2_2,
          env_3_0, env_3_1, env_3_2
        );
      }else{
        env_0_0 = svld1_gather_u64index_f64(pg1, base, index0);
        env_0_1 = svld1_gather_u64index_f64(pg1, base, index1);
        env_0_2 = svld1_gather_u64index_f64(pg1, base, index2);
        env_1_0 = svld1_gather_u64index_f64(pg1, base, index3);
        env_1_1 = svld1_gather_u64index_f64(pg1, base, index4);
        env_1_2 = svld1_gather_u64index_f64(pg1, base, index5);
        env_2_0 = svld1_gather_u64index_f64(pg1, base, index6);
        env_2_1 = svld1_gather_u64index_f64(pg1, base, index7);
        env_2_2 = svld1_gather_u64index_f64(pg1, base, index8);
        env_3_0 = svld1_gather_u64index_f64(pg1, base, index9);
        env_3_1 = svld1_gather_u64index_f64(pg1, base, index10);
        env_3_2 = svld1_gather_u64index_f64(pg1, base, index11);
      }

      env_0_0 = svmul_z(pg1, env_0_0, vpref_0);
      env_0_1 = svmul_z(pg1, env_0_1, vpref_0);
      env_0_2 = svmul_z(pg1, env_0_2, vpref_0);
      env_1_0 = svmul_z(pg1, env_1_0, vpref_1);
      env_1_1 = svmul_z(pg1, env_1_1, vpref_1);
      env_1_2 = svmul_z(pg1, env_1_2, vpref_1);
      env_2_0 = svmul_z(pg1, env_2_0, vpref_2);
      env_2_1 = svmul_z(pg1, env_2_1, vpref_2);
      env_2_2 = svmul_z(pg1, env_2_2, vpref_2);
      env_3_0 = svmul_z(pg1, env_3_0, vpref_3);
      env_3_1 = svmul_z(pg1, env_3_1, vpref_3);
      env_3_2 = svmul_z(pg1, env_3_2, vpref_3);

      // 
      tmp_0_0 = svmla_m(pg1, tmp_0_0, env_0_0, rij_0);
      tmp_0_1 = svmla_m(pg1, tmp_0_1, env_0_0, rij_1);
      tmp_0_2 = svmla_m(pg1, tmp_0_2, env_0_0, rij_2);
      tmp_0_3 = svmla_m(pg1, tmp_0_3, env_0_1, rij_0);
      tmp_0_4 = svmla_m(pg1, tmp_0_4, env_0_1, rij_1);
      tmp_0_5 = svmla_m(pg1, tmp_0_5, env_0_1, rij_2);
      tmp_0_6 = svmla_m(pg1, tmp_0_6, env_0_2, rij_0);
      tmp_0_7 = svmla_m(pg1, tmp_0_7, env_0_2, rij_1);
      tmp_0_8 = svmla_m(pg1, tmp_0_8, env_0_2, rij_2);
      tmp_1_0 = svmla_m(pg1, tmp_1_0, env_1_0, rij_0);
      tmp_1_1 = svmla_m(pg1, tmp_1_1, env_1_0, rij_1);
      tmp_1_2 = svmla_m(pg1, tmp_1_2, env_1_0, rij_2);
      tmp_1_3 = svmla_m(pg1, tmp_1_3, env_1_1, rij_0);
      tmp_1_4 = svmla_m(pg1, tmp_1_4, env_1_1, rij_1);
      tmp_1_5 = svmla_m(pg1, tmp_1_5, env_1_1, rij_2);
      tmp_1_6 = svmla_m(pg1, tmp_1_6, env_1_2, rij_0);
      tmp_1_7 = svmla_m(pg1, tmp_1_7, env_1_2, rij_1);
      tmp_1_8 = svmla_m(pg1, tmp_1_8, env_1_2, rij_2);
      tmp_2_0 = svmla_m(pg1, tmp_2_0, env_2_0, rij_0);
      tmp_2_1 = svmla_m(pg1, tmp_2_1, env_2_0, rij_1);
      tmp_2_2 = svmla_m(pg1, tmp_2_2, env_2_0, rij_2);
      tmp_2_3 = svmla_m(pg1, tmp_2_3, env_2_1, rij_0);
      tmp_2_4 = svmla_m(pg1, tmp_2_4, env_2_1, rij_1);
      tmp_2_5 = svmla_m(pg1, tmp_2_5, env_2_1, rij_2);
      tmp_2_6 = svmla_m(pg1, tmp_2_6, env_2_2, rij_0);
      tmp_2_7 = svmla_m(pg1, tmp_2_7, env_2_2, rij_1);
      tmp_2_8 = svmla_m(pg1, tmp_2_8, env_2_2, rij_2);
      tmp_3_0 = svmla_m(pg1, tmp_3_0, env_3_0, rij_0);
      tmp_3_1 = svmla_m(pg1, tmp_3_1, env_3_0, rij_1);
      tmp_3_2 = svmla_m(pg1, tmp_3_2, env_3_0, rij_2);
      tmp_3_3 = svmla_m(pg1, tmp_3_3, env_3_1, rij_0);
      tmp_3_4 = svmla_m(pg1, tmp_3_4, env_3_1, rij_1);
      tmp_3_5 = svmla_m(pg1, tmp_3_5, env_3_1, rij_2);
      tmp_3_6 = svmla_m(pg1, tmp_3_6, env_3_2, rij_0);
      tmp_3_7 = svmla_m(pg1, tmp_3_7, env_3_2, rij_1);
      tmp_3_8 = svmla_m(pg1, tmp_3_8, env_3_2, rij_2);
    }
  }  
  tmp_0_0 = svadd_z(pt, tmp_0_0, tmp_1_0);
  tmp_2_0 = svadd_z(pt, tmp_2_0, tmp_3_0);
  tmp_0_1 = svadd_z(pt, tmp_0_1, tmp_1_1);
  tmp_2_1 = svadd_z(pt, tmp_2_1, tmp_3_1);
  tmp_0_2 = svadd_z(pt, tmp_0_2, tmp_1_2);
  tmp_2_2 = svadd_z(pt, tmp_2_2, tmp_3_2);
  tmp_0_3 = svadd_z(pt, tmp_0_3, tmp_1_3);
  tmp_2_3 = svadd_z(pt, tmp_2_3, tmp_3_3);
  tmp_0_4 = svadd_z(pt, tmp_0_4, tmp_1_4);
  tmp_2_4 = svadd_z(pt, tmp_2_4, tmp_3_4);
  tmp_0_5 = svadd_z(pt, tmp_0_5, tmp_1_5);
  tmp_2_5 = svadd_z(pt, tmp_2_5, tmp_3_5);
  tmp_0_6 = svadd_z(pt, tmp_0_6, tmp_1_6);
  tmp_2_6 = svadd_z(pt, tmp_2_6, tmp_3_6);
  tmp_0_7 = svadd_z(pt, tmp_0_7, tmp_1_7);
  tmp_2_7 = svadd_z(pt, tmp_2_7, tmp_3_7);
  tmp_0_8 = svadd_z(pt, tmp_0_8, tmp_1_8);
  tmp_2_8 = svadd_z(pt, tmp_2_8, tmp_3_8);
  tmp_0_0 = svadd_z(pt, tmp_0_0, tmp_2_0);
  tmp_0_1 = svadd_z(pt, tmp_0_1, tmp_2_1);
  tmp_0_2 = svadd_z(pt, tmp_0_2, tmp_2_2);
  tmp_0_3 = svadd_z(pt, tmp_0_3, tmp_2_3);
  tmp_0_4 = svadd_z(pt, tmp_0_4, tmp_2_4);
  tmp_0_5 = svadd_z(pt, tmp_0_5, tmp_2_5);
  tmp_0_6 = svadd_z(pt, tmp_0_6, tmp_2_6);
  tmp_0_7 = svadd_z(pt, tmp_0_7, tmp_2_7);
  tmp_0_8 = svadd_z(pt, tmp_0_8, tmp_2_8);

  virial[0] = svaddv(pt, tmp_0_0);
  virial[1] = svaddv(pt, tmp_0_1);
  virial[2] = svaddv(pt, tmp_0_2);
  virial[3] = svaddv(pt, tmp_0_3);
  virial[4] = svaddv(pt, tmp_0_4);
  virial[5] = svaddv(pt, tmp_0_5);
  virial[6] = svaddv(pt, tmp_0_6);
  virial[7] = svaddv(pt, tmp_0_7);
  virial[8] = svaddv(pt, tmp_0_8);
}


void 
deepmd::prod_virial_a_cpu_sve(
    float * virial, 
    float * atom_virial, 
    const float * net_deriv, 
    const float * env_deriv, 
    const float * rij, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei)
{
  const int ndescrpt = 4 * nnei;

  float tmp_0_0 = 0.f;
  float tmp_0_1 = 0.f;
  float tmp_0_2 = 0.f;
  float tmp_0_3 = 0.f;
  float tmp_0_4 = 0.f;
  float tmp_0_5 = 0.f;
  float tmp_0_6 = 0.f;
  float tmp_0_7 = 0.f;
  float tmp_0_8 = 0.f;

  float tmp_1_0 = 0.f;
  float tmp_1_1 = 0.f;
  float tmp_1_2 = 0.f;
  float tmp_1_3 = 0.f;
  float tmp_1_4 = 0.f;
  float tmp_1_5 = 0.f;
  float tmp_1_6 = 0.f;
  float tmp_1_7 = 0.f;
  float tmp_1_8 = 0.f;

  float tmp_2_0 = 0.f;
  float tmp_2_1 = 0.f;
  float tmp_2_2 = 0.f;
  float tmp_2_3 = 0.f;
  float tmp_2_4 = 0.f;
  float tmp_2_5 = 0.f;
  float tmp_2_6 = 0.f;
  float tmp_2_7 = 0.f;
  float tmp_2_8 = 0.f;

  float tmp_3_0 = 0.f;
  float tmp_3_1 = 0.f;
  float tmp_3_2 = 0.f;
  float tmp_3_3 = 0.f;
  float tmp_3_4 = 0.f;
  float tmp_3_5 = 0.f;
  float tmp_3_6 = 0.f;
  float tmp_3_7 = 0.f;
  float tmp_3_8 = 0.f;

  // compute virial of a frame
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;
    // deriv wrt neighbors
    const int* nlist_i = &nlist[i_idx * nnei];
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist_i[jj];
      if (j_idx < 0) break;
      // 3 * 8
      float rij_0 = rij[i_idx * nnei * 3 + jj * 3 + 0];
      float rij_1 = rij[i_idx * nnei * 3 + jj * 3 + 1];
      float rij_2 = rij[i_idx * nnei * 3 + jj * 3 + 2];

      // 4 * 8
      float pref_0 = net_deriv[i_idx * ndescrpt + jj * 4 + 0];
      float pref_1 = net_deriv[i_idx * ndescrpt + jj * 4 + 1];
      float pref_2 = net_deriv[i_idx * ndescrpt + jj * 4 + 2];
      float pref_3 = net_deriv[i_idx * ndescrpt + jj * 4 + 3];

      // 12 * 8
      float env_0_0 = pref_0 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 0 * 3 + 0];
      float env_0_1 = pref_0 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 0 * 3 + 1];
      float env_0_2 = pref_0 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 0 * 3 + 2];
      float env_1_0 = pref_1 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 1 * 3 + 0];
      float env_1_1 = pref_1 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 1 * 3 + 1];
      float env_1_2 = pref_1 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 1 * 3 + 2];
      float env_2_0 = pref_2 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 2 * 3 + 0];
      float env_2_1 = pref_2 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 2 * 3 + 1];
      float env_2_2 = pref_2 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 2 * 3 + 2];
      float env_3_0 = pref_3 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 3 * 3 + 0];
      float env_3_1 = pref_3 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 3 * 3 + 1];
      float env_3_2 = pref_3 * env_deriv[i_idx * ndescrpt * 3 + jj * 4 * 3 + 3 * 3 + 2];

      tmp_0_0 += env_0_0 * rij_0;
      tmp_0_1 += env_0_0 * rij_1;
      tmp_0_2 += env_0_0 * rij_2;
      tmp_0_3 += env_0_1 * rij_0;
      tmp_0_4 += env_0_1 * rij_1;
      tmp_0_5 += env_0_1 * rij_2;
      tmp_0_6 += env_0_2 * rij_0;
      tmp_0_7 += env_0_2 * rij_1;
      tmp_0_8 += env_0_2 * rij_2;
      tmp_1_0 += env_1_0 * rij_0;
      tmp_1_1 += env_1_0 * rij_1;
      tmp_1_2 += env_1_0 * rij_2;
      tmp_1_3 += env_1_1 * rij_0;
      tmp_1_4 += env_1_1 * rij_1;
      tmp_1_5 += env_1_1 * rij_2;
      tmp_1_6 += env_1_2 * rij_0;
      tmp_1_7 += env_1_2 * rij_1;
      tmp_1_8 += env_1_2 * rij_2;
      tmp_2_0 += env_2_0 * rij_0;
      tmp_2_1 += env_2_0 * rij_1;
      tmp_2_2 += env_2_0 * rij_2;
      tmp_2_3 += env_2_1 * rij_0;
      tmp_2_4 += env_2_1 * rij_1;
      tmp_2_5 += env_2_1 * rij_2;
      tmp_2_6 += env_2_2 * rij_0;
      tmp_2_7 += env_2_2 * rij_1;
      tmp_2_8 += env_2_2 * rij_2;
      tmp_3_0 += env_3_0 * rij_0;
      tmp_3_1 += env_3_0 * rij_1;
      tmp_3_2 += env_3_0 * rij_2;
      tmp_3_3 += env_3_1 * rij_0;
      tmp_3_4 += env_3_1 * rij_1;
      tmp_3_5 += env_3_1 * rij_2;
      tmp_3_6 += env_3_2 * rij_0;
      tmp_3_7 += env_3_2 * rij_1;
      tmp_3_8 += env_3_2 * rij_2;
    }
  }  
  virial[0] = tmp_0_0 + tmp_1_0 + tmp_2_0 + tmp_3_0;
  virial[1] = tmp_0_1 + tmp_1_1 + tmp_2_1 + tmp_3_1;
  virial[2] = tmp_0_2 + tmp_1_2 + tmp_2_2 + tmp_3_2;
  virial[3] = tmp_0_3 + tmp_1_3 + tmp_2_3 + tmp_3_3;
  virial[4] = tmp_0_4 + tmp_1_4 + tmp_2_4 + tmp_3_4;
  virial[5] = tmp_0_5 + tmp_1_5 + tmp_2_5 + tmp_3_5;
  virial[6] = tmp_0_6 + tmp_1_6 + tmp_2_6 + tmp_3_6;
  virial[7] = tmp_0_7 + tmp_1_7 + tmp_2_7 + tmp_3_7;
  virial[8] = tmp_0_8 + tmp_1_8 + tmp_2_8 + tmp_3_8;
}

#endif
