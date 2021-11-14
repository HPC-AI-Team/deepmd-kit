#include <stdexcept>
#include <cstring>
#include "prod_force_opt.h"
#include "lib_tools.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 


void 
deepmd::
prod_force_a_cpu_sve(
    double * force, 
    const double * net_deriv, 
    const double * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei) 
{
  const int ndescrpt = 4 * nnei;
  memset(force, '\0', sizeof(double) * nall * 3);
  // compute force of a frame

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

  svbool_t pt = svptrue_b64();

  svfloat64_t 
    env_deriv_0_0, env_deriv_0_1, env_deriv_0_2,
    env_deriv_1_0, env_deriv_1_1, env_deriv_1_2,
    env_deriv_2_0, env_deriv_2_1, env_deriv_2_2,
    env_deriv_3_0, env_deriv_3_1, env_deriv_3_2;


  for (int i_idx = 0; i_idx < nloc; ++i_idx) {
    const int* nlist_i = &nlist[i_idx * nnei];
    const double* net_deriv_i = &net_deriv[i_idx * ndescrpt];
    const double* env_deriv_i = &env_deriv[i_idx * ndescrpt * 3];
    // deriv wrt neighbors
    
    svfloat64_t force_i_0 = svdup_f64(0.);
    svfloat64_t force_i_1 = svdup_f64(0.);
    svfloat64_t force_i_2 = svdup_f64(0.);

    for (int jj = 0; jj < nnei; jj += svcntd()) {

      svbool_t pg0 = svwhilelt_b64(jj, nnei);
      svint64_t vj_idx = svld1sw_s64(pg0, nlist_i + jj);

      svbool_t ge_zero = svcmpge(pg0,vj_idx,0);
      svbool_t pg1 = svand_z(pt,pg0,ge_zero);
      // pg1 all false
      
      if(!svptest_any(pt, pg1)){
        continue;
      }

      svint64_t vj_idx_0 = svmul_z(pg1, vj_idx, 3);
      svint64_t vj_idx_1 = svadd_z(pg1, vj_idx_0, 1);
      svint64_t vj_idx_2 = svadd_z(pg1, vj_idx_0, 2);

      // 4*8
      svfloat64x4_t vnet_deriv = svld4(pg1, &net_deriv_i[jj * 4]);
      svfloat64_t net_deriv_0 = svget4(vnet_deriv, 0);
      svfloat64_t net_deriv_1 = svget4(vnet_deriv, 1);
      svfloat64_t net_deriv_2 = svget4(vnet_deriv, 2);
      svfloat64_t net_deriv_3 = svget4(vnet_deriv, 3);
      
      // 12*8
      const double* base = &env_deriv_i[jj * 12];
      
      if(svptest_last(pt, pg1)){
        env_deriv_0_0 = svld1_vnum(pt, base, 0);
        env_deriv_0_1 = svld1_vnum(pt, base, 1);
        env_deriv_0_2 = svld1_vnum(pt, base, 2);
        env_deriv_1_0 = svld1_vnum(pt, base, 3);
        env_deriv_1_1 = svld1_vnum(pt, base, 4);
        env_deriv_1_2 = svld1_vnum(pt, base, 5);
        env_deriv_2_0 = svld1_vnum(pt, base, 6);
        env_deriv_2_1 = svld1_vnum(pt, base, 7);
        env_deriv_2_2 = svld1_vnum(pt, base, 8);
        env_deriv_3_0 = svld1_vnum(pt, base, 9);
        env_deriv_3_1 = svld1_vnum(pt, base, 10);
        env_deriv_3_2 = svld1_vnum(pt, base, 11);
        sve_aos2soa_12x8_inplace(
          env_deriv_0_0, env_deriv_0_1, env_deriv_0_2,
          env_deriv_1_0, env_deriv_1_1, env_deriv_1_2,
          env_deriv_2_0, env_deriv_2_1, env_deriv_2_2,
          env_deriv_3_0, env_deriv_3_1, env_deriv_3_2
        );
      }else{
        env_deriv_0_0 = svld1_gather_u64index_f64(pg1, base, index0);
        env_deriv_0_1 = svld1_gather_u64index_f64(pg1, base, index1);
        env_deriv_0_2 = svld1_gather_u64index_f64(pg1, base, index2);
        env_deriv_1_0 = svld1_gather_u64index_f64(pg1, base, index3);
        env_deriv_1_1 = svld1_gather_u64index_f64(pg1, base, index4);
        env_deriv_1_2 = svld1_gather_u64index_f64(pg1, base, index5);
        env_deriv_2_0 = svld1_gather_u64index_f64(pg1, base, index6);
        env_deriv_2_1 = svld1_gather_u64index_f64(pg1, base, index7);
        env_deriv_2_2 = svld1_gather_u64index_f64(pg1, base, index8);
        env_deriv_3_0 = svld1_gather_u64index_f64(pg1, base, index9);
        env_deriv_3_1 = svld1_gather_u64index_f64(pg1, base, index10);
        env_deriv_3_2 = svld1_gather_u64index_f64(pg1, base, index11);
      }
      svfloat64_t force_0_0 = svmul_z(pg1, net_deriv_0, env_deriv_0_0);
      svfloat64_t force_0_1 = svmul_z(pg1, net_deriv_0, env_deriv_0_1);
      svfloat64_t force_0_2 = svmul_z(pg1, net_deriv_0, env_deriv_0_2);

      svfloat64_t force_1_0 = svmul_z(pg1, net_deriv_1, env_deriv_1_0);
      svfloat64_t force_1_1 = svmul_z(pg1, net_deriv_1, env_deriv_1_1);
      svfloat64_t force_1_2 = svmul_z(pg1, net_deriv_1, env_deriv_1_2);
      
      svfloat64_t force_2_0 = svmul_z(pg1, net_deriv_2, env_deriv_2_0);
      svfloat64_t force_2_1 = svmul_z(pg1, net_deriv_2, env_deriv_2_1);
      svfloat64_t force_2_2 = svmul_z(pg1, net_deriv_2, env_deriv_2_2);
      
      svfloat64_t force_3_0 = svmul_z(pg1, net_deriv_3, env_deriv_3_0);
      svfloat64_t force_3_1 = svmul_z(pg1, net_deriv_3, env_deriv_3_1);
      svfloat64_t force_3_2 = svmul_z(pg1, net_deriv_3, env_deriv_3_2);

      force_0_0 = svadd_z(pg1, force_0_0,  force_1_0);
      force_2_0 = svadd_z(pg1, force_2_0,  force_3_0);
      force_0_1 = svadd_z(pg1, force_0_1,  force_1_1);
      force_2_1 = svadd_z(pg1, force_2_1,  force_3_1);
      force_0_2 = svadd_z(pg1, force_0_2,  force_1_2);
      force_2_2 = svadd_z(pg1, force_2_2,  force_3_2);
      force_0_0 = svadd_z(pg1, force_0_0,  force_2_0);
      force_0_1 = svadd_z(pg1, force_0_1,  force_2_1);
      force_0_2 = svadd_z(pg1, force_0_2,  force_2_2);

      svfloat64_t force_j_0 = svld1_gather_s64index_f64(pg1, force, vj_idx_0);
      svfloat64_t force_j_1 = svld1_gather_s64index_f64(pg1, force, vj_idx_1);
      svfloat64_t force_j_2 = svld1_gather_s64index_f64(pg1, force, vj_idx_2);

      force_j_0 = svadd_z(pg1, force_j_0, force_0_0);
      force_j_1 = svadd_z(pg1, force_j_1, force_0_1);
      force_j_2 = svadd_z(pg1, force_j_2, force_0_2);

      svst1_scatter_s64index_f64(pg1, force, vj_idx_0, force_j_0);
      svst1_scatter_s64index_f64(pg1, force, vj_idx_1, force_j_1);
      svst1_scatter_s64index_f64(pg1, force, vj_idx_2, force_j_2);

      force_i_0 = svadd_m(pg1, force_i_0, force_0_0);
      force_i_1 = svadd_m(pg1, force_i_1, force_0_1);
      force_i_2 = svadd_m(pg1, force_i_2, force_0_2);

    }

    force[i_idx * 3 + 0] -= svaddv(pt, force_i_0);
    force[i_idx * 3 + 1] -= svaddv(pt, force_i_1);
    force[i_idx * 3 + 2] -= svaddv(pt, force_i_2);
  }
}

void 
deepmd::
prod_force_a_cpu_sve(
    float * force, 
    const float * net_deriv, 
    const float * env_deriv, 
    const int * nlist, 
    const int nloc, 
    const int nall, 
    const int nnei) 
{
  const int ndescrpt = 4 * nnei;

  memset(force, 0.0, sizeof(float) * nall * 3);
  // compute force of a frame
  for (int i_idx = 0; i_idx < nloc; ++i_idx) {
    // deriv wrt center atom
    for (int aa = 0; aa < ndescrpt; ++aa) {
      force[i_idx * 3 + 0] -= net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
      force[i_idx * 3 + 1] -= net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
      force[i_idx * 3 + 2] -= net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
    }
    // deriv wrt neighbors
    for (int jj = 0; jj < nnei; ++jj) {
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      int aa_start, aa_end;
      for (int aa = jj * 4; aa < jj * 4 + 4; ++aa) {
        force[j_idx * 3 + 0] += net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 0];
        force[j_idx * 3 + 1] += net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 1];
        force[j_idx * 3 + 2] += net_deriv[i_idx * ndescrpt + aa] * env_deriv[i_idx * ndescrpt * 3 + aa * 3 + 2];
      }
    }
  }
}

#endif




