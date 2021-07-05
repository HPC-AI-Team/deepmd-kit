#include "env_mat.h"
#include "switcher.h"
#include "tools.h"
#include <string.h>

#ifdef __ARM_FEATURE_SVE 
#include <arm_sve.h> 

inline void spline5_switch_vector (
  svbool_t pg,
  svfloat64_t & vv,
	svfloat64_t & dd,
	svfloat64_t xx, 
	float64_t rmin, 
	float64_t rmax)
{
    svbool_t lt_rmin = svcmplt(pg, xx, rmin);
    svbool_t ge_rmax = svcmpge(pg, xx, rmax);
    svbool_t ge_rmin_lt_rmax = svnor_z(pg, lt_rmin, ge_rmax);
    if(svptest_any(pg,ge_rmin_lt_rmax)){
      float64_t max_diff = rmax - rmin;
      float64_t du = 1. / max_diff;
      svfloat64_t vneg_ten = svdup_f64(-10.);
      svfloat64_t vneg_six = svdup_f64(-6.);
      svfloat64_t fifteen = svdup_f64(15.);
      svfloat64_t one = svdup_f64(1.);
      svfloat64_t three = svdup_f64(3.);
      svfloat64_t zero = svdup_f64(0.);
      svfloat64_t vneg_twelve = svdup_f64(-12.);

      svfloat64_t vuu_1 = svsub_z(ge_rmin_lt_rmax,xx,rmin);
      vuu_1 = svdiv_z(ge_rmin_lt_rmax,vuu_1,max_diff);
      svfloat64_t vuu_2 = svmul_z(ge_rmin_lt_rmax,vuu_1,vuu_1);
      svfloat64_t vuu_3 = svmul_z(ge_rmin_lt_rmax,vuu_2,vuu_1);
      svfloat64_t tmp1 = svmad_z(ge_rmin_lt_rmax, fifteen, vuu_1, vneg_ten); // 15 * uu - 10
      svfloat64_t tmp2 = svmad_z(ge_rmin_lt_rmax, vneg_six, vuu_2, tmp1); // (-6 * uu*uu + 15 * uu - 10)
      vv = svmad_z(ge_rmin_lt_rmax, vuu_3, tmp2 , one); //uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1;
      // TODO 
      svfloat64_t tmp3 = svmad_z(ge_rmin_lt_rmax, vneg_twelve, vuu_1, fifteen); // (-12 * uu + 15) )
      svfloat64_t tmp4 = svmul_z(ge_rmin_lt_rmax, vuu_3, tmp3);  // uu*uu*uu * (-12 * uu + 15)
      svfloat64_t tmp5 = svmul_z(ge_rmin_lt_rmax, vuu_2, three); // 3 * uu*uu
      svfloat64_t tmp6 = svmad_z(ge_rmin_lt_rmax, tmp5, tmp2, tmp4); // ( 3 * uu*uu * (-6 * uu*uu + 15 * uu - 10) + uu*uu*uu * (-12 * uu + 15) ) 
      dd = svmul_z(ge_rmin_lt_rmax, tmp6, du);
    }
    dd = svdup_f64_m(dd,lt_rmin,0.);
    vv = svdup_f64_m(vv,lt_rmin,1.);
    dd = svdup_f64_m(dd,ge_rmax,0.);
    vv = svdup_f64_m(vv,ge_rmax,0.);
}

#define sve_transpose_12x8_inplace(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11){      \
    svfloat64_t _tmp0 = svzip1(v0,v4);                                          \
    svfloat64_t _tmp1 = svzip1(v1,v5);                                          \
    svfloat64_t _tmp2 = svzip1(v2,v6);                                          \
    svfloat64_t _tmp3 = svzip1(v3,v7);                                          \
    svfloat64_t _tmp4 = svzip2(v0,v4);                                          \
    svfloat64_t _tmp5 = svzip2(v1,v5);                                          \
    svfloat64_t _tmp6 = svzip2(v2,v6);                                          \
    svfloat64_t _tmp7 = svzip2(v3,v7);                                          \
    v0 = svzip1(_tmp0,_tmp2);                                                   \
    v1 = svzip1(_tmp1,_tmp3);                                                   \
    v2 = svzip2(_tmp0,_tmp2);                                                   \
    v3 = svzip2(_tmp1,_tmp3);                                                   \
    v4 = svzip1(_tmp4,_tmp6);                                                   \
    v5 = svzip1(_tmp5,_tmp7);                                                   \
    v6 = svzip2(_tmp4,_tmp6);                                                   \
    v7 = svzip2(_tmp5,_tmp7);                                                   \
    _tmp0 = svzip1(v0,v1);                                                      \
    _tmp1 = svzip2(v0,v1);                                                      \
    _tmp2 = svzip1(v2,v3);                                                      \
    _tmp3 = svzip2(v2,v3);                                                      \
    _tmp4 = svzip1(v4,v5);                                                      \
    _tmp5 = svzip2(v4,v5);                                                      \
    _tmp6 = svzip1(v6,v7);                                                      \
    _tmp7 = svzip2(v6,v7);                                                      \
    svfloat64_t _tmp8 = svzip1(v8,v10);                                         \
    svfloat64_t _tmp9 = svzip1(v9,v11);                                         \
    svfloat64_t _tmp10 = svzip2(v8,v10);                                        \
    svfloat64_t _tmp11 = svzip2(v9,v11);                                        \
    v8 = svzip1(_tmp8,_tmp9);                                                   \
    v9 = svzip2(_tmp8,_tmp9);                                                   \
    v10 = svzip1(_tmp10,_tmp11);                                                \
    v11 = svzip2(_tmp10,_tmp11);                                                \
    v1 = svzip1(v8,_tmp1);                                                      \
    v2 = svzip2(_tmp1,v8);                                                      \
    v4 = svzip1(v9,_tmp3);                                                      \
    v5 = svzip2(_tmp3,v9);                                                      \
    v7 = svzip1(v10,_tmp5);                                                     \
    v8 = svzip2(_tmp5,v10);                                                     \
    v10 = svzip1(v11,_tmp7);                                                    \
    v11 = svzip2(_tmp7,v11);                                                    \
    v0 = _tmp0;                                                                 \
    v3 = _tmp2;                                                                 \
    v6 = _tmp4;                                                                 \
    v9 = _tmp6;                                                                 \
    uint64_t indices[8] = {0,2,4,6,1,3,5,7};                                    \
    svuint64_t vindices = svld1(svptrue_b64(), indices);                        \
    v1 = svtbl(v1, vindices);                                                   \
    v2 = svtbl(v2, vindices);                                                   \
    v4 = svtbl(v4, vindices);                                                   \
    v5 = svtbl(v5, vindices);                                                   \
    v7 = svtbl(v7, vindices);                                                   \
    v8 = svtbl(v8, vindices);                                                   \
    v10 = svtbl(v10, vindices);                                                 \
    v11 = svtbl(v11, vindices);                                                 \
}

void deepmd::env_mat_a_cpu_sve (
  double*	        descrpt_a,
  double*	        descrpt_a_deriv,
  double*	        rij_a,
  const std::vector<double > &	  posi,
  const std::vector<int > &		    type,
  const int &				              i_idx,
  const int *		                  fmt_nlist_a,
  const std::vector<int > &		    sec_a, 
  const float &			              rmin,
  const float &			              rmax
  )  {
    
  // // compute the diff of the neighbors
  // rij_a.resize (sec_a.back() * 3);
  // fill (rij_a.begin(), rij_a.end(), 0.0);
  // // 1./rr, cos(theta), cos(phi), sin(phi)
  // descrpt_a.resize (sec_a.back() * 4);
  // fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
  // // deriv wrt center: 3
  // descrpt_a_deriv.resize (sec_a.back() * 4 * 3);
  // fill (descrpt_a_deriv.begin(), descrpt_a_deriv.end(), 0.0);

  memset(rij_a,'\0',sec_a.back() * 3 * sizeof(double));
  memset(descrpt_a,'\0',sec_a.back() * 4 * sizeof(double));
  memset(descrpt_a_deriv,'\0',sec_a.back() * 4 * 3 * sizeof(double));

  svfloat64_t vposi_0 = svdup_f64(posi[i_idx*3]);
  svfloat64_t vposi_1 = svdup_f64(posi[i_idx*3+1]);
  svfloat64_t vposi_2 = svdup_f64(posi[i_idx*3+2]);
  svfloat64_t vone = svdup_f64(1.0);
  svfloat64_t vtwo = svdup_f64(2.0);
  for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter){
    for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter+1]; nei_iter += 8){
      svbool_t pg0 = svwhilelt_b64(nei_iter, sec_a[sec_iter+1]);
      svint64_t vj_idx = svld1sw_s64(pg0, &fmt_nlist_a[nei_iter]);
      svbool_t lt_zero = svcmplt(pg0,vj_idx,0);
      svbool_t pg1 = svbrkb_z(pg0,lt_zero);
      // pg1 all false
      if(!svptest_first(pg0, pg1)){
        break;
      }
      svint64_t vj_idx0 = svmul_z(pg1, vj_idx, 3);
      svint64_t vj_idx1 = svadd_z(pg1, vj_idx0, 1);
      svint64_t vj_idx2 = svadd_z(pg1, vj_idx0, 2);
      svfloat64_t vposj_0 = svld1_gather_s64index_f64(pg1,&posi[0],vj_idx0);
      svfloat64_t vposj_1 = svld1_gather_s64index_f64(pg1,&posi[0],vj_idx1);
      svfloat64_t vposj_2 = svld1_gather_s64index_f64(pg1,&posi[0],vj_idx2);
      svfloat64_t vr_0 = svsub_z(pg1,vposj_0,vposi_0);
      svfloat64_t vr_1 = svsub_z(pg1,vposj_1,vposi_1);
      svfloat64_t vr_2 = svsub_z(pg1,vposj_2,vposi_2);

      // 写回rij_a 3*8 -> 8*3
      svfloat64x3_t vr = svcreate3(vr_0,vr_1,vr_2);
      svst3(pg1,&rij_a[nei_iter*3],vr);

      svfloat64_t vr2_0 = svmul_z(pg1,vr_0,vr_0);
      svfloat64_t vr2_1 = svmul_z(pg1,vr_1,vr_1);
      svfloat64_t vr2_2 = svmul_z(pg1,vr_2,vr_2);
      svfloat64_t vnr2 = svadd_z(pg1,vr2_0,vr2_1);
      vnr2 = svadd_z(pg1,vnr2,vr2_2);

      // 近似算法
      // svfloat64_t vinr = svrsqrte(vnr2);
      svfloat64_t vinr = svsqrt_z(pg1,vnr2);
      vinr = svdiv_z(pg1,vone,vinr);

      svfloat64_t vnr = svmul_z(pg1, vnr2, vinr);
      svfloat64_t vinr2 = svmul_z(pg1, vinr, vinr);
      svfloat64_t vinr4 = svmul_z(pg1, vinr2, vinr2);
      svfloat64_t vinr3 = svmul_z(pg1, vinr4, vnr);
      // spline5 swich 展开
      svfloat64_t vsw,vdsw;
      spline5_switch_vector(pg1, vsw, vdsw, vnr, rmin, rmax);
      int idx_value = nei_iter * 4;
      int idx_deriv = nei_iter * 4 * 3;
      // 4 value components
      // svfloat64_t vdescrpt_a_0 = svdiv_z(pg1, vone, vnr);
      svfloat64_t vdescrpt_a_0 = vinr;
      svfloat64_t vdescrpt_a_1 = svdiv_z(pg1, vr_0, vnr2);
      svfloat64_t vdescrpt_a_2 = svdiv_z(pg1, vr_1, vnr2);
      svfloat64_t vdescrpt_a_3 = svdiv_z(pg1, vr_2, vnr2);

      // deriv of component 1/r
      // descrpt_a_deriv[idx_deriv + 0] = rr[0] * (inr3 * sw - descrpt_a[idx_value + 0] * dsw * inr);
      // descrpt_a_deriv[idx_deriv + 1] = rr[1] * (inr3 * sw - descrpt_a[idx_value + 0] * dsw * inr);
      // descrpt_a_deriv[idx_deriv + 2] = rr[2] * (inr3 * sw - descrpt_a[idx_value + 0] * dsw * inr);

      // deriv of component x/r2
      // descrpt_a_deriv[idx_deriv + 3] = rr[0] * (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr) - inr2 * sw;
      // descrpt_a_deriv[idx_deriv + 4] = rr[1] * (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr);
      // descrpt_a_deriv[idx_deriv + 5] = rr[2] * (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr);

      // deriv of component y/r2
      // descrpt_a_deriv[idx_deriv + 6] = rr[0] * (2. * rr[1] * inr4 * sw - descrpt_a[idx_value + 2] * dsw * inr);
      // descrpt_a_deriv[idx_deriv + 7] = rr[1] * (2. * rr[1] * inr4 * sw - descrpt_a[idx_value + 2] * dsw * inr) - inr2 * sw;
      // descrpt_a_deriv[idx_deriv + 8] = rr[2] * (2. * rr[1] * inr4 * sw - descrpt_a[idx_value + 2] * dsw * inr);

      // deriv of component z/r2
      // descrpt_a_deriv[idx_deriv + 9] = rr[0] * (2. * rr[2] * inr4 * sw - descrpt_a[idx_value + 3] * dsw * inr);
      // descrpt_a_deriv[idx_deriv +10] = rr[1] * (2. * rr[2] * inr4 * sw - descrpt_a[idx_value + 3] * dsw * inr);
      // descrpt_a_deriv[idx_deriv +11] = rr[2] * (2. * rr[2] * inr4 * sw - descrpt_a[idx_value + 3] * dsw * inr) - inr2 * sw;

      // common 
      // inr3 * sw
      // dsw * inr 
      // 2. * inr4 * sw
      // inr2 * sw
      svfloat64_t tmp1 = svmul_z(pg1, vinr3, vsw); // inr3 * sw
      svfloat64_t tmp2 = svmul_z(pg1, vdsw, vinr);   // dsw * inr 

      svfloat64_t tmp3 = svmul_z(pg1, vinr4, vsw); // inr4 * sw
      svfloat64_t tmp4 = svmul_z(pg1, vtwo, tmp3); // 2. * inr4 * sw
      svfloat64_t tmp5 = svmul_z(pg1, vinr2, vsw); // inr2 * sw

      svfloat64_t tmp6 = svmul_z(pg1, vr_0, tmp4); // 2. * rr[0] * inr4 * sw
      svfloat64_t tmp7 = svmul_z(pg1, vr_1, tmp4); // 2. * rr[1] * inr4 * sw
      svfloat64_t tmp8 = svmul_z(pg1, vr_2, tmp4); // 2. * rr[2] * inr4 * sw

      svfloat64_t tmp9 = svmsb_z(pg1, vdescrpt_a_0, tmp2, tmp1);   // (inr3 * sw - descrpt_a[idx_value + 0] * dsw * inr)
      svfloat64_t tmp10 = svmsb_z(pg1, vdescrpt_a_1, tmp2, tmp6);  // (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr)
      svfloat64_t tmp11 = svmsb_z(pg1, vdescrpt_a_2, tmp2, tmp7);  // (2. * rr[1] * inr4 * sw - descrpt_a[idx_value + 2] * dsw * inr)
      svfloat64_t tmp12 = svmsb_z(pg1, vdescrpt_a_3, tmp2, tmp8);  // (2. * rr[2] * inr4 * sw - descrpt_a[idx_value + 3] * dsw * inr)

      svfloat64_t vdescrpt_a_deriv_0_0 = svmul_z(pg1, vr_0, tmp9);  // rr[0] * (inr3 * sw - descrpt_a[idx_value + 0] * dsw * inr)
      svfloat64_t vdescrpt_a_deriv_0_1 = svmul_z(pg1, vr_1, tmp9);  // rr[1] * (inr3 * sw - descrpt_a[idx_value + 0] * dsw * inr)
      svfloat64_t vdescrpt_a_deriv_0_2 = svmul_z(pg1, vr_2, tmp9);  // rr[2] * (inr3 * sw - descrpt_a[idx_value + 0] * dsw * inr)

      svfloat64_t vdescrpt_a_deriv_1_0 = svmul_z(pg1, vr_0, tmp10);  // rr[0] * (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr)
      svfloat64_t vdescrpt_a_deriv_1_1 = svmul_z(pg1, vr_1, tmp10);  // rr[1] * (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr)
      svfloat64_t vdescrpt_a_deriv_1_2 = svmul_z(pg1, vr_2, tmp10);  // rr[2] * (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr)

      svfloat64_t vdescrpt_a_deriv_2_0 = svmul_z(pg1, vr_0, tmp11);  // rr[0] * (2. * rr[1] * inr4 * sw - descrpt_a[idx_value + 2] * dsw * inr)
      svfloat64_t vdescrpt_a_deriv_2_1 = svmul_z(pg1, vr_1, tmp11);  // rr[1] * (2. * rr[1] * inr4 * sw - descrpt_a[idx_value + 2] * dsw * inr)
      svfloat64_t vdescrpt_a_deriv_2_2 = svmul_z(pg1, vr_2, tmp11);  // rr[2] * (2. * rr[1] * inr4 * sw - descrpt_a[idx_value + 2] * dsw * inr)

      svfloat64_t vdescrpt_a_deriv_3_0 = svmul_z(pg1, vr_0, tmp12);  // rr[0] * (2. * rr[2] * inr4 * sw - descrpt_a[idx_value + 3] * dsw * inr)
      svfloat64_t vdescrpt_a_deriv_3_1 = svmul_z(pg1, vr_1, tmp12);  // rr[1] * (2. * rr[2] * inr4 * sw - descrpt_a[idx_value + 3] * dsw * inr)
      svfloat64_t vdescrpt_a_deriv_3_2 = svmul_z(pg1, vr_2, tmp12);  // rr[2] * (2. * rr[2] * inr4 * sw - descrpt_a[idx_value + 3] * dsw * inr)

      vdescrpt_a_deriv_1_0 = svsub_z(pg1, vdescrpt_a_deriv_1_0, tmp5); // rr[0] * (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr) - inr2 * sw
      vdescrpt_a_deriv_2_1 = svsub_z(pg1, vdescrpt_a_deriv_2_1, tmp5); // rr[0] * (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr) - inr2 * sw
      vdescrpt_a_deriv_3_2 = svsub_z(pg1, vdescrpt_a_deriv_3_2, tmp5); // rr[0] * (2. * rr[0] * inr4 * sw - descrpt_a[idx_value + 1] * dsw * inr) - inr2 * sw

      vdescrpt_a_0 = svmul_z(pg1, vdescrpt_a_0, vsw);
      vdescrpt_a_1 = svmul_z(pg1, vdescrpt_a_1, vsw);
      vdescrpt_a_2 = svmul_z(pg1, vdescrpt_a_2, vsw);
      vdescrpt_a_3 = svmul_z(pg1, vdescrpt_a_3, vsw);

      // 写回到descrpt_a 4*8 -> 8*4;
      svfloat64x4_t vdescrpt_a = svcreate4(vdescrpt_a_0,vdescrpt_a_1,vdescrpt_a_2,vdescrpt_a_3);
      svst4(pg1,&descrpt_a[idx_value],vdescrpt_a);
      
      // 写回descrpt_a_deriv 12*8 -> 8*12
      double* base = &descrpt_a_deriv[idx_deriv];

      svbool_t ptrue = svptrue_b64();
      // sve transpose 实现
      if(svptest_last(ptrue, pg1)){
        sve_transpose_12x8_inplace(
          vdescrpt_a_deriv_0_0,vdescrpt_a_deriv_0_1,vdescrpt_a_deriv_0_2,
          vdescrpt_a_deriv_1_0,vdescrpt_a_deriv_1_1,vdescrpt_a_deriv_1_2,
          vdescrpt_a_deriv_2_0,vdescrpt_a_deriv_2_1,vdescrpt_a_deriv_2_2,
          vdescrpt_a_deriv_3_0,vdescrpt_a_deriv_3_1,vdescrpt_a_deriv_3_2
        );
        svst1(ptrue, base   , vdescrpt_a_deriv_0_0);
        svst1(ptrue, base+ 8, vdescrpt_a_deriv_0_1);
        svst1(ptrue, base+16, vdescrpt_a_deriv_0_2);
        svst1(ptrue, base+24, vdescrpt_a_deriv_1_0);
        svst1(ptrue, base+32, vdescrpt_a_deriv_1_1);
        svst1(ptrue, base+40, vdescrpt_a_deriv_1_2);
        svst1(ptrue, base+48, vdescrpt_a_deriv_2_0);
        svst1(ptrue, base+56, vdescrpt_a_deriv_2_1);
        svst1(ptrue, base+64, vdescrpt_a_deriv_2_2);
        svst1(ptrue, base+72, vdescrpt_a_deriv_3_0);
        svst1(ptrue, base+80, vdescrpt_a_deriv_3_1);
        svst1(ptrue, base+88, vdescrpt_a_deriv_3_2);
      }else{
        // scater 实现
        // compute index
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
        svst1_scatter_u64index_f64(pg1,base,index0,vdescrpt_a_deriv_0_0);
        svst1_scatter_u64index_f64(pg1,base,index1,vdescrpt_a_deriv_0_1);
        svst1_scatter_u64index_f64(pg1,base,index2,vdescrpt_a_deriv_0_2);
        svst1_scatter_u64index_f64(pg1,base,index3,vdescrpt_a_deriv_1_0);
        svst1_scatter_u64index_f64(pg1,base,index4,vdescrpt_a_deriv_1_1);
        svst1_scatter_u64index_f64(pg1,base,index5,vdescrpt_a_deriv_1_2);
        svst1_scatter_u64index_f64(pg1,base,index6,vdescrpt_a_deriv_2_0);
        svst1_scatter_u64index_f64(pg1,base,index7,vdescrpt_a_deriv_2_1);
        svst1_scatter_u64index_f64(pg1,base,index8,vdescrpt_a_deriv_2_2);
        svst1_scatter_u64index_f64(pg1,base,index9,vdescrpt_a_deriv_3_0);
        svst1_scatter_u64index_f64(pg1,base,index10,vdescrpt_a_deriv_3_1);
        svst1_scatter_u64index_f64(pg1,base,index11,vdescrpt_a_deriv_3_2);
      }
      // lt_zero
      if(svptest_any(pg0, lt_zero)){
        break;
      }
    }
  }
}


void deepmd::env_mat_a_cpu_sve (
    float*	        descrpt_a,
    float*        descrpt_a_deriv,
    float*	        rij_a,
    const std::vector<float > &	  posi,
    const std::vector<int > &		  type,
    const int &				            i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		  sec_a, 
    const float &			            rmin,
    const float &			            rmax) {
    // // compute the diff of the neighbors
    // rij_a.resize (sec_a.back() * 3);
    // fill (rij_a.begin(), rij_a.end(), 0.0);
    // // 1./rr, cos(theta), cos(phi), sin(phi)
    // descrpt_a.resize (sec_a.back() * 4);
    // fill (descrpt_a.begin(), descrpt_a.end(), 0.0);
    // // deriv wrt center: 3
    // descrpt_a_deriv.resize (sec_a.back() * 4 * 3);
    // fill (descrpt_a_deriv.begin(), descrpt_a_deriv.end(), 0.0);
    
    memset(rij_a,'\0',sec_a.back() * 3 * sizeof(float));
    memset(descrpt_a,'\0',sec_a.back() * 4 * sizeof(float));
    memset(descrpt_a_deriv,'\0',sec_a.back() * 3 * 4 * sizeof(float));

    for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
        for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter + 1]; ++nei_iter) {
            if (fmt_nlist_a[nei_iter] < 0) break;
            const int & j_idx = fmt_nlist_a[nei_iter];
            for (int dd = 0; dd < 3; ++dd) {
                rij_a[nei_iter * 3 + dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
            }
            const float * rr = &rij_a[nei_iter * 3];
            float nr2 = deepmd::dot3(rr, rr);
            float inr = 1./sqrt(nr2);
            float nr = nr2 * inr;
            float inr2 = inr * inr;
            float inr4 = inr2 * inr2;
            float inr3 = inr4 * nr;
            float sw, dsw;
            deepmd::spline5_switch(sw, dsw, nr, rmin, rmax);
            int idx_deriv = nei_iter * 4 * 3;	// 4 components time 3 directions
            int idx_value = nei_iter * 4;	// 4 components
            // 4 value components
            descrpt_a[idx_value + 0] = 1./nr;
            descrpt_a[idx_value + 1] = rr[0] / nr2;
            descrpt_a[idx_value + 2] = rr[1] / nr2;
            descrpt_a[idx_value + 3] = rr[2] / nr2;
            // deriv of component 1/r
            descrpt_a_deriv[idx_deriv + 0] = rr[0] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[0] * inr;
            descrpt_a_deriv[idx_deriv + 1] = rr[1] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[1] * inr;
            descrpt_a_deriv[idx_deriv + 2] = rr[2] * inr3 * sw - descrpt_a[idx_value + 0] * dsw * rr[2] * inr;
            // deriv of component x/r2
            descrpt_a_deriv[idx_deriv + 3] = (2. * rr[0] * rr[0] * inr4 - inr2) * sw - descrpt_a[idx_value + 1] * dsw * rr[0] * inr;
            descrpt_a_deriv[idx_deriv + 4] = (2. * rr[0] * rr[1] * inr4	) * sw - descrpt_a[idx_value + 1] * dsw * rr[1] * inr;
            descrpt_a_deriv[idx_deriv + 5] = (2. * rr[0] * rr[2] * inr4	) * sw - descrpt_a[idx_value + 1] * dsw * rr[2] * inr;
            // deriv of component y/r2
            descrpt_a_deriv[idx_deriv + 6] = (2. * rr[1] * rr[0] * inr4	) * sw - descrpt_a[idx_value + 2] * dsw * rr[0] * inr;
            descrpt_a_deriv[idx_deriv + 7] = (2. * rr[1] * rr[1] * inr4 - inr2) * sw - descrpt_a[idx_value + 2] * dsw * rr[1] * inr;
            descrpt_a_deriv[idx_deriv + 8] = (2. * rr[1] * rr[2] * inr4	) * sw - descrpt_a[idx_value + 2] * dsw * rr[2] * inr;
            // deriv of component z/r2
            descrpt_a_deriv[idx_deriv + 9] = (2. * rr[2] * rr[0] * inr4	) * sw - descrpt_a[idx_value + 3] * dsw * rr[0] * inr;
            descrpt_a_deriv[idx_deriv +10] = (2. * rr[2] * rr[1] * inr4	) * sw - descrpt_a[idx_value + 3] * dsw * rr[1] * inr;
            descrpt_a_deriv[idx_deriv +11] = (2. * rr[2] * rr[2] * inr4 - inr2) * sw - descrpt_a[idx_value + 3] * dsw * rr[2] * inr;
            // 4 value components
            descrpt_a[idx_value + 0] *= sw;
            descrpt_a[idx_value + 1] *= sw;
            descrpt_a[idx_value + 2] *= sw;
            descrpt_a[idx_value + 3] *= sw;
        }
    }
}


#endif