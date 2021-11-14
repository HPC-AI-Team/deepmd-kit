#include "env_mat_opt.h"
#include "switcher.h"
#include <string.h>


template<typename FPTYPE> 
void 
deepmd::
env_mat_a_cpu_normalize (
    FPTYPE*	                        descrpt_a,
    FPTYPE*	                        descrpt_a_deriv,
    FPTYPE*               	        rij_a,
    const FPTYPE*                     	posi,
    const int*                    		type,
    const int &				i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			rmin,
    const float &			rmax,
    const FPTYPE * avg, 
    const FPTYPE * std
    ) 
{  
    memset(rij_a,'\0',sec_a.back() * 3 * sizeof(FPTYPE));
    memset(descrpt_a_deriv,'\0',sec_a.back() * 3 * 4 * sizeof(FPTYPE));
    
    const int nnei = sec_a.back();
    const int nem = nnei * 4;
    const FPTYPE * AVG = &avg[type[i_idx] * nem];
    const FPTYPE * STD = &std[type[i_idx] * nem];

    for (int jj = 0; jj < nem; ++jj) {
      descrpt_a[jj] = - AVG[jj] / STD[jj];
    }

    for (int sec_iter = 0; sec_iter < int(sec_a.size()) - 1; ++sec_iter) {
        for (int nei_iter = sec_a[sec_iter]; nei_iter < sec_a[sec_iter + 1]; ++nei_iter) {
            if (fmt_nlist_a[nei_iter] < 0) break;
            const int & j_idx = fmt_nlist_a[nei_iter];
            for (int dd = 0; dd < 3; ++dd) {
                rij_a[nei_iter * 3 + dd] = posi[j_idx * 3 + dd] - posi[i_idx * 3 + dd];
            }

            const FPTYPE * rr = &rij_a[nei_iter * 3];
            FPTYPE nr2 = deepmd::dot3(rr, rr);
            FPTYPE inr = 1./sqrt(nr2);
            FPTYPE nr = nr2 * inr;
            FPTYPE inr2 = inr * inr;
            FPTYPE inr4 = inr2 * inr2;
            FPTYPE inr3 = inr4 * nr;
            FPTYPE sw, dsw;
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

            descrpt_a[idx_value + 0] = (descrpt_a[idx_value + 0] - AVG[idx_value + 0]) / STD[idx_value + 0];
            descrpt_a[idx_value + 1] = (descrpt_a[idx_value + 1] - AVG[idx_value + 1]) / STD[idx_value + 1];
            descrpt_a[idx_value + 2] = (descrpt_a[idx_value + 2] - AVG[idx_value + 2]) / STD[idx_value + 2];
            descrpt_a[idx_value + 3] = (descrpt_a[idx_value + 3] - AVG[idx_value + 3]) / STD[idx_value + 3];

            descrpt_a_deriv[idx_deriv + 0] /=  STD[idx_value + 0];
            descrpt_a_deriv[idx_deriv + 1] /=  STD[idx_value + 0];
            descrpt_a_deriv[idx_deriv + 2] /=  STD[idx_value + 0];

            descrpt_a_deriv[idx_deriv + 3] /=  STD[idx_value + 1];
            descrpt_a_deriv[idx_deriv + 4] /=  STD[idx_value + 1];
            descrpt_a_deriv[idx_deriv + 5] /=  STD[idx_value + 1];

            descrpt_a_deriv[idx_deriv + 6] /=  STD[idx_value + 2];
            descrpt_a_deriv[idx_deriv + 7] /=  STD[idx_value + 2];
            descrpt_a_deriv[idx_deriv + 8] /=  STD[idx_value + 2];

            descrpt_a_deriv[idx_deriv + 9] /=  STD[idx_value + 3];
            descrpt_a_deriv[idx_deriv + 10] /=  STD[idx_value + 3];
            descrpt_a_deriv[idx_deriv + 11] /=  STD[idx_value + 3];
        }
    }
}


template
void 
deepmd::
env_mat_a_cpu_normalize<double> (
    double*	                        descrpt_a,
    double*	                        descrpt_a_deriv,
    double*               	        rij_a,
    const double*                     	posi,
    const int*                    		type,
    const int &				i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			rmin,
    const float &			rmax,
    const double * avg, 
    const double * std
    ) ;

template
void 
deepmd::
env_mat_a_cpu_normalize<float> (
    float*	                        descrpt_a,
    float*	                        descrpt_a_deriv,
    float*               	        rij_a,
    const float*                     	    posi,
    const int*                    		type,
    const int &				i_idx,
    const int *		                fmt_nlist_a,
    const std::vector<int > &		sec_a, 
    const float &			rmin,
    const float &			rmax,
    const float * avg, 
    const float * std
    ) ;