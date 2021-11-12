#pragma once
#include <omp.h>
#include <cstdlib>
#ifdef __ARM_FEATURE_SVE 
#include <arm_sve.h> 
#endif

static bool get_env_preprocessed(){
    char *var = getenv("HAVE_PREPROCESSED");
    if(var == NULL){
        return false;
    }else{
        return true;
    }
}

#ifdef __ARM_FEATURE_SVE 

#define sve_soa2aos_12x8_inplace(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11){        \
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

#define sve_aos2soa_12x8_inplace(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11){        \
    svfloat64_t _tmp0 = svzip1(v0,v6);                                          \
    svfloat64_t _tmp1 = svzip2(v0,v6);                                          \
    svfloat64_t _tmp2 = svzip1(v1,v7);                                          \
    svfloat64_t _tmp3 = svzip2(v1,v7);                                          \
    svfloat64_t _tmp4 = svzip1(v2,v8);                                          \
    svfloat64_t _tmp5 = svzip2(v2,v8);                                          \
    svfloat64_t _tmp6 = svzip1(v3,v9);                                          \
    svfloat64_t _tmp7 = svzip2(v3,v9);                                          \
    svfloat64_t _tmp8 = svzip1(v4,v10);                                         \
    svfloat64_t _tmp9 = svzip2(v4,v10);                                         \
    svfloat64_t _tmp10 = svzip1(v5,v11);                                        \
    svfloat64_t _tmp11 = svzip2(v5,v11);                                        \
    v0 = svzip1(_tmp0,_tmp6);                                                   \
    v1 = svzip2(_tmp0,_tmp6);                                                   \
    v2 = svzip1(_tmp1,_tmp7);                                                   \
    v3 = svzip2(_tmp1,_tmp7);                                                   \
    v4 = svzip1(_tmp2,_tmp8);                                                   \
    v5 = svzip2(_tmp2,_tmp8);                                                   \
    v6 = svzip1(_tmp3,_tmp9);                                                   \
    v7 = svzip2(_tmp3,_tmp9);                                                   \
    v8 = svzip1(_tmp4,_tmp10);                                                  \
    v9 = svzip2(_tmp4,_tmp10);                                                  \
    v10 = svzip1(_tmp5,_tmp11);                                                 \
    v11 = svzip2(_tmp5,_tmp11);                                                 \
    _tmp0 = svzip1(v0,v6);                                                      \
    _tmp1 = svzip2(v0,v6);                                                      \
    _tmp2 = svzip1(v1,v7);                                                      \
    _tmp3 = svzip2(v1,v7);                                                      \
    _tmp4 = svzip1(v2,v8);                                                      \
    _tmp5 = svzip2(v2,v8);                                                      \
    _tmp6 = svzip1(v3,v9);                                                      \
    _tmp7 = svzip2(v3,v9);                                                      \
    _tmp8 = svzip1(v4,v10);                                                     \
    _tmp9 = svzip2(v4,v10);                                                     \
    _tmp10 = svzip1(v5,v11);                                                    \
    _tmp11 = svzip2(v5,v11);                                                    \
    v0 = _tmp0;                                                                 \
    v1 = _tmp1;                                                                 \
    v2 = _tmp2;                                                                 \
    v3 = _tmp3;                                                                 \
    v4 = _tmp4;                                                                 \
    v5 = _tmp5;                                                                 \
    v6 = _tmp6;                                                                 \
    v7 = _tmp7;                                                                 \
    v8 = _tmp8;                                                                 \
    v9 = _tmp9;                                                                 \
    v10 = _tmp10;                                                               \
    v11 = _tmp11;                                                               \
}


#endif