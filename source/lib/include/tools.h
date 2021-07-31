#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <omp.h>
#include <iomanip>
#include <cstdlib>
#ifdef __ARM_FEATURE_SVE 
#include <arm_sve.h> 
#endif

// Assert -------------------------------------------------------------------
#ifdef GZQ_ASSERT
#define gzq_assert(p)                                       \
    if(!(p)){                                               \
        std::cerr << "Assert Failed In " << __FILE__        \
                  << " Line : " << __LINE__                 \
                  << " For " << #p << endl;                 \
        abort();                                            \
    }                                                      
#else
#define gzq_assert(p) ((void)0)
#endif

// Timing --------------------------------------------------------------------
#ifdef GZQ_TIME
#define gzq_get_wtime() _dtime()
#define gzq_print_wtime(start,end,name) _print_time(start,end,name)
#else 
#define gzq_get_wtime() (0.)
#define gzq_print_wtime(start,end,name) ((void)0)
#endif


inline double _dtime(){
    return omp_get_wtime();
}

inline void _print_time(double start,double end,const char* name){
    std::cout << "Wall time of " << name << " : "
              << end - start 
              << std::endl;
}

// IO --------------------------------------------------------------------

static void write_to_file(const double* data,int len,std::string file_path){
    std::ofstream fw(file_path, std::ios::out);
    for(int i = 0; i < len; i++){
        fw << i << " " << std::setprecision(10) << data[i] << std::endl;
    }
}

// rand env variable

static bool get_env_print_time(){
    char *var = getenv("PRINT_TIME");
    if(var == NULL){
        return false;
    }else{
        return true;
    }
}

static bool get_env_preprocessed(){
    char *var = getenv("HAVE_PREPROCESSED");
    if(var == NULL){
        return false;
    }else{
        return true;
    }
}

static int get_env_num_threads(){
    char *var = getenv("DEEPMD_NUM_THREADS");
    int ret = 12;
    if(var != NULL){
        ret = atoi(var);
    }
    // std::cout << "thread num : " << var << ", " << ret << std::endl;
    return ret;
}

#ifdef __ARM_FEATURE_SVE 

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

#endif