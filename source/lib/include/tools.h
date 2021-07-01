#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <omp.h>
#include <iomanip>

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