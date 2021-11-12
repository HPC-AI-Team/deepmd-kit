#pragma once
#include <omp.h>
#include <cstdlib>

static int get_env_num_threads(){
    char *var = getenv("DEEPMD_NUM_THREADS");
    int ret = 12;
    if(var != NULL){
        ret = atoi(var);
    }
    // std::cout << "thread num : " << var << ", " << ret << std::endl;
    return ret;
}

