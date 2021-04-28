#pragma once

#include <cstdlib>

bool get_print_time(){
    char *var = getenv("PRINT_TIME");
    if(var == NULL){
        return false;
    }else{
        return true;
    }
}