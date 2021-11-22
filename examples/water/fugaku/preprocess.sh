#!/bin/bash -e
#PJM -L  "node=1"                           # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L  "rscgrp=int"                     # Specify resource group
#PJM -L  "elapse=00:10:00"                  # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=1"            # Maximum number of MPI processes created per node
export PLE_MPI_STD_EMPTYFILE=off

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
    exit -1
fi
source $deepmd_root/script/a64fx_fj/env.sh

export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

set -ex
model_path=../model
# python_file_path=$deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py
rm $model_path/double/compress-preprocess/* -rf

name_list=(baseline gemm gemm_tanh test)
precision_list=(double float)

for precision in ${precision_list[*]}
do
    for name in ${name_list[*]}
    do
        origin_model=$model_path/$precision/compress/graph-compress-$name.pb
        target_model=$model_path/$precision/compress-preprocess/graph-compress-preprocess-$name.pb
        if [ -e $origin_model ]
        then
            dp preprocess -i $origin_model -o $target_model
            # python $python_file_path $origin_model $target_model
        else
            echo "$origin_model_path not exist !!!"
            # exit -1
        fi
    done
done
