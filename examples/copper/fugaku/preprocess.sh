#!/bin/bash -e


source $deepmd_root/script/fugaku/env.sh
# bash $deepmd_root/script/fugaku/build_deepmd.sh


model_path=../model
python_file_path=$deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py

cp $deepmd_root/deepmd/entrypoints/preprocess.py $python_file_path

set -ex

rm $model_path/double/preprocess/* -rf

name_list=(baseline gemm gemm_tanh gemm_tanh_fusion)
precision_list=(double float)

for precision in ${precision_list[*]}
do
    for name in ${name_list[*]}
    do
        origin_model=$model_path/$precision/compress/graph-compress-$name.pb
        target_model=$model_path/$precision/compress-preprocess/graph-compress-preprocess-$name.pb
        if [ -e $origin_model ]
        then
            python $python_file_path $origin_model $target_model
        else
            echo "$origin_model_path not exist !!!"
            # exit -1
        fi
    done
done
