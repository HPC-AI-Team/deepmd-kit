#!/bin/bash -e

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi
source $deepmd_root/script/x86_64/env.sh


model_path=../model
python_file_path=$deepmd_root/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py

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
