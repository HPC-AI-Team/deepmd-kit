#!/bin/bash -e


if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
fi



set -ex
model_path=../model
python_file_path=$deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py
rm $model_path/double/compress-preprocess/* -rf

name_list=(baseline gemm gemm_tanh)
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
