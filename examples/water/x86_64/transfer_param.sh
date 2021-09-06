#/bin/bash -e

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh

set -x

model_path=$deepmd_root/model/water

mkdir -p $model_path/double/original
mkdir -p $model_path/double/compress
mkdir -p $model_path/double/compress-preprocess

cp $model_path/double/graph.pb $model_path/double/original/graph-original-baseline.pb
# dp transfer -O $model_path/double/graph.pb -r $model_path/double/original/graph-original-gemm.pb -o $model_path/double/original/graph-original-gemm.pb
# dp transfer -O $model_path/double/graph.pb -r $model_path/double/original/graph-original-gemm_tanh.pb -o $model_path/double/original/graph-original-gemm_tanh.pb
# dp transfer -O $model_path/double/graph.pb -r $model_path/double/original/graph-original-gemm_tanh_fusion.pb -o $model_path/double/original/graph-original-gemm_tanh_fusion.pb

dp compress ../se_e2_a/input_double.json -i $model_path/double/original/graph-original-baseline.pb -o $model_path/double/compress/graph-compress-baseline.pb
# dp compress ../se_e2_a/input_double.json -i $model_path/double/original/graph-original-gemm.pb -o $model_path/double/compress/graph-compress-gemm.pb
# dp compress ../se_e2_a/input_double.json -i $model_path/double/original/graph-original-gemm_tanh.pb -o $model_path/double/compress/graph-compress-gemm_tanh.pb
# dp compress ../se_e2_a/input_double.json -i $model_path/double/original/graph-original-gemm_tanh_fusion.pb -o $model_path/double/compress/graph-compress-gemm_tanh_fusion.pb

# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py $model_path/double/compress/graph-compress-baseline.pb $model_path/double/compress-preprocess/graph-compress-preprocess-baseline.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py $model_path/double/compress/graph-compress-gemm.pb $model_path/double/compress-preprocess/graph-compress-preprocess-gemm.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py $model_path/double/compress/graph-compress-gemm_tanh.pb $model_path/double/compress-preprocess/graph-compress-preprocess-gemm_tanh.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py $model_path/double/compress/graph-compress-gemm_tanh_fusion.pb $model_path/double/compress-preprocess/graph-compress-preprocess-gemm_tanh_fusion.pb


# mkdir -p $model_path/float/original
# mkdir -p $model_path/float/compress
# mkdir -p $model_path/float/compress-preprocess

# cp $model_path/float/graph.pb $model_path/float/original/graph-original-baseline.pb
# dp compress ../se_e2_a/input_float.json -i $model_path/double/original/graph-original-baseline.pb -o $model_path/float/compress/graph-compress-baseline.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py $model_path/float/compress/graph-compress-baseline.pb $model_path/float/compress-preprocess/graph-compress-preprocess-baseline.pb

# dp transfer -O $model_path/float/graph.pb -r $model_path/float/original/graph-original-gemm.pb -o $model_path/float/original/graph-original-gemm.pb
# dp compress ../se_e2_a/input_float.json -i $model_path/float/original/graph-original-gemm.pb -o $model_path/float/compress/graph-compress-gemm.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py $model_path/float/compress/graph-compress-gemm.pb $model_path/float/compress-preprocess/graph-compress-preprocess-gemm.pb

# dp transfer -O $model_path/float/graph.pb -r $model_path/float/original/graph-original-gemm_tanh.pb -o $model_path/float/original/graph-original-gemm_tanh.pb
# dp compress ../se_e2_a/input_float.json -i $model_path/float/original/graph-original-gemm_tanh.pb -o $model_path/float/compress/graph-compress-gemm_tanh.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py $model_path/float/compress/graph-compress-gemm_tanh.pb $model_path/float/compress-preprocess/graph-compress-preprocess-gemm_tanh.pb

# dp transfer -O $model_path/float/graph.pb -r $model_path/float/original/graph-original-gemm_tanh_fusion.pb -o $model_path/float/original/graph-original-gemm_tanh_fusion.pb
# dp compress ../se_e2_a/input_float.json -i $model_path/float/original/graph-original-gemm_tanh_fusion.pb -o $model_path/float/compress/graph-compress-gemm_tanh_fusion.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py $model_path/float/compress/graph-compress-gemm_tanh_fusion.pb $model_path/float/compress-preprocess/graph-compress-preprocess-gemm_tanh_fusion.pb
