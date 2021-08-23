#/bin/bash -e

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh

set -x

# mkdir -p ../model/double/original
# mkdir -p ../model/double/compress
# mkdir -p ../model/double/compress-preprocess
# cp ../model/double/graph.pb ../model/double/original/graph-original-baseline.pb
# dp transfer -O ../model/double/graph.pb -r ../model/double/original/graph-original-gemm.pb -o ../model/double/original/graph-original-gemm.pb
# dp transfer -O ../model/double/graph.pb -r ../model/double/original/graph-original-gemm_tanh.pb -o ../model/double/original/graph-original-gemm_tanh.pb
# dp transfer -O ../model/double/graph.pb -r ../model/double/original/graph-original-gemm_tanh_fusion.pb -o ../model/double/original/graph-original-gemm_tanh_fusion.pb

# dp compress ../se_e2_a/input_double.json -i ../model/double/original/graph-original-baseline.pb -o ../model/double/compress/graph-compress-baseline.pb
# dp compress ../se_e2_a/input_double.json -i ../model/double/original/graph-original-gemm.pb -o ../model/double/compress/graph-compress-gemm.pb
# dp compress ../se_e2_a/input_double.json -i ../model/double/original/graph-original-gemm_tanh.pb -o ../model/double/compress/graph-compress-gemm_tanh.pb
# dp compress ../se_e2_a/input_double.json -i ../model/double/original/graph-original-gemm_tanh_fusion.pb -o ../model/double/compress/graph-compress-gemm_tanh_fusion.pb

# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/double/compress/graph-compress-baseline.pb ../model/double/compress-preprocess/graph-compress-preprocess-baseline.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/double/compress/graph-compress-gemm.pb ../model/double/compress-preprocess/graph-compress-preprocess-gemm.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/double/compress/graph-compress-gemm_tanh.pb ../model/double/compress-preprocess/graph-compress-preprocess-gemm_tanh.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/double/compress/graph-compress-gemm_tanh_fusion.pb ../model/double/compress-preprocess/graph-compress-preprocess-gemm_tanh_fusion.pb


mkdir -p ../model/float/original
mkdir -p ../model/float/compress
mkdir -p ../model/float/compress-preprocess

cp ../model/float/graph.pb ../model/float/original/graph-original-baseline.pb
dp compress ../se_e2_a/input_float.json -i ../model/double/original/graph-original-baseline.pb -o ../model/float/compress/graph-compress-baseline.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/float/compress/graph-compress-baseline.pb ../model/float/compress-preprocess/graph-compress-preprocess-baseline.pb

# dp transfer -O ../model/float/graph.pb -r ../model/float/original/graph-original-gemm.pb -o ../model/float/original/graph-original-gemm.pb
# dp compress ../se_e2_a/input_float.json -i ../model/float/original/graph-original-gemm.pb -o ../model/float/compress/graph-compress-gemm.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/float/compress/graph-compress-gemm.pb ../model/float/compress-preprocess/graph-compress-preprocess-gemm.pb

# dp transfer -O ../model/float/graph.pb -r ../model/float/original/graph-original-gemm_tanh.pb -o ../model/float/original/graph-original-gemm_tanh.pb
# dp compress ../se_e2_a/input_float.json -i ../model/float/original/graph-original-gemm_tanh.pb -o ../model/float/compress/graph-compress-gemm_tanh.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/float/compress/graph-compress-gemm_tanh.pb ../model/float/compress-preprocess/graph-compress-preprocess-gemm_tanh.pb

# dp transfer -O ../model/float/graph.pb -r ../model/float/original/graph-original-gemm_tanh_fusion.pb -o ../model/float/original/graph-original-gemm_tanh_fusion.pb
# dp compress ../se_e2_a/input_float.json -i ../model/float/original/graph-original-gemm_tanh_fusion.pb -o ../model/float/compress/graph-compress-gemm_tanh_fusion.pb
# python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/float/compress/graph-compress-gemm_tanh_fusion.pb ../model/float/compress-preprocess/graph-compress-preprocess-gemm_tanh_fusion.pb
