#/bin/bash -e

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh
bash $deepmd_root/script/x86_64/build_deepmd.sh

set -x

# dp transfer -O ../model/graph_baseline.pb -r ../model/graph_gemm.pb -o ../model/graph_gemm.pb
# dp transfer -O ../model/graph_baseline.pb -r ../model/graph_gemm_tanh.pb -o ../model/graph_gemm_tanh.pb
# dp transfer -O ../model/graph_baseline.pb -r ../model/graph_gemm_tanh_fushion.pb -o ../model/graph_gemm_tanh_fushion.pb

# dp transfer -O ../model/graph-compress_baseline.pb -r ../model/graph-compress_gemm.pb -o ../model/graph-compress_gemm.pb
# dp transfer -O ../model/graph-compress_baseline.pb -r ../model/graph-compress_gemm_tanh.pb -o ../model/graph-compress_gemm_tanh.pb
# dp transfer -O ../model/graph-compress_baseline.pb -r ../model/graph-compress_gemm_tanh_fushion.pb -o ../model/graph-compress_gemm_tanh_fushion.pb

python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/graph-compress_baseline.pb ../model/graph-compress-preprocess_baseline.pb
python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/graph-compress_gemm.pb ../model/graph-compress-preprocess_gemm.pb
python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/graph-compress_gemm_tanh.pb ../model/graph-compress-preprocess_gemm_tanh.pb
python /data/home/guozhuoqiang/deepmd-kit/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/entrypoints/preprocess.py ../model/graph-compress_gemm_tanh_fushion.pb ../model/graph-compress-preprocess_gemm_tanh_fushion.pb
