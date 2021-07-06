#!/bin/bash -e

deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh
# bash $deepmd_root/script/fugaku/build_deepmd.sh

cp $deepmd_root/deepmd/entrypoints/preprocess.py $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py

set -ex

rm ../model/graph-compress-preprocess_* -rf

# name=baseline
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py  ../model/graph-compress_$name.pb  ../model/graph-compress-preprocess_$name.pb

# name=gemm
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py  ../model/graph-compress_$name.pb  ../model/graph-compress-preprocess_$name.pb

name=gemm_tanh
python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py  ../model/graph-compress_$name.pb  ../model/graph-compress-preprocess_$name.pb

python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/pb2pbtxt.py ../model/graph-compress_$name.pb ../model/graph-compress_$name.pbtxt

# name=gemm_tanh_fusion
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py  ../model/graph-compress_$name.pb  ../model/graph-compress-preprocess_$name.pb
