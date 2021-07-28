#/bin/bash -e

deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh
bash $deepmd_root/script/fugaku/build_deepmd.sh

export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=-1
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3

set -ex
export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

mkdir -p ../model/double/original
mkdir -p ../model/double/compress
mkdir -p ../model/double/compress-preprocess
cp ../model/double/graph.pb ../model/double/original/graph-original-baseline.pb
# dp transfer -O ../model/double/graph.pb -r ../model/double/original/graph-original-gemm.pb -o ../model/double/original/graph-original-gemm.pb
# dp transfer -O ../model/double/graph.pb -r ../model/double/original/graph-original-gemm_tanh.pb -o ../model/double/original/graph-original-gemm_tanh.pb
# dp transfer -O ../model/double/graph.pb -r ../model/double/original/graph-original-gemm_tanh_fusion.pb -o ../model/double/original/graph-original-gemm_tanh_fusion.pb

dp compress ../se_e2_a/input_double.json -i ../model/double/graph.pb -o ../model/double/compress/graph-compress-baseline.pb
# dp compress ../se_e2_a/input_double.json -i ../model/double/original/graph-original-gemm.pb -o ../model/double/compress/graph-compress-gemm.pb
# dp compress ../se_e2_a/input_double.json -i ../model/double/original/graph-original-gemm_tanh.pb -o ../model/double/compress/graph-compress-gemm_tanh.pb
# dp compress ../se_e2_a/input_double.json -i ../model/double/original/graph-original-gemm_tanh_fusion.pb -o ../model/double/compress/graph-compress-gemm_tanh_fusion.pb

python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py ../model/double/compress/graph-compress-baseline.pb ../model/double/compress-preprocess/graph-compress-preprocess-baseline.pb
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py ../model/double/compress/graph-compress-gemm.pb ../model/double/compress-preprocess/graph-compress-preprocess-gemm.pb
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py ../model/double/compress/graph-compress-gemm_tanh.pb ../model/double/compress-preprocess/graph-compress-preprocess-gemm_tanh.pb
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py ../model/double/compress/graph-compress-gemm_tanh_fusion.pb ../model/double/compress-preprocess/graph-compress-preprocess-gemm_tanh_fusion.pb

# mkdir -p ../model/float/original
# mkdir -p ../model/float/compress
# mkdir -p ../model/float/compress-preprocess

# dp transfer -O ../model/double/graph.pb -r ../model/float/original/graph-original-baseline.pb -o ../model/float/original/graph-original-baseline.pb
# dp compress ../se_e2_a/input_float.json -i ../model/double/original/graph-original-baseline.pb -o ../model/float/compress/graph-compress-baseline.pb
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py ../model/float/compress/graph-compress-baseline.pb ../model/float/compress-preprocess/graph-compress-preprocess-baseline.pb

# dp transfer -O ../model/double/graph.pb -r ../model/float/original/graph-original-gemm.pb -o ../model/float/original/graph-original-gemm.pb
# dp compress ../se_e2_a/input_float.json -i ../model/float/original/graph-original-gemm.pb -o ../model/float/compress/graph-compress-gemm.pb
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py ../model/float/compress/graph-compress-gemm.pb ../model/float/compress-preprocess/graph-compress-preprocess-gemm.pb

# dp transfer -O ../model/double/graph.pb -r ../model/float/original/graph-original-gemm_tanh.pb -o ../model/float/original/graph-original-gemm_tanh.pb
# dp compress ../se_e2_a/input_float.json -i ../model/float/original/graph-original-gemm_tanh.pb -o ../model/float/compress/graph-compress-gemm_tanh.pb
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py ../model/float/compress/graph-compress-gemm_tanh.pb ../model/float/compress-preprocess/graph-compress-preprocess-gemm_tanh.pb

# dp transfer -O ../model/double/graph.pb -r ../model/float/original/graph-original-gemm_tanh_fusion.pb -o ../model/float/original/graph-original-gemm_tanh_fusion.pb
# dp compress ../se_e2_a/input_float.json -i ../model/float/original/graph-original-gemm_tanh_fusion.pb -o ../model/float/compress/graph-compress-gemm_tanh_fusion.pb
# python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py ../model/float/compress/graph-compress-gemm_tanh_fusion.pb ../model/float/compress-preprocess/graph-compress-preprocess-gemm_tanh_fusion.pb
