#!/bin/bash
#PJM -L  "node=1"                          # Number of assign node 8 (1 dimention format)
#PJM -L  "freq=2200"                         
#PJM -L "rscgrp=small"         # Specify resource group
#PJM -L  "elapse=01:00:00"                 # Elapsed time limit 1 hour
#PJM --mpi "shape=1"
#PJM --mpi "max-proc-per-node=48"          # Maximum number of MPI processes created per node
#PJM -s                                    # Statistical information output

set -ex

name=""
if [ $# == 1 ]
then
    name=$1
else
    echo "usage : $0 name"
    exit -1
fi

deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh
bash $deepmd_root/script/fugaku/build_deepmd.sh

export PLE_MPI_STD_EMPTYFILE=off
# export PRINT_TIME=1
export OMP_NUM_THREADS=1

export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=1
export TF_PROFILE=1
export TF_CPP_MIN_LOG_LEVEL=3


likwid-pin -c 0 lmp_serial -echo screen -in ../lmp/in.water_1 &> output/output.$name
likwid-perfctr -C 0 -g FLOPS_DP lmp_serial -echo screen -in ../lmp/in.water_1 &> likwid/likwid.$name
rm -f profiler.json_*
likwid-pin -c 0 dp test -m ../model/graph.pb -s ../data/data_3 -n 1 &> prof/prof.$name
python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/profiler_visualization_topk.py profiler >> prof/prof.$name


likwid-pin -c 0 lmp_serial -echo screen -in ../lmp/in.water_compress_1 &> output/output.compress-$name
likwid-perfctr -C 0 -g FLOPS_DP lmp_serial -echo screen -in ../lmp/in.water_compress_1 &> likwid/likwid.compress-$name
rm -f profiler.json_*
likwid-pin -c 0 dp test -m ../model/graph-compress.pb -s ../data/data_3 -n 1 &> prof/prof.compress-$name
python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/profiler_visualization_topk.py profiler >> prof/prof.compress-$name


export HAVE_PREPROCESSED=1

likwid-pin -c 0 lmp_serial -echo screen -in ../lmp/in.water_compress_preprocess_1 &> output/output.compress-preprocess-$name
likwid-perfctr -C 0 -g FLOPS_DP lmp_serial -echo screen -in ../lmp/in.water_compress_preprocess_1 &> likwid/likwid.compress-preprocess-$name
rm -f profiler.json_*
likwid-pin -c 0 dp test -m ../model/graph-compress-preprocess.pb -s ../data/data_3 -n 1 &> prof/prof.compress-preprocess-$name
python $deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/profiler_visualization_topk.py profiler >> prof/prof.compress-preprocess-$name