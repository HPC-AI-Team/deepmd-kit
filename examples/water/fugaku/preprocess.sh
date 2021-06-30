#!/bin/bash

set -ex

deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh
# bash $deepmd_root/script/fugaku/build_deepmd.sh

cp /home/hp200266/u01036/gzq/deepmd-kit/deepmd/entrypoints/preprocess.py /home/hp200266/u01036/gzq/deepmd-kit/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py

python /home/hp200266/u01036/gzq/deepmd-kit/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/entrypoints/preprocess.py /home/hp200266/u01036/gzq/deepmd-kit/examples/water/model/graph-compress.pb /home/hp200266/u01036/gzq/deepmd-kit/examples/water/model/graph-compress-preprocess.pb

# python /home/hp200266/u01036/gzq/deepmd-kit/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/pb2pbtxt.py /home/hp200266/u01036/gzq/deepmd-kit/examples/water/model/graph-compress.pb /home/hp200266/u01036/gzq/deepmd-kit/examples/water/model/graph-compress.pbtxt

# python /home/hp200266/u01036/gzq/deepmd-kit/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/pb2pbtxt.py /home/hp200266/u01036/gzq/deepmd-kit/examples/water/model/graph-compress-preprocess.pb /home/hp200266/u01036/gzq/deepmd-kit/examples/water/model/graph-compress-preprocess.pbtxt