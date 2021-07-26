#!/bin/bash

deepmd_root=$HOME/gzq/deepmd-kit
source $deepmd_root/script/fugaku/env.sh

python_script=$deepmd_root/_skbuild/linux-aarch64-3.8/cmake-install/deepmd/tools/pb2pbtxt.py

export LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1

python $python_script ../model/float/compress/graph-compress-baseline.pb ../model/float/compress/graph-compress-baseline.pbtxt