#!/bin/bash

export deepmd_root=$HOME/deepmd-kit
source $deepmd_root/script/x86_64/env.sh

python $deepmd_root/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/tools/pb2pbtxt.py ../model/float/original/graph-original-baseline.pb ../model/float/original/graph-original-baseline.pbtxt
python $deepmd_root/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/tools/pb2pbtxt.py ../model/float/compress/graph-compress-baseline.pb ../model/float/compress/graph-compress-baseline.pbtxt