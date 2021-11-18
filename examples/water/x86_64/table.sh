#!/bin/bash

set -ex

source $deepmd_root/script/x86_64/env.sh

python_file_path=$deepmd_root/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/tools/table_visualization.py

cp $deepmd_root/deepmd/tools/table_visualization.py $deepmd_root/_skbuild/linux-x86_64-3.7/cmake-install/deepmd/tools/table_visualization.py

python $python_file_path $deepmd_root/../model/water/graph-compress.pb