#!/bin/bash
#PJM -L "node=1"               # Number of node
#PJM -L "rscgrp=eap-small"     # Specify resource group
#PJM -L "elapse=60:00"         # Job run time limit value
#PJM -S                        # Direction of statistic information file output

set -e

. ~/gzq/fj_env.sh

cd ~/gzq/deepmd-kit/source/build
make -j48 && make install

cd ~/gzq/deepmd-kit/
python ./setup.py install

DIR=$HOME/gzq/water
cd $DIR

export PLE_MPI_STD_EMPTYFILE=off # Do not create a file if there is no output to stdout/stderr.

#fipp -C -d profiling_data -Icall,cpupa lmp_serial -echo screen < in.water
#fapp -C -d ./tmp -Icpupa -Hevent=statistics  lmp_serial -echo screen < in.water
fapp -C -d ./rep1 -Hevent=pa1 lmp_serial -echo screen < in.water
