#!/bin/bash

#!/bin/sh -x
#PJM -L  "node=256"                          # Number of assign node 8 (1 dimention format)
#PJM -L "rscgrp=small"         # Specify resource group
#PJM -L  "elapse=01:00:00"                 # Elapsed time limit 1 hour
#PJM --mpi "shape=256"
#PJM --mpi "max-proc-per-node=48"          # Maximum number of MPI processes created per node
#PJM -s                                    # Statistical information output

source $HOME/gzq/fj_env.sh
export PLE_MPI_STD_EMPTYFILE=off
mpiexec lmp_mpi -echo screen -in in.water_48x256