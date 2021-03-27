#!/bin/bash
#PJM -L "node=4"                  # 4 nodes
#PJM -L "rscgrp=eap-small"        # Specify resource group
#PJM -L "elapse=01:00:00"
#PJM --mpi "max-proc-per-node=1"  # Upper limit of number of MPI process created at 1 node
#PJM -s

export PLE_MPI_STD_EMPTYFILE=off # Do not create a file if there is no output to stdout/stderr.

#source $HOME/gzq/fj_env.sh

. ~/gzq/fj_env.sh

mpiexec $HOME/gzq/lammps/src/lmp_mpi -echo screen -in in.water

