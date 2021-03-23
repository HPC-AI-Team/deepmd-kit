#!/bin/bash

export OMP_NUM_THREADS=4

mpirun -n 4  lmp_mpi -echo screen < in.water_double

