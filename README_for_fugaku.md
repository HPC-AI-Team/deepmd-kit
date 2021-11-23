# DeepMD for Fugaku

## build deepmd for fugaku

```bash
# in login node 

cd <samewhere you want to put deepmd>
mkdir DeepMD
cd DeepMD

# tensroflow for fugaku

mkdir package
# copy from group shared directory or download from [google driver](https://drive.google.com/file/d/1BF3ereji7g0Aj0X_q4tjwC1wjbKB7zqd/view?usp=sharing)
cp /share/hp210260/TensorFlow-2.2.0.tar.gz ./package

mkdir dependents
cd dependents
tar -xzf ../package/TensorFlow-2.2.0.tar.gz

cd ..
git clone git@github.com:gzq942560379/deepmd-kit.git

cd deepmd-kit
git checkout -b fugaku-v2.0.3 origin/fugaku-v2.0.3

# The following two lines will write to ~/.bashrc for convenience.(Only execute once)
echo "export deepmd_root=$(pwd)" >> ~/.bashrc
echo 'alias "interact=pjsub --interact -L node=1 -L freq=2200 --sparam wait-time=600 "' >> ~/.bashrc

source ~/.bashrc

# build lammps in plugin mode
interact $deepmd_root/script/a64fx_fj/build_lammps.sh

# build deepmd c++
interact $deepmd_root/script/a64fx_fj/build_deepmd.sh

# build deepmd python
interact $deepmd_root/script/a64fx_fj/build_python.sh
```


## use deepmd for fugaku

### 1. prepare model

Prepare a trained model and its training config and data. (non-compressed)

Training on fugaku is not optimized, you can train the model on other platforms.

### 2. optimize model for fugaku

```bash
# in login node 

# modify the paths in script/a64fx_fj/0-model_optimization_for_fugaku.sh
# the Training step (numb_steps or stop_betch) in training config file set to 100. because we just need a model pattern and the model weights will be transfered from your trained model.

# This step will replace some ops in the model, such as matmul, tanh, etc., because they are too slow on fugaku.
interact $deepmd_root/script/a64fx_fj/0-model_optimization_for_fugaku.sh

# modify the paths in script/a64fx_fj/1-model_compress.sh

# model compression. use Tubulation of embedding net.
interact $deepmd_root/script/a64fx_fj/1-model_compress.sh

# modify the paths in script/a64fx_fj/2-model_preprocess.sh

# Preprocessing compressed model to speed up tubulation op, which will adjust table layout of the tabulation method.
interact $deepmd_root/script/a64fx_fj/2-model_preprocess.sh

# It is a better practice to copy these three scripts to a specific system and modify their paths.
# There is an example in water system. (examples/water/fugaku)
```

### 3. run lammps+deepmd with optimized model

#### job script example 

more example can be found in water example. (examples/water/fugaku)

```bash
#!/bin/bash -e
#PJM -L "node=1"                        # Number of assign node
#PJM -L "freq=2200"                    
#PJM -L "rscgrp=small"                  # Specify resource group
#PJM -L "elapse=00:05:00"               # Elapsed time limit 1 hour
#PJM --mpi "max-proc-per-node=16"       # Maximum number of MPI processes created per node
#PJM -s                                 # Statistical information output

# modify for your path -------------------------
export deepmd_root=<your deepmd path>
# ----------------------------------------------

if [ -z $deepmd_root ]
then
    echo "not found envoriment variable : deepmd_root"
    exit -1
fi

source $deepmd_root/script/a64fx_fj/env.sh

# These environment variables are required and do not need to be modified.
export OMPI_MCA_plm_ple_memory_allocation_policy=bind_local
export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=1
export TF_INTER_OP_PARALLELISM_THREADS=-1
export TF_INTRA_OP_PARALLELISM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3

# If you use a preprocessed model, this environment variable is required. Otherwise, comment out this line.
export HAVE_PREPROCESSED=1
# The number of threads used by each process.
# The product of max-proc-per-node and DEEPMD_NUM_THREADS should be 48. 16x3 is a recommended configuration.
export DEEPMD_NUM_THREADS=3

# Running lammps uses 1 node, 16 processes, and each process has 3 threads.
mpiexec lmp -echo screen -in <lammps input file>
```

#### run job
```bash
# in login node

pjsub run.sh

```



