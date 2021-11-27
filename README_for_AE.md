# DeepMD for Fugaku

## build deepmd for fugaku

```bash
# in login node 
cd <samewhere you want to put deepmd>

mkdir DeepMD

cd DeepMD

tar -xzvf deepmd-kit-fugaku.tar.gz

# tensroflow for fugaku
mkdir package
# download from [google driver](https://drive.google.com/file/d/1BF3ereji7g0Aj0X_q4tjwC1wjbKB7zqd/view?usp=sharing)
cp <where tensorflow you put>/TensorFlow-2.2.0.tar.gz ./package

mkdir dependents
cd dependents
tar -xzf ../package/TensorFlow-2.2.0.tar.gz

cd ..

# The following two lines will write to ~/.bashrc for convenience.(Only execute once)
echo "export deepmd_root=$(pwd)" >> ~/.bashrc
echo 'alias "interact=pjsub --interact -L node=1 -L freq=2200 --sparam wait-time=600 "' >> ~/.bashrc

source ~/.bashrc

# build deepmd c++
interact $deepmd_root/script/a64fx_fj/build_deepmd.sh

# build deepmd python
interact $deepmd_root/script/a64fx_fj/build_python.sh

# build lammps
interact $deepmd_root/script/a64fx_fj/build_lammps.sh
```

## run script example

```bash
# water
cd $deepmd_root/examples/water/fugaku

# run lammps + deepmd with 1 core for water
interact run_compress_preprocess_1_1.sh

# copper
cd $deepmd_root/examples/copper/fugaku
# run lammps + deepmd with 1 core for water
interact run_compress_preprocess_1_1.sh
```

## run script and output data


```bash
pjsub run.sh 
```

### step-by-step speedup (Fig.8)

water: examples/water/fugaku_one_node_optimazation

copper: examples/copper/fugaku_one_node_optimazation

### strong scaling (Fig.9 Fig.10)

water: examples/water/fugaku_strong_scale_16x3

copper: examples/copper/fugaku_strong_scale_16x3

### weak scaling (Fig.11)

water: examples/water/fugaku_weak_scale_16x3_

copper: examples/copper/fugaku_weak_scale_16x3_
