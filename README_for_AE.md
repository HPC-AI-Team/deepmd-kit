# DeepMD for Fugaku

## build deepmd for fugaku

```bash
# in login node 

cd <samewhere you want to put deepmd>
mkdir DeepMD
cd DeepMD

# tensroflow for fugaku

mkdir package
# download from [google driver](https://drive.google.com/file/d/1BF3ereji7g0Aj0X_q4tjwC1wjbKB7zqd/view?usp=sharing)
cp <where tensorflow you put>/TensorFlow-2.2.0.tar.gz ./package

mkdir dependents
cd dependents
tar -xzf ../package/TensorFlow-2.2.0.tar.gz

cd ..
git clone git@github.com:gzq942560379/deepmd-kit.git

cd deepmd-kit
git checkout -b fugaku origin/fugaku

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






