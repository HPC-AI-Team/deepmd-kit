### Fugaku 安装 deepmd baseline


##### in login node
```bash
git clone git@github.com:gzq942560379/deepmd-kit.git
cd deepmd-kit
git checkout baseline

# add deepmd_root to ~/.bashrc
echo "export deepmd_root=$(pwd)" >> ~/.bashrc
```

#### 申请一个计算节点
```bash
pjsub --interact -L node=1 -L elapse=4:00:00 --sparam wait-time=600
```

#### in compute node
```bash
cd $deepmd_root
source ./script/fugaku/env.sh
bash ./script/fugaku/install.sh
```
