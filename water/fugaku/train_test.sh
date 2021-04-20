
cd ~/gzq/deepmd-kit/source/build
make -j48 && make install

cd ~/gzq/deepmd-kit/
python ./setup.py install

cd ~/gzq/water

dp train water_se_a.json
dp freeze -o graph.pb
dp test -m graph.pb -s /vol0004/hp200266/u01036/gzq/deepmd-kit/examples/water/data -n 1
rm -rf model.ckpt.*
rm -rf checkpoint
