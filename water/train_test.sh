#/bin/bash

set -ex

dp train train/water_se_a.json
dp freeze -o model/graph.pb
dp test -m model/graph.pb -s ./data -n 1
rm -rf model.ckpt.*
rm -rf checkpoint
