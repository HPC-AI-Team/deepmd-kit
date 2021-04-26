#!/bin/bash

set -ex

pjsub ./run_1.sh
pjsub ./run_2.sh
pjsub ./run_4.sh
pjsub ./run_8.sh
pjsub ./run_16.sh
pjsub ./run_32.sh
pjsub ./run_64.sh
pjsub ./run_128.sh
pjsub ./run_256.sh