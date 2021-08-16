#!/bin/bash -ex

rm -rf model.ckpt.*
rm -f checkpoint
rm -f *.json
rm -f *.log
rm -f *.dump
rm -f water.lmp

rm -f profiler.json_*