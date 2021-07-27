#!/bin/bash


pjsub run_compress_preprocess_8x6x50_864x50.sh
pjsub run_compress_preprocess_48x1x4560_864x50.sh

watch pjstat