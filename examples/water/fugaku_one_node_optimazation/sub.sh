#!/bin/bash

pjsub run_48x1x1_48.sh
pjsub run_compress_48x1x1_48.sh
# pjsub run_compress_preprocess_48x1x1_48.sh

# pjsub run_48x1x1_144.sh
# pjsub run_48x1x1_192.sh
# pjsub run_48x1x1_240.sh
# pjsub run_48x1x1_288.sh



# pjsub run_48x1x1_96.sh
# pjsub run_compress_48x1x1_96.sh
# pjsub run_compress_preprocess_48x1x1_96.sh

watch pjstat
