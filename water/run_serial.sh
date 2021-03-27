#!/bin/bash
#PJM -L "node=1"               # Number of node
#PJM -L "rscgrp=eap-small"     # Specify resource group
#PJM -L "elapse=60:00"         # Job run time limit value
#PJM -S     

lmp_serial -echo screen -in ./lmp/in.water

