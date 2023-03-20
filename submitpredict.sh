#!/bin/sh

options="-N mlpredict -q LUNG -e /Users/ewallat/mltest/logs/predict -o /Users/ewallat/mltest/logs/predict/ -l ngpus=1 -l gpu.cuda.0.mem_free=30G  -pe smp 40 -cwd"

qsub ${options} predict.job
