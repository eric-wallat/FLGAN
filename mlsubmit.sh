#!/bin/sh
fn=$1
options="-N ${fn} -q LUNG -e /Users/ewallat/mltest/logs/ -o /Users/ewallat/mltest/logs/  -l gpu.cuda.0.mem_free=30G -l ngpus=1 -pe smp 8 -cwd"

qsub ${options} train.job
