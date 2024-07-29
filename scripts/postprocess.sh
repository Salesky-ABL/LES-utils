#!/bin/bash
# postprocess code
conda activate LES
# arguments
d=$1
s=$2
# run process.py
pypath=/home/bgreene/anaconda3/envs/LES/bin/python
pyscript=/home/bgreene/LES-utils/scripts/process.py
# command
$pypath $pyscript -d $d -s $s