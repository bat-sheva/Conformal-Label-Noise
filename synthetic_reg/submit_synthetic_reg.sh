#!/bin/bash


source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate project

python3 main_synthetic_reg.py $seed $sigma
