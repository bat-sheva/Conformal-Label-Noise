#!/bin/bash

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate conf

python ./main_scripts/main.py --noise_probability $noise_probability --noise_type $noise_type