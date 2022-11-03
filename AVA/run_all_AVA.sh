#!/bin/bash

for seed in {13..24}
do
#for conv_base_lr in 1e-3
#do

sbatch -N1 -c1 --gres=gpu:A4000:1 -w plato2 -o slurm-test.out -J exp --export=seed=$seed submit_AVA.sh
#sbatch -c 2 -A galileo -p galileo --gres=gpu:1 -o slurm-test.out -J exp --export=seed=$seed submit_AVA.sh
#sbatch -N1 -c1 --gres=gpu:A40:1 -w nlp-a40-1 -o slurm-test.out -J exp --export=seed=$seed submit_AVA.sh

#done
done
