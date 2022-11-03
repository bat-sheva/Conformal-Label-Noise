#!/bin/bash

for seed in {1..50}
do
for sigma in 0, 0.01, 0.1, 1
#for sigma in 0
do

sbatch -N1 -c1 --gres=gpu:0 -o slurm-test.out -J exp --export=seed=$seed,sigma=$sigma submit_synthetic_reg.sh

done
done
