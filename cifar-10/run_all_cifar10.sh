#!/bin/bash

for seed in {1..30}
do

#sbatch -N1 -c1 --gres=gpu:A4000:1 -w plato2 -o slurm-test.out -J exp --export=seed=$seed submit_cifar10.sh
sbatch -c 2 -A galileo -p galileo --gres=gpu:1 -o slurm-test.out -J exp --export=seed=$seed submit_cifar10.sh
#sbatch -N1 -c1 --gres=gpu:A40:1 -w nlp-a40-1 -o slurm-test.out -J exp --export=seed=$seed submit_cifar10.sh

done
