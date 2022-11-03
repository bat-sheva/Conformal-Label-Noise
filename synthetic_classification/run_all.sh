#!/bin/bash

for noise_probability in 0.1 0.05 0.01
do
for noise_type in None Uniform Confusion_Matrix Rare_to_Common Common_Mistake Wrong_to_Right Adversarial_HPS Adversarial_APS
do
sbatch -c 2 --gres=gpu:0 -o slurm-test.out -J exp --export=noise_type=$noise_type,noise_probability=$noise_probability ./submit.sh
done
done
