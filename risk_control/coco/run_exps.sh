

srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.1 --noise_type independent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.2 --noise_type independent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.3 --noise_type independent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.4 --noise_type independent_even --model_trained_on_noisy 0 &
#srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.7 --noise_type independent_even --model_trained_on_noisy 0 &
#srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.9 --noise_type independent_even --model_trained_on_noisy 0 &

#srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.1 --noise_type dependent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.2 --noise_type dependent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.3 --noise_type dependent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.4 --noise_type dependent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.7 --noise_type dependent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.9 --noise_type dependent_even --model_trained_on_noisy 0 &

srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.1 --noise_type partial_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.2 --noise_type partial_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.3 --noise_type partial_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.4 --noise_type partial_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.7 --noise_type partial_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.9 --noise_type partial_even --model_trained_on_noisy 0 &


srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.1 --noise_type independent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.2 --noise_type independent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.3 --noise_type independent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.4 --noise_type independent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.7 --noise_type independent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.9 --noise_type independent_uneven --model_trained_on_noisy 0 &

srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.1 --noise_type dependent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.2 --noise_type dependent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.3 --noise_type dependent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.4 --noise_type dependent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.7 --noise_type dependent_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.9 --noise_type dependent_uneven --model_trained_on_noisy 0 &

srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.1 --noise_type partial_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.2 --noise_type partial_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.3 --noise_type partial_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.4 --noise_type partial_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.7 --noise_type partial_uneven --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.9 --noise_type partial_uneven --model_trained_on_noisy 0 &






srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.7 --noise_type independent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.9 --noise_type independent_even --model_trained_on_noisy 0 &
srun -c 3 --gres=gpu:1 -J plsNoKil --exclude=dym-lab python risk_histogram.py --noise_level 0.1 --noise_type dependent_even --model_trained_on_noisy 0 &

