# Regression- adding different noises

import torch
import numpy as np

def add_Y_noise_additive(Y, probability=1.0, sigma=0.1, positive=False, gaussian=False, gumbel=False):
    u = np.random.uniform(0,1,(len(Y),))
    indices_to_change = np.where(u <= probability)[0]
    Y_noisy = Y.copy()
    if positive:
      if gaussian:
        Y_noisy[indices_to_change] = Y_noisy[indices_to_change] + np.abs(sigma*np.random.randn(len(indices_to_change)))
      elif gumbel:
        Y_noisy[indices_to_change] = Y_noisy[indices_to_change] + np.abs(sigma*np.random.gumbel(0, 1, len(indices_to_change)))
      else:
        Y_noisy[indices_to_change] = Y_noisy[indices_to_change] + np.abs(sigma*np.random.standard_t(1,len(indices_to_change)))
    else:
      if gaussian:
        Y_noisy[indices_to_change] = Y_noisy[indices_to_change] + sigma*np.random.randn(len(indices_to_change))
      elif gumbel:
        gumbel_noise = np.random.gumbel(0, 1, len(indices_to_change))
        print(np.mean(gumbel_noise))
        print(np.std(gumbel_noise))
        gumbel_noise_normalized = (gumbel_noise - np.mean(gumbel_noise))/np.std(gumbel_noise)
        print(np.mean(gumbel_noise_normalized))
        print(np.std(gumbel_noise_normalized))
        Y_noisy[indices_to_change] = Y_noisy[indices_to_change] + sigma*gumbel_noise_normalized
      else:
        Y_noisy[indices_to_change] = Y_noisy[indices_to_change] + sigma*np.random.standard_t(1,len(indices_to_change))

    return Y_noisy
    
    
def add_Y_noise_dependant(Y, probability=1.0, contractive=True):
    u = np.random.uniform(0,1,(len(Y),))
    indices_to_change = np.where(u <= probability)[0]
    Y_mean = np.mean(Y)
    Y_dist = Y-Y_mean
    Y_noisy = Y.copy()
    if contractive:
      Y_noisy[indices_to_change] = Y_noisy[indices_to_change] - Y_dist*np.random.uniform(0,0.5,(len(indices_to_change),))
    else:
      Y_noisy[indices_to_change] = Y_noisy[indices_to_change] + Y_dist*np.random.uniform(0,0.5,(len(indices_to_change),))
    return Y_noisy