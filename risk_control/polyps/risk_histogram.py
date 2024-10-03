import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio as io
import matplotlib.pyplot as plt
import pandas as pd
from polyp_utils import *
from PraNet.lib.PraNet_Res2Net import PraNet
from PraNet.utils.dataloader import test_dataset
import pathlib
import random
from scipy.stats import norm
from skimage.transform import resize
import seaborn as sns
from tqdm import tqdm
import pdb
from core import get_lhat 

def get_example_loss_and_size_tables(cache_path, regions, masks, lambdas_example_table, num_calib):
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'{cache_path}/{lam_low}_{lam_high}_{lam_len}_example_loss_table.npy'
    fname_sizes = f'{cache_path}/{lam_low}_{lam_high}_{lam_len}_example_size_table.npy'
    try:
        loss_table = np.load(fname_loss)
        sizes_table = np.load(fname_sizes)
    except:
        print("computing loss and size table")
        loss_table = np.zeros((regions.shape[0], lam_len))
        sizes_table = np.zeros((regions.shape[0], lam_len))
        for j in tqdm(range(lam_len)):
        # for j in tqdm(range(2)):
            est_regions = (regions >= -lambdas_example_table[j])
            loss_table[:,j] = loss_perpolyp_01(est_regions, regions, masks) 
            sizes_table[:,j] = est_regions.sum(dim=1).sum(dim=1)/masks.sum(dim=1).sum(dim=1)

        np.save(fname_loss, loss_table)
        np.save(fname_sizes, sizes_table)

    return loss_table, sizes_table

def trial_precomputed(clean_example_loss_table, clean_example_sizes_table, noisy_example_loss_table, noisy_example_sizes_table, alpha, num_calib, num_lam, lambdas_example_table):

    perm = torch.randperm(clean_example_loss_table.shape[0])
    clean_example_loss_table = clean_example_loss_table[perm]
    noisy_example_loss_table = noisy_example_loss_table[perm]
    clean_example_size_table = clean_example_sizes_table[perm]
    noisy_example_size_table = noisy_example_sizes_table[perm]

    calib_losses = noisy_example_loss_table[0:num_calib]

    test_clean_losses = clean_example_loss_table[num_calib:]
    test_clean_sizes = clean_example_size_table[num_calib:]

    test_noisy_losses = noisy_example_loss_table[num_calib:]
    test_noisy_sizes = noisy_example_size_table[num_calib:]

    lhat = get_lhat(calib_losses[:, ::-1], lambdas_example_table[::-1], alpha)

    clean_test_losses_lhat = test_clean_losses[:, np.argmax(lambdas_example_table == lhat)]
    clean_test_sizes_lhat = test_clean_sizes[:, np.argmax(lambdas_example_table == lhat)]

    noisy_test_losses_lhat = test_noisy_losses[:, np.argmax(lambdas_example_table == lhat)]
    noisy_test_sizes_lhat = test_noisy_sizes[:, np.argmax(lambdas_example_table == lhat)]

    return lhat, noisy_test_losses_lhat.mean(), clean_test_losses_lhat.mean(), noisy_test_sizes_lhat.mean(), clean_test_sizes_lhat.mean()
    # return lhat, losses.mean(), size

def plot_histograms(df, alpha, output_dir):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,3))
    axs[0].hist(df['risk'].to_numpy(), alpha=0.7, density=True)

    normalized_size = df['sizes'].to_numpy()
    axs[1].hist(normalized_size, bins=60, alpha=0.7, density=True)

    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=10)
    axs[0].axvline(x=alpha,c='#999999',linestyle='--',alpha=0.7)
    axs[0].set_ylabel('density')
    axs[1].set_xlabel('set size as a fraction of polyp size')
    axs[1].locator_params(axis='x', nbins=10)
    axs[1].set_yscale('log')
    #axs[1].legend()
    sns.despine(top=True, right=True, ax=axs[0])
    sns.despine(top=True, right=True, ax=axs[1])
    plt.tight_layout()
    plt.savefig( output_dir + (f'{alpha}_polyp_histograms').replace('.','_') + '.pdf'  )
    print(f"The mean and standard deviation of the risk over {len(df)} trials are {df['risk'].mean()} and {df['risk'].std()} respectively.")

def experiment(alphas, num_trials, num_calib, num_lam, lambdas_example_table, noise_level, noise_type, model_trained_on_noisy):
    set_seeds(0)
    cache_path = get_cache_path(noise_type, noise_level, model_trained_on_noisy=model_trained_on_noisy)  # the test data is may noisy or clean - depends on what we would like to check...
    clean_img_names, clean_sigmoids, clean_masks, clean_regions, clean_num_components = get_data(cache_path, noise_level, noise_type, model_trained_on_noisy=model_trained_on_noisy, is_clean=True)
    noisy_img_names, noisy_sigmoids, noisy_masks, noisy_regions, noisy_num_components = get_data(cache_path, noise_level, noise_type, model_trained_on_noisy=model_trained_on_noisy, is_clean=False)

    clean_masks[clean_masks > 1] = 1
    noisy_masks[noisy_masks > 1] = 1

    clean_example_loss_table, clean_example_sizes_table = get_example_loss_and_size_tables(cache_path, clean_regions,
                                                                                         clean_masks,
                                                                                         lambdas_example_table,
                                                                                         num_calib)
    noisy_example_loss_table, noisy_example_sizes_table = get_example_loss_and_size_tables(cache_path, noisy_regions,
                                                                                       noisy_masks,
                                                                                       lambdas_example_table,
                                                                                       num_calib)

    for alpha in alphas:
        alpha = np.round(alpha, 3)
        fname = cache_path + f'{alpha}_{num_calib}_{num_lam}_{noise_level}_{noise_type}_{model_trained_on_noisy}_dataframe'.replace('.','_') + '.csv'
        try:
            pd.read_csv(fname)
            print('Dataframe loaded')
        except:
            local_df_list = []
            for i in tqdm(range(num_trials)):
                set_seeds(i)
                lhat, noisy_risk, clean_risk, noisy_sizes, clean_sizes = trial_precomputed(
                    clean_example_loss_table, clean_example_sizes_table,
                                                      noisy_example_loss_table, noisy_example_sizes_table,
                                                      alpha, num_calib, num_lam, lambdas_example_table)

                dict_local = {
                                "$\\hat{\\lambda}$": lhat,
                                "noisy risk": noisy_risk,
                                "clean risk": clean_risk,
                                "noisy sizes": noisy_sizes,
                                "clean sizes": clean_sizes,
                                "alpha": alpha,
                             }
                df_local = pd.DataFrame(dict_local, index=[i])
                local_df_list = local_df_list + [df_local]
            df = pd.concat(local_df_list, axis=0, ignore_index=True)
            df.to_csv(fname)

    # return df



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_cache_path(noise_type, noise_level, model_trained_on_noisy):
    return f'./.cache/{noise_type}_{noise_level}_model_trained_on_noisy={model_trained_on_noisy}/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--noise_level', type=float,
                        default=.2)
    parser.add_argument('--noise_type', type=str,
                        default='uniform')
    parser.add_argument('--model_trained_on_noisy', type=int,
                        default=1)
    opt = parser.parse_args()
    opt.model_trained_on_noisy = opt.model_trained_on_noisy > 0
    with torch.no_grad():
        sns.set(palette='pastel', font='serif')
        sns.set_style('white')
        fix_randomness()

        num_trials = 1000
        num_calib = 500
        num_lam = 1500
        lambdas_example_table = np.linspace(-1, 0, 1000)
        noise_level = opt.noise_level
        noise_type = opt.noise_type
        model_trained_on_noisy = opt.model_trained_on_noisy

        cache_path = get_cache_path(noise_type, noise_level, model_trained_on_noisy)
        pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
        alphas = np.arange(0., 1.001, 0.01)
        experiment(alphas, num_trials, num_calib, num_lam, lambdas_example_table, noise_level, noise_type, model_trained_on_noisy)


        # plot_histograms(df, alpha, output_dir)
