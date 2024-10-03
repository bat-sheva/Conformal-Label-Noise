import os, sys
from itertools import product

import pandas as pd
import torch
import numpy as np
import argparse
import six
from tqdm import tqdm

from helper import set_seeds
from datasets.datasets import data_train_test_split, DataGeneratorFactory
from utils.models import PredictionIntervalModel
from losses import batch_qr_loss, batch_interval_loss, Loss, MiscoverageLoss, \
    SmoothedMiscoverageLoss
import helper

sys.modules['sklearn.externals.six'] = six
# np.warnings.filterwarnings('ignore')

os.environ["MKL_CBWR"] = 'AUTO'

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def get_loss_fn(loss_name):
    if loss_name == 'batch_qr' or loss_name == 'batch_wqr':
        fn = batch_qr_loss
    elif loss_name == 'batch_int':
        fn = batch_interval_loss
    else:
        raise ValueError('loss arg not valid')

    return fn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')

    parser.add_argument('--seed_begin', type=int, default=None,
                        help='random seed')
    parser.add_argument('--seed_end', type=int, default=None,
                        help='random seed')

    parser.add_argument('--dataset_name', type=str, default='',
                        help='dataset to use')
    parser.add_argument('--noise', type=str, default='normal',
                        help='noise to use')

    parser.add_argument('--num_q', type=int, default=30,
                        help='number of quantiles you want to sample each step')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu num to use')

    parser.add_argument('--num_ep', type=int, default=10000,
                        help='number of epochs')
    parser.add_argument('--nl', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--hs', type=int, default=64,
                        help='hidden size')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout ratio of the dropout level')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--bs', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--wait', type=int, default=200,
                        help='how long to wait for lower validation loss')

    parser.add_argument('--loss', type=str,
                        help='specify type of loss')

    parser.add_argument('--ds_type', type=str, default="",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')

    parser.add_argument('--test_ratio', type=float, default=0.4,
                        help='ratio of test set size')

    parser.add_argument('--max_data_size', type=int, default=35000,
                        help='')
    parser.add_argument('--method', type=str, default='oqr',
                        help='')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    args.device = device

    return args


def get_lambda_hat(loss: Loss, nominal_loss_level, y_cal, intervals):  # 0.0001
    y_cal = y_cal.squeeze()
    if isinstance(loss, MiscoverageLoss):
        E_i = torch.max(intervals[:, 0] - y_cal, y_cal - intervals[:, 1])
        Q = torch.quantile(E_i, q=min((1 - nominal_loss_level) * (1 + 1 / len(E_i)), 0.999))
        return Q
    lambdas = torch.arange(-3, 3, 0.0005, device=intervals.device)
    lambdas = lambdas.flip(0)
    n_lambdas = len(lambdas)
    n = y_cal.shape[0]
    lambdas_rep = lambdas.unsqueeze(1).repeat(1, n)
    y_cal_rep = y_cal.unsqueeze(0).repeat(n_lambdas, 1).flatten(0, 1)
    intervals_rep = intervals.clone()
    intervals_rep = intervals_rep.unsqueeze(0).repeat(n_lambdas, 1, 1)  # .flatten(0,1)
    intervals_rep[..., 0] -= lambdas_rep
    intervals_rep[..., 1] += lambdas_rep
    intervals_rep = intervals_rep.flatten(0, 1)
    losses = loss.compute_loss(y_cal_rep, intervals_rep)
    unflatten = torch.nn.Unflatten(0, (n_lambdas, n))
    losses = unflatten(losses).mean(dim=1)
    idx = max(
        torch.argmax((((n / (n + 1)) * losses + loss.max_val() / (n + 1)) >= nominal_loss_level).float()).item() - 1, 0)
    return lambdas[idx]


def get_results_dict(model: PredictionIntervalModel, miscoverage_level, data, args):
    cal_intervals = model.estimate_interval(data['x_cal'], miscoverage_level, use_best_va_model=True)
    lambda_hat = get_lambda_hat(MiscoverageLoss(), miscoverage_level, data['noisy_y_cal'], cal_intervals)
    intervals = model.estimate_interval(data['x_test'], miscoverage_level, use_best_va_model=True)
    calibrated_intervals = intervals.clone()
    calibrated_intervals[..., 0] -= lambda_hat
    calibrated_intervals[..., 1] += lambda_hat
    cal_intervals[..., 0] -= lambda_hat
    cal_intervals[..., 1] += lambda_hat

    miscoverage_loss = MiscoverageLoss()
    smoothed_miscoverage_loss = SmoothedMiscoverageLoss()

    noisy_miscoverage = miscoverage_loss.compute_loss(data['noisy_y_te'].squeeze(), calibrated_intervals).mean().item()
    clean_miscoverage = miscoverage_loss.compute_loss(data['clean_y_te'].squeeze(), calibrated_intervals).mean().item()
    noisy_smoothed_miscoverage = smoothed_miscoverage_loss.compute_loss(data['noisy_y_te'].squeeze(),
                                                                        calibrated_intervals).mean().item()
    clean_smoothed_miscoverage = smoothed_miscoverage_loss.compute_loss(data['clean_y_te'].squeeze(),
                                                                        calibrated_intervals).mean().item()
    q = smoothed_miscoverage_loss.min_second_derivative(intervals=calibrated_intervals)
    epsilon_var_bound = 0.1
    intersection_point = smoothed_miscoverage_loss.get_intersection_point()
    clean_smoothed_miscoverage_bound = noisy_smoothed_miscoverage - 0.5 * q.mean().item() * epsilon_var_bound
    clean_miscoverage_bound = clean_smoothed_miscoverage_bound / intersection_point

    best_params = get_tightest_clean_miscoverage_bounds(data['noisy_y_cal'].squeeze(), data['clean_y_cal'].squeeze(),
                                                        cal_intervals, epsilon_var_bound)
    smoothed_miscoverage_loss_best_params = SmoothedMiscoverageLoss(**best_params)
    best_bounds = get_clean_miscoverage_bounds(smoothed_miscoverage_loss_best_params, data['noisy_y_te'].squeeze(),
                                               data['clean_y_te'].squeeze(), intervals, epsilon_var_bound)
    return {
        'nominal_miscoverage_level': miscoverage_level,
        'noisy_miscoverage': noisy_miscoverage,
        'clean_miscoverage': clean_miscoverage,
        'noisy_smoothed_miscoverage': noisy_smoothed_miscoverage,
        'clean_smoothed_miscoverage': clean_smoothed_miscoverage,
        'clean_miscoverage_bound': clean_miscoverage_bound,
        'clean_smoothed_miscoverage_bound': clean_smoothed_miscoverage_bound,
        'noisy_best_smoothed_miscoverage': smoothed_miscoverage_loss_best_params.compute_loss(
            data['noisy_y_te'].squeeze(),
            calibrated_intervals).mean().item(),
        'clean_best_smoothed_miscoverage': smoothed_miscoverage_loss_best_params.compute_loss(
            data['clean_y_te'].squeeze(),
            calibrated_intervals).mean().item(),
        'clean_miscoverage_best_bound': best_bounds['clean_miscoverage_bound'],
        'clean_smoothed_miscoverage_best_bound': best_bounds['clean_smoothed_miscoverage_bound'],
        'cheat_clean_miscoverage_best_bound': best_bounds['cheat_clean_miscoverage_bound'],
        'cheat_improved_markov_clean_miscoverage_best_bound': best_bounds[
            'cheat_improved_markov_clean_miscoverage_bound'],
        'best_p': best_bounds['p']
    }


def get_simple_markov_clean_miscoverage_bound(smoothed_miscoverage_loss, noisy_y_cal, intervals, epsilon_var_bound):
    intersection_point = smoothed_miscoverage_loss.get_intersection_point()
    q = smoothed_miscoverage_loss.min_second_derivative(intervals=intervals)
    noisy_smoothed_miscoverage = smoothed_miscoverage_loss.compute_loss(noisy_y_cal.squeeze(), intervals).mean().item()
    clean_smoothed_miscoverage_bound = noisy_smoothed_miscoverage - 0.5 * q.mean().item() * epsilon_var_bound
    clean_miscoverage_bound = clean_smoothed_miscoverage_bound / intersection_point
    return clean_miscoverage_bound, clean_smoothed_miscoverage_bound


def get_best_cheat_clean_miscoverage_bound(clean_y_cal, intervals):
    cs = [0.1, 0.5 + 0.01, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 100]
    ds = [0.001, 0.1, 0.3, .5, .75, 1, 2.5, 2, 2.5, 3, 3.5, 4, 5, 10]
    best_bound = None
    best_params = None
    for c, d in tqdm(list(product(cs, ds))):
        smoothed_miscoverage_loss = SmoothedMiscoverageLoss(d=d, c=c)
        clean_smoothed_miscoverage = smoothed_miscoverage_loss.compute_loss(clean_y_cal, intervals).mean().item()
        intersection_point = smoothed_miscoverage_loss.get_intersection_point()
        cheat_clean_miscoverage_bound = clean_smoothed_miscoverage / intersection_point
        if best_bound is None or cheat_clean_miscoverage_bound < best_bound:
            best_params = {'c': c, 'd': d}
            best_bound = cheat_clean_miscoverage_bound

    return best_bound, best_params


def get_clean_miscoverage_bounds(smoothed_miscoverage_loss, noisy_y_cal, clean_y_cal, intervals, epsilon_var_bound):
    simple_markov_clean_miscoverage_bound, clean_smoothed_miscoverage_bound = get_simple_markov_clean_miscoverage_bound(
        smoothed_miscoverage_loss,
        noisy_y_cal, intervals,
        epsilon_var_bound)

    cheat_clean_miscoverage_bound, best_cheat_params = get_best_cheat_clean_miscoverage_bound(clean_y_cal, intervals)

    def find_best_p():
        smoothed_miscoverage_loss = SmoothedMiscoverageLoss(**best_cheat_params, p=1)
        intersection_point = smoothed_miscoverage_loss.get_intersection_point()
        clean_smoothed_miscoverages = smoothed_miscoverage_loss.compute_loss(noisy_y_cal, intervals)
        ps = torch.arange(0.1, 15.1, 0.1).to(clean_smoothed_miscoverages.device)
        ps_rep = ps.unsqueeze(1).repeat(1, clean_smoothed_miscoverages.shape[0])
        clean_smoothed_miscoverages_rep = clean_smoothed_miscoverages.unsqueeze(0).repeat(len(ps), 1)
        pth_moment = torch.pow(clean_smoothed_miscoverages_rep, ps_rep).mean(dim=1)
        improved_markov_bounds = pth_moment / (intersection_point ** ps)
        improved_markov_bounds[improved_markov_bounds.isnan()] = np.inf
        best_idx = improved_markov_bounds.argmin().item()
        best_p = ps[best_idx].item()
        best_bound = improved_markov_bounds[best_idx].item()
        return best_p, best_bound

    best_p, cheat_improved_markov_clean_miscoverage_bound = find_best_p()

    return {
        'clean_miscoverage_bound': simple_markov_clean_miscoverage_bound,
        'clean_smoothed_miscoverage_bound': clean_smoothed_miscoverage_bound,
        'cheat_clean_miscoverage_bound': cheat_clean_miscoverage_bound,
        'cheat_improved_markov_clean_miscoverage_bound': cheat_improved_markov_clean_miscoverage_bound,
        'p': best_p
    }


def get_tightest_clean_miscoverage_bounds(noisy_y_cal, clean_y_cal, intervals, epsilon_var_bound):
    best_params = {'c': None, 'd': None}
    best_bound = None
    ps = [0.1, 0.5 + 0.01, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10]
    ds = [0.3, .5, .75, 1, 2.5, 2, 2.5, 3, 3.5, 4, 5, 10]
    rnd_idx = np.random.permutation(len(noisy_y_cal))[:7000]
    shuffled_noisy_y_cal = noisy_y_cal[rnd_idx]
    shuffled_intervals = intervals[rnd_idx]
    for p, d in tqdm(list(product(ps, ds))):
        smoothed_miscoverage_loss = SmoothedMiscoverageLoss(d=d, p=p)
        bound, _ = get_simple_markov_clean_miscoverage_bound(smoothed_miscoverage_loss, shuffled_noisy_y_cal,
                                                             shuffled_intervals, epsilon_var_bound)
        if best_bound is None or bound <= best_bound:
            best_bound = bound
            best_params = {'p': p, 'd': d}

    return best_params


def main():
    args = parse_args()

    if 'syn' in args.ds_type.lower():
        is_real_data = False
        data_type = 'syn'
    elif 'real' in args.ds_type.lower():
        is_real_data = True
        data_type = 'real'
    else:
        raise RuntimeError('Must decide dataset type!')
    args.is_real = is_real_data

    if args.seed is not None:
        seeds = [args.seed]
    elif args.seed_begin is not None and args.seed_end is not None:
        seeds = range(args.seed_begin, args.seed_end)
    else:
        seeds = range(0, 20)

    print('DEVICE: {}'.format(args.device))

    train_all_q = False
    synthetic_datasets = ['extreme_bimodal_x_dim_10', 'smooth_bimodal_x_dim_10']
    real_datasets = ['normal_noise_bio', 'normal_noise_meps_19']
    if is_real_data:
        dataset_names = real_datasets
    else:
        dataset_names = synthetic_datasets
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        set_seeds(0)
        data_generator = DataGeneratorFactory.get_data_generator(args.dataset_name, is_real_data, args)
        x, clean_y, noisy_y = data_generator.generate_data(n_samples=30000, add_noise=True)
        print("total data size: ", len(x))
        for s in seeds:
            print(f"seed: {s}")
            args.seed = s
            set_seeds(args.seed)

            X = torch.Tensor(x)
            clean_Y = torch.Tensor(clean_y).reshape(len(clean_y), 1)
            noisy_Y = torch.Tensor(noisy_y).reshape(len(noisy_y), 1)
            data1 = data_train_test_split(clean_Y, X, device=device, test_ratio=0.2, val_ratio=0.1,
                                          calibration_ratio=0.5, seed=s, scale=True, dim_to_reduce=None,
                                          is_real=args.is_real, y_prefix='clean_')
            data2 = data_train_test_split(noisy_Y, X, device=device, test_ratio=0.2, val_ratio=0.1,
                                          calibration_ratio=0.5, seed=s, scale=True, dim_to_reduce=None,
                                          is_real=args.is_real, y_prefix='noisy_')
            data = {**data1, **data2}
            x_te = data['x_test']
            x_va = data['x_val']
            x_tr = data['x_train']
            x_ca = data['x_cal']

            y_va = data['noisy_y_val']
            y_tr = data['noisy_y_train']
            print("train size: ", x_tr.shape)
            print("val size: ", x_va.shape)
            print("cal size: ", x_ca.shape)
            print("test size: ", x_te.shape)
            dim_x = x_tr.shape[-1]
            dim_y = y_tr.shape[-1]

            # miscoverage_level = 0.1
            for miscoverage_level in [0.01, 0.02, 0.05, 0.1, 0.15] + list(np.arange(0.2, 0.9, 0.1)) + [0.9, 0.95, 0.98,
                                                                                                       0.99]:
                miscoverage_level = np.round(miscoverage_level, 3)
                print(f"s={s}, miscoverage level={miscoverage_level}")
                quantiles = None  # torch.Tensor([0.05, 0.95])  # torch.arange(0.03, 0.97, 0.01, device=device)
                model = PredictionIntervalModel(dim_x, dim_y, train_all_q, quantiles, miscoverage_level, args.hs,
                                                args.nl, args.dropout, args.lr, args.wd, device, args)
                model_save_dir = f"models/{args.dataset_name}/miscoverage_level={miscoverage_level}/s={s}"
                helper.create_folder_if_it_doesnt_exist(model_save_dir)
                try:
                    model.model.load_state_dict(torch.load(f"{model_save_dir}/model"))
                    model.best_va_model.load_state_dict(torch.load(f"{model_save_dir}/best_va_model"))
                except:
                    model.train(x_tr, y_tr, x_va, y_va, args)
                    torch.save(model.model.state_dict(), f"{model_save_dir}/model")
                    torch.save(model.best_va_model.state_dict(), f"{model_save_dir}/best_va_model")
                # continue
                results = get_results_dict(model, miscoverage_level, data, args)
                save_dir = f"results/{data_type}/{args.dataset_name}/"
                save_name = f"miscoverage_level={miscoverage_level}_seed={s}.csv"
                helper.create_folder_if_it_doesnt_exist(save_dir)
                pd.DataFrame(results, index=[s]).to_csv(f"{save_dir}/{save_name}")


if __name__ == '__main__':
    main()
