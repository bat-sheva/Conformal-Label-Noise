import os, sys, inspect

import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from utils import *
from core import get_lhat
import seaborn as sns
from ml_decoder.models import create_model
from helper_functions import CIFARDetection, CocoDetection, WebVisionDetection

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--pic-path', type=str, default='./pics/dog.jpg')
parser.add_argument('--model-name', type=str, default='tresnet_l')
parser.add_argument('--model_trained_on_noisy', type=int, default=1)
parser.add_argument('--dataset_name', type=str, default='coco')
parser.add_argument('--input-size', type=int, default=640)
# parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=80)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)
parser.add_argument('--noise_level', type=float,
                    default=.2)
parser.add_argument('--noise_type', type=str,
                    default='uniform')


def get_example_loss_and_size_tables(scores, labels, lambdas_example_table, num_calib, noise_type, noise_level,
                                     is_clean):
    cache_path = get_cache_path(args)
    lam_len = len(lambdas_example_table)
    lam_low = min(lambdas_example_table)
    lam_high = max(lambdas_example_table)
    fname_loss = f'{cache_path}/{lam_low}_{lam_high}_{lam_len}_is_clean={is_clean}_example_loss_table.npy'
    fname_sizes = f'{cache_path}/{lam_low}_{lam_high}_{lam_len}_is_clean={is_clean}_example_size_table.npy'
    try:
        loss_table = np.load(fname_loss)
        sizes_table = np.load(fname_sizes)
        # raise Exception("e")
    except:
        loss_table = np.zeros((scores.shape[0], lam_len))
        sizes_table = np.zeros((scores.shape[0], lam_len))
        print("caching loss and size tables")
        for j in tqdm(range(lam_len)):
            est_labels = scores >= lambdas_example_table[j]
            loss, sizes = get_metrics_precomputed(est_labels, labels)
            loss_table[:, j] = loss
            sizes_table[:, j] = sizes

        np.save(fname_loss, loss_table)
        np.save(fname_sizes, sizes_table)

    return loss_table, sizes_table


def trial_precomputed(clean_example_loss_table, clean_example_size_table,
                      noisy_example_loss_table, noisy_example_size_table,
                      lambdas_example_table, alpha, num_lam, num_calib,
                      batch_size):
    perm = torch.randperm(clean_example_loss_table.shape[0])
    clean_example_loss_table = clean_example_loss_table[perm]
    noisy_example_loss_table = noisy_example_loss_table[perm]
    clean_example_size_table = clean_example_size_table[perm]
    noisy_example_size_table = noisy_example_size_table[perm]

    calib_losses = noisy_example_loss_table[0:num_calib]

    test_clean_losses = clean_example_loss_table[num_calib:]
    test_clean_sizes = clean_example_size_table[num_calib:]

    test_noisy_losses = noisy_example_loss_table[num_calib:]
    test_noisy_sizes = noisy_example_size_table[num_calib:]

    lhat = get_lhat(calib_losses, lambdas_example_table, alpha)

    clean_test_losses_lhat = test_clean_losses[:, np.argmax(lambdas_example_table == lhat)]
    clean_test_sizes_lhat = test_clean_sizes[:, np.argmax(lambdas_example_table == lhat)]

    noisy_test_losses_lhat = test_noisy_losses[:, np.argmax(lambdas_example_table == lhat)]
    noisy_test_sizes_lhat = test_noisy_sizes[:, np.argmax(lambdas_example_table == lhat)]

    return clean_test_losses_lhat.mean(), noisy_test_losses_lhat.mean(), np.mean(
        clean_test_sizes_lhat), np.mean(noisy_test_sizes_lhat), lhat


def plot_histograms(df, alpha):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

    minrisk = df['risk'].min()
    maxrisk = df['risk'].max()

    risk_bins = np.arange(minrisk, maxrisk, 0.001)
    size = df['size'].to_numpy()
    d = np.diff(np.unique(size)).min()
    lofb = size.min() - float(d) / 2
    rolb = size.max() + float(d) / 2
    size_bins = np.arange(lofb, rolb + d, d)

    axs[0].hist(df['risk'], risk_bins, alpha=0.7, density=True)

    axs[1].hist(size, size_bins, alpha=0.7, density=True)

    axs[0].set_xlabel('risk')
    axs[0].locator_params(axis='x', nbins=4)
    axs[0].set_ylabel('density')
    # axs[0].set_yticks([0,100])
    axs[0].axvline(x=alpha, c='#999999', linestyle='--', alpha=0.7)
    axs[1].set_xlabel('size')
    sns.despine(ax=axs[0], top=True, right=True)
    sns.despine(ax=axs[1], top=True, right=True)
    # axs[1].legend()
    plt.tight_layout()
    os.makedirs('../outputs/histograms/', exist_ok=True)
    plt.savefig('../' + (f'outputs/histograms/{alpha}_coco_histograms').replace('.', '_') + '.pdf')
    print(f"Average threshold: ", df["$\\hat{\\lambda}$"].mean())
    print(
        f"The mean and standard deviation of the risk over {len(df)} trials are {df['risk'].mean()} and {df['risk'].std()} respectively.")


def get_scores_and_labels(model, dataset, corr, noise_type, noise_level, is_clean, n_labels, compute_scores):
    cache_path = get_cache_path(args)
    set_seeds(0)
    dataset_fname = f'{cache_path}/coco_val_is_clean={is_clean}.pkl'
    if os.path.exists(dataset_fname):
        dataset_precomputed = pkl.load(open(dataset_fname, 'rb'))
        print(f"Precomputed dataset loaded. Size: {len(dataset_precomputed)}")
    else:
        dataset_precomputed = get_scores_targets(model,
                                                 torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True),
                                                 corr,
                                                 n_labels, compute_scores)
        pkl.dump(dataset_precomputed, open(dataset_fname, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
    scores, labels = dataset_precomputed.tensors
    scores = torch.rand_like(scores) * 0.001 + scores
    if compute_scores:
        return scores, labels
    else:
        return labels


def experiment(alphas, num_lam, num_calib, lambdas_example_table, num_trials, batch_size, noise_type, noise_level, model_trained_on_noisy):
    fix_randomness(seed=0)
    cache_path = get_cache_path(args)
    # dataset

    # dataset = tv.datasets.CocoDetection(coco_val_2017_directory, coco_instances_val_2017_json,
    #                                     transform=tv.transforms.Compose(
    #                                         [tv.transforms.Resize((args.input_size, args.input_size)),
    #                                          tv.transforms.ToTensor()]))
    instances_path_val = os.path.join("../asl/data", 'annotations/instances_val2014.json')
    data_path_val = f'../asl/data/val2014'  # args.data
    if args.dataset_name == 'coco':
        clean_dataset = CocoDetection(data_path_val,
                                      instances_path_val,
                                      transforms.Compose([
                                          transforms.Resize((448, 448)),
                                          transforms.ToTensor(),
                                      ]),
                                      noise_level=noise_level,
                                      noise_type=noise_type,
                                      is_clean=True)
        noisy_dataset = CocoDetection(data_path_val,
                                      instances_path_val,
                                      transforms.Compose([
                                          transforms.Resize((448, 448)),
                                          transforms.ToTensor(),
                                      ]),
                                      noise_level=noise_level,
                                      noise_type=noise_type,
                                      is_clean=False)
    elif args.dataset_name == 'cifar':
        dataset = torchvision.datasets.CIFAR10(root='../asl/cifar_10', train=False, download=True)

        cifar10h_raw = pd.read_csv(r'./cifar10h-raw.csv')
        cifar10_test_idx = np.asarray(cifar10h_raw.iloc[0:, 8]).astype(int)
        _, unique_indices = np.unique(cifar10_test_idx, return_index=True)
        unique_indices = unique_indices[1:]
        cifar10h_unique = cifar10h_raw.iloc[unique_indices, :]
        noisy_labels = np.asarray(cifar10h_unique.iloc[:, 6]).astype(int)
        clean_labels = np.asarray(cifar10h_unique.iloc[:, 5]).astype(int)

        seed = get_seed()
        set_seeds(0)
        idx = np.random.permutation(len(dataset))
        noisy_labels = noisy_labels[idx]
        clean_labels = clean_labels[idx]
        images = dataset.data[idx]
        clean_dataset = CIFARDetection(images, clean_labels, transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]))
        noisy_dataset = CIFARDetection(images, noisy_labels, transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]))
        set_seeds(seed)
    # elif args.dataset_name == 'wordvision':
    #     transform = transforms.Compose([
    #         transforms.Resize((64, 64)),
    #         transforms.ToTensor(),
    #     ])
    #     test_image_path = './wordvision/test_images'
    #     test_labels_path = './wordvision/test_labels'
    #     test_dataset = WebVisionDetection(test_image_path, test_labels_path, transform=transform)
    #     cal_dataset = WebVisionDetection(test_image_path, test_labels_path, transform=transform)

    else:
        raise Exception("invalid data")
    print('Dataset loaded')

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    print(f"loading model from path: {args.model_path}")
    state = torch.load(args.model_path, map_location='cpu')
    try:
        model.load_state_dict(state, strict=True)
    except:
        model.load_state_dict(state['model'], strict=True)

    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = model.cuda().eval()
    model.eval()
    print('Model Loaded')
    corr = None  # get_correspondence(classes_list, cal_dataset.coco.cats)
    n_labels = 10 if args.dataset_name == 'cifar' else 80
    noisy_labels = get_scores_and_labels(model, noisy_dataset, corr, noise_type, noise_level,
                                         is_clean=False, n_labels=n_labels, compute_scores=False)
    scores, clean_labels = get_scores_and_labels(model, clean_dataset, corr, noise_type, noise_level,
                                                 is_clean=True, n_labels=n_labels, compute_scores=True,
                                                 )


    # get the loss and size table
    clean_example_loss_table, clean_example_size_table = get_example_loss_and_size_tables(scores, clean_labels,
                                                                                          lambdas_example_table,
                                                                                          num_calib,
                                                                                          noise_type, noise_level,
                                                                                          True)
    noisy_example_loss_table, noisy_example_size_table = get_example_loss_and_size_tables(scores, noisy_labels,
                                                                                          lambdas_example_table,
                                                                                          num_calib,
                                                                                          noise_type, noise_level,
                                                                                          False)

    for alpha in alphas:
        alpha = np.round(alpha, 3)
        local_df_list = []
        print(f"\n\n\n ============           NEW EXPERIMENT alpha={alpha}           ============ \n\n\n")

        fname = f'{cache_path}/{noise_type}_{noise_level}_{alpha}_{num_calib}_{num_lam}_{num_trials}_{model_trained_on_noisy}_dataframe.csv'
        os.makedirs(cache_path, exist_ok=True)

        if os.path.exists(fname):
            continue
        for i in tqdm(range(num_trials)):
            set_seeds(i)
            clean_risk, noisy_risk, clean_size, noisy_size, lhat = trial_precomputed(clean_example_loss_table,
                                                                                     clean_example_size_table,
                                                                                     noisy_example_loss_table,
                                                                                     noisy_example_size_table,
                                                                                     lambdas_example_table, alpha,
                                                                                     num_lam, num_calib, batch_size)
            dict_local = {"$\\hat{\\lambda}$": lhat,
                          "clean risk": clean_risk,
                          "noisy risk": noisy_risk,
                          "clean size": clean_size,
                          "noisy size": noisy_size,
                          "alpha": alpha
                          }
            df_local = pd.DataFrame(dict_local, index=[i])
            local_df_list = local_df_list + [df_local]
        df = pd.concat(local_df_list, axis=0, ignore_index=True)
        df.to_csv(fname)


def get_cache_path(args):
    noise_level = args.noise_level
    noise_type = args.noise_type
    dataset_name = args.dataset_name
    path = f'../.cache/dataset_name={dataset_name}/noise_type={noise_type}_noise_level={noise_level}_model_trained_on_noisy={args.model_trained_on_noisy}'
    return path


def main():
    import platform
    is_windows = 'windows' in platform.system().lower()
    noise_level = args.noise_level
    noise_type = args.noise_type
    if args.dataset_name == 'cifar':
        args.noise_level = None
        args.noise_type = None
        args.model_name = 'tresnet_m'
        args.num_classes = 10
        num_calib = 800
        args.model_path = f'../asl/models/cifar10/model-highest.ckpt'

    elif args.dataset_name == 'wordvision':
        args.noise_level = None
        args.noise_type = None
        args.model_name = 'tresnet_m'
        args.num_classes = 12
        num_calib = 2000
        args.model_path = f'../asl/models/wordvision/model-highest.ckpt'

    elif args.dataset_name == 'coco':
        args.model_name = 'tresnet_l'
        args.num_classes = 80
        num_calib = 10000
        if args.model_trained_on_noisy:
            args.model_path = f'../asl/models/noise_type={noise_type}_noise_level={noise_level}/model-highest.ckpt'
        else:
            # args.model_path = '../asl/models/MS_COCO_TRresNet_L_448_86.6.pth'
            args.model_path = '../asl/models/MS_COCO_TRresNet_L_448_86.6.pth'
    else:
        raise Exception("invalid dataset name")

    args.do_bottleneck_head = False
    cache_path = get_cache_path(args)
    pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
    num_trials = 50
    batch_size = 100

    with torch.no_grad():
        # alphas = [0.005, 0.01, 0.02, 0.03, 0.04] + list(np.arange(0.05, 0.25, 0.025))
        # num_lam = 2000
        # lambdas_example_table = np.linspace(0., 0.01, num_lam)  # np.array([0] + list(np.geomspace(1e-20, 0.0001, num_lam)))
        # experiment(alphas, num_lam, num_calib, lambdas_example_table, num_trials, batch_size, noise_type, noise_level)

        alphas = [0.005, 0.01, 0.02, 0.03, 0.04] + list(np.arange(0.05, 1.0001, 0.025))
        num_lam = 1500
        lambdas_example_table = np.array(list(np.linspace(0., 0.1, 1000)) + list(np.linspace(0.1, 1.1, 500)))
        experiment(alphas, num_lam, num_calib, lambdas_example_table, num_trials, batch_size, noise_type, noise_level, args.model_trained_on_noisy)


if __name__ == "__main__":
    args = parser.parse_args()
    args.model_trained_on_noisy = args.model_trained_on_noisy > 0
    main()
