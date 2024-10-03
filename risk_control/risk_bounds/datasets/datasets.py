"""
Part of the code is taken from https://github.com/yromano/cqr
"""
import abc
from abc import ABC
from argparse import Namespace

import torch
import numpy as np
import pandas as pd
import sys

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helper import set_seeds, get_current_seed
import re

from utils import DataScaler

sys.path.insert(1, '..')


def scale_data(x, y, seed, test_size=0.1):
    x_train, x_te, y_train, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed)
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_train, y_train, test_size=0.1, random_state=seed)

    s_tr_x = StandardScaler().fit(x_tr)
    s_tr_y = StandardScaler().fit(y_tr)

    x_tr = torch.Tensor(s_tr_x.transform(x_tr))
    x_va = torch.Tensor(s_tr_x.transform(x_va))
    x_te = torch.Tensor(s_tr_x.transform(x_te))

    y_tr = torch.Tensor(s_tr_y.transform(y_tr))
    y_va = torch.Tensor(s_tr_y.transform(y_va))
    y_te = torch.Tensor(s_tr_y.transform(y_te))
    y_al = torch.Tensor(s_tr_y.transform(y))

    x_train = torch.cat([x_tr, x_va], dim=0)
    y_train = torch.cat([y_tr, y_va], dim=0)
    out_namespace = Namespace(x_tr=x_tr, x_va=x_va, x_te=x_te,
                              y_tr=y_tr, y_va=y_va, y_te=y_te, y_al=y_al,
                              x_train=x_train, y_train=y_train)

    return out_namespace


class DataGeneratorFactory:
    possible_syn_dataset_names = [
        'extreme_bimodal', 'smooth_bimodal', ]

    @staticmethod
    def get_data_generator(dataset_name, is_real, args):
        if is_real:
            return DataGeneratorFactory.get_real_data_generator(dataset_name, args)
        else:
            return DataGeneratorFactory.get_syn_data_generator(dataset_name, args)

    @staticmethod
    def get_real_data_generator(dataset_name, args):
        if 'y_dim' in dataset_name:
            data_y_dim = int(re.search(r'\d+', re.search(r'y_dim_\d+', dataset_name).group()).group())
        else:
            data_y_dim = 1
        return RealDataGenerator(dataset_name, args, data_y_dim)

    @staticmethod
    def get_syn_data_generator(dataset_name, args):

        assert any(
            possible_dataset in dataset_name for possible_dataset in DataGeneratorFactory.possible_syn_dataset_names)
        if 'x_dim' in dataset_name:
            x_dim = int(re.search(r'\d+', re.search(r'x_dim_\d+', dataset_name).group()).group())
        else:
            x_dim = 1

        if 'extreme_bimodal' in dataset_name:
            return ExtremeBimodalDataGenerator(x_dim, args=args)
        elif 'smooth_bimodal' in dataset_name:
            return SmoothBimodalDataGenerator(x_dim, args=args)
        else:
            assert False


class DataGenerator(ABC):
    @abc.abstractmethod
    def __init__(self, args, y_dim):
        self.args = args
        self.y_dim = y_dim

    @abc.abstractmethod
    def undo_n_steps(self, data_info, n):
        pass

    @abc.abstractmethod
    def __generate_data_aux(self, T, x=None, previous_data_info=None, **kwargs):
        pass

    def generate_data(self, T, x=None, previous_data_info=None, **kwargs):
        x, y, curr_data_info = self.__generate_data_aux(T + self.y_dim - 1, x, previous_data_info, **kwargs)
        if self.y_dim > 1:
            y_mat = torch.zeros(T, self.y_dim, device=y.device)
            for i in range(T):
                y_mat[i] = y[i:i + self.y_dim]
            y = y_mat
            x = x[:T]
            curr_data_info = self.undo_n_steps(curr_data_info, n=self.y_dim - 1)
        return x, y, curr_data_info


def symmetric_low_noise(y):
    return y + torch.randn_like(y) * y.mean() / 10


def symmetric_medium_noise(y):
    return y + torch.randn_like(y) * y.mean() / 4


def symmetric_high_noise(y):
    return y + torch.randn_like(y) * y.mean() / 2


def upper_skewed_low_noise(y):
    m = torch.distributions.Beta(torch.FloatTensor([5]), torch.FloatTensor([2]))
    s = m.sample(y.shape).squeeze()
    s -= s.mean()

    return y + s * y.mean() / 8


def upper_skewed_high_noise(y):
    m = torch.distributions.Beta(torch.FloatTensor([5]), torch.FloatTensor([2]))
    s = m.sample(y.shape).squeeze()
    s -= s.mean()

    return y + s * y.mean() / 2


def lower_skewed_low_noise(y):
    m = torch.distributions.Beta(torch.FloatTensor([2]), torch.FloatTensor([5]))
    s = m.sample(y.shape).squeeze()
    s -= s.mean()
    return y + s * y.mean() / 8


def lower_skewed_high_noise(y):
    m = torch.distributions.Beta(torch.FloatTensor([2]), torch.FloatTensor([5]))
    s = m.sample(y.shape).squeeze()
    s -= s.mean()
    return y + s * y.mean() / 2


def small_shift_noise(y):
    return y + y.mean() / 8


def large_shift_noise(y):
    return y + y.mean() / 2


def uniform_noise(y, noise_magnitude=.1):
    noise = torch.rand_like(y)
    noise /= noise.std()
    noise *= y.std()
    return y + noise * noise_magnitude


def normal_noise(y, noise_magnitude=.1):
    noise = torch.randn_like(y)
    noise /= noise.std()
    noise *= y.std()
    return y + noise * noise_magnitude


"""
m = torch.distributions.Beta(torch.FloatTensor([5]), torch.FloatTensor([2]))
s = m.sample((10000,)) ; print(s.mean()) ; s -= s.mean()
a,b = np.histogram(s.squeeze()) ; plt.plot(b[:-1], a) ; plt.show()
"""


def get_noise_func(noise):
    if 'symmetric_low_noise' == noise:
        return symmetric_low_noise
    if 'symmetric_medium_noise' in noise:
        return symmetric_medium_noise
    if 'symmetric_high_noise' in noise:
        return symmetric_high_noise
    if 'upper_skewed_low_noise' in noise:
        return upper_skewed_low_noise
    if 'upper_skewed_high_noise' in noise:
        return upper_skewed_high_noise
    if 'lower_skewed_low_noise' in noise:
        return lower_skewed_low_noise
    if 'lower_skewed_high_noise' in noise:
        return lower_skewed_high_noise
    if 'small_shift_noise' in noise:
        return small_shift_noise
    if 'large_shift_noise' in noise:
        return large_shift_noise
    if 'uniform' in noise:
        return uniform_noise
    if 'normal' in noise:
        return normal_noise
    raise Exception("invalid noise")


class RealDataGenerator(DataGenerator):
    def __init__(self, dataset_name: str, args, y_dim):
        super().__init__(args, y_dim)
        self.dataset_name = dataset_name
        self.load_data()

    def load_data(self):
        dataset_name = self.dataset_name[self.dataset_name.index('noise') + len('noise') + 1:]
        self.x, self.real_y = GetDataset(dataset_name, 'datasets/real_data/')
        self.x = torch.Tensor(self.x)
        self.real_y = torch.Tensor(self.real_y)
        noise_func = get_noise_func(self.dataset_name)
        self.y = noise_func(self.real_y)
        self.max_data_size = self.x.shape[0]

    def _DataGenerator__generate_data_aux(self, T, x=None, previous_data_info=None, device='cpu'):

        if previous_data_info is None:
            starting_time = 0
        else:
            starting_time = previous_data_info['ending_time'] + 1

        current_process_info = {'ending_time': starting_time + T - 1}
        return self.x[starting_time: starting_time + T].cpu().to(device), \
               self.y[starting_time:starting_time + T].cpu().to(device), current_process_info

    def undo_n_steps(self, data_info, n):
        data_info = {'ending_time': data_info['ending_time'] - n}
        return data_info


class SynDataGenerator(ABC):
    def __init__(self):
        self.max_data_size = np.inf

    @abc.abstractmethod
    def generate_data(self, T, x=None, previous_data_info=None, n_samples=1, device='cpu',
                      current_process_info=None, use_constant_seed=True):
        pass

    @abc.abstractmethod
    def get_y_given_x_and_uncertainty(self, x, uncertainty, previous_data_info=None):
        pass

    @abc.abstractmethod
    def get_oracle_quantiles(self, x_test, alpha, previous_data_info=None, current_process_info=None):
        pass


class BimodalDataGenerator(SynDataGenerator, abc.ABC):
    def __init__(self, x_dim, shift, args=None):
        super().__init__()
        self.beta = torch.rand(x_dim)
        self.beta /= self.beta.norm(p=1)
        self.beta2 = torch.rand(x_dim)
        self.beta2 /= self.beta2.norm(p=1)
        self.x_dim = x_dim
        self.dataset_name = 'extreme_bimodal'
        self.noise = 'uniform' if args is None else args.noise
        self.shift = shift

    def generate_x(self, z, n_samples, device, previous_data_info=None):
        x = torch.rand(n_samples, self.x_dim, device=device)
        return x

    def generate_data(self, T=None, x=None, previous_data_info=None, n_samples=1, device='cpu',
                      current_process_info=None, use_constant_seed=True, add_noise=False):
        if x is None:
            x = self.generate_x(None, n_samples, device)
        z = torch.randn(n_samples, 3, device=device)
        clean_y = self.get_y_given_x_and_uncertainty(x, z)
        if add_noise:
            noise_func = get_noise_func(self.noise)
            noisy_y = noise_func(clean_y, np.sqrt(.1))
            return x, clean_y, noisy_y
        else:
            return x, clean_y

    def get_y_given_x_and_uncertainty(self, x, uncertainty, previous_data_info=None):
        var1 = torch.sin(x @ self.beta.to(x.device) * 2 * np.pi) + 2
        var2 = torch.sin(x @ self.beta2.to(x.device) * 2 * np.pi) + 2
        shift1_x_factor = abs(torch.sin(x @ self.beta.to(x.device) * 2 * np.pi)) / 8
        shift2_x_factor = abs(torch.sin(x @ self.beta2.to(x.device) * 2 * np.pi)) / 8
        uncertainty[:, 0] = var1 * uncertainty[:, 0] + self.shift / 2 * (shift1_x_factor + 1)
        uncertainty[:, 1] = var2 * uncertainty[:, 1] - self.shift / 2 * (shift2_x_factor + 1)
        y_type = (uncertainty[:, 2] > 0).float()
        y = y_type * uncertainty[:, 0] + (1 - y_type) * uncertainty[:, 1]
        return y

    def get_oracle_quantiles(self, x_test, alpha, previous_data_info=None, current_process_info=None):
        return NotImplementedError()


class SmoothBimodalDataGenerator(BimodalDataGenerator):
    def __init__(self, x_dim, args=None):
        shift = 7
        super().__init__(x_dim, shift, args)


class ExtremeBimodalDataGenerator(BimodalDataGenerator):
    def __init__(self, x_dim, args=None):
        shift = 125
        super().__init__(x_dim, shift, args)


class DataSet:

    def __init__(self, data_generator, is_real_data, device, T, test_ratio):
        self.data_generator = data_generator
        self.is_real_data = is_real_data
        self.device = device
        self.test_ratio = test_ratio
        self.generate_data(T)
        self.data_size = T

    def generate_data(self, T):
        validation_ratio = 0
        train_ratio = 1 - self.test_ratio - validation_ratio
        self.data_generator = self.data_generator
        self.is_real_data = self.is_real_data
        self.x_train, self.y_train, self.training_data_info = self.data_generator.generate_data(int(train_ratio * T),
                                                                                                device=self.device)
        if len(self.y_train.shape) == 1:
            self.y_train = self.y_train.unsqueeze(-1)

        self.data_scaler = DataScaler()
        self.data_scaler.initialize_scalers(self.x_train, self.y_train)
        # self.y_train = self.data_scaler.scale_y(self.y_train)

        if validation_ratio is not None and validation_ratio > 0:
            self.x_val, self.y_val, self.pre_test_data_info = self.data_generator.generate_data(
                int(validation_ratio * T),
                device=self.device,
                previous_data_info=self.training_data_info)
            self.y_val = self.y_val.unsqueeze(1)
            self.starting_test_time = self.x_train.shape[0] + self.x_val.shape[0]
            # self.y_val = self.data_scaler.scale_y(self.y_val)

        else:
            self.y_val = None
            self.pre_test_data_info = self.training_data_info
            self.starting_test_time = self.x_train.shape[0]

        self.x_test, self.y_test, self.test_data_info = self.data_generator.generate_data(int(self.test_ratio * T),
                                                                                          device=self.device,
                                                                                          previous_data_info=self.pre_test_data_info)
        if len(self.y_test.shape) == 1:
            self.y_test = self.y_test.unsqueeze(-1)

        if self.y_val is not None:
            all_y = torch.cat([self.y_train, self.y_val, self.y_test], dim=0)
        else:
            all_y = torch.cat([self.y_train, self.y_test], dim=0)

        all_y_scaled = self.data_scaler.scale_y(all_y.cpu())

        self.y_scaled_min = all_y_scaled.min().item()
        self.y_scaled_max = all_y_scaled.max().item()
        self.y_dim = self.y_train.shape[-1]
        self.x_dim = self.x_train.shape[-1]
        # self.y_test = self.data_scaler.scale_y(self.y_test)


def GetDataset(name, base_path):
    """ Load a dataset

    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"

    Returns
    -------
    X : features (nXp)
    y : labels (n)

	"""
    if name == 'energy':
        df = pd.read_csv(base_path + 'energy.csv')
        y = np.array(df['Appliances'])
        X = df.drop(['Appliances', 'date'], axis=1)
        date = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)

    if name == 'tetuan_power':
        df = pd.read_csv(base_path + 'tetuan_power.csv')
        y = np.array(df['Zone 1 Power Consumption'])
        X = df.drop(['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption', 'DateTime'],
                    axis=1)
        date = pd.to_datetime(df['DateTime'].apply(lambda datetime: datetime.replace(' 0:', ' 00:')),
                              format='%m/%d/%Y %H:%M')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)

    if name == 'traffic':
        df = pd.read_csv(base_path + 'traffic.csv')
        df['holiday'].replace(df['holiday'].unique(),
                              list(range(len(df['holiday'].unique()))), inplace=True)
        df['weather_description'].replace(df['weather_description'].unique(),
                                          list(range(len(df['weather_description'].unique()))), inplace=True)
        df['weather_main'].replace(['Clear', 'Haze', 'Mist', 'Fog', 'Clouds', 'Smoke', 'Drizzle', 'Rain', 'Squall',
                                    'Thunderstorm', 'Snow'],
                                   list(range(len(df['weather_main'].unique()))), inplace=True)
        y = np.array(df['traffic_volume'])
        X = df.drop(['date_time', 'traffic_volume'], axis=1)
        date = pd.to_datetime(df['date_time'].apply(lambda datetime: datetime.replace(' 0:', ' 00:')),
                              format='%Y-%m-%d %H:%M:%S')
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        # X['minute'] = date.dt.minute
        X['day_of_week'] = date.dt.dayofweek

        X = np.array(X)

    if name == 'wind':
        df = pd.read_csv(base_path + 'wind_power.csv')
        date = pd.to_datetime(df['dt'], format='%Y-%m-%d %H:%M:%S')
        X = df.drop(['dt', 'MW'], axis=1)
        y = np.array(df['MW'])[1:]
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['minute'] = date.dt.minute
        X['hour'] = date.dt.hour
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)[:-1]

    if name == 'prices':
        df = pd.read_csv(base_path + 'Prices_2016_2019_extract.csv')
        # 15/01/2016  4:00:00
        date = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
        X = df.drop(['Date', 'Spot', 'hour'], axis=1)
        y = np.array(df['Spot'])[1:]
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X['hour'] = date.dt.hour
        X['day_of_week'] = date.dt.dayofweek
        X = np.array(X)[:-1]

    try:
        X = X.astype(np.float32)
        y = y.astype(np.float32)

    except Exception as e:
        raise Exception("invalid dataset")

    return X, y


def data_train_test_split(Y, X=None, device='cpu', test_ratio=0.2, val_ratio=0.2,
                          calibration_ratio=0., seed=0, scale=False, dim_to_reduce=None, is_real=True, y_prefix=''):
    data = {}
    is_conditional = X is not None
    if X is not None:
        X = X.cpu()
    Y = Y.cpu()

    y_names = [f'{y_prefix}y_train', f'{y_prefix}y_val', f'{y_prefix}y_te']
    if is_conditional:
        x_names = ['x_train', 'x_val', 'x_test']

        x_train, x_test, y_train, y_te = train_test_split(X, Y, test_size=test_ratio, random_state=seed)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_ratio, random_state=seed)

        if calibration_ratio > 0:
            x_train, x_cal, y_train, y_cal = train_test_split(x_train, y_train, test_size=calibration_ratio,
                                                              random_state=seed)
            data['x_cal'] = x_cal
            data[f'{y_prefix}y_cal'] = y_cal
            x_names += ['x_cal']
            y_names += [f'{y_prefix}y_cal']

        data['x_train'] = x_train
        data['x_val'] = x_val
        data['x_test'] = x_test

        if scale:
            s_tr_x = StandardScaler().fit(x_train)
            data['s_tr_x'] = s_tr_x
            for x in x_names:
                data[f"untransformed_{x}"] = data[x]
                data[x] = torch.Tensor(s_tr_x.transform(data[x]))

        if (is_real and x_train.shape[1] > 70) or (dim_to_reduce is not None and x_train.shape[1] > dim_to_reduce):
            if dim_to_reduce is None:
                n_components = 50 if x_train.shape[1] < 150 else 100
            else:
                n_components = dim_to_reduce
            pca = PCA(n_components=n_components)
            pca.fit(data['x_train'])
            for x in x_names:
                data[x] = torch.Tensor(pca.transform(data[x].numpy()))

        for x in x_names:
            data[x] = data[x].to(device)

    else:
        y_train, y_te = train_test_split(Y, test_size=test_ratio, random_state=seed)
        y_train, y_val = train_test_split(y_train, test_size=val_ratio, random_state=seed)
        if calibration_ratio > 0:
            y_train, y_cal = train_test_split(y_train, test_size=calibration_ratio, random_state=seed)
            y_names += [f'{y_prefix}y_cal']
            data[f'{y_prefix}y_cal'] = y_cal

    data[f'{y_prefix}y_train'] = y_train
    data[f'{y_prefix}y_val'] = y_val
    data[f'{y_prefix}y_te'] = y_te

    if scale:
        s_tr_y = StandardScaler().fit(y_train)
        data[f'{y_prefix}s_tr_y'] = s_tr_y

        for y in y_names:
            data[f"{y_prefix}untransformed_{y}"] = data[y]
            data[y] = torch.Tensor(s_tr_y.transform(data[y]))

    for y in y_names:
        data[y] = data[y].to(device)

    return data
